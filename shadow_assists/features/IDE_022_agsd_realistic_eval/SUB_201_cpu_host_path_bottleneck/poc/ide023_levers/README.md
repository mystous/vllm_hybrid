# IDE_023 — 13 New Lever PoC (SUB_201)

## 목적

IDE_023 에서 도출한 13 개 신규 lever 의 **net-throughput 영향** 을 단일 모델 / 단일 워크로드로 빠르게 가늠하는 1차 PoC.

- 모델 : `meta-llama/Llama-3.1-8B-Instruct` (32 heads → TP=8 정합)
- HW   : B200 ×8, Xeon Platinum 8570 (224 thread, 2 NUMA, AVX-512 + AMX native)
- 워크로드 : sharegpt 200p × conc=16 × max-tok=512
- baseline = **Optimal Config** = vanilla + cudagraph `FULL_AND_PIECEWISE` (B3 FaP) + `VLLM_PREFETCH_TOKENIZE=1` (L2) + `VLLM_BURST_AWARE_ADMISSION=1` (L10)
- 각 lever 는 `VLLM_LEVER_N{X}=1` 환경변수로 ON, default OFF (regression 보호)

## 13 lever

| ID | Lever | 위치 | 외부 의존 |
|---|---|---|---|
| N1 | AVX-512 BPE encode | tiktoken warmup + hint env | tiktoken 0.12.0 (OK) |
| N4 | SoA paged attention layout | KV tile bytes env hint (4MiB) | 없음 |
| N5 | SMT-pair pinning scheduler | sched_setaffinity to SMT pairs | 없음 |
| N6 | Lock-free priority queue | scheduler deque-path hint | 없음 |
| N7 | Huge pages 2MB for KV | madvise(MADV_HUGEPAGE) hook | libc (OK) |
| N8 | NUMA-local draft state | numa_run_on_node(local) | libnuma.so.1 (OK) |
| N9 | DSA memcpy host↔pinned | /dev/dsa/wq* probe | **N/A — WQ 미설정** |
| N10 | AVX-512 simdjson parse | json.loads → simdjson | pysimdjson 7.0.2 (OK) |
| N11 | AVX-512 base64 streaming | base64 → pybase64 hook | pybase64 1.4.3 (OK) |
| N14 | Prefetch suffix tree | ARCTIC_SUFFIX_PREFETCH=1 | 없음 |
| N17 | CMT-driven priority | /sys/fs/resctrl 활성 확인 | resctrl visible (entries=0) |
| N19 | AVX-512 SSE writer | 핸들러 hint env | 없음 |
| N20 | LogGP admission | scheduler cost-aware hint env | 없음 |

> 일부 lever 는 "**hint env 만 설정**" 인 단순화 PoC 이다 (단일 모델 단발 측정으로
> 도입 가치를 가린다). hint 만으로 의미 있는 Δ% 가 나오는 lever 만 다음
> 단계에서 깊은 구현으로 승격한다 (production gate).

## 구조

```
ide023_levers/
├── README.md             # 본 문서
├── SUMMARY.md            # 사후 결과 (aggregate.py 출력)
├── results.csv           # 사후 결과 (csv)
├── sharegpt200.parquet   # bench corpus (B3 8GPU 와 동일)
├── scripts/
│   ├── sweep.sh          # 14 boot (baseline + 13 lever) 순차 driver
│   └── aggregate.py      # runs/*.json → SUMMARY.md / results.csv
├── runs/                 # 각 tag 당 1개의 throughput summary json
└── logs/                 # boot/bench log
```

## 패치

vllm 본체 수정 :

1. `vllm/envs.py`
   - dataclass: `VLLM_LEVER_N{1,4,5,6,7,8,9,10,11,14,17,19,20}: bool = False`
   - environment_variables: 13 개 lambda 등록
2. `vllm/v1/spec_decode/ide023_levers.py` (신규)
   - 각 lever 의 `_apply_n*()` 함수 + `apply_ide023_levers()` 엔트리
   - 환경 미지원 lever (DSA / numa lib / resctrl) 는 자동 N/A 처리, boot 실패 없음
3. `vllm/v1/engine/core.py`
   - `EngineCore.__init__` 의 plugin load 직후 `apply_ide023_levers()` 호출

## 실행

```bash
cd /workspace/host_vllm_hybrid/shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/ide023_levers
# 전체 sweep (14 boot)
nohup bash scripts/sweep.sh > logs/sweep.log 2>&1 &
# 결과 집계
/workspace/vllm_dev_prj/bin/python scripts/aggregate.py
cat SUMMARY.md
```

## 정확도 게이트

본 PoC 는 **throughput-only** PoC 이다. 13 lever 중 어느 것도 GPU 산술 경로를
바꾸지 않으므로 (모두 host / scheduler / I/O lever) GPU↔비-GPU equiv 검증은
별도 단계로 분리한다. net-positive 판정된 lever 만 IDE_006/TST_003 protocol
(분포 유사성 : per-token logprob max abs diff + 시퀀스 PPL relative diff) 의
후속 검증으로 진입한다.

## 정의 / 제약

- 모든 vllm 명령 prefix : `LD_LIBRARY_PATH=/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib`
- TP=8 강제 (Llama-3.1-8B 32 heads / 8 = 4 OK)
- Δ% ≥ +3% (noise floor) 인 lever만 production-ready 권고에 포함
- 각 lever 6 시간 박스, 어떤 단계라도 막히면 다음 lever 로 즉시 이동
