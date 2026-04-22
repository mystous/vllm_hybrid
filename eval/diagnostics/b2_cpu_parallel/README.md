# B2 CPU Parallelism Diagnosis — 3-Phase 검증 키트

## 문제 (실측 요약)

Qwen2.5-32B × 16K/16K heavy workload 의 hybrid seqs=1 run 에서:

| | HEAVY seqs=1 | LIGHT seqs=1 |
|---|---|---|
| mean > 50% 인 코어 | **2개 (cpu0, cpu56)** | **60개** |
| mean < 10% 인 코어 | **94개** | **0개** |
| Top cores mean util | cpu0: **99.4%**, cpu56: **99.3%** | 60개가 94~99% |

즉 heavy 는 engine 당 1 master core 만 실행, 나머지 94 코어 idle. Light 는 정상 full utilization. **workload-dependent single-thread 화**.

## 가설 3종

| 가설 | 설명 |
|---|---|
| **A** | IPEX paged attention 이 long-ctx 에서 single-thread path 로 fallback |
| **B** | Python-level critical section (block table walk, KV index) 에서 GIL serialize |
| **C** | `sdpa_loop` / 유사 fallback path 가 선택되어 per-seq 순회 |

## 3-Phase 검증 워크플로우

### Phase 1 — 정적 코드 분석 (서버 불필요, 즉시)

cpu_attn.py 의 dispatch tree 를 소스에서 추출. heavy input shape
(`batch=1, num_tokens=1, num_kv_heads=8, ctx_len=16384`) 이 어느 path 를
타는지 조건 읽기.

```bash
python eval/diagnostics/b2_cpu_parallel/phase1_dispatch_static.py
```

**출력**: 각 `_trace_decode_path` 호출 site 의 gating 조건 + IPEX entry 조건.
이것만으로 C (fallback) 가 확정 or 배제되는 경우 많음.

---

### Phase 2 — TRACE counter 실측 (새 run 필요, 10~20분)

`VLLM_HYBRID_TRACE=1` 로 decode call 마다 path 를 로그에 찍음. 4 prompt × 256 output 으로 단축해서 빠르게.

```bash
bash eval/diagnostics/b2_cpu_parallel/phase2_run_trace.sh
```

**출력**: `[HYBRID-CPU-ATTN] totals={'custom_avx': a, 'ipex': b, 'sdpa_batched': c, 'sdpa_loop': d}`

이 숫자로:
- `sdpa_loop` dominant → C 확정
- `ipex` dominant 이면서 여전히 느림 → A 확정
- `custom_avx` dominant → custom_avx 커널 자체 parallelism 문제

---

### Phase 3 — stuck 프로세스 실시간 introspection

heavy 가 돌고 있는 상태에서 병렬로 실행. py-spy / perf / thread state 를
한 번에 캡처.

```bash
bash eval/diagnostics/b2_cpu_parallel/phase3_live_introspect.sh
```

의존성:
- `py-spy` (pip install py-spy)
- `perf` (apt install linux-tools-common linux-tools-generic)

**출력**: `snapshots/<ts>/` 에 engine 별 3개 파일 + `summary.md`.

판정:
- py-spy stack 이 Python attention 함수에 잡힘 → **B**
- py-spy stack 이 `<native>` + perf 가 `ipex_*_paged_attention` → **A**
- py-spy stack 이 `torch::sdpa` → **C**

---

## 권장 순서

**Phase 1 → 3 → 2** — 1 은 비용 0, 3 은 stuck 프로세스 살아 있으면 즉시,
2 는 10~20분 재실행 필요. 1+3 으로 대부분 판가름 남.

## 결과 해석 후 다음 조치

| 확정된 가설 | 조치 |
|---|---|
| **A** (IPEX fallback) | IPEX 버전 업그레이드 또는 `HYBRID_BATCH_AWARE_ATTN` flag 로 직접 구현 path 로 강제 |
| **B** (Python GIL) | Block table walk / KV indexing 을 C++ extension 으로 이전 |
| **C** (sdpa_loop) | dispatch 조건 수정 (num_tokens=1 case 를 ipex 로 강제 routing) |

## 파일 목록

| 파일 | 역할 |
|---|---|
| `README.md` | 이 문서 |
| `phase1_dispatch_static.py` | cpu_attn.py 정적 분석 |
| `g0_h100x8_qwen32b_longctx_trace.env` | Phase 2 run 용 env |
| `phase2_run_trace.sh` | Phase 2 orchestrator (env 로드 + run + 요약) |
| `phase3_live_introspect.sh` | Phase 3 live capture |
| `snapshots/` | Phase 3 출력 (gitignore 권장) |
