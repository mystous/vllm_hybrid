# IDE_025 — Parallelism · Precision · Comm Remap (신규 방법군)

> **status**: 활성 (2026-06-11 신설)
> **parent**: `TSK_020/SUB_072`
> **자식 TSK**: `TSK_047`
> **동기**: 기존 방법군 (spec-decode 변형 / cudagraph 모드 / DSA·AMX 오프로드 / KV tiering /
> oracle 라우팅) 의 실패 위험이 높다는 판단 → **전 sweep 이 단 한 번도 움직이지 않은 축**을 재조사.

---

## 0. 결정적 관찰 — 지금까지 모든 측정이 고정해 온 것

SUB_212 / TSK_042 의 **모든 420+ 셀**이 다음을 고정했다 (boot log 확인, 2026-06-11):

```
tensor_parallel_size=8, data_parallel_size=1, decode_context_parallel_size=1,
kv_cache_dtype=auto(bf16), enable_dbo 미사용, pass_config.enable_sp=False,
fuse_gemm_comms=False, compile_sizes=[]
```

그리고 본 빌드 (v1.7.dev16107) 에는 이 축들이 **전부 구현되어 있다**:

| 기능 | 코드 위치 | 상태 |
|---|---|---|
| Data Parallel + hybrid/external LB | `arg_utils.py:430,934-980` (`-dp`, `--data-parallel-hybrid-lb`) | 미사용 |
| DBO (dual-batch overlap, comm·compute 중첩) | `arg_utils.py:996` (`--enable-dbo`), `v1/worker/ubatching.py`, decode threshold 32 | 미사용 |
| Sequence-parallel + comm fusion pass | `config/compilation.py:129` (`pass_config.enable_sp`), `fuse_gemm_comms` | OFF |
| FP8 KV cache | `--kv-cache-dtype fp8` (+ `--kv-cache-dtype-skip-layers`) | auto(bf16) |
| Decode Context Parallel | `--decode-context-parallel-size` (dcp_comm_backend=ag_rs) | =1 |
| P/D 분리 (NIXL/offloading connector) | `distributed/kv_transfer/kv_connector/v1/nixl/` | 미사용 |

**왜 이것이 1순위인가** — SUB_201 프로파일이 이미 답을 줬다:
70B TP=8 의 시간 80% 가 memcpy(allreduce 경로), 7B 는 launch 36%.
즉 **병목의 대부분이 "모델 계산"이 아니라 "TP=8 이라는 분해 방식"이 만든 통신·런치 비용**이다.
기존 lever 들은 이 비용을 우회하려 했고, 신규 방법군은 **비용의 원인 자체를 제거**한다.

### 핵심 수치 (strong-scaling 관점)

| 모델 | 현 구성 | cluster tps | per-GPU tps | 이론 진단 |
|---|---|---:|---:|---|
| Llama-8B (16GB wt) | TP=8 | 27,851 (suf) | **3,481** | 8-way 분해가 통신 지배 — 1 GPU(192GB) 에 충분히 적재 가능 |
| Qwen-32B | TP=8 | (SUB_212) | — | TP=2~4 면 충분, 나머지는 DP 로 |
| Llama-70B | TP=8 | (SUB_212) | — | memcpy 80% — TP=4×DP2 또는 SP/DBO 로 중첩 |

---

## 1. 신규 방법군 카탈로그 (기존 시도 전면 제외)

| # | 방법 | 출처 분야 | 메커니즘 | 출력 등가 | 비용 | 실패 위험 |
|---|---|---|---|---|---|---|
| **N1** | **TP→DP/하이브리드 재매핑** | HPC strong-scaling / processor allocation (Amdahl·Gustafson) | 작은 모델의 TP 통신 제거, GPU 당 독립 replica | ✅ 수치 경로 동일 | 플래그 | **낮음** (업계 표준 관행) |
| **N2** | **FP8 KV cache** | 정밀도-대역폭 트레이드 (mixed precision, HPC) | decode 의 KV 읽기 대역폭 ½, KV 용량 2× | ⚠️ 분포 게이트 필요 | 플래그 | 중 (게이트 탈락 가능) |
| **N3** | **DBO — 마이크로배치 comm·compute 중첩** | double buffering / software pipelining (HPC) | batch 를 2 ubatch 로 쪼개 allreduce 와 GEMM 중첩 | ✅ | 플래그 | 중 (decode 소배치 효과 의문, threshold 32) |
| **N4** | **enable_sp + comm fusion pass** | communication-avoiding (Demmel) / async-TP (Flux) | allreduce → reduce-scatter+all-gather 분해 후 GEMM 와 융합 | ✅ | 플래그 | 중 |
| **N5** | **DCP (decode context parallel)** | 도메인 분해 (HPC) | 장문 decode 의 KV 를 rank 분할 — KV-heavy 전용 | ✅ | 플래그 | 중 |
| **N6** | **inductor compile_sizes + autotune** | 커널 단위 autotuning (ATLAS/FFTW) | 고정 decode shape (288/512) 에 커널 특화 | ✅ | 플래그+컴파일시간 | 낮음 (이득 소폭) |
| **N7** | **P/D 분리 (NIXL, in-node)** | DistServe/Splitwise (2024 serving) | prefill·decode 상호 간섭 제거, 단계별 최적 병렬화 | ✅ | 구조 변경 (중) | 중~높음 |
| **N8** | **호스트 코드 PGO/BOLT + 핫루프 네이티브화** | SE: profile-guided optimization | host-bound 경로 (scheduler/input-prep) 의 IPC 개선 | ✅ | 빌드 작업 (중) | 낮음 (이득 5~15% host 한정) |
| **N9** | **MLFQ 선점 스케줄링** | 큐잉 이론 / FastServe | 길이 예측 없이 long-job 강등 — p99 개선 | ✅ (순서만 변경) | 구현 (중) | 중 (tput 중립) |
| **N10** | **GPU clock lock + NCCL/IRQ steering 번들** | 시스템 튜닝 | DVFS 진동 제거, NCCL 스레드 NUMA 고정 | ✅ | 설정 (root) | 낮음 (이득 소폭) |
| N11 | persistent megakernel / CUDA graph conditional node | GPU 런타임 연구 (2024-25) | launch 경로 원천 제거, 가변 길이 spec 을 graph 내 분기로 | ✅ | **대수술** | 높음 (Tier-3 연구) |

기각 (검토 후): 압축 allreduce (분포 등가 위험+NVLink5 에서 무익), token dropping/H2O (출력 변경),
IAA/QAT 계열 (HW 부재 — IDE_024 §1.1).

## 2. 왜 이 방법군의 성공 확률이 더 높은가

1. **검증된 외부 실적**: N1(DP for small models)·N2(FP8 KV)·N3/N4(comm overlap) 는
   업스트림 vLLM/SGLang/TensorRT-LLM 프로덕션에서 상시 사용되는 기법 — 본 환경 특수성에
   기대지 않음. 기존 lever 들(DSA·AMX·tiering)은 특정 regime 가정이 필요했고 그 가정이 반복 기각됨.
2. **병목의 원인 제거 vs 우회**: 프로파일이 지목한 비용(TP 통신·launch)을 직접 없앤다.
3. **수치 경로 불변** (N2 제외): 정확도 게이트 리스크가 구조적으로 없음.
4. **측정 비용 저렴**: N1~N6 은 전부 부팅 플래그 — 기존 harness 그대로 1 sweep.

## 3. 공정 비교 원칙 (TSK_047 설계 핵심)

- DP=N 비교 시 **총 부하 동일** (conc 32 → 32×N 또는 부하 발생기 고정 RPS).
  Objective 가 cluster throughput 이므로 per-replica underfill 로 DP 가 불리해지는
  잘못된 비교를 금지.
- 모든 셀 cudagraph_mode **명시** 부팅 (SUB_212 confounder 교훈).
- FP8 KV 셀만 분포 유사성 게이트 (greedy seed 고정, per-token logprob diff) 추가.

## 4. CPU Objective 와의 접점

- DP=8 이면 EngineCore ×8 — scheduler/detok/tokenizer 가 8 프로세스로 늘어나
  유휴 224 thread 를 자연 소비 (단일 TP=8 인스턴스의 GIL-bound 한계 해소).
- N8 (PGO/BOLT) 은 CPU 효율 자체를 올려 같은 CPU 로 더 많은 replica 지탱.

## 5. 산출물

- `task.md` — TSK_047 sweep 설계 + N7~N11 후속
- `test.md` — 게이트
- `sweep_remap.sh` — (작성 예정) TP×DP×{fp8,dbo,sp} 매트릭스 runner
