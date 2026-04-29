**↑ 부모**: [`PLN_001`](PLN_001.md) · **연계 TSK**: [`TSK_002`](TSK_002.md) §4.6 / [`TSK_004`](TSK_004.md) · **↟ 조부**: [`IDE_006`](README.md)

---

# PLN_001 / TSK_002 §4.6 — overlap 작업 회귀 fix log

| 항목 | 값 |
|---|---|
| 산출물 ID | (PLN-deliverable, ID 미부여) |
| 부모 PLN | [`PLN_001`](PLN_001.md) |
| 연계 TSK | [`TSK_002`](TSK_002.md) §4.6 (async stream + Q chunk pipelining) / [`TSK_004`](TSK_004.md) (NUMA-aware) |
| 산출 시점 | 2026-04-27 |
| 목적 | TSK_002 §4.6 (overlap) 진입 직전·직후에 발견된 prod 회귀 / 성능 격차의 진단·fix 흐름을 시간순으로 기록. TSK_002 / TSK_004 의 단순 "구현 완료" 표기로 묻히는 추적 정보를 남겨 향후 동일 회귀 재발 시 첫 의심 지점이 되도록 함 |
| 입력 데이터 | `eval/results/20260427_*_simd_verify` 와 `_cold_verify` 회차의 stderr/stdout, monitor.csv, bench.json |

---

## 1. 배경 — 왜 이 노트가 필요한가

[`TSK_002`](TSK_002.md) §4.6 (async stream + Q chunk pipelining) 진입 게이트는 [`PLN_001`](PLN_001.md) §4.3 의 overlap 부등식

```
T_Q_transfer + T_CPU_partial + T_partial_transfer  ≤  T_GPU_hot_attn + ε
```

이 성립 가능한지 측정하는 것이다. 그런데 prod 머신에서 IDE_006 ON 으로 전환하자마자 이 부등식 측정 *이전 단계* 에서 다음 회귀들이 차례로 노출됐다:

1. **libgomp `pthread_create` EAGAIN** — engine 부팅 시점에 worker 가 죽음
2. **wrapper 5462 ms / call timeout** — sample_tokens RPC 가 5 분 안에 못 돌아옴
3. **device-mismatch RuntimeError** — prod 의 mixed device topology (query=cuda, query_positions=cpu) 에서 100/100 fail
4. **batch 전체 폭 D2H/alloc/H2D** — cold 가 필요한 row 가 16384 중 1 개여도 32 MB 메모리 이동 비용 매 layer
5. **GPU/CPU sequential 실행** — overlap 을 의도한 §4.6 가 미구현 상태라 baseline 대비 5.6× slowdown

본 노트는 (1)~(5) 의 진단·fix 흐름을 시간순으로 정리한다. 각 단계의 root cause / fix / 측정 데이터 / 영향받은 코드를 함께 둔다.

---

## 2. 회귀 fix 흐름

### 2.1 · libgomp pthread_create EAGAIN — TSK_004 (b') 로 흡수

| 항목 | 값 |
|---|---|
| 발생 시점 | prod TP=8, 모든 worker 가 같은 NUMA node 에 affinity 박혔을 때 |
| 증상 | engine 부팅 직후 `libgomp: Thread creation failed: Resource temporarily unavailable` → EngineDeadError |
| Root cause | `pin_threads_to_local_numa()` 가 worker 마다 노드 cpulist 전체 (~56 코어) 를 affinity 로 박음. C++ helper 의 `vllm_partial_attn_thread_count()` 가 `sched_getaffinity` 결과로 56 을 받아 `num_threads(56)` 를 요청. 한 node 에 4 worker 가 묶이면 4 × 56 = 224 thread 를 56 코어에 spawn 시도 → `RLIMIT_NPROC` / `kernel.threads-max` 한도 초과 |
| Fix | `numa_aware.py:_partition_node_cpus_for_rank()` — 같은 node 에 묶인 rank 들을 round-robin 으로 식별, 노드 cpulist 를 worker 수만큼 균등 분할. `pin_threads_to_local_numa` 가 슬라이스를 affinity 로 사용 |
| 측정 | prod 회차 `eval/results/20260427_030536_*_simd_verify`: EAGAIN 사라짐. `rank=N pinned threads to NUMA node M (28/112 cores: A~B)` × 8 worker. |
| 코드 | [`vllm/distributed/kv_transfer/kv_connector/v1/offloading/numa_aware.py`](../../../vllm/distributed/kv_transfer/kv_connector/v1/offloading/numa_aware.py) |
| 추적 | [`TSK_004`](TSK_004.md) §1.2 (b') 항 |

### 2.2 · wrapper 5462 ms / call timeout — TSK_004 (b''') 로 흡수

| 항목 | 값 |
|---|---|
| 발생 시점 | (2.1) 해소 후, prod 가 실 e2e 워크로드에 진입하면 sample_tokens RPC 가 5 분 timeout 으로 EngineDeadError |
| 증상 | 5 분 후 timeout. 그 사이 worker 는 살아있지만 forward 가 진척 안 함 |
| 진단 데이터 | `VLLM_PARTIAL_ATTN_PROFILE=1` 으로 단면 측정. stage=hot_cold 평균 **kernel_ms = 5462 ms**, 같은 호출 안의 C++ AMX 함수 평균 **9.4 ms**. 격차 5.45 초가 모두 Python wrapper 에서 새고 있음 |
| Root cause | `forward_partial_with_lse` 가 매 호출마다 `select_isa_path()` 재평가. 그 내부의 `_has_amx()` / `_has_avx512()` 가 `cpuinfo.get_cpu_info()` 를 매번 새로 실행 (py-cpuinfo 는 호출당 1~5 초 소요 — /proc/cpuinfo 파싱 + 일부 플랫폼에선 cpuid subprocess). TP=8 / 70B / 80 layer = forward 한 번당 8 × 80 = 640 회 dispatch → cpuid 누적 비용이 호출당 5+ 초로 발현 |
| Fix | (1) `_read_cpu_flags_once()` — 함수-속성 캐시. cpuinfo 1 회만 실행. (2) `select_isa_path._cache` — ISA path 결과도 1 회만 결정. (3) `_pin_fast_done` 모듈-level bool — `pin_threads_to_local_numa` 의 hot path 에서 Lock 진입까지 단축 |
| 측정 | prod 회차 `eval/results/20260427_044407_*_simd_verify`: stage=hot_cold kernel_ms median **12.6 ms** (이전 5171 ms 에서 410× 감소). gap (outer − inner) median **-0.03 ms** (사실상 0). engine 정상 종료 — `8 prompts in 17.4s` |
| 코드 | [`vllm/v1/attention/ops/cpu_partial_attention.py`](../../../vllm/v1/attention/ops/cpu_partial_attention.py) |
| 추적 | [`TSK_004`](TSK_004.md) §1.2 (b''') 항 |

### 2.3 · device-mismatch RuntimeError — TSK_002 §4.6 진입 직전 회귀

| 항목 | 값 |
|---|---|
| 발생 시점 | (2.4) per-seq 필터링 추가 후 첫 prod run.sh 회차에서 100 prompt 중 99 fail |
| 증상 | `RuntimeError: Expected all tensors to be on the same device, but got index is on cuda:5, different from other tensors on cpu (when checking argument in method wrapper_CUDA__index_select)` 첫 layer 호출 시점에 worker 죽음 → 모든 후속 request 가 EngineDeadError |
| Root cause | per-seq 필터의 index_select 코드가 query 가 GPU 면 query_positions 도 같은 device 라 가정하고 GPU index 텐서를 사용. 그런데 prod 의 dispatcher 는 `query_positions` 를 host 측에서 빌드해 CPU 텐서로 넘김. dev 의 모든 hot_cold_attention 테스트가 *모든 입력을 GPU 에 두는* case 만 커버해서 회귀가 통과해 push 됨 |
| Fix | `_index_rows_to_cpu(src)` / `_index_seqs_to_cpu(src)` helper 로 각 텐서의 실제 device 를 보고 같은 device 의 index 텐서를 사용. CPU/GPU 임의 조합 안전 |
| 회귀 테스트 | `tests/v1/cpu_partial_attention/test_hot_cold_attention_phase3b.py:test_hot_cold_split_mixed_device_inputs` — prod topology (query=cuda, query_positions=cpu, cold_block_ids=cpu) 그대로 재현. fix 전 코드에서는 RuntimeError, fix 후 통과 |
| 측정 | prod 회차 `eval/results/20260427_064134_*_simd_verify`: 8/8 success (`8 prompts in 14.4s`), 회귀 없음 |
| 코드 | [`vllm/v1/attention/backends/flash_attn.py`](../../../vllm/v1/attention/backends/flash_attn.py) `hot_cold_attention()` |

### 2.4 · per-seq 필터링 — TSK_002 §4.6 진척의 일부

| 항목 | 값 |
|---|---|
| 발생 시점 | (2.2) wrapper 캐싱으로 timeout 해소 후, prod 데이터에서 호출당 D2H/H2D 가 32 MB 인 패턴 확인 |
| 증상 | cold-block 보유 seq 가 1 개 (`cbl=[630, 0, 0, 0, 0]`) 여도 16384 row 모두에 대해 D2H 32 MB / 커널 alloc 32 MB / H2D 32 MB / merge 비용 지불. layer × worker × step 마다 누적 |
| 분석 | C++ 커널 안의 `if (n_cold_blocks <= 0) continue;` 는 *연산만* skip 함. *I/O 와 텐서 alloc 은 batch 전체 폭* 그대로 |
| Fix | `hot_cold_attention` 의 cold-path 진입부에서 host-side 마스크 (`num_cold_blocks > 0`) 로 cold-needing seq 만 추려, 그 seq 의 query row 만 `index_select` 후 CPU 로 D2H. 커널은 reduced batch 만 처리. 결과는 GPU 에서 `index_copy_` 로 원래 위치에 scatter, 나머지 row 의 cold_lse 는 -inf 로 두어 merge 가 hot 만 채택 |
| 의도된 효과 | D2H 32 MB → reduced row 만 (예: 1 row × 1 KB), 커널 alloc 도 reduced 사이즈, H2D 도 동일. cbl=[630,0,0,0,0] 케이스에선 99.99% 의 메모리 이동이 사라짐 |
| 측정 | prod 회차 `eval/results/20260427_070242_*_cold_verify` (NUM_PROMPTS=50): cold-path firing log 가 `reduced_rows=1/16384` 로 의도대로 동작. `[IDE_006/TSK_004 cold-path fired pid=N] #1/5 need_cold_seqs=1/5 reduced_rows=1/16384 max_cold_blocks=630` × 8 worker × 5 회 |
| 코드 | [`vllm/v1/attention/backends/flash_attn.py`](../../../vllm/v1/attention/backends/flash_attn.py) `hot_cold_attention()` Step 2~3 |

### 2.5 · GPU/CPU sequential 실행 — TSK_002 §4.6 의 본 작업

| 항목 | 값 |
|---|---|
| 발생 시점 | (2.4) 까지 끝낸 후 prod cold_verify 회차에서 baseline 과 throughput 비교 |
| 증상 | 50 prompts 처리 시간 IDE_006 ON = 190.7 s vs baseline ≈ 34 s (env 환산) → **5.6× slow**. CPU util 평균 30.9%, GPU util 평균 29.8% — 서로 idle 패턴 (sequential 실행의 시그너처) |
| 분석 | hot path FA 가 default stream 위에서 실행 → `query.index_select(...).cpu()` 가 default stream 동기화 강제 → cold 작업이 hot 끝난 *후* 시작. CPU 가 일하는 동안 GPU 도 idle, 그 반대도. PLN_001 §4.3 가 가정한 overlap 미작동 |
| Fix (1차 — stream 분리) | `_get_cold_path_stream(device)` — 디바이스 인덱스별 dedicated CUDA Stream 캐시. `hot_cold_attention` 의 cold-path GPU 영역을 `with torch.cuda.stream(cold_stream)` 으로 감쌈. hot path FA 는 default stream 그대로. merge 직전 `current_stream(device).wait_stream(cold_stream)` 으로 event-based 대기. opt-out env `VLLM_COLD_KV_DISABLE_OVERLAP=1` 로 단일 stream 동작 강제 가능 (A/B 비교 / 디버깅 용) |
| Q chunk pipelining (2차 — 보류) | PLN_001 §4.3 의 second knob. 1 차 stream 분리 효과 측정 후 부등식이 미달이면 추가 |
| 측정 | dev pytest 130 passed, 152 skipped — 회귀 없음. prod 측정은 별도 회차 (이 노트 작성 시점 기준 진행 예정) |
| 코드 | [`vllm/v1/attention/backends/flash_attn.py`](../../../vllm/v1/attention/backends/flash_attn.py) `_get_cold_path_stream` + `hot_cold_attention` |
| 추적 | [`TSK_002`](TSK_002.md) §4.6 항 |

---

## 3. 부수 산출물 — 진단 / 검증 인프라

회귀 fix 흐름과 함께 누적된 진단·검증 도구.

### 3.1 · 진단 instrumentation

- `VLLM_PARTIAL_ATTN_PROFILE=1` env — `_call_compiled_kernel` / `hot_cold_attention` 단면 wall-time 출력. 첫 128 회만 (per-process cap). 평상시 OFF
- `VLLM_AMX_TRACE=1` env — Python `_call_amx` per-call 트레이스 + permission grant 결과
- `[IDE_006/TSK_004 cold-path fired pid=N] #K/5` breadcrumb — per-process 첫 5 회만 stderr. cold path 실 발화 확인용. 그 이후엔 `_COLD_PATH_FIRING_LOG_DONE` 모듈 bool 로 ~10 ns short-circuit
- `[IDE_006/TSK_004 q_len_cap WARNING]` — q_len cap fired + cold-with-decode 보유 시 1 회 loud warning. CLAUDE.md 위반 가능성 표면화

### 3.2 · 검증 wrapper

- [`eval/run_prod_simd_verify.sh`](../../../eval/run_prod_simd_verify.sh) — TST_004 cross-check + e2e_quick (`--max-prompts 8`). cold-tier eviction 트리거 안 되는 워크로드라 SIMD numerical correctness 검증 위주
- [`eval/run_prod_cold_verify.sh`](../../../eval/run_prod_cold_verify.sh) — IDE_006 의 cold path 실 발화 검증. NUM_PROMPTS=100 (default) 로 GPU KV pool 한계 위. monitor.py 시계열 캡처. cold-path firing 카운트 + CPU/GPU avg util 자동 요약
- [`eval/monitor.py`](../../../eval/monitor.py) — 1 초 간격 CPU/GPU 사용률 CSV. `run.sh` / `run_prod_cold_verify.sh` 가 백그라운드로 호출

### 3.3 · q_len cap (escape hatch only)

`VLLM_PARTIAL_ATTN_MAX_QLEN` env 로 켤 수 있는 임시 cap. 기본 `-1` (OFF). 켜면 `max_query_len > N` 배치는 cold path 우회 → mixed prefill+decode chunk 가 `forward_partial_with_lse` 로 가는 비용을 일시 차단. 단 cold-with-decode seq 가 같은 배치에 있으면 그 row 의 cold prefix 머지가 빠지므로 CLAUDE.md 의 "결과 값이 달라져서는 안됨" 위반. **production 에서 켜지 않을 것** — staged 검증 / 디버깅 용도

### 3.4 · `VLLM_COLD_KV_DISABLE_OVERLAP` (A/B 비교용)

§4.6 의 stream 분리를 `1` 로 비활성. 단일 stream sequential 실행으로 즉시 회귀 가능. overlap 효과를 동일 워크로드에서 직접 비교하거나 회귀 의심 시 단일 stream 으로 단순화하기 위함

---

## 4. 다음 단계

| 항목 | 결정 게이트 |
|---|---|
| stream 분리 (§4.6 1차) 효과 측정 | prod cold_verify 회차 한 번 — overlap on/off 비교. PLN_001 §4.3 의 부등식 충족 여부 1차 판정 |
| Q chunk pipelining (§4.6 2차) | 1 차 측정에서 부등식 미달 시 진입. dev 검증 + prod 회차 |
| TST_003 풀 회차 — D-i / D-ii 정합성 게이트 | §4.6 안정화 후 `eval/run_prod_smoke.sh --push` 한 번. baseline vs split_on 분포 비교 |
| TST_002 풀 회차 — throughput / overlap profile | TST_003 통과 후. IDE_006 의 net-win 정량 결정점 |
| (병행) `eval/` wrapper 사용법 정리 | `shadow_assists/README.md` Part VII 또는 `eval/README.md` 신설 — simd_verify / cold_verify / prod_smoke 의 용도·소요·의미 분리 |

---

## 5. References

### 부모·연계 문서

- 부모 PLN: [`PLN_001`](PLN_001.md) (§4.3 overlap profile 가설)
- 연계 TSK: [`TSK_002`](TSK_002.md) §4.6 / [`TSK_004`](TSK_004.md)
- 자매 산출물: [`PLN_001_TSK_002_01_partition_api_survey.md`](PLN_001_TSK_002_01_partition_api_survey.md)

### 코드 참조

- [`vllm/v1/attention/backends/flash_attn.py`](../../../vllm/v1/attention/backends/flash_attn.py) — `hot_cold_attention()`, `_get_cold_path_stream()`, q_len cap, cold-path firing breadcrumb
- [`vllm/v1/attention/ops/cpu_partial_attention.py`](../../../vllm/v1/attention/ops/cpu_partial_attention.py) — `_read_cpu_flags_once()`, `select_isa_path()` 캐싱, `_PROFILE_ENABLED` / `_PY_AMX_TRACE_ENABLED` env gate
- [`vllm/distributed/kv_transfer/kv_connector/v1/offloading/numa_aware.py`](../../../vllm/distributed/kv_transfer/kv_connector/v1/offloading/numa_aware.py) — `_partition_node_cpus_for_rank()`, `_pin_fast_done` hot-path 단축
- [`csrc/cpu/partial_attention_portable.cpp`](../../../csrc/cpu/partial_attention_portable.cpp) / [`_avx512.cpp`](../../../csrc/cpu/partial_attention_avx512.cpp) / [`_amx.cpp`](../../../csrc/cpu/partial_attention_amx.cpp) — `vllm_partial_attn_thread_count()` (env override + affinity baseline `min` clamp)

### 검증 결과

- `eval/results/20260427_030536_*_simd_verify` — partition / EAGAIN 해소 확인
- `eval/results/20260427_044407_*_simd_verify` — dispatch 캐싱 효과 (5462 ms → 12 ms)
- `eval/results/20260427_064134_*_simd_verify` — device-mismatch fix 검증 (8/8 success)
- `eval/results/20260427_070242_*_cold_verify` — cold path 실 발화 검증 (50/50 success, cold path 40 회 발화, CPU avg 30.9% / GPU avg 29.8% — sequential 시그너처)

---

## 6. Change Log

| 날짜 | 변경 | 사유 |
|---|---|---|
| 2026-04-27 | 초안 작성 | TSK_002 §4.6 진입 직전·직후의 회귀 fix 흐름이 commit 메시지에만 남고 PLN/TSK 문서까지 못 갱신된 상황을 정리. 다섯 단계 (libgomp EAGAIN / wrapper timeout / device-mismatch / per-seq 필터링 / sequential→stream 분리) 의 진단·fix·측정 데이터 통합 기록 |

---

**↑ 부모**: [`PLN_001`](PLN_001.md) · **연계 TSK**: [`TSK_002`](TSK_002.md) §4.6 / [`TSK_004`](TSK_004.md) · **↟ 조부**: [`IDE_006`](README.md)
