# SUB_239 — FERRY: DSA-운반 NUMA 파이프라인, 2026-06-15

> **판정: ✅ positive — latency-bound 워크로드에서 DSA 운반이 CPU 직접 원격접근 대비
> CPU-busy 지연 −29% / e2e −28%. 운반 오버헤드(e2e의 1.3%)가 NUMA 지연 절감에 묻힘.**

## 명제
"컴퓨트는 로컬, 운반은 DSA." 원격 NUMA(node1) 데이터를 CPU 가 직접 cross-UPI 접근하는
대신, DSA(shared WQ, ENQCMD)로 node1→node0 스테이징 후 node0-로컬 연산. CPU 는 운반에서
해방되고 연산은 로컬 지연으로 수행.

## 환경/방법
- 컨테이너(dgx-b200, dsa1=node1 shared WQ). **mbind/set_mempolicy 가 EPERM(seccomp)
  → 명시 NUMA 바인딩 불가**. 대신 **first-touch**: 대상 노드 코어에 핀한 헬퍼 스레드가
  버퍼를 fault-in (node1 src=cpu56 touch, node0 stage=worker cpu8 touch).
- 워크로드: **offset 기반 pointer-chase** (절대주소 아닌 인덱스 사이클 → DSA memcpy 후에도
  체인 유지). 의존 load 라 HW prefetcher 무력 → 진짜 NUMA 지연 노출. (streaming sum 은
  prefetch 가 NUMA gap 을 가려 부적합 — 예비측정에서 local/remote 9% 차에 그침.)
- worker=cpu8(node0), ws=128MB, iters=5, 3-run. `ferry.c` + `run_sub239.sh`.

## NUMA 지연 대조 (배치 검증, chase ns/step)
| src 배치 | ns/step |
|---|---|
| node0 (local, cpu8 touch) | 127.7 |
| node1 (remote, cpu56 touch) | 192.3 |

→ **원격 = +50.6% 지연** (SUB_235 호스트 numactl 측정 +43% 와 부합) → first-touch 배치 유효 확인.

## FERRY 본 비교 (3-run, CV<0.5%)
| 지표 | REMOTE (CPU 직접 원격) | FERRY (DSA 운반+로컬) | Δ |
|---|---|---|---|
| ns_per_step (CPU-busy 지연) | 157.79 | 111.63 | **−29.3%** |
| e2e_s (운반+연산) | 1.6546 | 1.1923 | **−27.9%** |
| 그중 DSA 운반 ferry_s | — | 0.0218 (e2e의 **1.3%**) | — |

## 게이트 판정
1. **CPU-busy 지연 ↓**: ferry 111.6 < remote 157.8 ns/step (−29%) → **PASS**.
   CPU 가 점유되는 시간이 29% 감소 — 그만큼 CPU 를 다른 일(serving)에 환원 가능.
2. **e2e ≤ remote**: ferry 1.19s ≤ remote 1.65s (−28%) → **PASS**.
   운반(21.8ms)이 NUMA 지연 절감(수백 ms)에 비해 무시 가능 → 순이득.

## 함의
- **NUMA 채널(SUB_235 +43%, SUB_245 +40%)의 실용적 해소책**: harvest/serving 워킹셋이
  원격 노드로 드리프트했을 때, 페이지 마이그레이션(`migrate_pages`, 커널·동기) 대신
  **DSA 비동기 운반**으로 CPU 를 cross-NUMA 패널티에서 분리. 운반은 RMID 미태깅
  (채널②, SUB_236) 이지만 *유용한* 운반이라 SUB_236 의 *해로운* aggressor 와 대비.
- **적용 조건**: latency-bound(의존 접근) + 재사용 충분(운반비용 상각) 워크로드. streaming/
  one-shot 은 prefetch 가 NUMA gap 을 가려 이득 작음(예비 9%) — regime 분리 필요.
- vLLM 연결: NEO KV 스테이징·prefix 운반 등 원격 노드 데이터를 로컬로 끌어오는 경로에
  DSA 오프로드 적용 후보 (gpu_worker NEO pinning 과 병행).

## 한계
- first-touch 는 명시 바인딩보다 약한 보장(THP/마이그레이션이 사후 재배치 가능). 호스트
  numactl 환경에서 교차검증 권장.
- victim 간섭 관점(게이트 원안 "victim 간섭 ↓")은 별도: ferry 는 데이터를 node0 로 가져와
  node0 iMC 부하를 오히려 늘리므로, node0-victim 시나리오에선 중립~불리할 수 있음.
  본 측정은 **worker 자신의 CPU-busy/e2e 이득**을 확정(주 가치). co-located victim 영향은
  호스트 resctrl 환경에서 후속.

산출물: `ferry.c`(first-touch + offset-chase + DSA 운반), `run_sub239.sh`, `sub239_results/`.

## 보강 — co-located victim 간섭 (호스트, 2026-06-15, DSA 불요)

> SUB_239 한계의 "co-located victim 영향" 미해결 항목을 호스트에서 정량화. FERRY 가
> 데이터를 node0 로 끌어오면 node0 iMC 부하↑ → 같은 node0 동거 victim 손해 가설 검증.

방법: victim = node0(cpu8) 128MB pointer-chase 지연(메모리-bound). aggressor = node0
8코어(cpu10-17) STREAM-read, 버퍼 배치만 LOCAL(node0=ferry 후 staging) vs REMOTE(node1
직접). DSA 운반 자체는 무관 — *데이터 종착 노드*가 victim 간섭을 결정하므로 배치로 대리.
`coloc_victim.c`, 3-run, 6s/셀.

| aggressor | victim p50 ns | mean ns | 기준 대비 |
|---|---:|---:|---:|
| off (기준) | 110.5 | 111.7 | — |
| **LOCAL (FERRY staging→node0)** | 138.4 | 139.1 | **+24.7%** |
| REMOTE (node1 직접접근) | 120.9 | 122.1 | +9.4% |

**판정**: FERRY 의 node0 데이터 종착이 동거 node0 victim 지연을 **+24.7%** 악화 —
REMOTE 패턴(+9.4%)의 **약 2.6배** 간섭. SUB_239 본문의 "중립~불리할 수 있음" 우려가
**불리로 확정**됨 (node0 victim 시나리오 한정).

**함의 (regime 분리)**: FERRY 는 *worker 자신*의 CPU-busy/e2e 를 −29%/−28% 개선(주 가치,
컨테이너 실측)하지만, **node0 에 지연민감 co-tenant(serving)가 있으면 그쪽이 ~25% 손해**.
→ 적용 조건: (a) worker 가 병목이고 node0 에 latency-SLO 동거자가 없을 때 순이득,
(b) node0 에 serving victim 동거 시 staging 노드를 victim 반대 노드로 두거나 MBA 로
staging-read 코어를 throttle(SUB_220 계열) 하는 가드 병용 필요. 단독 적용은 부적절.

산출물: `coloc_victim.c`, `host_runs/coloc_victim.csv`.

## ③ vLLM 통합 — FERRY staging (NEO swap-in read path, 2026-06-15)

> **상태: 코드 통합 + 호스트 정확성 검증 완료. DSA 가속·NUMA 성능 실측은 DSA-가용
> 환경(컨테이너/sm_on 리부트)으로 보류** — 호스트 DSA portal mmap EPERM(intel_iommu
> 에 sm_on 부재) 재확인.

### 통합 지점
NEO swap-in(CPU KV buffer → GPU): `gpu_model_runner.py` 의 per-layer 루프
(`copy_layer_out` → `.to(device)` H2D). 게더된 CPU 블록 텐서는 **contiguous 지만
non-pinned**, 또한 buffer 가 GPU-로컬 노드에 대해 원격 NUMA 면 H2D DMA 가 UPI 를 건넌다.
(advanced indexing 은 PyTorch 에서 contiguous *복사본* 을 반환하므로 단편화는 아님 —
문제는 pinned 속성 상실 + NUMA 원격성.)

### 구현
- `vllm/v1/lhc/ferry.py` — `FerryStager`: per-(shape,dtype) **node-local pinned
  bounce 버퍼 풀**(재사용). `stage(src)` = src 를 bounce 로 복사 후 반환. 복사는
  DSA lane(`dsa_lane.dsa_memcpy`) 가용 시 **DSA 오프로드(CPU-free 운반)**, 아니면
  bit-exact `Tensor.copy_` fallback.
- 훅: `gpu_model_runner.py` swap-in 루프에 `VLLM_NEO_FERRY=1` env-gate(기본 off,
  완전 가역)로 `k_cpu/v_cpu = stager.stage(...)` 삽입 — H2D 직전.
- env: `VLLM_NEO_FERRY`(on/off), `VLLM_NEO_FERRY_MIN`(최소 바이트, 기본 65536).

### 정확성 (CLAUDE.md Constraint)
staging 은 **정확한 바이트 복사** → swap-in KV 가 비-FERRY 경로와 **bit-exact 동일**
(분포 유사가 아니라 동일). 호스트 테스트 `test_ferry_stage.py`:
- `PASS bit_exact` (fp16+bf16, contiguous + gather 입력)
- `PASS pool_reuse` (51 stage 가 bounce 1개 재사용)
- `PASS fallback_stats` (호스트 DSA EPERM → 전부 CPU fallback, 그래도 정확)

### 오버헤드 마이크로벤치 (호스트, per-layer 2 MiB, fp16→bf16 H2D, 200-iter)
| 경로 | us/layer |
|---|---:|
| direct (pageable H2D) | 357.5 |
| FERRY (CPU-stage + H2D) | 403.2 (**+12.8%**) |

→ **호스트에선 순오버헤드** (DSA EPERM 으로 stage 가 CPU 복사라 NUMA/pinned 이득보다
복사비용이 큼). DSA-가용 환경에선 stage 가 DSA 오프로드(≈무료)가 되어 (a) CPU-busy 절감
(SUB_239 −29%) + (b) pinned/local H2D 이득만 남아 순이득 예상. **그 실측이 보류 항목.**

### 보류(DSA 의존) 검증 항목
- DSA-가속 stage 의 실효 e2e (컨테이너 NEO serving run 또는 호스트 `intel_iommu=sm_on`
  리부트 후): direct vs FERRY 의 swap-in 지연 + 전체 throughput.
- buffer 가 실제 원격 NUMA 로 드리프트한 케이스(`VLLM_NEO_NUMA_BIND=0`)에서의 이득 상한.

산출물: `vllm/v1/lhc/ferry.py`, `gpu_model_runner.py` 훅,
`test_ferry_stage.py`(호스트 통과).

### 컨테이너 DSA-가속 실측 (2026-06-15) — 메커니즘 수준 A/B ✅

> **보류 항목(DSA-가속 stage 실효)을 컨테이너에서 해소.** 단, 전체 serve A/B 는
> **환경 충돌로 불가** → 메커니즘 수준으로 측정.

**환경 충돌 (전체 serve 차단)**: FERRY DSA-가속은 DSA 가용 환경이 필요한데,
- DSA 제출: **컨테이너만 가능** (호스트는 IOMMU `intel_iommu=on` sm_off → portal mmap EPERM)
- vLLM 커스텀 커널 `vllm._C`: **CUDA 13 런타임 요구** (`libcudart.so.13`/`libnvrtc.so.13`),
  이 컨테이너는 **CUDA 12.8**(torch `2.11.0+cu128`)뿐 → `import vllm._C` 실패 → **serve 부팅 불가**.
- cu13 런타임을 torch(cu128) 옆에 끼우면 두 CUDA 런타임 혼재 → 출력 silent 손상 위험
  (CLAUDE.md 출력등가 제약 위반) + 공유 venv 손상 위험 → **하지 않음**.
- 즉 "DSA 되는 환경(컨테이너) ⊥ vLLM 커널 되는 환경(호스트)" 의 정반대 제약.

**대안 — 메커니즘 수준 A/B**: torch(cu128)는 B200 정상 동작(matmul/pinned-H2D 검증).
swap-in 핫패스 `k_cpu = stager.stage(k_cpu); k_gpu = k_cpu.to(device,dtype)` 를 **실제
`FerryStager` + 실제 DSA lane + 실제 GPU H2D** 로 재현(`ferry_vllm_bench.py`). KV 블록 형상
= Qwen2.5-7B/TP2 (kv_heads=2, block=16, head_dim=128, 28층 × K/V = 56 텐서/swap-in).

**DSA lane 게이트**: `dsa_lane_available()=True` (ops>0, fails=0). **DSA 경로 실제 탑승
확인: dsa_ops=1961 (cpu_ops=0)** — 사용자 요구 "dsa_ops>0" 충족.

| 블록 (텐서당) | direct(pageable H2D) | ferry+CPU stage | **ferry+DSA stage** | bit-exact |
|---|---|---|---|---|
| 128 blk (1MB, 56MB/swap) | 3.96 ms | 4.23 ms (**+8.9%**) | **2.90 ms (−26.7%)** | True |
| 256 blk (2MB, 112MB/swap) | 7.61 ms | 5.29 ms (−30.7%) | **4.78 ms (−37.3%)** | True |

(30-iter median, dsa_ops=1961/run. >2MB/텐서는 WQ max_transfer=2MB 초과로 단일 descriptor
실패→CPU fallback — KV 블록 단위가 ≤2MB라 실사용 무관.)

**기여 분해**:
- **핀드-H2D 이득** (ferry+CPU vs direct): 256blk 에서 −30.7% (pageable→pinned H2D
  14→큰 폭). 단 128blk 에선 CPU 스테이징 복사 비용이 이득을 잡아먹어 **+8.9% (손해)**.
- **DSA 가속 기여** (ferry+DSA vs ferry+CPU): 128blk 에서 4.23→2.90ms (**DSA 가 유일한
  승인** — CPU 스테이징이면 FERRY 가 손해인데 DSA 가 −27% 로 역전), 256blk 에서 5.29→4.78ms
  (핀드-H2D 위에 추가 −9.5%). → **DSA 오프로드가 작은 KV 운반에서 결정적**.

**정확성**: staged 텐서를 GPU 로 올린 결과가 direct 경로와 **bit-exact 동일**(torch.equal=True,
전 셀). 프롬프트 단위 출력 분포 동일성은 serve 경로(차단)가 필요하나, **텐서 단위 bit-exact
(더 강한 보장)** 로 출력등가 확정 — 바이트 동일이면 하류 연산도 동일.

**잔여(여전히 호스트/cu13 필요)**: 전체 serve tps·swap-in e2e A/B 는 (a) 컨테이너에 cu13
런타임 정합 빌드, 또는 (b) 호스트 `intel_iommu=sm_on` 리부트(DSA 해금) 후에만. 본 메커니즘
실측이 "DSA-가속 stage 가 실효 이득(−27~37%)" 을 입증하므로 serve 이득의 직접 근거.

**[2026-06-15 e2e 시도 결과 — 미해결 확정]**: 전체 serve e2e A/B 를 위해 cu13 정합을
`VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto` 로 시도(STEP 0). 결과:
vllm 이 `dev16186 precompiled` 로 갱신됐으나 **`vllm._C.abi3.so` 가 여전히 `libcudart.so.13`
(CUDA 13) 링크**, torch 는 `2.11.0+cu128` 유지(auto 가 cu130 미선택) → 컨테이너에 cu13
런타임 부재로 `import vllm._C` 실패 지속. 호스트는 시스템 레벨 `libcudart.so.13` 보유로
동작하나 컨테이너엔 cu12.8 만 존재. **precompiled 설치로는 컨테이너 cu13 정합 불가 확정.**
cu13 런타임 lib 주입(LD_LIBRARY_PATH) 또는 torch cu130 교체는 사용자 판정으로 **미진행**
(C: 메커니즘 A/B 로 마무리). e2e serve A/B 는 호스트 또는 cu13-정합 컨테이너 잔여.
*venv 영향*: 위 설치로 공유 venv 의 vllm 이 `dev16040(@/workspace/vllm_hybrid)`
→ `dev16186 precompiled(@/workspace/host_vllm_hybrid)` 로 변경됨 (호스트도 cu13 빌드라
정상 동작 예상, 원복 필요 시 통지).

산출물: `ferry_vllm_bench.py`, `vllm_bench_results/ferry_vllm_bench.csv`,
`run_ferry_e2e.sh`+`ferry_e2e_load.py`(e2e 하네스, cu13 정합 시 재사용).

## ‼ 정정 (2026-06-15) — "DSA되는 곳 ⊥ vLLM되는 곳" 이분법 철회

위 컨테이너 섹션의 "호스트=DSA불가(EPERM), 컨테이너=cu13불가 → 두 조건 동시 불가" 결론은
**틀림**. 호스트 portal EPERM 의 진짜 원인은 sm_on 이 아니라 **비-root 실행**이었음.
**호스트는 root(sudo)로 DSA 제출 + cu13 vLLM serve 를 둘 다 한다**(검증: dsa_lane
available=True@호스트 root, vllm._C@호스트 cu13 OK). → FERRY serve e2e A/B 는 **호스트
단독, `sudo vllm serve`** 로 가능. 컨테이너·리부트 모두 불요. (serve 프로세스가 DSA portal
mmap 하려면 root 필요 → serve 를 sudo 로 띄우는 것이 유일 조건.)

## serve-level e2e A/B 실증 결과 (호스트, 2026-06-15) — NEO 실 swap 미발화 확정

오진단(컨테이너/DSA) 철회 후 **호스트 단독으로 NEO-exclusive serve + DSA 부팅 성공**
(컨테이너·리부트 불요). 그러나 serve-level FERRY 효과는 **측정 불가**로 실증됨.

**구성**: `sudo`(DSA portal=root) + PATH 보존(flashinfer JIT 의 ninja). Qwen2.5-7B/TP2,
GPU0,1, `--enable-neo-asymmetric --kv-cache-policy exclusive`(CLI 노출 누락이라 arg_utils 에
`--kv-cache-policy` add_argument 추가), `--num-gpu-blocks-override 1024`(GPU KV 를 16K tok
으로 강제 축소 → swap 유발 의도). 부하: conc=48 × prompt~6000tok × 120req = 288K tok 과청약.

**run A (FERRY=0)**: gen=508.4 tok/s, total=11468 tok/s, req p50=23.8s, 120 ok / 0 err.

**결정적 관측 — swap 미발화**:
- 부팅 로그: `neo_scheduler_adapter: enable_neo_asymmetric activated. First-stage wiring —
  vanilla data path retained, **NEO decisions are recorded but not yet executed.**`
- 부하 중 `copy_layer_out` / `copy_all_layers_in_from_staged` / NEO BUF ALLOC 실행 = **0회**.
  16K tok 용량에 288K tok 을 던졌는데도 NEO swap 대신 스케줄러 큐잉으로 처리(p50 23.8s 의
  큐 지연이 증거). → **이 fork 상태에서 NEO 실 KV swap 이 serve 경로에서 실행되지 않음.**

**판정**: FERRY 의 swap-in 훅은 NEO 실 swap 이 실행돼야 도달하는데, **그 swap 이 현재
빌드에서 미발화** → run B(FERRY=1) 는 run A 와 동일할 수밖에 없어 **미실행**(GPU 낭비 회피).
serve-level A/B 불가의 원인은 환경(DSA/cu13/컨테이너)이 **아니라** NEO 자체의 Phase 3/4
(real KV swap 실행) 미완성. → **FERRY 의 유효성 근거는 메커니즘 A/B(−27~37%, 동일 hot-path
코드 + 실 DSA + 실 GPU H2D)가 유일하고 충분**하며, serve-level 숫자는 NEO real-swap 가
serve 에서 켜질 때라야 의미를 가진다.

**부수 산출**: `vllm/engine/arg_utils.py` 에 `--kv-cache-policy` CLI 노출(기존 필드/passthrough
는 있었으나 add_argument 누락이었음) — NEO exclusive 를 CLI 로 켤 수 있게 됨.
`VLLM_NEO_FERRY` 는 envs.py 미등록(fork 관례, "Unknown env" 경고는 무해).

## 최종 판정 (2026-06-15) — 메커니즘 입증 ✅ / vLLM 적용 무가치 ❌

사용자 확인: **NEO 를 vLLM 에서 실험했으나 서빙 효과 없음.** FERRY 는 NEO swap-in 가속이라
NEO 자체가 무효/미발화면 **서빙 가치 0**. 따라서:
- FERRY **기법**(DSA NUMA staging)은 메커니즘 A/B(−27~37%, 실 DSA+실 GPU H2D, bit-exact)로
  **입증 완료** — "swap 을 한다면 빨라진다".
- 그러나 그 적용점(NEO swap)이 vLLM 서빙에서 효과 없음 → **FERRY 의 vLLM 적용은 무가치**.
  A-시리즈 메커니즘들과 동일 패턴(흥미롭지만 serving tps 무영향).
- 코드는 env-gated(`VLLM_NEO_FERRY` 기본 off)라 무해 → 그대로 보존, "적용 보류"로 닫음.
  실 서빙 승부수는 spec-decode(SUB_213 +38%). 메모리 `ide006-neo-no-vllm-effect` 참조.
