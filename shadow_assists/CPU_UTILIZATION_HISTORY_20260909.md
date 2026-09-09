# CPU/호스트 자원 활용 시도 이력 — 전체 정리 (2026-04-24 ~ 2026-09-09)

목표(CLAUDE.md Objective): GPU가 아닌 **CPU/호스트 자원을 최대로 써서 GPU 포함 서버 전체의 LLM 서빙 성능을 올린다.** 이 문서는 그 목표로 진행한 모든 시도를 시기순으로 정리하고, 각 시도가 **무엇을 했고 / 얼마나 개선됐고 / 왜 실패했는지**를 수치와 함께 기록한다. 수치는 각 결과 디렉토리·보고서의 값을 그대로 옮겼다(출처 표기).

하드웨어: 개발기 RTX 3090 + i9-12900KF(2026-04~05 일부), 운영기 **violet-h100-016** = Xeon 8480+ ×2 (AMX, 112코어, turbo OFF 2.0GHz 고정) + 2TB DDR5 + H100 ×8.

---

## 1. 한눈에 보는 연표

| 시기 | ID | 시도 (CPU가 맡은 일) | 결과 | 상태 |
|---|---|---|---|---|
| 04-21~25 | IDE_001~008 | 아이디어 8개 등록 (배치 planner, prefill-assist, drafter, cold-KV attention 등) | IDE_006만 착수, 나머지 "대기" | 대기 |
| 04-25~05-23 | **IDE_006** Cold-KV CPU Partial Attention → NEO 이식 | cold KV를 DRAM에 두고 **CPU가 partial attention 연산** | 4차 재정의 끝에 vanilla 대비 **2.1~4.1× 느림**, CPU 결과 활용률 0% | **기각** |
| 05-13 | 브레인스토밍 | 새 알고리즘 10 + low-level 13 후보 | 문서만, 미착수 | 대기 |
| 05-23~27 | (IDE_006/TSK_019 부산물) | vLLM 설정·ngram spec decode(GPU-only) | **10,956.6 tok/s (+134%)** — CPU 활용 5.5% | 채택(GPU 기법) |
| 08-27 | **IDE_023** MoE expert offload (SGLang+kt-kernel AMX) | 라우팅된 expert를 **CPU가 자기 DRAM의 INT4 가중치로 AMX 계산** | GPU-only 불가(OOM) 모델 서빙 성립: R1 0→19.67, 480B 44.5 tok/s | 채택(성립) |
| 08-27 | IDE_024 Co-location | CPU에 **별도 워크로드**(SHA256) 동거 | BG 56proc: GPU −0.5%, 112proc: −3.7%·CPU 55% | 부록 강등 |
| 08-27 | IDE_025 DRAM KV tier | DRAM을 **KV/prefix 저장 계층**으로 (연산 없음) | KV 압박 시 **+51.8%**, 비압박 −1.3% | 채택(운영), 논문 기여 아님 |
| 08-29 | IDE_026 SCED (spec depth × CPU expert) | CPU drafter/검증 배치로 CPU expert 연산 강도 조절 | spec 1.45~1.55×, 핵심 가설(K\* 이동) 기각 | 보류 |
| 08-29~30 | C1·Footprint·HeteroGuard·PlacementBound 외 11후보 | 신규성 탐색 | 11/11 선점·기각 (PlacementBound 오차 56%) | 기각 |
| 08-30 | **IDE_030** hot expert GPU 배치 + decode graph | CPU는 cold expert만 (CPU 일 감소) | 480B C32 **56.3 → 152.6 (+171%) → 324 (+475%)** | 채택 |
| 08-30~09-07 | IDE_030 spec×graph / 교차 실행 | CPU 대기 40ms를 두 micro-batch로 숨기기 | −13.6% / −20.4%, 동시 재생 미생존 | 기각 |
| 09-08 | IDE_030 hot-96 + deferred 4 | | C32 **490.9 (+53.5%)**, GSM100 97.0 | 채택 (기존 최종) |
| 09-08~09 | IDE_031/PLN_008 성능 모델 | 마이크로벤치 파라미터로 TPOT 사전 예측 | 8셀 중앙값 오차 8.3%, 순위 100% | 통과 |
| 09-09 | IDE_032 배치-인지 deferral | cold 전량 다음 층 지연 | +3~4% | 기각 |
| 09-09 | **IDE_033** callback-free 핸드오프 + 빈 immediate 제거 + 코어 배치 | CPU↔GPU 동기화 비용 제거 | 505 → 600 (+19%) | 채택 |
| 09-09 | **IDE_034** phase 가중 hot set (α=0.25) | CPU cold 작업 감소 | 600 → **661 (+10%)**, C64 884 | 채택 |
| 09-09 | **IDE_035** τ deferral | 정확도 보존 | **572 / 769, GSM100 97.0** | 채택(무손실 구성) |
| 09-09 | IDE_036 빈 deferred 생략 | | 효과 0, 층당 고정비 77µs 하한 실측 | 기각 |
| 09-09 | M3 235B / 30B | | GPU-only가 +16~24% 우세 (반례) | 판정 |

---

## 2. 세대별 상세

### 2.1 IDE_006 — Cold-KV CPU Partial Attention → NEO 비대칭 파이프라인 (2026-04-25 ~ 05-23)

**목표**: GPU 메모리에서 밀려난 cold KV를 CPU DRAM에 두고, **CPU가 그 위에서 partial attention을 직접 계산**해 (partial_output, LSE)만 GPU로 올려 online-softmax 병합. 가치 축은 "시스템 throughput" 단일(GPU 메모리 절약은 사용자 지시로 제외).

**시도** (4차 재정의)
1. 1차 Cold-KV staging → 업스트림(vLLM CPU KV offloading #37160) 중복으로 기각.
2. 2차 "GPU reload 없이 CPU partial attention" → partition/fallback 경로 모두 같은 cold KV에서 발산.
3. 3차(04-28) deadline 기반 GPU full-FA fallback으로 throughput 하한 보장(TSK_011).
4. 4차(04-29) **NEO(arXiv 2411.01142) 이식** — 요청 단위 GPU/CPU 배타 소유 + 비대칭 파이프라인, AVX-512/AMX C++ 커널(`csrc/cpu/pacpu/`), SUB_007~028(SIMD·AMX·async swap·staging·persistent OMP team 등).

**측정** (`features/IDE_006/TSK_019/`)
- 최적화 0~5단계(04-28): partition 경로 233.2s → 136.4s(−41%), 그래도 baseline 186.5s 대비 **3.1× 느림**.
- NEO 최고치 S1-S9: **2,238.6 tok/s** vs vanilla **4,690.7** (vanilla가 2.1× 빠름). AMX qk_product 적용 −4.3%, Step 5 −2.35% 회귀.
- 결정 대조(SUB_036~042): NEO 1,778.8 vs vanilla 4,680.8(**2.63×**), decode-heavy 2,039 vs 8,160(**4.0×**). NEO CPU busy 11.9%(목표 미달), 효율 tps/CPU% = vanilla 1,002 vs NEO 141.
- 부산물(GPU 기법): ngram spec decode + 설정 → **10,956.6 tok/s (+134.1%)**, CPU busy 5.5% (`vllm_config_perf` 산출물, 05-27).

**실패 원인**
- **Q-dependency dilemma**: partial = softmax(Q·K_coldᵀ)·V_cold의 Q는 해당 층 QKV projection 뒤에야 나옴 → CPU partial(~6.4ms)이 GPU hot attention(~0.6ms) 창에 들어갈 수 없음. 6회차 통합 검증에서 **모든 시나리오 merged 0.00%**(CPU 결과가 한 번도 쓰이지 않음), 속도 1.08~1.10× 느림. 교차 층(TSK_005)으로 옮겨도 층 N+1 진입 시 GPU가 진짜 Q를 가져 full attention이 가능해 CPU 결과 무용.
- **정확도 발산**: deadline 100ms/1000ms 모두 worst_max_abs_logprob ~3.43, PPL 상대차 ~0.24(동일 cold KV 원천).
- **자원 비율**: H100 대비 Xeon AMX BF16 GEMM 원시 처리량 차이로 층 단위 attention을 CPU로 옮기면 순손실.
- 운영 사고: OOB 로그 1.09M줄/214MB로 shm_broadcast 포화, async cdec EngineDeadError.

**상태**: **기각**. 사용자 결정 05-23 "CPU 활용 거의 없음"으로 중단. 이 시기의 실제 채택물은 CPU가 아닌 GPU-only spec decode.

### 2.2 2026-05 브레인스토밍 (미착수 기록)
- `brainstorming/new_algorithms_2026-05.md`: AMX sampling penalty, CPU prepare-inputs lookahead, async detokenize/logprob CPU pool, **AMX CPU drafter + GPU verifier(+20~40% 추정)**, CPU constrained decoding mask 등 10후보.
- `brainstorming/low_level_optimizations_2026-05.md`: AMX/AVX-512 prefetch, huge page, pinned pool, fused H2D, decode-only graph 등 13후보.
- 전부 추정치만 있고 착수·측정 없음(6~8월 공백; 8월부터 운영기 캠페인으로 전환).

### 2.3 2026-08-27 캠페인 (PLN_003, violet-h100-016) — IDE_023 / 024 / 025

**IDE_023 MoE expert offload (SGLang 0.5.18 + kt-kernel AMXINT4)** — 기록: `features/IDE_023/COMPREHENSIVE_REPORT_20260827.md`
- CPU가 하는 일: 라우팅된 sparse expert를 **자기 DRAM에 있는 INT4 가중치로 AMX GEMM 계산**(전송 아님 → PCIe·Q-dependency 무관).
- Qwen3-30B-A3B: GPU-only 1,461.8 vs 전 expert CPU **96.1 tok/s(15× 손해)** — GPU에 들어가는 모델에서는 무의미 확인.
- DeepSeek-R1-0528(642GB): GPU-only OOM → hybrid **19.67 tok/s**(TTFT 11.2s, TPOT 305ms, CPU 42.7%). 출력 품질 결함 → SUB_167(DeepSeek 특이 변환 결함, 미해결·R1 튜닝 금지).
- Qwen3-Coder-480B: GPU-only TP=4 OOM, TP=8 부팅 불가 → hybrid TP=4 **44.5 tok/s**(C16), 품질 정상. 480B ×2 동시 서빙 간섭 ≈0.
- 실패·정정: 동시 실행 초기 **1.6 tok/s(115× 붕괴)** = kt CPUInfer가 numactl을 무시하고 core 0부터 절대 pin → cgroup cpuset 격리로 해결. "분할이 통합보다 +37%"는 부하 불일치 비교였고 정합 후 +2~12%로 정정, 다시 threadpool-numa 불일치 발견 후 결론 철회.
- 사용자 판정(08-29 campaign_assessment): 메커니즘은 KTransformers 것, 우리 기여는 패치·결함 발견·측정 → "당장 논문감 없음".

**IDE_024 Co-location** — CPU가 GPU를 돕지 않고 **별도 워크로드**를 수행. 70B TP=8 + SHA256 BG: solo 4,921.0 / BG56 4,896.3(−0.50%, CPU 29%) / BG56 격리 4,827.5(−1.9%, 격리가 불리) / BG112 4,739.6(−3.69%, CPU 54.6%). 손실 ≤1%와 CPU ≥50% 동시 충족 구성 미확보. "유휴 CPU에 일을 얹으면 CPU가 오른다"는 자명 → 부록 강등.

**IDE_025 DRAM KV/prefix tier** — CPU는 계산하지 않고 DRAM을 저장 계층으로. KV 압박(GPU pool 153K ≪ prefix 262K) 시 418.0 → **634.4 tok/s(+51.8%)**, TTFT p50 347 → 122ms(−65%), DRAM→GPU reload 91.3GB 실증. 비압박 −1.31%(게이트 ≤1% 소폭 초과). 사용자 판정: **vLLM 내장 기능 활용이라 신규성 없음, 운영 권고로만 보존.**

### 2.4 2026-08-29~30 신규성 탐색 (11 후보, 전부 기각) — `brainstorming/problem_search_20260829.md`
- **IDE_026 SCED**(spec depth K로 CPU expert의 연산 강도 조절): E0~E4 하루 완주(30B + 0.6B draft). H1 통과(토큰당 expert 비용 n_e 1→512에서 43~53× 하락, knee 128~256), **H3 기각**(hybrid와 GPU의 최적 K가 모두 3, 차이 0), spec 이득 1.45~1.55×, 정적 (G,K) 표면 ±4%. 부산물: curvature 역전 관측. 사용자 결정: 안 A(draft-in-bubble)·안 B 착수 없이 보존.
- C1 탐색-네이티브 서빙: K2 +0.0%p, K3 음수 → 기각. Footprint-Elastic: 교차점 없음 → 기각. HeteroGuard(유휴 CPU를 SDC 감시자로): 4기둥 전부 선행(Ekka 등) → 기각. 훈련-서빙 동거, zero-copy 복제, dLLM 오프로딩(TIDE 선점), 배치=품질 변수, CPU-draft/GPU-verify(Dovetail 선점), INT4 자기-초안(QuantSpec 선점), 서빙-인지 라우터 재학습(ReMoE + 부정결과 논문까지 존재) → 전부 기각.
- **IDE_029 PlacementBound**(HPC 통신 하한 이식): 사전 등록 예측 오차 중앙값 **56%**(게이트 30%), 방향 예측 반대 → 기각. 기전 4개(id 고정 배치, overlap 바닥, 층 고정비, 소켓 공유 대역폭)만 승계.

### 2.5 IDE_030 — Build 트랙 (2026-08-30 ~ 09-08) — `features/IDE_030/FINAL_RESULT_20260830.md`, `FINAL_RESULT_20260908.md`
Qwen3-Coder-480B, TP=4 H100 + Xeon 96스레드, sonnet 512/128.

| 단계 | 구성 | C32 tok/s | TPOT | 품질 |
|---|---|---|---|---|
| 기준 | expert 전량 CPU | 56.3 | 436ms | GSM40 85.0 |
| hot expert GPU 배치 (EPLB 맵 주입) | hot-64 | **152.6 (+171%)** | 177 | 95.0 |
| + decode CUDA graph | hot-80 | **324 (+475%)** | — | 97.5 |
| + hot-96 + deferred 4 | | **490.9 (+53.5% vs 319.9)** | 51.5 | GSM100 97.0 |

- 성립 전제였던 버그 3건 수리(SGLang expert_location 미전달로 조용한 출력 붕괴, kt_ep_wrapper의 gpu_experts_mask=None으로 CPU 전량 계산 + GPU 중복 덧셈, threadpool 수 ≠ 변환 numa로 무경고 붕괴) + kt 절대-pin 결함. upstream 리포트 5건.
- **기각 2건**: spec × graph 결합(C32 279.8 = −13.6%, C16 −18%: TPOT 82ms 체제에선 draft 왕복 > 검증 절약); 교차 실행 dual micro-batch(순차 254.5 = −20.4%; 동시 재생은 두 통신그룹 collective 상호 spin으로 미생존 — 신호식 대기 C++·격리 수리 5건은 부산물).
- 기전 확인: CPU 층 호출 = DDR 천장(430GB/s) 스트리밍; hot-80에서 스텝 66ms = GPU 26 + **CPU 대기 40**; hot-96에서 GPU 27.7 + CPU 대기 9.9. **CPU 노출을 줄일수록 빨라졌다.** hot-104는 GPU 메모리 불가(TP4·80GB 상한). 자기 판정: "구성 요소는 모두 기존 기법, 기여는 엔지니어링".

### 2.6 IDE_031 성능 모델 + IDE_032~036 메커니즘 (2026-09-08 ~ 09-09) — `eval/results/SUMMARY_20260909.md`
- **모델(PLN_008)**: 마이크로벤치·라우팅 트레이스 파라미터만으로 TPOT 사전 예측. 첫 M1 실패(중앙값 52%)의 원인은 트레이스 불일치였고, 실측 트레이스 투입 후 **8셀 중앙값 8.3%, 순위 26/26**; 새 메커니즘용 v3 **10셀 15.0%, 순위 41/41**. in-situ 법칙: CPU 층 작업 ≈ 40 + 90 + 71·D_c µs(expert당 332GB/s = DDR 천장의 ~80%).
- **IDE_032** cold 전량 deferral: +3~4% → 기각(프로파일로 기전 특정: host 콜백 노드 바닥·단일 FIFO 큐).
- **IDE_033** callback-free 핸드오프(+5.1%) → 빈 immediate 제거(+8.3%) → AMX 워커 HT 형제에 있던 스핀 스레드 재배치(+4.3%, TTFT 1674→1304): 505 → **600**.
- **IDE_034** phase 가중 hot set: prompt 트레이스 hot set의 decode 커버리지가 96%(prompt 99.2%)임을 phase 분해로 발견. decode-only는 TTFT ×2로 기각, α=0.25: **661 / C64 884**.
- 정확도 귀속: N=4 97.0×2 vs N=8 전량 95/93/94 → 전량 deferral 비용 −2~4점 실재.
- **IDE_035** τ=0.25 가중치 임계 deferral + callback-free: **572 / 769, GSM100 97.0**(무손실). 
- **IDE_036** 빈 deferred 생략: 효과 0 → 기각. hot-128 프로브로 하이브리드 경로 **층당 고정비 77µs(CF) / 126µs(host 콜백)** 실측(구조적 하한).
- **M3**: 235B는 GPU-only 929.6/1915.8 vs hybrid 804/1549 → **GPU-only 우세(+16/+24%)**; 30B GPU-only 2456/4213, hybrid 부팅 segfault(미측정).

### 2.7 IDE_037~050 심층조사 트랙 + 오후 트랙 (2026-09-09 11:00 ~ 18:40) — `eval/results/SUMMARY_20260909.md` §11
- **IDE_037** EPOCH 오라클: 고정 동시성 no-go (읽기당 행 중앙값 1). 부산물 = rows 스윕에서 **AMX 절벽** 발견 (qlen>80 이면 AMX, 행당 18→2.3µs; 1~2행은 AMX 타일이 오히려 느림).
- **IDE_038-b** expert 별 행 수 기준 AMX 선택 (m≥3): C96 +8%. **IDE_039** FP8 KV (실KV 37k 예약 발견) + graph 160: **C160 982.4 / TPOT 119.5, GSM40 97.5** (정오 최고).
- **IDE_040** hot-88/80 고동시성: 토큰당 CPU 바이트 증가 > KV 이득 → 기각. **IDE_041** 도착률 특성화 → **IDE_043-c** 스트리밍-인지 prefill 배칭 (delayer + 실행 배치 128 미만 게이트): 경부하 TPOT −27%, 과부하 TTFT p99 −76% (처리량 −7%, 운영 옵션). **IDE_042** 청크 확대 OOM, **IDE_044** hot set 재도출 (커버리지 포화) 기각.
- **IDE_045** huge page: 적용 확인 (133→245 GB) 했으나 효과 없음 (TLB 아님). 부산물: prefix cache 전량 적중 시 C64 +43% → prefill 이 처리량의 ~30%.
- **IDE_046** prefill AMX GEMM 커널 (B 타일 언팩 캐시·A 양자화 병렬화, bit 동일): 층당 −3~6%, 서빙 동일. GEMM 은 L2 타일 적재 한계 (AMX 활용 24%). 스레드 절반 실험: TTFT +27% (CPU 는 prefill 임계의 1/4), TPOT +24~26% (decode 는 CPU 노출 큼) → 기각.
- **IDE_047** EAGLE3 투기 디코딩 (lmsys draft): 수락 길이 2.0 (sonnet); C160 706, C64 461 → 기각. 구조적 원인: cold slot 이 서로 다른 expert 에 떨어져 검증 행 4배 → 스트리밍 expert 수 2~3배 + GPU 투기 오버헤드 ~135 ms/스텝.
- **IDE_048** AVX-512 vec 커널 레지스터 블로킹 (bit 동일): 1행 −8%, 16행 −24%; 서빙 C64 782. **IDE_050** B 스트림 SW prefetch: 1행 −6% 추가 (누적 −13%, ≈190 GB/s/소켓). 채택 (기본 켬).
- **IDE_049** CUDA graph 버킷 6~7종 명시로 GPU 메모리 회수 → KV 110k → 123~127k → graph 192/224: **C192 1049, C224 1067** (+8.6% vs 정오). 채택.
- **최종 재측정 (`*_final_sweep/`)**: 운영점 A (graph 192·청크 8192·KV 122,880) C192 1048.7 / C160 1012.2 / C128 961.9 / C64 796.6 / C32 578.0, GSM100 96.0 (−1문항, logprob 비교로 분포 동등성 확인 중). 운영점 B (graph 224·청크 4096·KV 127,309) C224 1066.5, GSM40 97.5.
- 하드웨어 사실: 소켓당 DDR5 8ch×2DPC 4400 → 이론 282 GB/s, torch 읽기 195, 커널 ≈190 (67%). 벤치 규칙: 192 프롬프트 뒤의 480 프롬프트 C160 은 prefix cache 로 +17% 부풀려짐 → C160 은 fresh 또는 480 뒤에만.

---

## 3. 성능 개선 누적 (Qwen3-Coder-480B, C32 sonnet 512/128, TP=4)

| 시점 | 구성 | tok/s | 누적 배수 | GSM |
|---|---|---|---|---|
| 08-29 | hybrid 성립 (expert 전량 CPU) | 44.5 → 56.3 | 1.0× | — / 85.0 (40) |
| 08-30 | hot-64 GPU 배치 | 152.6 | 2.7× | 95.0 (40) |
| 08-30 | + decode graph, hot-80 | 324 | 5.8× | 97.5 (40) |
| 09-08 | hot-96 + deferred 4 | 490.9 | 8.7× | 97.0 (100) |
| 09-09 | + IDE_033/034/035 (무손실) | **572.1** (C64 769) | **10.2×** | **97.0 (100)** |
| 09-09 | + 전량 deferral (최고 처리량) | **658.7** (C64 883.7) | 11.7× | 93~94 (100) |
| 09-09 정오 | + AMX 선택 + FP8 KV + graph 160 (무손실) | C160 **982.4** | 17.4× | 97.5 (40) |
| 09-09 저녁 | + AVX 레지스터 블로킹·prefetch + graph 버킷 축소·KV 확대 | **C192 1048.7 / C224 1066.5** (C64 796.6, C32 578.0) | **19.0×** | 96.0 (100) / 97.5 (40) |

70B(GPU에 들어가는 모델)에서의 CPU 관련 이득: DRAM KV tier **+51.8%**(KV 압박 워크로드 한정). IDE_006 계열(CPU attention)은 개선 0(전부 손실).

---

## 4. 실패 유형 분류

| 유형 | 사례 | 교훈 |
|---|---|---|
| **물리 한계** (자원 비율) | IDE_006 CPU attention 2.6~4× 열세; 30B/235B에서 hybrid < GPU-only; IDE_036 층당 77µs 하한 | CPU는 GPU 대비 대역폭 33배·연산 40배 열세. GPU와 같은 종류의 일을 나눠 받으면 진다. 이기는 조건 = GPU가 못 담는 용량(480B) 또는 저장 계층(KV tier) |
| **의존성 구조** | IDE_006 Q-dependency(merged 0%); 교차 실행 collective 교착 | 임계 경로 안에서 CPU 결과를 기다려야 하는 설계는 창이 없다 |
| **구현 결함** (성립 자체를 막음) | kt 절대-pin(115× 붕괴), expert_location 미전달, gpu_experts_mask=None, threadpool-numa 불일치, 마스크 H2D 캡처 오류, spec×graph 폭 불일치 segfault | 수리 6건 이상, upstream 리포트 5건. 결함 수리 전 성능 수치는 무효였던 경우 다수 |
| **측정 오류 정정** | "+37% 분할 우세"(부하 불일치), "deferred 2× 지연"(평균/중앙값 비교), "wrapper 0.2ms"(동일), M1 52% 실패(트레이스 불일치) | 사전 등록 + 동일 조건 재측정으로 4건 정정 |
| **신규성 선점** | SCED, C1, Footprint, HeteroGuard, PlacementBound, CPU-draft, ReMoE 계열 등 11후보 | 아이디어 수준 신규성은 발상 속도 < 출판 속도. 남는 길은 측정·사전 등록 서사 |
| **정확도 비용** | 전량 deferral −2~4점(GSM100) | 처리량 증분의 일부는 정확도와 교환된 것 — 분리 측정 필수 |

---

## 5. 현재 판단 (2026-09-09 기준, 사실만)

- CPU가 서빙 처리량에 기여한 방식은 두 가지뿐이다: **(a) GPU 메모리에 들어가지 않는 모델을 성립시키고(480B: 0 → 572~659 tok/s), (b) DRAM을 KV 저장 계층으로 쓰는 것(+51.8%, 압박 시)**. CPU를 임계 경로의 연산자로 넣은 시도(attention, 교차 실행)는 전부 손실이었다.
- **09-09 저녁 갱신**: 최고 1067 tok/s (C224). decode 는 CPU cold expert 스트리밍에 묶여 있으며 (스레드 절반 → TPOT +25%), 스트리밍은 DDR 실용 상한의 ~90% (190 / 195~225 GB/s). 남은 CPU 측 여지 ≈ 5~10% (스트리밍) + 5% (층당 위상 장벽). 다음 큰 폭은 GPU 메모리 (KV) 확보 = hot expert 의 GPU 측 양자화 (손실, 사용자 판단) 또는 더 큰 GPU 메모리.
- 480B 하이브리드에서 CPU의 현 상태: 큐는 99% 바쁘지만 일의 98%가 가중치 스트리밍(332GB/s)이고 AMX 연산은 expert당 1.5~6행으로 거의 놀고 있다. 처리량이 오른 기법들은 대부분 CPU의 일을 줄이거나 대기를 없앤 것이다.
- 사용자 지시(09-09): 목표는 CPU를 더/더 효율적으로 써서 시스템을 올리는 것이며 지표·목표 재정의는 하지 않는다. 미착수 후보 중 이 방향에 맞는 것: expert 친화 admission(같은 expert에 행 모으기), 고동시성에서 hot expert를 CPU로 더 넘기고 KV를 키우는 분할, 스트리밍 효율(332→400GB/s). CPU drafter 계열은 IDE_026/9라운드에서 선점·부분 기각 이력이 있으므로 재제안 시 그 기록을 먼저 볼 것.

---

## 6. 파일 색인
- 레지스트리·트리: `shadow_assists/id_registry.md`, `shadow_assists/README.md`
- IDE_006: `features/IDE_006/README.md`, `features/IDE_006/TSK_019/` (INDEX, Best_*.md)
- 2026-05 브레인스토밍: `shadow_assists/brainstorming/{new_algorithms_2026-05.md, low_level_optimizations_2026-05.md}`
- 08-27 캠페인: `features/IDE_023/{COMPREHENSIVE_REPORT_20260827.md, campaign_assessment_20260829.md, dual480b_results_20260829.md, PLN_003.md}`, `features/IDE_024/`, `features/IDE_025/`, `eval/results/20260827_*`
- 신규성 탐색: `shadow_assists/brainstorming/{paper_novelty_candidates_20260827.md, problem_search_20260829.md}`, `features/IDE_026/{PLN_004.md, PROGRESS_20260829.md}`, `features/IDE_029/PLN_006.md`
- Build 트랙: `features/IDE_030/{FINAL_RESULT_20260830.md, FINAL_RESULT_20260908.md, interleave_design.md, specgraph_worklog.md, upstream_reports/}`
- 09-08~09: `features/IDE_031/{PLN_008.md, paper_outline.md, K1_DELTA.md, model/}`, `eval/results/SUMMARY_20260909.md`, `eval/results/20260909_*`, 오후 하네스 `eval/harness/20260909_afternoon/`
