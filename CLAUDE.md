Answer to me always Korean
Write your task on CLAUDE.md file
Don't commit or push without explicit command

# vLLM Hybrid 프로젝트

## 프로젝트 개요
vLLM의 CPU/GPU 하이브리드 추론 최적화 포크. GPU와 CPU를 **별도 프로세스**에서 동시에 실행하여 `total_throughput = GPU + CPU` 달성.

## 타겟 하드웨어
- **GPU**: NVIDIA H100 x8 (TP=8)
- **CPU**: Intel Xeon Platinum 8480+ (Sapphire Rapids), 2소켓 112코어
- **RAM**: 2TB DDR5, NUMA 멀티소켓
- **ISA**: AVX-512, AVX-512 VNNI, AMX-BF16/INT8

---

## 핵심 아키텍처: Dual-Process Parallel-Batch

```
HybridAsyncMPClient (CapacityAwareRouter로 라우팅)
├─ input_socket (ZMQ ROUTER)
│   ├─ GPU: identity=b'\x00\x00' (engine_index=0)
│   └─ CPU: identity=b'\x01\x00' (engine_index=1)
├─ output_socket (ZMQ PULL) ← GPU/CPU 결과 비동기 수집
└─ CapacityAwareRouter: CPU 슬롯 여유시 CPU, 가득차면 GPU

GPU EngineCoreProc [별도 프로세스]
├─ EngineCore → MultiprocExecutor (8x H100)
└─ KV Cache: GPU VRAM

CPU EngineCoreProc [별도 프로세스]
├─ EngineCore → UniProcExecutor (CPUWorker)
└─ KV Cache: NUMA-aware DRAM
```

### 설계 원칙
1. **core.py 무수정** — hybrid 코드는 hybrid_core.py와 core_client.py에만 존재
2. **별도 프로세스** — GPU/CPU 각 독립 PID, GIL, busy loop
3. **CapacityAwareRouter** — CPU 슬롯 기반 라우팅 (RequestRouter 대비 CPU 활용률 극대화)
4. **자동 감지** — cpu_max_num_seqs, kvcache, threads 모두 0(auto) 기본값

---

## 핵심 파일

### Hybrid 엔진
| 파일 | 역할 |
|------|------|
| `vllm/v1/engine/hybrid_core.py` | CapacityAwareRouter, _resolve_cpu_params, _setup_cpu_process_env, run_cpu_engine_core, launch_hybrid_engines |
| `vllm/v1/engine/core_client.py` | HybridAsyncMPClient, HybridSyncMPClient, _HybridEngineLauncherMixin |
| `vllm/v1/engine/core.py` | EngineCore/EngineCoreProc (hybrid 코드 없음) |

### CPU 최적화
| 파일 | 역할 |
|------|------|
| `vllm/platforms/intel_cpu_utils.py` | Intel CPU 감지, NUMA, AMX/AVX-512, OpenMP 설정 |
| `vllm/v1/worker/cpu_worker.py` | CPUWorker (NUMA 스레드 바인딩, IPEX 감지) |
| `vllm/v1/attention/backends/cpu_attn.py` | CPU PagedAttention (IPEX 최적 커널) |
| `vllm/config.py` (HybridConfig) | 하이브리드 설정 (기본값 0=auto) |
| `vllm/engine/arg_utils.py` | CLI 인자 정의 |

### AVX-512 C++ 커널 (하이브리드 빌드)
| 파일 | 역할 |
|------|------|
| `csrc/cpu/gemm_vnni.cpp` | VNNI INT8 GEMM (6x16 마이크로커널) |
| `csrc/cpu/quant_q8_0.cpp` | Q8_0 양자화 |
| `csrc/cpu/decode_gemv.cpp` | BF16/FP32 Decode GEMV |
| `csrc/cpu/batch_attention.cpp` | 16-시퀀스 배치 Attention |
| `csrc/cpu/mem_opt.cpp` | NT memcpy, NUMA 할당, 프리페치 |
| `cmake/cpu_hybrid_extension.cmake` | _C_cpu_ops 타겟 빌드 |

### 기타 컴포넌트
| 파일 | 역할 |
|------|------|
| `vllm/model_executor/layers/fused_moe/expert_offload.py` | MoE Expert Offload (미래용) |
| `vllm/v1/spec_decode/ngram_proposer_dynamic.py` | N-gram Speculative Decode (미래용) |
| `vllm/engine/disaggregated/` | Disaggregated Serving (미래용) |

---

## CLI 옵션

```bash
# 기본 실행 (자동 감지 — 권장)
vllm serve <model> \
  --tensor-parallel-size 8 \
  --hybrid-mode parallel-batch

# 수동 오버라이드
vllm serve <model> \
  --tensor-parallel-size 8 \
  --hybrid-mode parallel-batch \
  --hybrid-cpu-max-seqs 28 \
  --hybrid-cpu-kvcache-gb 800 \
  --hybrid-cpu-threads 112 \
  --hybrid-cpu-max-batched-tokens 7168
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--hybrid-mode` | none | parallel-batch / moe-hybrid / none |
| `--hybrid-cpu-ratio` | auto | CPU 비율 (CapacityAwareRouter 사용 시 무시됨) |
| `--hybrid-cpu-max-seqs` | 0 (auto) | CPU 최대 동시 시퀀스 (auto: 물리코어/4) |
| `--hybrid-cpu-kvcache-gb` | 0 (auto) | CPU KV cache GB (auto: 총메모리*0.4) |
| `--hybrid-cpu-threads` | 0 (auto) | CPU 스레드 수 (auto: NUMA 노드 물리코어) |
| `--hybrid-cpu-max-batched-tokens` | 0 (auto) | CPU 배치 토큰 수 (auto: seqs*256) |
| `--hybrid-numa-aware` / `--no-hybrid-numa-aware` | True | NUMA 최적화 |
| `--hybrid-numa-node` | auto | NUMA 노드 지정 |
| `--hybrid-routing-strategy` | capacity | 라우팅 전략: capacity / length-aware / throughput-adaptive |
| `--hybrid-cpu-prefill-threshold` | 512 | length-aware/throughput-adaptive에서 CPU 최대 프롬프트 토큰 수 |
| `--hybrid-warmup-requests` | 10 | 워밍업 프로파일링 요청 수 (디바이스당, 0=비활성화) |
| `--hybrid-stats-log-interval` | 50 | 통계 로깅 간격 (완료 요청 수, 0=비활성화) |

---

## 환경 변수 (하이브리드 모드에서 자동 설정됨)

```bash
# _setup_cpu_process_env()에서 자동 설정:
CUDA_VISIBLE_DEVICES=""              # GPU 격리
VLLM_CPU_KVCACHE_SPACE=<auto>        # CPU KV cache GB
OMP_NUM_THREADS=<auto>               # NUMA 노드 물리 코어 수
VLLM_CPU_OMP_THREADS_BIND=auto       # 스레드 자동 바인딩
VLLM_CPU_NUM_OF_RESERVED_CPU=0       # 예약 코어 없음

# configure_intel_optimizations()에서 자동 설정:
KMP_AFFINITY=granularity=fine,compact,1,0
KMP_BLOCKTIME=1
MKL_ENABLE_INSTRUCTIONS=AVX512
ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX   # AMX 가용 시
```

---

## 빌드

```bash
# CUDA + CPU 하이브리드 빌드
pip install -e . --config-settings="cmake.args=-DVLLM_TARGET_DEVICE=cuda"

# 빌드 결과: _C.abi3.so (CUDA) + _C_cpu_ops.abi3.so (CPU 커널)
```

## 검증

```bash
# CPU 기능 감지
python -c "
from vllm.platforms.intel_cpu_utils import detect_intel_cpu_features
f = detect_intel_cpu_features()
print(f'{f.model_name}: {f.num_sockets}S x {f.cores_per_socket}C x {f.threads_per_core}T')
print(f'AVX-512={f.avx512f}, VNNI={f.avx512_vnni}, AMX={f.amx_bf16}')
"

# 프로세스 확인
ps aux | grep -E "GPU_EngineCore|CPU_EngineCore"
```

---

## 문서 참조

| 문서 | 내용 |
|------|------|
| `Deployment.md` | 배포 가이드 (빌드/설치/실행/트러블슈팅) |
| `docs/HYBRID_OPTIONS_IMPLEMENTATION_PLAN.md` | 하이브리드 옵션 상세 설계 |
| `docs/HETEROGENEOUS_CPU_OPTIMIZATIONS.md` | CPU 최적화 상세 |
| `docs/AVX512_OPTIMIZATION_IMPLEMENTATION_PLAN.md` | AVX-512 커널 구현 계획 |
| `analysis/overview.md` | 시스템 아키텍처 분석 |

---

## 호환성

| 환경 | AVX | NUMA | IPEX | AMX | 상태 |
|------|-----|------|------|-----|------|
| H100 서버 (Xeon 8480+) | AVX-512 | 멀티소켓 | 지원 | 지원 | 최적화 |
| 개발 머신 (i9-12900KF) | AVX2 | 단일 | 미설치 | 없음 | 호환 |
| 일반 x86_64 | AVX2 | - | - | - | 호환 |

모든 기능은 graceful fallback 지원. IPEX/NUMA/AMX 없어도 정상 동작.

---

*마지막 업데이트: 2026-03-10 (세션 6)*

---

## 작업 이력

### 2026-03-10: Abstract 교체 및 본문 교차 검증 (세션 6)
- 사용자 제공 장문 Abstract로 교체 (6줄 → 14줄)
- 본문 우선 원칙에 따라 3건 수정: "pure energy waste"→"significant energy inefficiency", 중복 "cleanly eliminates" 통합, "all sources"→"software-level" (Claim 1 condition ii 반영)
- Introduction L111 "every source"→"software-level sources" 동일 수정
- 11개 주장 교차 검증 완료: Major 불일치 0건
- 수정 파일: `docs/paper/main.tex`

### 2026-03-10: 3인 페르소나 피어 리뷰 반복 (세션 5)
- 3명 리뷰어 페르소나: A(시스템/OSDI), B(HPC/SC), C(에너지/MLSys)
- 3회 반복으로 Major 0건 달성 (1회차 8건 → 2회차 1건 → 3회차 0건)
- 핵심 수정: Abstract 축약, Theorem→Claim 격하, HPC 유추 톤다운, Roofline KV cache 확장
- 독립 2-인스턴스 baseline 비교 섹션 추가, EMA α→γ 기호 변경
- Table 예측값 Caution 경고, 73% 프레이밍 균형화, Granite/Blackwell 비율 분석
- BSP 형식주의 → "Why separate processes?" 실용적 근거로 교체
- 수정 파일: `docs/paper/main.tex`

### 2026-03-10: 반복 검증 루프로 논문-코드 정합성 완전 검증 (세션 4)
- 4회 반복 (2 에이전트 × 4회 = 8회 검증), 총 16건 수정으로 Major/Critical 불일치 0 달성
- Critical: TP=8 시스템 기여도 9.2% → 1.15% (B_CPU/(k×B_GPU))
- Major: H100 TF32→BF16 dense 정정 (989=BF16 dense, 495=TF32 dense, 1979=BF16 sparse)
- Major: hwloc "defined by" → "conceptual model" (실제 /proc/cpuinfo, lscpu, libnuma 사용)
- Major: FP32 peak 표기 명확화 (2 FMA × 16 elements × 2 ops/FMA = 64 FP32 ops/cycle)
- Major: P_GPU 명시적 정의 추가 (Energy Efficiency Corollary)
- Major: Theorem 1 → "Proof sketch" + 분해 가정의 실험 검증 위임 명시
- Major: Low-load starvation caveat 추가 (CPU-first 시 λ≤N이면 GPU 미사용)
- Major: ZMQ PULL 출력 경로 경합 분석 추가
- Major: GPU tok/s 출처(vLLM 벤치마크) 명시
- Major: Abstract "pure waste" → "significant energy inefficiency"
- Major: 프로세스 시작 방법 → get_mp_context() 동적 선택
- Major: FP32 attention → "per-sequence without batch-16 grouping"
- Major: N_max → 사용자 오버라이드 + 2N_max 초과 설명
- Major: ZMQ identity 표기 일관성 통일 (b'\x00\x00' / b'\x01\x00')
- Major: Q8_0 커널도 VNNI 의존성 필요 → "(1)과 (2) 모두 VNNI 필요"
- Major: Figure 3 mutually exclusive 명확화 + 본문 설명 보강
- 수정 파일: `docs/paper/main.tex`

### 2026-03-10: 논문 비판점 12건 대응 수정 (세션 3)
- **N1(엔지니어링 기여)**: Contribution #1에 "systems integration"으로 재포지셔닝, practical value 강조
- **N3(Theorem 자명)**: Theorem 1에 비간섭 조건 (i)프로세스 격리 (ii)HW 경쟁 없음 명시 + 조건(ii) 논증 단락 추가 (NUMA 분리, HBM/DDR 독립, PCIe 미미)
- **N2(HPC 과장)**: BSP→"Relation to parallel models"로 톤다운, 라우팅 전략 설명을 "inspired by" + 차이점 명시 (work stealing은 proactive dispatch, HEFT는 single threshold, StarPU는 single parameter)
- **T1(메모리 경쟁)**: Theorem proof 뒤에 hardware contention 분석 단락 추가 (NUMA 분리, HBM/DDR 독립 경로, PCIe 미미, 실험 검증 계획)
- **T2(Prefill latency)**: Limitations에 CPU TTFT 10-50x 느림 정량화, length-aware 완화, 실험 우선 계획 명시
- **T4(EMA α=0.3)**: EMA half-life 계산(1.9회), 높은/낮은 α 트레이드오프, sensitivity analysis 계획
- **T5(모델 정확도)**: Limitations에 수치 일관성 논의 추가 (동일 precision, 독립 요청이므로 누적 없음)
- **T6(에너지 미측정)**: Corollary 2 + Discussion에 "theoretical estimate" 명시, RAPL 검증 계획
- **T7(LogGP 불필요)**: LogGP 수식 3개 제거 → 실용적 1문장 결론으로 간소화
- **P1(과잉 형식화)**: "formalize" → "model", "strictly additive" → "additive" 톤다운
- **P3(인용 불균형)**: Related Work에 PowerInfer/HeteGen 정량 비교 추가 + 트레이드오프 명시
- 수정 파일: `docs/paper/main.tex`

### 2026-03-10: 코드 완성도 문제 4건 해결 (세션 3)
- **C++ 커널 연결**: `cpu_attn.py`의 CPU decode 경로에 `_C_cpu_ops.batch16_paged_attention_v1` 디스패치 추가. `HAS_CPU_OPS=True` + BF16/FP32 + num_tokens==num_seqs 조건 충족 시 커스텀 AVX-512 커널 사용, 실패 시 PyTorch SDPA fallback
- **compute_qk_batch16 스텁 제거**: `batch_attention.cpp`에서 미사용 더미 함수 삭제, 파일 헤더 주석을 실제 구현에 맞게 수정 (시퀀스 순차 + head dim SIMD)
- **ParallelBatchExecutor 폐기 표시**: V0 레거시 코드에 deprecated 경고 + 활성 경로(hybrid_core.py) 안내 docstring 추가
- **단위 테스트 30개 작성**: `tests/v1/engine/test_hybrid_core.py` 신규 생성
  - CapacityAwareRouter: capacity(6), length-aware(5), throughput-adaptive(5), warmup(3), fault-tolerance(2)
  - _resolve_cpu_params: auto-detection 공식(5), manual override(1), ResolvedCpuParams(1)
  - 전체 30/30 통과
- 수정 파일: `cpu_attn.py`, `batch_attention.cpp`, `parallel_batch_executor.py`, `test_hybrid_core.py`

### 2026-03-10: 논문 ↔ 코드 교차 검증 및 불일치 수정 (세션 3)
- 논문(main.tex)과 실제 코드 구현을 다각도로 교차 검증
- **불일치 4건 발견 및 논문 수정**:
  1. EMA 스무딩 계수: 논문 β=0.9 → 코드 α=0.3에 맞춰 수정 (수식/표기 전체)
  2. KV cache 상한: 논문 "800GB" → 코드 `max(32, min(512, mem×0.4))`에 맞춰 수정 (Table + 본문)
  3. 동적 슬롯 수식: 논문 `N_max × ratio` → 코드 `clamp(N_max×(1+ratio), 2, 2N_max)`에 맞춰 수정
  4. ONEDNN fallback: non-AMX 시 `AVX512_CORE_VNNI` fallback 추가 기술
- **ParallelBatchExecutor 존재 확인**: 코드 완전 구현(1033줄)이나 실제 호출 경로 없음 (V0 호환용, 미사용)
- **`vllm serve` hybrid 경로 연결 수정**: `make_async_mp_client()`에 `is_hybrid_mode()` 분기 추가 → `HybridAsyncMPClient` 도달 가능하도록 수정. 이전에는 AsyncLLM이 hybrid 분기를 우회하여 GPU-only로 동작했음
- RESEARCH.md 신규 생성 (코드 구현 현황 종합 문서)
- **2차 교차 검증 (소스→논문, 논문→소스 양방향) → 불일치 12건 발견 및 논문 수정**:
  - [높음] 라우팅 우선순위: 논문 "GPU default" → 코드 "CPU-first". Definition 1, Theorem proof, Property 2 전면 수정
  - [높음] NUMA 노드: 논문 "CPU→NUMANode 1" → 코드 "rank=0→NUMANode 0". Fig.4 caption/label, Package 설명 수정
  - [높음] Batch Attention SIMD: 논문 "16 시퀀스×16 lanes 동시" → 코드 "시퀀스 순차, SIMD는 head dim". 설명 수정
  - [중간] 307 GB/s "aggregate" → "per-socket" 수정 (CPU 엔진은 단일 NUMA 노드 사용)
  - [중간] KV cache total_mem → eff_mem (NUMA 노드 메모리) 명시, 예시 수치 수정
  - [중간] 워밍업 강제 완료 메커니즘 (GPU 2W + CPU 1개) 추가 기술
  - [중간] CPU 크래시 장애 허용: "built-in" → "implicit (부수적 효과)" 수정
  - [낮음] ZMQ identity: 0x00/0x01 → 0x0000/0x0100 (2바이트 LE)
  - [낮음] max_seqs: 최소값 4 추가 (Table + 본문)
  - [낮음] cpu_threads: NUMA 비활성화 시 전체 물리 코어 사용 명시
  - [낮음] ISA 감지: "빌드타임만" → "빌드타임(CMake)+런타임(Python)" 수정
- 수정 파일: `docs/paper/main.tex`, `core_client.py`, `CLAUDE.md`, `RESEARCH.md`
- **3차 교차 검증 → 불일치 10건 발견 및 논문 수정**:
  - Fig.4 NUMANode 대역폭: 153.6 → 307 GB/s (8채널)
  - Fig.4 색상 반전 수정: CPU=orange, GPU=green 일관성 복원
  - Fig.3 caption: "layered refinements" → 전략은 mutually exclusive 명시
  - §2.3 전력비: $158K → $18.4K (계산식 명시)
  - §5 프로세스: "spawn" → "fork" 기본값, spawn은 CLI 옵션
  - §5 VNNI: "fused multiply-add" → "VPDPBUSD dot-product accumulation"
  - §6.1 전력비: $15K → $18.4K
  - §1 전력량: 1,500 MWh → 153 MWh (계산식 추가)
  - Fig.1 Xeon peak: 2.0 → 3.6 TFLOP/s (ridge point 6.5→11.7)
  - Table 1 BW: "307 GB/s" → "307 GB/s (per-socket)"
- **4차 교차 검증 (3에이전트: 소스→논문, 논문→소스, 비판적 지도교수) → 7건 수정**:
  - [높음] Property 2 fault tolerance: crash 시 C<N 가능성 명시, health-check watchdog 필요성 추가
  - [높음] Batch Attention: "16 시퀀스 SIMD 동시처리" → "시퀀스 순차 + head dim SIMD" 정확히 수정
  - [중간] throughput-adaptive에 length threshold 적용 사실 누락 → 명시 추가
  - [중간] Fig.3 설명: 시각적 계층 표현이 mutually exclusive와 모순 → 설명문 보강
  - [중간] Roofline 73% 오해 유발 → per-GPU vs system-wide 명확 구분
  - [낮음] "CLI configuration" → 환경변수 VLLM_WORKER_MULTIPROC_METHOD 정정
  - [낮음] NUMA fallback (라이브러리 미가용 시 전체 코어) 명시

### 2026-02-27: 논문 그림 크기 수정 및 논리 흐름 개선 (세션 2)
- §4.3/§4.4 상세 설명 작성 → `docs/DETAIL_INFO_4_3_4_4.md` 저장
- 상세 설명을 논문(main.tex)에 반영: §4.3 HPC 계보 확장, §4.4 NUMA/SMT/hwloc 상세화
- 새 BibTeX 4개 추가: lameter2013numa, eyerman2010smt, vaswani2017attention, hunter1986ema
- TikZ 그림 5개 생성 (fig_script/ 독립 파일 + main.tex 인라인)
  - fig1: Dual-Process Architecture, fig2: Roofline Model, fig3: Routing Flow
  - fig4: hwloc Topology, fig5: Throughput Bar Chart
- Abstract~§4 논리 흐름 검토 → 8개 문제 식별 및 전부 수정
  - §2.4/§2.5 삭제 (중복), Table을 §3.2로 이동, Intro 단축, Roofline 전방/후방 참조
  - Introduction 수식 번호 제거, §4.2 전환 보강, 자기참조 수정
- 그림 크기 수정: Fig.2 resizebox, Fig.4 figure* 두컬럼, Fig.5 annotation axis cs 좌표
- 수정 파일: `docs/paper/main.tex`, `docs/paper/references.bib`, `docs/DETAIL_INFO_4_3_4_4.md`, `docs/paper/fig_script/*.tex`

### 2026-03-10: 6차 검증 결과 논문 수정 (15건)
- H100 989 TFLOP/s가 TF32임을 각주로 명시 (BF16 1,979와 구분, 결론 불변 설명)
- "fork" 기본 → spawn 자동 전환 가능성 명시
- Conclusion "proven" → "analyzed...to be validated empirically"로 톤다운
- Property 2 "CPU is first target" → capacity 전략 한정, 다른 전략 언급
- EMA 수식: per-request throughput 명시, N 변동 범위(0.02-0.10) 및 하향 조정 역할 설명
- Limitations: 모델 가중치 중복 로딩 (~70GB, KV cache 감소, startup 2배) 상세화
- Limitations: continuous batching 상호작용 미분석 사항 추가
- Abstract "zero interference" → "software-level interference + subject to HW contention" 톤다운
- batch-16 attention → BF16 경로 한정 명시 (FP32는 비배치 구현)
- Roofline Corollary에 KV cache 접근 비용 미포함 한계 명시
- TTFT 불일치에 SLO 관점 추가 (latency inconsistency, p99 TTFT 바운딩 어려움)
- Property 2 fault tolerance → "incidental, not designed" 명시
- Fig.2에 Q8_0 Decode 점(OI=2) 추가 (GPU/CPU 양쪽)
- AVX-512 커널 조건부 빌드: "subset selection" → AVX-512F 필수, VNNI 추가 조건 명시
- 에너지 분석: 실 TDP 60-70% 미달 + DRAM 전력 증가 언급
- BibTeX 검증: neo2025/apex2025/dovetail2024 저자 교체, zhao2024hetegen 제목, patel2024splitwise Best Paper Nominee
- cite key 완전 일치 확인 (0 mismatch)
- 수정 파일: `docs/paper/main.tex`, `docs/paper/references.bib`

### 2026-03-10: 5차 검증 결과 논문 수정 (코드 무수정, 논리/자료 기반)
- Fig.2 Prefill(CPU) 좌표 오류 수정: (128, 2.0) → (128, 7.2) (CPU compute ceiling에 일치)
- Xeon 8480+ peak FP32 수정: 3.6 → 7.2 TFLOP/s (2 AVX-512 FMA units, 56C × 2.0GHz × 64 ops)
- Ridge point 수정: 11.7 → 23 FLOP/byte (7.2/0.307)
- Q8_0/BF16 Roofline 불일치 해소: Corollary에 BF16(OI=1)과 Q8_0(OI=2) 모두 명시
- Fig.2 caption에 Q8_0 OI 추가
- "five kernels" → "up to five" + 조건부 빌드(ISA 가용성) 명시
- 350W 전력 명확화: 1소켓 TDP + 1소켓 idle로 분리 설명
- DGX H100 NUMA 토폴로지 caveat 추가: single-socket에서의 DDR5 contention 경고
- Decode-only 분석 한계 명시: end-to-end throughput은 prefill 포함 필요
- Fig.3 caption 개선: capacity 전략이 length/adaptive 단계를 건너뛰는 것을 명시
- Corollary의 throughput ratio에 양자화 형식 독립성 강조
- 수정 파일: `docs/paper/main.tex`

### 2026-02-27: §4 System Design에 HPC 기법 적용 Refinement (Draft v4)
- §4.1: Heterogeneous BSP 형식화 — Dual-Process를 Relaxed HBSP로 위치 지정 [Valiant, CACM 1990]
- §4.2: Roofline 기반 α 상한 Corollary 추가 (B_CPU/B_GPU = 9.2%, TP=8시 73% GPU-equivalent) [Williams et al., CACM 2009]
- §4.2: Corollary 1(GPU Latency Preservation)을 LogGP 모델로 강화 (T_route ≈ 11μs, < 0.04%) [Culler/Alexandrov]
- §4.3: 3가지 라우팅 전략의 HPC 스케줄링 계보 확립 (Work Stealing → HEFT → StarPU)
- §4.3: Algorithm 2 (Length-Aware Strategy) 의사코드 추가, HEFT 원리 연결
- §4.4: hwloc 토폴로지 계층(Machine→NUMANode→Package→L3Cache→Core→PU) 기반 자동 설정 서술
- references.bib에 8개 HPC 참고문헌 추가
- 수정 파일: `docs/paper/main.tex`, `docs/paper/references.bib`

### 2026-02-27: HPC 기법 문헌 조사 (CPU-GPU 이기종 추론 시스템 적용)
- 10개 HPC 주제 영역에 걸친 학술 문헌 및 기술 조사 수행
- Topic 1: Work Stealing & Task Scheduling (Cilk, TBB, StarPU, OmpSs, XKaapi, CoreTSAR)
- Topic 2: HPC Communication Patterns (BSP, HBSP, MPI Asynchronous Progress)
- Topic 3: Roofline Model (Williams et al., Heterogeneous Roofline for MI300A/GH200)
- Topic 4: NUMA-aware Scheduling (hwloc, libnuma, thread/memory affinity)
- Topic 5: Pipeline Parallelism & Overlap (computation-communication overlap, non-blocking MPI)
- Topic 6: Lock-free Data Structures (LCRQ, Michael-Scott Queue)
- Topic 7: Heterogeneous Load Balancing (HEFT, PEFT, E-HEFT)
- Topic 8: Memory Bandwidth Optimization (non-temporal stores, prefetch, Intel CAT cache partitioning)
- Topic 9: Batch Scheduling (SLURM backfill, bin packing)
- Topic 10: Performance Modeling (LogP, LogGP, HLogGP/mHLogGP, PRAM)
- 각 기법에 대해 핵심 개념, CPU-GPU LLM 서빙 매핑, 주요 인용문헌 제공

### 2026-02-26: IEEE 논문 Related Work 섹션 문헌 조사
- 6개 카테고리에 걸친 CPU-GPU 하이브리드 LLM 추론 관련 문헌 조사 수행
- Category 1: LLM Serving Systems (vLLM, FlexGen, DeepSpeed, Orca, AlpaServe, TGI, LightLLM)
- Category 2: Heterogeneous CPU-GPU Inference (PowerInfer, PowerInfer-2, LLM in a Flash, Dovetail, APEX, HGCA, HeteGen)
- Category 3: Speculative Decoding (Leviathan et al., Medusa, EAGLE, EAGLE-2)
- Category 4: Disaggregated/Distributed Serving (Mooncake, DistServe, Splitwise, TetriInfer)
- Category 5: Green AI / Energy-Efficient Computing (Patterson et al., Strubell et al., Wu et al.)
- Category 6: Memory Offloading (DeepSpeed-Inference, ZeRO-Inference, FlexGen)
- 각 논문에 대해 full citation, BibTeX entry, relevance summary 제공

### 2026-02-26: GPU 처리량 실제 프로파일링
- CapacityAwareRouter에 워밍업 프로파일링 페이즈 추가 (첫 N개 요청으로 실측 처리량 수집)
- 워밍업 완료 시 throughput-adaptive EMA 초기화 → 즉시 adaptive 슬롯 조정
- 주기적 통계 로깅 (N 요청마다 GPU/CPU 처리량, cpu_ratio, in_flight 출력)
- CLI 옵션 `--hybrid-warmup-requests`, `--hybrid-stats-log-interval` 추가
- 수정 파일: `config.py`, `arg_utils.py`, `hybrid_core.py`, `core_client.py`

### 2026-02-26: CPU 스케줄링 고도화 + KV Cache 인라인 프리페치
- CapacityAwareRouter에 3가지 라우팅 전략 추가: `capacity` (기존), `length-aware`, `throughput-adaptive`
- CLI 옵션 `--hybrid-routing-strategy`, `--hybrid-cpu-prefill-threshold` 추가
- EMA 처리량 기반 동적 CPU 슬롯 조정 (`throughput-adaptive` 전략)
- `batch_attention.cpp` 6개 블록 루프에 `_mm_prefetch` 인라인 프리페치 삽입 (K/V cache → L2)
- 수정 파일: `config.py`, `arg_utils.py`, `hybrid_core.py`, `core_client.py`, `batch_attention.cpp`

### 2026-02-24: 학술 논문 및 기술 참고문헌 조사
- vLLM Hybrid 프로젝트 관련 15개 주제에 대한 학술 논문 및 기술 참고문헌 웹 검색 수행
- 결과를 `docs/REFERENCES.md`에 정리
