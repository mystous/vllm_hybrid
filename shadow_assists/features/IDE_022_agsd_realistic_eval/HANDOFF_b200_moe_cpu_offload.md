# HANDOFF — B200에서 이어갈 작업: MoE CPU offload 검증

> **목적**: 이 문서는 dev 박스(RTX 3090 + i9, AMX 없음) 세션에서 **B200 컨테이너 세션으로 작업을 넘기기** 위한 핸드오프다. dev 박스에서는 충실한 측정이 불가(아래 §환경)하여, 결정적 측정을 B200에서 진행해야 한다.
> **작성**: 2026-06-06 (dev 박스 세션)
> **연관 ID**: `TSK_043`(Host-Side Slack Reclamation, 활성) / `TSK_044`(AGSD 분류기, **기각**) / `SUB_201`(host-bound 재정립, §5 verdict 완료) / `SUB_202~211`(lever PoC)

---

## 0. 한 줄 요약
spec-decode 특성화(TSK_042, 222셀)로 "CPU host-path reclamation"은 abundant-B200에서 이득이 작음이 거의 확정됐고, **유일하게 검증값 있는 CPU→throughput 길은 "MoE expert를 CPU/AMX로 offload"** 뿐이다. 단 이는 abundant-HBM(B200)에선 per-model 손해 가능성이 크고, **"GPU 대수 절감 → 클러스터 capacity"** 축에서만 net 이득 가능성이 있다. 이를 **B200에서 짧게 측정**해 확정하는 것이 다음 작업.

---

## 1. 지금까지의 결론 (측정 기반)

### 1.1 특성화 (TSK_042, B200 222셀, 확정)
- spec-decode(suffix)가 지배 lever: mix +83~232% (8/10 모델 net-positive).
- 부호는 **모델 유형**이 결정: 추론/MoE 모델(DS-Llama-70B, R1-671B)은 chat·code **둘 다** 음, 그 외는 양.
- **코드 워크로드가 저하 원인이 아님** — 실 코드는 오히려 suffix를 더 잘 받음(반복적 구문). code Δ ≥ chat Δ (8/10).
- **gate/분류기(TSK_044)는 무가치** → 기각. method 선택은 "모델당 정적 규칙"(배포시 결정)이지 runtime 분류기 불요. → 논문에선 *발견(negative)*으로 둠.

### 1.2 CPU reclamation lever (SUB_201 §5 + PoC, 측정)
- §5 verdict: 3모델 전부 **host-bound** (Qwen-7B launch 36%, Llama-70B memcpy 80%, R1 launch+memcpy 70%).
- lever PoC 결과:
  - **B3 = FaP(`cudagraph_mode=FULL_AND_PIECEWISE`) +30%** → 단 이건 **vLLM-native GPU-side 설정**(기본값), spec-decode에 가산·직교. **우리 CPU 기여 아님**.
  - B1 detok +0.68%(negligible), A1 CPU draft 조건부, A2 KV tier −16.2%, B2 기각.
  - → **abundant-HBM에서 CPU 직접 기여는 작음**(과거 NEO net-negative와 일치).

### 1.3 CPU→throughput 검증 문헌 (regime 주의)
- KTransformers(SOSP'25, **SGLang 통합**): MoE expert CPU/AMX offload, AMX 5.4 TFLOPS, decode **1.25–4.09×**, trillion-MoE 220 tok/s. **단 A100-40G(메모리제약) 측정** — full-HBM GPU 대비가 아님.
- Fiddler(ICLR'25), MoE-Lens, MoE-Gen, FastDecode, NanoFlow — 전부 **메모리제약/저-GPU regime**에서 측정.
- 즉 **검증된 CPU 이득은 전부 "VRAM 부족/저-GPU" regime**. abundant-B200으로 직접 transfer 안 됨.

### 1.4 dev 박스 microbench (참고용, 충실하지 않음)
- MoE expert GEMM (R1형, B=64): GPU 3090 457k tok/s vs **CPU MKL fp32 4.9k → ~93×**. (torch CPU bf16은 6 tok/s로 깨짐, 무시.)
- ⚠ 이 CPU엔 **AMX 없음** → 타깃 Xeon-8570(AMX)이면 ~30×로 좁혀짐. 3090≠B200, compute-ratio지 decode-bandwidth 아님. **결론 근거로 못 씀** — B200 실측 필요.

---

## 2. 열린 질문 (B200에서 측정할 것)
**Q: R1-671B(또는 작은 MoE)를 B200에서 expert만 CPU/AMX로 offload하면, per-model 및 클러스터(tps-per-GPU) 관점에서 net 이득인가?**
- 가설 A(예상): abundant-HBM이라 per-model은 손해(HBM 8TB/s ≫ DRAM 0.7TB/s). → measured negative면 그것도 논문 카드.
- 가설 B(희망): GPU 대수 절감(예 8→4 GPU + AMX)으로 tps-per-GPU↑ → 남는 GPU로 타모델 → 클러스터 throughput↑.

---

## 3. SHORT 측정 프로토콜 (B200 컨테이너)
**전체 매트릭스 금지. 신호만 짧게.**

### 3.1 1단계 — 작은 MoE proxy (수 분)
- 모델: **Qwen3-30B-A3B**(권장, MoE 성질 동일·부팅 빠름) 또는 Mixtral-8x7B.
- 입력: **20 prompt, max_tokens=256**, decode 위주, conc=8.
- A/B 2회만:
  - **(A)** full-GPU baseline (vLLM 또는 SGLang 기본, FaP on).
  - **(B)** SGLang + KTransformers `kt-kernel`로 **expert→CPU(AMX) offload**.
- 측정: decode **tps**, 사용 **GPU 수**, **tps-per-GPU**.
- 판정: B/A < 1 이면 "abundant offload 손해" 확정 → §2 가설 A. B/A ≥ 1 또는 tps-per-GPU↑면 → 2단계.

### 3.2 2단계 — R1-671B (1단계가 유망할 때만)
- R1-671B: (A) 8-GPU full-HBM vanilla(현 1,538 tps mix 기준) vs (B) N-GPU(2/4)+CPU-AMX expert offload.
- 동일 입력 소량(20~50p). tps-per-GPU 비교. 남은 GPU로 2nd 모델 동시 서빙 시 클러스터 tps까지.

### 3.3 설치 (B200 컨테이너, `/workspace/vllm_dev_prj`)
- `uv pip install sglang` + KTransformers kt-kernel (LMSYS 블로그 https://lmsys.org/blog/2025-10-22-KTransformers/ 절차).
- AMX 확인: `lscpu | grep amx` (Xeon 8570 = Emerald Rapids, AMX native).
- 함정: vllm serve PID/pgroup kill (CLAUDE.md), `--disable-log-requests` 미지원, completions에 실모델명.

### 3.4 산출물
- `SUB_201_cpu_host_path_bottleneck/poc/moe_offload/MEASUREMENTS.md` 에 A/B tps·GPU수·tps-per-GPU + 판정.
- 유망하면 신규 **TSK_045 (MoE CPU offload)** 발급(id_registry 다음 TSK_045), 아니면 measured-negative로 논문 §discussion에 기록.

---

## 4. 환경 메모
- **이 세션(dev)**: RTX 3090 24GB + i9-12900KF(**AMX 없음**), vllm dev. → MoE offload 충실 측정 **불가**.
- **B200 컨테이너**: 8× B200 183GB sm_100 + Xeon Platinum 8570(224스레드, AMX), 2TB DRAM. venv `/workspace/vllm_dev_prj`. 빌드 메모리 `b200-vllm-build`.
- 데이터/코드: TSK_042 raw `vllm_config_perf/gating/realistic_eval/runs/`, reclamation 코드 `vllm/v1/spec_decode/cpu_amx*.py`, `vllm/v1/core/kv_dram_tiering.py`.

---

## 5. 논문 현황 (paper/)
- B200 특성화 + host-side reclamation(방향, TBD)으로 재구성 완료. FaP는 vLLM-native(직교)로 분리. gate는 llm-d 대체 "발견"으로.
- 비어있는 TBD: reclamation 성능(tbl_reclamation), gate regret(tbl_gate). → 위 측정/TSK_044 기각으로 일부는 "측정 안 함"으로 정리될 수 있음.

---

## 6. B200 세션 시작 프롬프트 (이 문서 기준)
아래 §프롬프트를 B200 컨테이너 세션에 붙여넣어 이어간다.
