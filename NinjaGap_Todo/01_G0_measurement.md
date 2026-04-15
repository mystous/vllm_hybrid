# 01. G0 — 계측 재정의 (모든 후속 기법의 전제)

**Tier**: -1 (계측)
**상태**: ⭕ 미구현
**우선순위**: 최우선 — **다른 모든 기법의 선행 조건**

---

## 왜 필요한가

현재 `T_hybrid` 가 `T_gpu_only` 대비 26–143× 느린 구체적 원인이 계측으로 분해되어 있지 않다. 가설은:
- **실패 1**: batch scaling 제로 (batch=1 3079ms → batch=16 16390ms, 5.3×. 선형 기대 16× 대비 3× 실패)
- **실패 2**: ISA 경직 (AMX 고정 시 batch=1 에서 AVX-512 대비 2.22× 손해)
- **실패 3**: Dataflow 미설계 (sublayer 8개 체인 = 독립 kernel = DDR 왕복 8회 × batch)

이 중 **어느 것이 실측에서 주된 원인인지** 확정 안 되면 kernel 투자 방향이 틀릴 수 있다. Codex 규율: "계측 전엔 어떤 gain 도 % 로 단정하지 않는다".

---

## 기술적 배경

### 측정해야 할 지표

**CPU-side scaling**:
- `step_ms(batch=1)`, `step_ms(batch=N)` — full forward 시간
- `batch_scaling_ratio = step_ms(batch=N) / step_ms(batch=1)` — 1.0 에 가까울수록 좋음 (완벽 amortization)
- `per_req_cost = step_ms / active_reqs`

**Sublayer breakdown** (매 layer 당):
- QKV projection (fused vs split)
- Attention (prefill QKV · decode single-query KV cached)
- O projection
- RMSNorm (pre-attn, post-attn)
- Gate / Up projection (SwiGLU)
- SiLU activation
- Down projection
- Residual add

**OMP 동작**:
- barrier entry/exit time per parallel region
- thread team create/destroy count (persistent region 이면 0)
- chunk schedule distribution

**Memory**:
- weight read bandwidth (`perf stat -e uncore_imc/mem_bw`)
- packing/repacking count per step (amx tile ↔ zmm)
- L2/L3 cache miss ratio (`perf stat -e l2_rqsts.miss,llc_misses`)

### Profiler 계층

Intel 엔지니어가 권장하는 3단 접근:

1. **Python-level coarse**: `cpu_worker.py` forward hook — 현재 구현됨, 세분화 필요
2. **Intel VTune Profiler**: microarchitectural top-down (frontend / backend / memory bound). [Intel VTune docs](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2024-2/)
3. **Linux perf**: hardware counter 저수준 (cache miss, dTLB load miss, 포트 이용률)

---

## 관련 참고 문헌

- **Codex playbook §6 Tier -1 계측 재정의**: `/vllm_hybrid/ideation/20260415_094148_codex_ninja_gap_modification_playbook.md`
- **Claude 3겹 실패 모델**: `/vllm_hybrid/ideation/20260415_094130_claude_ninja_gap_comprehensive_plan.md` §1-1
- **Intel VTune Top-Down Analysis**: Yasin, A. (2014). "A top-down method for performance analysis and counters architecture." *ISPASS*. https://ieeexplore.ieee.org/document/6844459
- **perf Tutorial (Brendan Gregg)**: https://www.brendangregg.com/perf.html
- **KTransformers micro-analysis**: [SOSP'25 paper](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) — CPU compute limit 원인 분해 방식 참조

---

## 구체 작업

- [ ] `eval/cpu_profile*.sh` 에 `num_seqs=1/2/4/8/16` sweep 고정 (dev + H100x8 동일)
- [ ] CPU-only 와 hybrid CPU engine 동일 shape 비교 harness 구축
- [ ] `cpu_worker.py` 의 `attn/mlp` coarse hook 을 sublayer 수준으로 세분화:
  - QKV (split or fused)
  - O projection
  - RMSNorm × 2
  - Gate projection
  - Up projection
  - SiLU
  - Down projection
  - Residual add
- [ ] per-step barrier/sync time 측정 marker (`omp_get_wtime()` 기반)
- [ ] memory wait 측정 (packing/repacking count)
- [ ] H100x8 + dev (RTX3090) 동일 CSV schema 로 저장
- [ ] Intel VTune run 1회 (top-down metric: frontend bound / backend memory bound / backend core bound / retiring)
- [ ] Linux perf run 1회 (L2/L3 miss, dTLB miss, uncore BW)

---

## 성공 조건

산출물:
1. `batch_scaling_ratio` 가 `num_seqs = 1/2/4/8/16` 각각에 대해 측정됨
2. `per_req_cost` 그래프 — batch 증가에 따른 개선 여부 확인
3. Sublayer breakdown 으로 **top-2 bottleneck sublayer** 식별
4. `num_seqs` 증가 시 어느 sublayer 가 **선형적으로 폭증** 하는지 특정
5. OMP barrier/sync overhead 가 step 시간 중 차지하는 비율

이 5개가 나오기 전에는 §06 이후 어떤 kernel 수정도 시작 금지.

---

## 의존성

- **선행**: 없음 (G0)
- **후속**: 모든 기법 (§02~§22) 이 본 계측 결과에 근거해 우선순위 조정

---

## 리스크

- **계측 overhead 자체가 측정 왜곡**: hook/marker 비용을 빼고 기록하거나, enable/disable 비교로 bias 측정
- **VTune / perf 컨테이너 권한**: `perf_event_paranoid` + `CAP_SYS_ADMIN` 필요
- **sublayer 분해가 안 됨** (IPEX 가 layer 를 하나로 묶어서 hook point 없음): Intel VTune "Hotspots" 로 자동 분해 시도, 또는 IPEX optimize 우회 path 에서 측정

---

## 관련 코드 위치

- `vllm/v1/worker/cpu_worker.py` — forward hook
- `eval/cpu_profile.sh`, `eval/cpu_profile_dev.sh` — sweep harness
- `eval/monitor.py` — 1Hz CSV (이미 raw per-logical-CPU schema)
- `eval/basic/H100x8/analysis_h100.ipynb` — 분석 템플릿
