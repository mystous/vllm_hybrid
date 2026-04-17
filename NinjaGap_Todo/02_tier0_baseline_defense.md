# 02. [Tier 0] 기준선 방어

**Tier**: 0
**상태**: ✅ 완료 (2026-04-17, 22afea529)

---

## 왜 필요한가

Ninja Gap 본선 (G1~G3) 에 들어가기 전, **잘못된 routing/tail 증폭이 후속 kernel 실험의 신호를 흐리지 않도록** baseline 을 고정한다. `cpu_max_num_seqs > 1` 이나 `wave-batch` 가 켜진 상태로 kernel 수정을 하면 "개선 효과가 있는지" 를 판단 불가.

**Codex 원칙**: "wave-batch 제거로 tail 이 줄어도 CPU 처리량 기여가 없으면 성공으로 보지 않는다" — Tier 0 자체는 Ninja Gap 이 아니며, **이후 실험의 신뢰도를 위한 장치**일 뿐.

---

## 기술적 배경

### 왜 `cpu_max_num_seqs=1` 이 baseline 인가

현재 H100x8 실측에서 batch scaling 실패가 확인됨 (batch=1 3079ms → batch=16 16390ms, 5.3×). 즉 `cpu_max_num_seqs` 확대가 per-req cost 를 낮추지 못하고 tail 만 키움. 따라서:

- **batch scaling kernel (§07~§11) 확립 전까지는 per-NUMA 1 seq** 고정이 가장 깨끗한 비교 기준
- `cpu_max_num_seqs=1` 은 `_resolve_cpu_params()` 의 auto 기본값이므로 이미 적용됨
- 수동 override (`--hybrid-cpu-max-seqs=4` 등) 는 **경고 로그 + 실험 결과 분리 기록**

### `wave-batch` vs `throughput-adaptive` vs strict continuous

현재 router 옵션:
- **capacity** (기본): `cpu_in_flight < cpu_max_num_seqs` 면 CPU, 아니면 GPU (논문 Algorithm 1)
- **wave-batch**: `cpu_max_num_seqs` 개를 한 번에 묶어 CPU 로 보내고 drain 까지 admit 중단 → **tail 최대화 위험**
- **throughput-adaptive**: EMA 기반 expected finish time 비교 (`cpu_finish ≤ gpu_finish` iff CPU)
- **strict continuous**: capacity 와 유사하지만 wave close 로 batch 형성 안 함

**기준선 방어 지침**:
- `wave-batch` 를 기본 전략에서 제외 (비교 대상으로만 유지)
- 동일 workload 에서 `throughput-adaptive` vs strict continuous 비교 harness 설정
- 실험 결과는 각 전략마다 별도 디렉토리에 격리

---

## 관련 참고 문헌

- **논문 §3 Algorithm 1** (capacity), **Algorithm 2** (length-aware), **Algorithm 3** (throughput-adaptive): `/vllm_hybrid/docs/paper/main.tex`
- **Codex playbook §6 Tier 0 기준선 방어**: `/vllm_hybrid/ideation/20260415_094148_codex_ninja_gap_modification_playbook.md`
- **H100x8 worst-case timeframe**: `/vllm_hybrid/eval/basic/H100x8/20260415_031045_worst_case_timeframe.md` — wave-batch 로 CPU tail 99.2% 지배한 실측
- **실측 비교** (3 runs max_seqs=1/16 × threads=32/56): `/vllm_hybrid/eval/basic/H100x8/20260415_*/`

---

## 구체 작업

- [ ] **기본 실험 설정 확정**: `cpu_max_num_seqs=1` / strategy=`capacity` / priority=`cpu-first` (현재 auto 와 동일, 명시적 env 파일로 박제)
- [ ] **`wave-batch` 를 기본 전략에서 내림**: `_default_strategy` 가 `capacity` 임을 재확인. 실험용 env 파일 (`h100x8_*_wavebatch.env`) 로 분리
- [ ] **비교 harness**: 동일 model × workload 에서 `capacity` vs `throughput-adaptive` vs strict continuous 3개 전략 스윕 스크립트
- [ ] **수동 override 경고 강화**: `_resolve_cpu_params()` 에 `cpu_max_num_seqs > 1` override 시 `[HYBRID-WARN] principle violation` 로그 (이미 부분 구현)
- [ ] **전략별 측정 폴더 분리**: 라우팅 전략 비교 시 `HYBRID_ROUTING_STRATEGY` / `HYBRID_ROUTING_PRIORITY` 를 별도 run 으로 돌려 `measurement_results/<HW>/g0_<NN>_capacity/`, `g0_<NN>_wavebatch/` 등으로 수동 이동
- [ ] **Huge Pages / WoQ INT8 은 별도 low-risk 실험** 으로 분리 (§03, §04 에서 다룸)

---

## 성공 조건

- 동일 모델 × workload 에서 `capacity` vs `throughput-adaptive` 간 **wall 차이가 통계적으로 유의** 한지 확인
- `cpu_max_num_seqs=1` 고정 상태에서 tail 이 `wave-batch(16)` 대비 얼마나 줄어드는지 정량화
- 이후 §06~§17 kernel 실험의 **baseline (pre-fix) 수치** 확정

---

## 의존성

- **선행**: §01 G0 계측 (baseline 수치 확정 필요)
- **병행**: §03, §04, §05 Tier 0 low-risk 실험
- **후속**: §06~§22 모두 본 baseline 대비 개선을 측정

---

## 리스크

- **baseline 자체가 shifting**: Tier 0 Quick Wins (Huge Pages / WoQ) 가 들어가면 baseline 재측정 필요. 해결: Tier 0 완료 후 "post-tier0-baseline" 을 새 기준으로 고정
- **`throughput-adaptive` 가 현재 workload 에서 항상 GPU 수렴**: router fix 이후 CPU 가 실제로 도움 되는 영역이 좁아짐. `capacity` 만 유의미할 수 있음. dev (RTX3090) 에서 weak GPU 상태 교차 검증

---

## 관련 코드 위치

- `vllm/v1/engine/hybrid_core.py` — `CapacityAwareRouter`, `_resolve_cpu_params`
- `vllm/engine/arg_utils.py` — `hybrid_routing_strategy`, `hybrid_cpu_max_seqs`
- `eval/envs/h100x8_*.env` — 전략별 실험 env
- `eval/bench.sh` — 벤치 harness
