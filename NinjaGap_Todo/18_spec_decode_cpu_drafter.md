# 18. Spec Decode CPU Drafter (DuoDecoding 스타일)

**Tier**: 병행 트랙
**상태**: 🔶 부분 구현 (GPU-only spec decode 프레임워크만 존재)
**예상 이득**: TPOT **2.1–2.61×** (DuoDecoding 실측). TTFT 17% 감소
**Ninja Gap 기여도**: **매우 큼** — wall 공식 자체 변경

---

## 왜 필요한가

경로 1 (CPU 자체 가속, §02–§17) 만으로 Ninja Gap 도달 실패 시 **Plan B**. 경로 1 이 cost_cpu / cost_gpu 를 28× → 15–20× 까지만 줄여도, Spec Decode 가 wall 공식을 "CPU 가 전체 처리" 에서 "CPU 가 draft 만" 으로 바꿔 **tail 소멸**.

**권장 시나리오 (확률 50%)**: "경로 1 + Spec Decode 조합". 경로 1 15-20× + DuoDecoding 2× 추가 → Ninja Gap.

---

## 기술적 배경

### Speculative Decoding 원리

**Target model** (큰 모델, 예: Qwen2.5-7B) 의 다음 token 을 검증:
1. **Drafter** (작은 모델, 예: Qwen2.5-0.5B) 가 k 토큰 **미리** 생성
2. Target 이 k 토큰 **한 번에** verify (parallel forward, GPU 에 유리)
3. Accept 된 앞부분 + 첫 reject 직후 target 의 correction token → 최종 output
4. Accept rate ~70% 이면 평균 **k-1 토큰 무료**

**Wall 공식**:
```
Standard:   T = N_tokens × cost_target_decode
Spec:       T = N_tokens × cost_target_decode / (1 + accept_rate × k)
            + k × cost_drafter × N_tokens × (1 - accept_rate × k)
```

DuoDecoding 은 **CPU 가 drafter, GPU 가 target** — drafter/target 를 병렬 실행. CPU drafter 시간이 GPU verify 와 겹치면 drafter cost 감춰짐.

### CPU Drafter 의 Balance 조건

**전제 (DuoDecoding)**: CPU drafter 의 throughput ≥ GPU target 의 per-token budget.
- GPU target (7B) per-token ≈ 25ms
- CPU drafter (0.5B) per-token 필요 ≤ 25ms
- 현재 CPU 성능 기준 (7B per-token 1980s / 16 req / 128 tok = ~2000s/tok 극단) → 0.5B 는 ~300s/tok (1/7 비율)
- **전혀 못 맞춤** → 경로 1 필수 선행

경로 1 후 CPU 가 ~15× 빠르다고 가정: 0.5B per-token 20ms → GPU 25ms 에 근접 → 성립

### vLLM 기존 spec decode

`vllm/v1/spec_decode/` 에 proposer 구조:
- `ngram_proposer.py` — n-gram 기반 drafter (외부 model 없음)
- `eagle_proposer.py` — EAGLE 방법
- `medusa_proposer.py` — Medusa heads
- 모두 **GPU-on-GPU** — drafter 도 GPU 에서 실행

**CPU drafter 경로 없음**. 별도 프로세스 + ZMQ IPC 필요.

### 아키텍처 — 3rd EngineCore

기존 hybrid dual-process:
- GPU EngineCore (identity `\x00\x00`)
- CPU EngineCore (identity `\x01\x00`, per NUMA `\x02\x00`...)

추가:
- CPU Drafter EngineCore (identity `\x03\x00` or 재지정)
- Router 가 요청을 **GPU (verify) + Drafter (draft)** 양쪽에 fan-out

### Accept/Reject 로직

```
Drafter 출력: [d_1, d_2, ..., d_k]
Target 출력: [t_1, t_2, ..., t_k, t_{k+1}]  (verify 후, 마지막은 correction)

for i in range(k):
    if t_i == d_i:
        accepted.append(t_i)
    else:
        output = accepted + [t_i]  # first mismatch replaced by target
        break
else:
    output = accepted + [t_{k+1}]  # all accepted, bonus token
```

샘플링 포함 시: target 의 확률 분포와 drafter 분포를 **rejection sampling** (Leviathan et al. 2023).

### Accept Rate 의존

- 동일 family (Qwen-7B + Qwen-0.5B): accept ~70-80%
- Cross-family: accept 저하 → 효과 축소
- Temperature 높을수록 accept 저하

---

## 관련 참고 문헌

- **DuoDecoding (arXiv 2503.00784)**: Lv et al. "DuoDecoding: Hardware-aware Heterogeneous Speculative Decoding with Dynamic Multi-Sequence Drafting" https://arxiv.org/abs/2503.00784
- **DuoDecoding GitHub**: https://github.com/KaiLv69/DuoDecoding
- **Leviathan et al. (2023) "Fast Inference from Transformers via Speculative Decoding"**: https://arxiv.org/abs/2211.17192
- **Chen et al. (2023) "Accelerating Large Language Model Decoding with Speculative Sampling"**: https://arxiv.org/abs/2302.01318
- **EAGLE (Li et al. 2024)**: https://arxiv.org/abs/2401.15077
- **Medusa (Cai et al. 2024)**: https://arxiv.org/abs/2401.10774
- **vLLM Speculative Decoding docs**: https://docs.vllm.ai/en/latest/usage/spec_decode.html
- **vLLM spec decode code**: `vllm/v1/spec_decode/`
- **TODO v5 A1 원본**: `/vllm_hybrid/old_doc/TODO_v5_20260415.md` §1.1

---

## 구체 작업

### 설계
- [ ] **Implementation plan 문서**: `docs/SPEC_DECODE_CPU_DRAFTER_PLAN.md`
  - HybridConfig 확장 필드
  - launch_hybrid_engines 3rd engine spawn 절차
  - _route_speculative fanout 로직
  - accept/reject 구현
- [ ] **Drafter model 선정**: Qwen2.5-0.5B-Instruct (동일 family)
- [ ] **Draft length `k` 결정**: 4~8 범위 실험

### Config
- [ ] **`HybridConfig.spec_decode_draft_model: str | None`** 필드 추가 (`vllm/config.py`)
- [ ] **`HybridConfig.spec_decode_k: int = 4`** 필드
- [ ] **CLI**: `--hybrid-spec-decode-draft-model`, `--hybrid-spec-decode-k`

### Third EngineCore
- [ ] **`launch_hybrid_engines` 확장**: drafter model 지정 시 3rd CPU engine 프로세스 spawn
- [ ] **ZMQ identity 할당**: `b'\x03\x00'` (drafter engine)
- [ ] **Drafter engine 자체 NUMA 분배**: 별도 NUMA 노드 or 기존 engine 과 공유 (아직 미정)
- [ ] **Drafter 가 CPU 전용 (GPU=none)**: 모델만 0.5B 이므로 memory 여유

### Router
- [ ] **`_route_speculative`**: 모든 요청을 GPU (verifier) + drafter engine 에 fan-out
- [ ] **Request 복제 관리**: 동일 request_id 가 2 engine 에서 처리
- [ ] **Token stream 동기화**: drafter 가 k 토큰 완료 대기 → target 으로 전달 → verify 후 accept/reject

### Accept/Reject
- [ ] **`process_engine_outputs` 확장**: GPU verify result + drafter tokens 결합
- [ ] **Greedy accept**: target top-1 과 drafter top-1 비교
- [ ] **Sampling accept** (rejection sampling): Leviathan 알고리즘
- [ ] **V0 `vllm.spec_decode` 코드 차용**: `vllm/spec_decode/metrics.py`, `vllm/spec_decode/spec_decode_worker.py` 참고

### 로깅
- [ ] **`[HYBRID-SPEC-STATS] accept=N/M rate=0.xx k=K`** 주기 출력

### 검증
- [ ] **정확도**: spec decode on/off 의 output token 동일성 (same seed, greedy)
- [ ] **TPOT 측정**: 32B + 0.5B drafter 조합 — DuoDecoding 목표 2.1-2.61×
- [ ] **Accept rate 측정**: 실제 70% 대 달성 여부
- [ ] **1.5B / 7B non-regression**: drafter 없을 때 기존 성능 유지

---

## 성공 조건

1. ✅ Drafter engine 별도 프로세스 spawn + ZMQ 통신 확인
2. ✅ Accept rate 60% 이상 (common workload)
3. ✅ TPOT 2× 이상 개선 (32B target 기준)
4. ✅ 정확도 baseline 대비 무변화 (sampling 동일 random seed)
5. ✅ CPU drafter balance 충족: drafter per-token ≤ GPU target per-token

---

## 의존성

- **선행**: 경로 1 kernel 투자 (§06, §07, §08, §13) 가 CPU 를 충분히 fast 하게 만들어야 drafter balance 성립
- **구현 리소스 배분**: 설계는 Tier 2 중 병행, 구현은 Tier 2 완료 후
- **병행**: 기존 hybrid infra (ZMQ router, dual-process) 재사용

---

## 리스크

- **⚠ CPU drafter balance 실패**: CPU 가 충분히 빠르지 않으면 drafter 가 bottleneck 이 되어 GPU verify 가 기다림 → 역효과. 경로 1 진전 없이는 의미 없음
- **Accept rate 낮음**: model family mismatch, temperature 높음. 0.3B 로 교체 or drop
- **엔진 경계 복잡도**: request_id 가 2 engine 에서 처리, 동기화 버그 가능성
- **Drafter quality 가 target 과 너무 다르면**: accept ≈ 0% → overhead 만 증가
- **Memory 부담**: drafter 0.5B = 1GB BF16, multiple NUMA engine 마다 로드 필요

---

## 스택 호환성

- **경로 1 과 독립 trajectory**: 경로 1 이 CPU 를 빠르게 만들수록 drafter 효과 확대
- §04 WoQ INT8 / §13 T-MAC INT4: drafter model 에 적용 → drafter throughput 추가 2-4×
- **현 workload 에서 최대 Ninja Gap 잠재력**

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `VLLM_HYBRID_PROFILE=1` | 측정 모드 | manifest + sublayer hook 활성 |
| `HYBRID_SPEC_DECODE_CPU` | `0` (기본) / `1` | CPU drafter + GPU verifier 경로 |
| `HYBRID_SPEC_DRAFT_MODEL` | `""` (기본) / `Qwen2.5-0.5B-Instruct` 등 | Drafter 모델 id |
| `HYBRID_SPEC_K` | `4` (기본) | Draft 길이 |

전체 flag 테이블: [README.md](./README.md) "기법 Feature Flag 테이블" 참조.

---

## 관련 코드 위치

- `vllm/v1/spec_decode/` — 기존 (참조)
- `vllm/v1/engine/hybrid_core.py` — `launch_hybrid_engines`, `_route_speculative` (신규)
- `vllm/v1/engine/core_client.py` — 3rd engine launcher
- `vllm/config.py` — `HybridConfig` 확장
- `vllm/engine/arg_utils.py` — CLI
- `docs/SPEC_DECODE_CPU_DRAFTER_PLAN.md` — (신규 설계 문서)
