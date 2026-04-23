# 22. NEO Asymmetric Batch Split

**Tier**: 장거리 (70B / HBM 압박 workload) / **근거 Tier 1 후보** (선행 연구 실측 수치 보유)
**상태**: ⭕ 미구현
**예상 이득**: **H100 70B 14.3%** (MLSys'25 실측, 우리와 동일 HW + 모델 규모)
**근거 등급**: **B** (H100 + 70B 직접 실측). Jiang et al. MLSys'25 "NEO" + GitHub 구현.
**Ninja Gap 기여도**: 7B 제한적, 32B/70B 급에서 효과. 우리 타겟 (Qwen2.5-32B) 에 적합.
**우선순위 근거**: 2026-04-20 Tier 1 후보 정리 시 선정. 우리 측정 환경 (H100x8 + 32B) 과 논문 실측 조건 가장 가까움.

---

## 왜 필요한가

**NEO** (MLSys'25) 은 decode step 의 batch 를 **GPU sub-batch + CPU sub-batch** 로 분할:
- GPU: linear projection (weight-heavy, 병렬성 큼)
- CPU: attention (memory-heavy, KV access)
- **Overlap**: GPU linear 실행 중 CPU 가 attention → 동시 진행

현 hybrid 는 **request-level partition** — 한 request 는 GPU 또는 CPU 전용. NEO 는 **step-level partition** — 같은 request 의 sublayer 가 HW 분할.

70B 에서 14.3% 실측. 7B 에서는 축소.

---

## 기술적 배경

### NEO 의 분할 스키마

Decode step 한 layer:
```
Layer forward(hidden):
    Linear_QKV(hidden) → Q, K, V       # GPU (weight matmul)
    Attention(Q, K, V) → attn_out      # CPU (KV scan, memory-bound)
    Linear_O(attn_out) → out            # GPU
    RMSNorm + Residual                  # CPU or GPU
    Linear_Gate, Linear_Up              # GPU
    SiLU                                # CPU (small)
    Linear_Down                         # GPU
```

**GPU linear 실행 중 CPU 가 이전 layer 의 attention 완료** → 2-stage pipeline.

### Load-aware Scheduling

Workload 의 linear 비중 / attention 비중이 context 에 따라 다름:
- Short context: linear 비중 > attention → GPU 일 많이, CPU 적게
- Long context: attention 비중 > linear → CPU 많이

**Dynamic schedule**: request 별로 linear/attention 비율 측정 후 HW 할당 조정.

### Asymmetric pipeline

Linear (GPU) 와 attention (CPU) 이 **다른 속도** → stage 간 gap 이 throughput 결정:
- `T_stage = max(T_gpu_linear, T_cpu_attention)`
- Balance 가 핵심. CPU attention 이 GPU linear 보다 느리면 overall 느려짐

### CPU 의 attention 역할 재정의

현재 우리 구조: CPU = request 전용 (prefill + decode 전부)
NEO 구조: CPU = **attention 전용 워커** (모든 request 의 attention 만)

이 변경은 **hybrid_core.py 의 routing 원리 재설계**. `_split_batch_asymmetric` 이 step 단위 분할.

---

## 관련 참고 문헌

- **NEO (MLSys'25)**: Jiang et al. "NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference" https://openreview.net/forum?id=umgy9tWBLA
- **NEO GitHub**: https://github.com/NEO-MLSys25/NEO
- **FlexGen (Sheng et al. ICML'23)**: https://arxiv.org/abs/2303.06865 — heterogeneous offload 원리
- **KTransformers MoE hybrid (SOSP'25)**: expert offload 원리 (유사 아이디어)
- **Orca (Yu et al. OSDI'22)**: continuous batching baseline
- **Codex 1630 superset Part 4-5**: `/vllm_hybrid/ideation/20260415_1630_ninja_gap_superset.md`

---

## 구체 작업

### 사전 평가
- [ ] **70B workload 구축** (§TODO H100 §3.1)
- [ ] **Linear / Attention 시간 비율 측정** (각 모델, 각 context 길이에서)
- [ ] **CPU attention 이 GPU linear 와 balance 될 수 있는 shape 탐색**

### 설계
- [ ] **`_split_batch_asymmetric` 알고리즘**: request 별 sublayer 분할 결정
- [ ] **Inter-layer dependency 관리**: GPU linear 결과 → CPU attention 전달 (PCIe DMA)
- [ ] **Pipeline depth**: step 당 2-stage vs 더 deep
- [ ] **Load-aware schedule**: runtime 비율 측정 후 동적 조정

### 구현
- [ ] **`vllm/v1/engine/hybrid_core.py`**: `_split_batch_asymmetric` 신규 routing
- [ ] **CPU worker 역할 재정의**: "attention 전용 워커" 모드
- [ ] **Cross-device data transfer**: Q/K/V 를 GPU → CPU, attn_out 을 CPU → GPU. pinned memory + CUDA stream
- [ ] **Pipeline barrier**: step 경계에서 동기화

### 검증
- [ ] **70B batch 500 throughput**: baseline vs NEO split
- [ ] **Latency**: TPOT 악화 여부 (pipeline overhead)
- [ ] **CPU / GPU util 비율**: top + nvidia-smi
- [ ] **7B 에서도 시도**: 효과 축소 확인

---

## 성공 조건

1. ✅ 70B throughput 14% 이상 개선 (NEO 원 수치)
2. ✅ CPU attention 과 GPU linear 가 overlap 됨 (profile 에서 동시성 확인)
3. ✅ Latency 악화 <5%
4. ✅ Load-aware schedule 이 context 변화에 반응

---

## 의존성

- **선행**: §TODO 70B workload, §11 Batch-aware decode attention (CPU attention 성능 확보), §06 hot path wiring
- **대안 경로**: 7B / 짧은 context 에서는 본 기법 낮은 우선순위

---

## 리스크

- **7B 에서 이득 작음**: 14% 가 아니라 수% 이하 가능성. 70B+ 전제
- **CPU attention 이 GPU linear 보다 느리면 역효과**: balance 실패 시 overall 느려짐
- **현 hybrid 구조 대대적 재설계**: request-level → step-level 은 scheduler 철학 변경
- **PCIe BW**: 매 step 마다 Q/K/V 전송 → 누적 bandwidth 부담
- **NEO 원 구현 open-source 상태 불명**: 포팅 난이도 미확정

---

## 스택 호환성

- §19 P/D disaggregation 과 독립 idea (prefill vs decode 분리 vs step 내 분리)
- §20 KV offload 와 조합: CPU 가 attention 담당하면 KV 가 CPU 에 있음 (offload 자연스러움)
- §21 ScoutAttention 과 유사 방향, 다른 scheme

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `VLLM_HYBRID_PROFILE=1` | 측정 모드 | manifest + sublayer hook 활성 |
| `HYBRID_NEO_ASYMMETRIC` | `0` (기본) / `1` | NEO step-level batch split 활성 |

전체 flag 테이블: [README.md](./README.md) "기법 Feature Flag 테이블" 참조.

---

## 관련 코드 위치

- `vllm/v1/engine/hybrid_core.py` — `_split_batch_asymmetric` (신규)
- `vllm/v1/worker/cpu_worker.py` — attention-only mode
- `vllm/v1/attention/backends/cpu_attn.py` — 수신 path
- 별도 설계 문서: `docs/NEO_ASYMMETRIC_DESIGN.md` (사전 필수)
