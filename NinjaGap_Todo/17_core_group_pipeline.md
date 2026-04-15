# 17. Core Group Systolic Pipeline

**Tier**: 3
**상태**: ⭕ 미구현
**예상 이득**: 2–3× latency (4 layer 동시 실행)
**근거 등급**: D → 강한 가설

---

## 왜 필요한가

현재 CPU 경로는 **모든 core 가 동일 sublayer 를 병렬 처리** (pure data parallel). 56 core 가 QKV GEMM 하고, 끝나면 다 같이 Attention, 다 같이 MLP... → **sublayer 간 sync 가 전체 완료 대기**.

GPU SM cluster 는 **다른 SM 들이 다른 layer 를 동시 처리** (pipeline parallelism). CPU 에도 56 core 를 4 group 으로 나누어:
- Group A: Layer i 의 QKV
- Group B: Layer i 의 Attn
- Group C: Layer i 의 MLP
- Group D: Layer i+1 의 QKV (pipeline 진입)

**L3 cache 로 group 간 data 전달** (L3 BW ~수 TB/s, DDR ~300GB/s 의 10× 이상).

---

## 기술적 배경

### Systolic Array 원리

Eyeriss (MIT) 의 systolic 설계:
- Weight stationary: weight 를 processing element 에 고정, input 만 흘림
- Input stationary: input 고정, weight 흘림
- Output stationary: partial sum 누적

CPU L3 로 흘리는 data 는 **activation**. Weight 는 각 group 에 local NUMA/L3 resident (L3 per socket 210MB → 큰 모델 layer subset 만 수용).

### Intra-socket vs Inter-NUMA

Xeon 8480+ L3 는 **per-socket** (not per-core):
- Socket 0: 56 core 공유 L3 210MB
- Socket 1: 56 core 공유 L3 210MB

Systolic 은 **intra-socket** (단일 NUMA 내) 만 효율적. Inter-NUMA L3 access 는 UPI 경유 → 느림.

따라서:
- NUMA 0 engine: 56 core 를 4 group (14 cores each)
- NUMA 1 engine: 동일, independent pipeline
- 두 engine 은 request-level partition (기존 구조)

### Group 간 data 전달

L3 로 전달:
- Group A 의 output (next group 의 input) 을 L3 cache line 으로 flush
- Group B 가 L3 hit 으로 읽음
- `CLFLUSHOPT` / `CLDEMOTE` intrinsic 으로 L2 → L3 강제 이동

**대안**: Group 간 shared buffer 를 L3 에 고정. 단 L3 은 inclusive (L1/L2 subset) → 의도적 eviction 필요.

### Pipeline depth vs latency

Depth 4 → 4 layer 가 동시 in-flight → 초기 pipeline fill 시간 (3 layer cost) 후 steady state.

Long sequence (80 layer) 에서 fill cost 는 전체의 3/80 ≈ 4% → 무시 가능. Short sequence 에서는 부담.

### Dependency 와 barrier

Same-layer sublayer 간 dependency:
- QKV → Attn (Q/K/V 필요)
- Attn → O
- O → RMSNorm → Gate/Up
- Gate/Up → SiLU → Down
- Down → Residual → 다음 layer

이를 4 stage 로 분할:
- Stage A: QKV + Attn (내부 sync)
- Stage B: O + Residual
- Stage C: RMSNorm + Gate/Up
- Stage D: SiLU + Down + Residual

그러나 layer i 의 Stage A 입력은 layer i-1 의 Stage D 출력. Layer 마다 4-stage pipeline 이 in-flight.

### Coherence 비용

L3 shared → cache line 이 다른 core 에서 modified 되면 coherence invalidation. Systolic 에서 producer → consumer 패턴은 false sharing 가능성. **Cache line 정렬** 필수 (64B align).

---

## 관련 참고 문헌

- **Eyeriss (Chen et al. 2017) "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks"**: https://ieeexplore.ieee.org/document/7738524
- **Systolic array theory (Kung 1982)**: "Why Systolic Architectures?"
- **Intel Optimization Reference Manual §"Sharing Data"**: L3 cache line transfer
- **Intel `CLDEMOTE` instruction**: SPR 이상 지원, L1/L2 → L3 강제 이동
- **NPU multi-core pipeline (T-MAN)**: https://arxiv.org/html/2511.11248v1 — 3-stage NPU pipeline 원리 multi-core 로 확장
- **Pipeline parallelism in LLM inference (GPipe, PipeDream)**: layer 단위 pipeline 의 원리 참조
- **Claude Part 2-13 Core Group Pipeline**: `/vllm_hybrid/ideation/20260415_094130_claude_ninja_gap_comprehensive_plan.md`

---

## 구체 작업

### 사전 검증
- [ ] **§01 G0 계측 결과에서 sublayer barrier 가 실제 bottleneck 인지 확인**
- [ ] **Tier 2 (§08, §11, §14) 완료 후 진입** — 기반 kernel 이 fast 해야 group-level 이득 가시화

### 설계
- [ ] **Group 분할 스키마**:
  - 56 core → 14 × 4 groups
  - 각 group 이 맡을 sublayer 배정
  - Layer-stage matrix: layer i 의 stage j → group k
- [ ] **L3 buffer layout**: 각 group 간 shared buffer 위치 (cache line aligned)
- [ ] **Sync 메커니즘**: lockless ring buffer with producer/consumer flag
- [ ] **Pipeline fill/drain 처리**: first/last layer 의 empty slot 관리

### 구현
- [ ] **`vllm/v1/worker/cpu_pipeline_runner.py`** (신규) — 4-group scheduler
- [ ] **`csrc/cpu/pipeline_stage.cpp`** — stage 별 kernel wrapper
- [ ] **Thread team split**: 기존 56-thread OMP team → 4 sub-teams of 14
- [ ] **`CLDEMOTE` 기반 L3 transfer**: stage 경계에서 activation 강제 flush

### 검증
- [ ] **정확도**: pipeline vs sequential 결과 동일
- [ ] **Latency**: 동일 request 의 TTFT, TPOT 비교
- [ ] **Throughput**: batch 당 step time 비교
- [ ] **L3 hit rate**: `perf stat -e offcore_response.*.llc_hit` 측정
- [ ] **Coherence cost**: `perf stat -e mem_load_retired.l3_hit` vs `l3_miss`

---

## 성공 조건

1. ✅ Pipeline pivot 구현 후 정확도 유지
2. ✅ TPOT 30-50% 감소 (4 layer 동시 → 전체 대기 감소)
3. ✅ L3 hit rate 90%+ (group 간 data 가 L3 에서 전달)
4. ✅ DDR BW 소비 감소 (weight 는 각 group local, activation 은 L3)
5. ✅ Throughput 2-3× (batch 크기 유지)

**Stop 조건**: L3 coherence cost 가 DDR 감소분을 상쇄 → systolic 포기, binary dispatch 유지

---

## 의존성

- **선행**: §11, §14 (Tier 2 완료) — 각 stage 의 kernel 이 fast
- **선행**: §12 Barrier reduction — stage 내 sub-barrier 는 최소
- **후속**: CPU scheduler 재설계 (이 규모는 별도 프로젝트)

---

## 리스크

- **매우 높음**: L3 coherence 비용이 DDR 절감분 상회 가능. DDR BW vs L3 BW 실측 차이가 core count, NUMA 구조에 따라 다름
- **구현 복잡도 최상**: vLLM core scheduler + worker + 기존 attention/mlp kernel 재구성. 6주+ 예상 (기간 가정 제거 원칙 상 정성 평가)
- **L3 210MB 한계**: 큰 모델 layer 여러 개 동시 resident 어려움. Qwen 7B layer 128MB 근처 → 1-2 layer 만 L3 resident 가능
- **Layer-stage 매칭 오류 시 정확도 폭락**: sync 실수 → race condition → garbage output
- **Single-request latency 오히려 악화 가능**: pipeline fill cost 가 short request 에 부담

---

## 스택 호환성

- §11 Batch-aware decode attention: stage A (attention) 의 primitive
- §14 AVX/AMX Cascade: single-core cascade → group-level systolic 으로 자연 확장
- §12 Barrier reduction: stage 내 sync 최소화와 동일 원리

---

## 관련 코드 위치

- `vllm/v1/worker/cpu_pipeline_runner.py` — (신규)
- `csrc/cpu/pipeline_stage.cpp` — (신규)
- `vllm/v1/engine/hybrid_core.py` — CPU engine group scheduling
- 별도 설계 문서 필요: `docs/CPU_SYSTOLIC_PIPELINE_DESIGN.md` (사전 설계 필수)
