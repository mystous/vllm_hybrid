# RBC-Attn v0.1 — Reduction-Budgeted Coalescing Attention

**작성: 2026-09-11. 산출물: C++ CPU 플래너 + Triton GPU 실행기 + 실행 벤치마크.**

## 0. 이번 구현의 계약

RBC-Attn은 이미 주어진 block-sparse attention을 실행하는 새로운 **작업 분해·결합 알고리즘의 프로토타입**이다. CPU는 희소 연결을 분석해 GPU 작업을 재구성하고, GPU는 실제 QK/softmax/PV 연산을 수행한다. 양자화, 가중치·KV 압축, 새 pruning, 모델 학습 변경, 선택된 KV 삭제를 사용하지 않는다.

대상은 **긴 문맥의 sparse prefill 및 여러 query가 이미 준비된 attention**이다. 과거 Qwen3-Coder-480B의 cold-expert 커널을 자동 교체하는 패키지가 아니다. 임의의 dense 모델을 sparse 모델로 바꾸는 기능도 없다. 여러 요청은 cache identity별로 분리해야 한다. 서로 다른 요청의 같은 KV block 번호를 같은 데이터로 합치면 안 된다.

현재 구현 범위: BF16/FP16 forward, GQA group 내 mask 공유, KV head별 서로 다른 mask, 절대 위치 causal mask, tail KV block, 빈 선택 행. Q/K/V에 이미 적용된 RoPE 등의 처리는 유지한다. Attention sink, ALiBi, softcap, dropout, backward, MLA 고유 표현, TP collective 통합은 구현하지 않았다. 이 기능이 있는 모델에 그대로 교체하면 안 된다.

**검증 상태:** C++ 빌드 및 호스트 테스트 460개 통과. CUDA 테스트 24개는 작성되어 있으나 작성 환경에 CUDA가 없어 미실행이다. GPU 커널 컴파일·성능과 전체 모델 성능은 미확인이다. 합성 사례의 바이트 집계는 GPU 가속 실측이 아니다.

## 1. 제거할 낭비

Sparse attention의 대표 실행 방향은 두 가지다.

- Q-outer: query 묶음이 선택한 KV 합집합을 순회한다. partial output은 작지만 서로 다른 선택을 큰 직사각형으로 합치면 마스킹될 계산이 늘 수 있다. 작은 query 묶음은 KV를 반복 방문한다.
- KV-outer: KV block을 사용하는 query를 모아 실행한다. KV 재사용과 Tensor Core 타일 충전은 좋지만, 여러 KV block을 고른 query는 block별 partial output/softmax state를 쓰고 다시 읽어 합쳐야 한다.

RBC는 **여러 KV block이 같은 query 집합에 기여하는 구조**를 찾아 한 GPU 작업에 묶는다. 작업 내부에서 online softmax 상태를 유지하고, 작업 끝에만 partial output을 쓴다. 단순히 작업 순서를 바꾸는 것이 아니라 **출력·재입력해야 할 중간 tensor를 제거**한다.

## 2. 알고리즘

입력은 동일 cache identity의 `Q[Hkv,Nq,G,D]`, `K,V[Hkv,Nk,D]`, query별 selected KV block CSR이다. 한 계획 그룹의 query 위치 수는 최대 `min(64, max_m/G)`이다.

```text
각 KV head, 준비된 query 그룹에 대해:
  1. 각 선택 KV block b의 query membership bitset S[b] 생성
  2. 동일 S[b]를 갖는 KV block을 모아 signature rectangle 생성
  3. 너무 작은 rectangle은 아래 순비용이 양수인 경우에만 병합
  4. 전체 Q-outer 합집합이 더 저렴한 경우 그 경로 선택
전체 작업에 대해:
  5. 너무 적은 CTA가 남으면 긴 rectangle을 KV 방향으로 분할
  6. task별 query/KV/mask descriptor와 query별 reduction CSR 생성
GPU:
  7. 한 CTA에서 여러 KV block 순회, FP32 softmax 상태/출력 누산 유지
  8. 실제 sparse edge만 계산 결과에 반영
  9. query별 partial state 결합
```

C++ 구현은 64-bit membership, cache-resident incidence scratch, signature 정렬, bounded greedy 병합을 사용한다. Query 그룹은 OpenMP로 병렬 처리한다. 임시 구조를 만들기 전에 병합 이득을 먼저 계산해 불필요한 메모리 할당을 줄였다. 기본 `RBC_THREADS=1`이며 서버에서 8/16/32를 비교할 수 있다. 최대 worker 112 제한은 안전 상한이지 권장값이 아니다.

### 2.1 비용과 결정 규칙

작업 t의 query 수를 q, KV block 수를 k, GQA group 크기를 G, head dimension을 D, KV block size를 BK라고 하자.

```text
Q logical bytes      = 2 q G D
KV logical bytes     = 4 k BK D
partial write+read   = 8 q G (D+2)  # FP32 u,m,l
padded dot FLOPs     = 4 round_power2(max(16,qG)) k BK D
```

비용은 위 바이트/연산량의 가중합이다. 현재 기본값 `byte_weight=1, flop_weight=0.015, task_weight=0, partial_weight=1`은 **교정되지 않은 결정 휴리스틱**이다. 시간이 아니며 roofline 예측이나 논문의 검증된 성능 모델이 아니다. `configs/cost.example.json`을 사용하면 이를 바꿀 수 있다. 실제 계수를 맞출 경우 평가 입력과 다른 calibration 입력을 사용해야 한다.

같은 query를 공유하는 두 rectangle을 합칠 때 절약하는 Q·partial bytes와 추가 padded/masked FLOPs를 비교한다. 정확히 같은 signature를 합칠 때는 선택 edge 연산을 늘리지 않고 partial 수를 줄일 수 있다. 유사 signature를 합칠 때는 추가 masking 비용을 지불한다. 병합은 최대 32개 atom을 가진 작은 그룹에서 최대 32회만 탐색한다. 그 이상이면 signature-only 경로를 사용한다.

`min_tasks`는 작업 수를 유지하는 일반적 병렬성 방어다. 신규 기여로 주장하지 않는다. 벤치는 모든 내부 비교군에 같은 조건으로 기본 `2×SM 수`를 적용한다. 작은 toy 사례에서는 이를 0으로 해 구조 자체를 본다.

### 2.2 정확성

CPU plan은 원래 sparse edge를 중복 없이 정확히 한 번 배정한다. 병합 직사각형의 빈 영역은 mask로 제외된다. query i의 task별 상태를 `(u_r,m_r,l_r)`라 하면:

\[
m=\max_r m_r,\quad
l=\sum_r e^{m_r-m}l_r,\quad
u=\sum_r e^{m_r-m}u_r,\quad O=u/l.
\]

이는 동일한 선택 KV에 대한 attention의 실수 연산 결과를 보존한다. **부동소수점 누산 순서가 바뀌므로 bit 동일성은 주장하지 않는다.** BF16/FP16 Q/K/V와 FP32 누산을 사용한다. GPU 테스트는 FP64/FP32 reference와 비교하며 NaN/Inf 오염은 실패로 처리한다. 빈 행은 output=0, LSE=-Inf로 정의한다.

패키지 수치 허용오차는 초기 커널 회귀검사용이다. 이것만으로 모델 품질 비열등성이나 장문 생성 동일성이 증명되지 않는다. sink 등 미지원 의미론이 있는 모델은 별도 구현이 필요하다.

## 3. 구조적으로 양의 절감이 발생하는 사례

8개의 query 위치, G=16, D=128, BK=128을 생각한다. 각 query는 공통 8개 KV block과 자기만 사용하는 8개 block을 선택한다. 개인 block은 다른 query와 겹치지 않는다.

| 동일 mask 실행 | GPU task 수 | query별 partial 슬롯 총합 | KV block 방문 | padded 계산/선택 edge 계산 |
|---|---:|---:|---:|---:|
| Q-outer, M128 | 1 | 8 | 72 | 4.5배 |
| Q-outer, M16 | 8 | 8 | 128 | 1배 |
| KV-outer, M128 | 72 | 128 | 72 | 1배 |
| RBC, M128 | 9 | 16 | 72 | 1배 |

RBC는 공통 8개 block을 8개 query의 작업 하나로, 개인 8개 block은 각 query의 작업 하나로 만든다.

KV-outer 대비 partial write/read는 2,129,920→266,240 bytes이고, Q·KV·partial 합계의 논리 바이트는 7,372,800→5,050,368 bytes다. **이는 소프트웨어의 논리적 메모리 접근량이며 HBM 실측이 아니다.** L2 cache, register spill, occupancy, CPU 준비비 때문에 시간 이득과 다를 수 있다. 9개 CTA만으로 H100 전체를 채울 수 없으므로 성능 벤치는 충분한 독립 그룹 및 병렬성 guard를 사용한다. 단독 toy의 87.5% partial 절감을 전체 속도 8배라고 제시하면 안 된다.

`structural_demo.py`는 random/clustered/disjoint 반례도 함께 기록한다. clustered 동일 mask에서 Q-outer도 잘할 수 있으므로 이를 RBC 독점 성과로 사용하지 않는다.

## 4. CPU·GPU 마이크로아키텍처의 역할

CPU는 작은 정수 bitset/비정형 집합을 처리해 **GPU가 쓸 실행 기술서**를 만든다. AMX에서 attention을 계산시키지 않는다. 준비된 다음 query chunk의 plan을 만드는 동안 GPU가 이전 chunk의 QK/PV/softmax를 계산한다. 아직 생성되지 않은 autoregressive token이나 다음 layer의 Q를 미리 알 필요가 없다.

GPU는 같은 Q 묶음을 유지하며 여러 KV block을 순회한다. FP32 `u,m,l`을 작업 안에 유지해 global-memory partial 기록 횟수를 줄인다. 대신 작업이 너무 커지면 register pressure, shared-memory 사용, 긴 CTA가 문제가 된다. M=16/32/64/128, 최대 KV block 수, 최소 task 수를 비교군과 동등하게 탐색한다.

**Triton 구현은 `tl.load`와 `tl.dot` 기반이다. 수동 TMA나 warp-specialization을 구현했다고 주장하지 않는다.** Hopper에서 register spill과 실제 generated MMA 경로는 Nsight Compute로 확인해야 한다. 양 장치 사이에는 mask/descriptor가 오가며 Q/K/V 전체를 CPU로 복사하지 않는다. 벤치 준비를 위한 실제 tensor capture는 성능 구간 밖의 관찰 도구다.

## 5. 가장 가까운 선행과 차이

- **MiniMax Sparse Attention, §4.2**: 선택 KV block 중심으로 query를 모아 Tensor Core 타일을 구성하고 block별 partial을 결합한다. 우리의 내부 `kv_outer`는 이 실행 방향을 통제하는 자체 구현이지 저자 커널을 그대로 재현한 것이라고 주장하지 않는다.
- **FlashInfer**: CPU plan/GPU run 분리, split-KV, partial attention composition, balanced scheduling은 기존 기술이다. CPU planner·softmax 결합·CTA 분할은 RBC의 새 기여가 아니다.
- **RBC의 제안 차이**: 동일/유사 query-incidence를 가진 **여러 KV block을 하나의 누산 작업으로 묶되**, partial-output 비용과 padded MMA 비용을 동시에 고려해 exact edge partition을 선택한다.

기존 biclique grouping/희소 행렬 타일링 전체에 대한 최초성 증명은 아니다. 논문 기여는 **이 결정 규칙과 실행기 때문에 강한 기준선보다 추가 이득이 나오는지**에 달려 있다. 같은 grouping을 GPU에서 수행하는 정책과도 비교해야 CPU 배치의 추가 가치를 판단할 수 있다. GPU planner는 이 v0.1에 구현하지 않았다.

기준 문헌:
1. MiniMax Sparse Attention (2026), https://arxiv.org/html/2606.13392v1 — §4.2.
2. FlashInfer (2025), https://arxiv.org/html/2501.01005v1 — planning, load balance, attention composition.
3. NVIDIA Hopper Tuning Guide, https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html — 자원·occupancy.
4. FlashInfer API, https://docs.flashinfer.ai/api/attention.html — 실제 native 비교 wrapper.

## 6. 빌드 및 실행

CUDA PyTorch가 이미 설치된 **별도의 실험용 환경**에서 실행한다. 기존 SGLang 환경의 PyTorch를 임의로 교체하지 않는다.

```bash
cd RBC_Attn
python -m pip install -r requirements.txt
bash run_host_tests.sh

# 반드시 CUDA와 Triton 확인. CPU 경로로 가짜 GPU 벤치하지 않는다.
python -c "import torch,triton; assert torch.cuda.is_available(); print(torch.__version__,triton.__version__,torch.cuda.get_device_name())"
RBC_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close \
  CUDA_VISIBLE_DEVICES=0 bash run_gpu.sh
```

Native FlashInfer는 별도 설치된 `flashinfer-python`을 사용한다. `--native optional`로 없으면 상태를 명시하며, 이때 SOTA 우위는 판정할 수 없다. 실제 native 대조를 강제할 때:

```bash
RBC_THREADS=16 CUDA_VISIBLE_DEVICES=0 python bench.py \
  --pattern common_private --nq 512 --nk 65536 --g 16 \
  --fresh --native required --native-backend fa3 \
  --out results/common_native_required.json

RBC_THREADS=16 CUDA_VISIBLE_DEVICES=0 python bench.py \
  --pattern random --fresh --native required \
  --out results/random_native_required.json

RBC_THREADS=16 CUDA_VISIBLE_DEVICES=0 python stream_bench.py \
  --routing-source gpu --chunk-q 256 \
  --out results/stream_gpu_routing.json
```

Native adapter는 원래 KV block을 page로 매핑하고 `(KV head,query position)`을 논리 request로 취급한다. K/V를 query마다 복제하지 않는다. 이는 동등 연산의 native control이지만 최상위 MSA 전용 커널 전체를 대체하는 비교는 아니다. 저자 커널/실제 서비스 kernel을 추가해야 한다. native API가 해당 환경에서 실패하면 원인을 기록하고 같은 Tensor를 느린 구현으로 바꾸지 않는다. 빈 선택 행 native mapping은 명시적으로 미지원 처리한다.

## 7. 실제 모델 입력 연결

기존 sparse attention의 indexer 결과를 그대로 CSR로 만든다. GQA group 안에서 head별 mask가 다르면 이 API에 억지로 합치지 않는다. cache identity와 layer, RoPE 적용 여부, causal positions, precision을 보존한다.

```python
from rbc.data import save_capture
# q_t: [Hkv,Nq,G,D], k_t/v_t: [Hkv,Nk,D], 원래 BF16 예
save_capture('captures/layer12.npz',
    q_t.detach().float().cpu().numpy(),
    k_t.detach().float().cpu().numpy(),
    v_t.detach().float().cpu().numpy(),
    ptr_numpy, block_ids_numpy, bk=128,
    q_positions=absolute_query_positions,
    causal=True, source_dtype='bfloat16')
```

```bash
RBC_THREADS=16 CUDA_VISIBLE_DEVICES=0 python bench.py \
  --capture captures/layer12.npz --fresh --native required \
  --out results/layer12_native.json
RBC_THREADS=16 CUDA_VISIBLE_DEVICES=0 python stream_bench.py \
  --capture captures/layer12.npz --routing-source gpu \
  --out results/layer12_stream.json
```

NPZ의 FP32 container에는 BF16 값을 정확히 저장할 수 있다. source_dtype에 따라 원래 표현으로 되돌리며 성능을 위해 정밀도를 바꾸지 않는다. Capture를 위해 CPU로 옮긴 Q/K/V는 관찰 비용이며 실제 실행은 GPU tensor를 직접 받는다.

## 8. 측정 해석

`bench.py`는 모든 내부 정책에 같은 M 탐색 기회를 준다. Q-outer도 동일한 reorder를 사용한다. mask, Q/K/V, dtype는 모두 동일하다. 각 계획의 결과를 검사한 뒤 무작위 순서로 반복 측정한다.

- `gpu_ms`: 미리 계획된 동일 구조의 CUDA graph 실행. CPU plan/H2D를 포함하지 않는다.
- `fresh_wall_ms`: 새 CPU plan + 할당 + descriptor H2D + GPU 본체 + merge + 동기화. mask는 반복 간 고정하지만 매번 다시 계획한다. JIT 컴파일은 warmup으로 제외한다.
- `stream_bench`: 이미 준비된 query를 chunk로 나누어 CPU plan/GPU 실행을 중첩한다. 기본 GPU-origin indices의 D2H는 포함한다. 현재 CSR 길이/offset은 host에 이미 있다는 계약이다. 실제 indexer에서 이것도 GPU 산출물이라면 추가 복사 비용을 통합해야 한다. Indexer 계산 자체는 포함하지 않는다.

CPU/GPU 시간은 중첩될 수 있으므로 합해서 wall time이라고 쓰지 않는다. CPU 계획 비용이 큰 경우 GPU-only 이득이 사라질 수 있다. native fresh 비교에는 이 Python adapter의 준비비도 포함되므로 production native setup 비용의 최저치라고 간주하면 안 된다. 계획 재사용은 실제 동일 mask인 경우만 허용한다.

**성공 기준:** 동일 입력에서 native 및 내부 최강 Q-outer/KV-outer보다 fresh wall time 또는 실제 pipeline wall time이 감소해야 한다. GPU kernel-only 개선을 CPU–GPU 공동 가속이라고 보고하지 않는다. 최초 확장 목표 10%는 관측치가 아니라 선택적 사전등록 기준이다. 전체 모델 tok/s는 모델 통합 뒤에만 주장한다.

## 9. 파일 목록

| 파일 | 기능 |
|---|---|
| `src/planner.cpp` | 실제 C++/OpenMP exact-edge compiler |
| `rbc/kernels.py` | GPU multi-block attention + partial merge |
| `rbc/runtime.py` | pinned descriptor upload, CUDA workspace lifetime |
| `rbc/native.py` | FlashInfer native paged-attention 대조 |
| `bench.py` | 동일 executor 비교, native 비교, fresh/run-only 측정 |
| `stream_bench.py` | 준비된 query chunk의 CPU/GPU 동시 실행 |
| `rbc/data.py` | synthetic mask / 실제 capture 저장 |
| `tests/test_host.py` | edge partition 및 FP64 수식 검산 |
| `tests/test_native_metadata.py` | native page mapping/절대 causal 경계 검증 |
| `tests/test_gpu.py` | GPU 정확성 검사 24개; 현재 미실행 |
| `results/validation.json` | 실제 수행한 검증 범위 |
| `AGENT_TASK.md` | 서버 작업 지시 |

이 v0.1은 attention 연산자 단위 forward prototype이며, SGLang drop-in patch나 논문 성능 완료본이 아니다. 운영 서버 설정·가중치·서비스 파일을 자동 변경하지 않는다.
