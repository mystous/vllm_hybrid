# 서버 에이전트 작업 지시 — RBC-Attn v0.1

## 목적

제공된 CPU sparse-edge compiler와 GPU attention kernel을 실제 서버에서 검증한다. 새로운 아이디어를 다시 설계하는 일이 아니라 **이 구현의 정확성·실제 속도·기존 native 대비 차이**를 확인한다. 양자화, 압축, mask 삭제/추가, 모델 변경 금지. 기존 운영 서비스에 자동 패치하지 않는다.

대상: 긴 문맥의 기존 block-sparse attention. 과거 480B INT4 expert 서버를 그대로 다시 측정하는 작업이 아니다. 원 모델이 dense attention이면 임의 top-k를 새로 도입해 RBC 이득이라고 보고하지 않는다.

## 1. 환경과 빌드

1. 별도 디렉터리/가상환경 또는 컨테이너를 사용한다. `nvidia-smi -q`, `nvidia-smi topo -m`, `lscpu`, `numactl --hardware`, CPU governor/주파수, torch/triton/FlashInfer/CUDA/compiler 버전을 기록한다.
2. `bash run_host_tests.sh` 실행. 배포 시 로컬 검증은 460 pass였다.
3. CUDA PyTorch와 Triton이 있어야 한다. `python -m pytest -q tests/test_gpu.py`를 실행한다. CUDA 없음으로 skipped된 것은 통과가 아니다.
4. GPU 코드 v0.1은 작성 환경에서 컴파일되지 않았다. 버전 관련 API/JIT 오류를 고치면 원문 diff를 보존한다. 알고리즘이나 dtype/mask를 바꾸어 오류를 우회하지 않는다.
5. GPU 수치 테스트 실패 시 속도 측정을 진행하지 않는다. tolerance를 결과에 맞춰 넓히지 않는다. sink/softcap 등 미지원 기능이 필요한 입력은 미지원으로 표시한다.

## 2. 첫 실행

```bash
bash build.sh
python -m pytest -q tests/test_host.py tests/test_native_metadata.py
python -m pytest -q tests/test_gpu.py
RBC_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores CUDA_VISIBLE_DEVICES=0 \
  python bench.py --pattern common_private --fresh --native required \
  --out results/common_private.json
RBC_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores CUDA_VISIBLE_DEVICES=0 \
  python bench.py --pattern random --fresh --native required \
  --out results/random.json
RBC_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores CUDA_VISIBLE_DEVICES=0 \
  python stream_bench.py --routing-source gpu --out results/pipeline.json
```

`--native required`에서 FlashInfer가 없으면 설치된 지원 환경을 별도로 준비한다. 불가능하면 명시적으로 `--native off`인 개발 결과만 보존하며 native/SOTA 우위 판정은 미완료로 둔다.

## 3. 빠른 비교 행렬

공통: 같은 QKV, 원래 mask, 같은 dtype, 같은 GPU, 같은 반복 시드. 통과한 작은 구간부터 확장한다.

- query 수 256/512/2048, context 16k/64k/128k, G=4/8/16, selected KV blocks=8/16/32.
- 먼저 common/private 합성과 random 반례를 측정. 이후 실제 모델에서 선택한 여러 층·프롬프트의 mask 및 QKV capture를 재생한다. 유리한 mask만 추려 주 결과로 쓰지 않는다.
- CPU worker 8/16/32: 실제 GPU root와 가까운 NUMA domain에 배치하되 코어 번호는 장비에서 확인한다. OpenMP threadpool과 운영 AMX pool을 중첩하지 않는다.
- 모든 내부 정책은 M=16/32/64/128 탐색 기회를 동일하게 갖는다. Q-outer reorder on/off도 포함한다.
- CPU 계획 시간, metadata 크기, H2D/D2H, GPU main/merge, 실제 wall time을 분리한다. 겹친 시간을 합산하지 않는다.

## 4. 가장 중요한 세 대조

1. **같은 executor:** Q-outer / KV-outer / signature-only / RBC. 앞 둘보다 빨라야 실행 분해의 효과다. signature-only와 같으면 추가 cost-aware merge의 기여는 확인되지 않은 것으로 쓴다.
2. **native:** FlashInfer paged baseline, 실제 사용 중인 sparse kernel, 가능하면 MSA 저자 kernel. 패키지 KV-outer는 저자 커널 자체가 아니다. 코드 작성 기술 차이와 알고리즘 차이를 분리한다.
3. **계획비 포함:** run-only, fresh wall, CPU/GPU chunk pipeline. pipeline은 whole-query 최강 GPU baseline과 비교한다. 일부러 직렬화한 chunk baseline만 이겨서는 안 된다.

CPU 구현의 역할까지 논문 기여로 주장하려면 같은 분해를 GPU에서 만드는 planner도 추가 비교해야 한다. 해당 GPU planner는 v0.1에 없다. 그 비교 전에는 “CPU 배치가 최적”이라고 쓰지 않는다.

## 5. 실제 입력

README의 `save_capture`를 사용한다. Q는 `[Hkv,Nq,G,D]`, K/V는 `[Hkv,Nk,D]`. mask CSR는 Hkv-major/query-major이며 중복 없이 정렬한다. `source_dtype`와 절대 query 위치를 반드시 기록한다. 다른 요청의 cache identity는 분리한다. QKV는 RoPE 등 전처리 완료 후 입력이다.

```bash
RBC_THREADS=16 CUDA_VISIBLE_DEVICES=0 python bench.py \
  --capture captures/layer12.npz --fresh --native required \
  --out results/layer12.json
RBC_THREADS=16 CUDA_VISIBLE_DEVICES=0 python stream_bench.py \
  --capture captures/layer12.npz --routing-source gpu \
  --out results/layer12_pipeline.json
```

현재 GPU-origin metadata 측정은 selected indices D2H를 포함하고 CSR offset/길이는 host 준비 상태다. 실제 모델이 offset도 GPU에서 만든다면 그 비용을 추가한다. 준비되지 않은 다음 layer/token의 mask를 미리 사용하지 않는다. 같은 mask인 것을 확인하지 않고 layer 간 plan 재사용을 하지 않는다.

## 6. 하드웨어 계측

Nsight Systems로 `CPU_RBC_plan_*` 구간과 실제 GPU 실행이 겹치는지 확인한다. Nsight Compute에서는 해당 GPU/버전에서 제공되는 metric 목록을 조회한 뒤 실제 DRAM/L2 bytes, register/spill, shared-memory, occupancy, tensor 실행, kernel 시간을 수집한다. metric 이름을 추측해 0 값을 만들지 않는다.

partial tensor 논리 bytes 감소와 HBM bytes 감소는 다르다. L2에 머물 수 있고, 긴 CTA로 occupancy가 떨어질 수 있다. 매우 큰 M에서 spill이 생기면 M을 낮추되 비교군도 같은 탐색 기회를 갖는다. 작은 toy의 9개 CTA만으로 H100 성능을 주장하지 않는다.

## 7. 결과와 판정

반환 파일:

```text
results/environment.txt
results/host_pytest.txt
results/gpu_pytest.txt
results/*json  # 반복별 원시 값, 실행 순서, 실제 실패 포함
results/native_versions.txt
results/source_changes.diff
RESULT.md
```

`RESULT.md` 첫 표는 입력별로 다음을 보인다.

| 입력 | 실제/합성 | 최강 native ms | 최강 내부 대조 ms | RBC run-only ms | RBC fresh ms | RBC pipeline ms | 정확성 | 판정 |

GPU-only 커널만 좋아지고 CPU 준비를 넣으면 느려지면 실패 결과를 그대로 보고한다. logical partial 87.5% 감소를 8배 가속으로 쓰지 않는다. 이 단계 결과는 operator latency이며 모델 tok/s가 아니다. 실제 모델의 동일 출력 계약·전체 임계경로를 검증한 뒤 전체 가속을 주장한다.

선택적 진입 목표는 실제 fresh/pipeline wall 10% 개선이며 예측/보장 수치가 아니다. 이득이 없으면 native가 이미 재사용하는지, CPU 계획비가 큰지, masked padding/occupancy/merge 중 무엇이 원인인지 원시 계측으로 분류한다. 모델/양자화/선택 block 수를 변경해 성공 결과를 만들지 않는다.
