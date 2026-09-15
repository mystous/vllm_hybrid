# IDE_069 상세 보고서 — 480B TP4 하이브리드 처리량 향상 (HBM 추가 사용 + hot expert 배치)

> 부모 IDE_068. 사용자 지시 (2026-09-15): "GPU 4개를 사용한 480B 실험의 성능을 GPU HBM 추가 사용 및 hot expert 배치를 통해서 향상시켜 봐. 다양한 시도를 통해서 metric 을 향상시켜 봐" → 이어서 "TP4+CPU 오프로딩 ×2 와 GPU-only 8장 비교 (hot 상주·비상주 둘 다)", "라우터 실험도 진행".
> 브랜치 `feat/480b-tp4-hot-expert`. 생성: `eval/ide069/make_full_report.py`. **모든 CPU 수치는 turbo OFF 2.0 GHz 하한.**

## 1. 한 줄 요약

| | GPU | C32 tok/s | C64 tok/s |
|---|---|---|---|
| 출발점 — expert 전량 CPU (IDE_068 b1) | 4 | 43.39 (C16) | — |
| **최종 단일 — hot96(hotmap) + deferral 4 + cuda graph + KV 40,960** | 4 | 465 ± 6 (KV24k: 520/487) | **642 ± 7** |
| dual ×2 (소켓별 cpuinfer 48) | 8 | 519.33 | **702.51** |
| 라우터 단일 엔드포인트 (cache_aware / round_robin) | 8 | 487.8 / 478.1 | 649.0 / 633.8 |
| **GPU-only TP8+EP8** | 8 | **1,029.75** | **2,031.11** |

출발점 대비 **14.8×** (GPU 4장). GPU 8장이면 GPU-only 가 dual 하이브리드의 2.9배. 품질: 전 셀 greedy 4/4, GSM 40 = 39/40 (deferral 0·4 동일).

## 2. 하드웨어 구성 (실측 프로브, `hwsw/`)

### 2.1 시스템

| 항목 | 값 |
|---|---|
| 호스트명 | violet-h100-016 |
| 제조사 / 제품 | HPE / HPE Cray XD670 |
| BIOS | American Megatrends International, LLC. CUXD670_5.32_v2.05 (04/11/2025) |
| OS | Red Hat Enterprise Linux 9.4 (Plow) |
| 커널 | 5.14.0-427.124.1.el9_4.x86_64 |
| 역할 | Kubernetes 워커 (containerd). Docker Engine 없음 — `~/bin/docker` = `sudo nerdctl` 셔임 |

### 2.2 CPU

| 항목 | 값 |
|---|---|
| 모델 | Intel(R) Xeon(R) Platinum 8480+ |
| 소켓 / 코어 / 스레드 | 2 소켓 · 소켓당 56 코어 · 코어당 2 스레드 = 논리 CPU 224 |
| NUMA | 2 노드 — node0 0-55,112-167 / node1 56-111,168-223 |
| L1d / L2 / L3 | 5.3 MiB (112 instances) / 224 MiB (112 instances) / 210 MiB (2 instances) |
| 클럭 정책 | `intel_pstate/no_turbo` = 1 (**turbo OFF**), scaling_max_freq = 2000 MHz, governor = performance |
| 행렬 ISA | amx_bf16 amx_int8 amx_tile avx512_bf16 avx512_bitalg avx512bw avx512cd avx512dq avx512f avx512_fp16 avx512ifma avx512vbmi avx512_vbmi2 avx512vl avx512_vnni avx512_vpopcntdq |
| 가상화 / 하이퍼바이저 | 없음 (베어메탈) |

### 2.3 메모리

| 항목 | 값 |
|---|---|
| DIMM 구성 (dmidecode) | 32 	Size: 64 GB; 32 	Configured Memory Speed: 4400 MT/s; 32 	Type: DDR5 |
| 총량 (`free -g`) | 2014 GB |
| 배치 | 소켓당 1 TB (NUMA node 0 / 1) |

### 2.4 GPU

| index | name | uuid | memory.total [MiB] | driver_version | vbios_version | pcie.link.gen.max | pcie.link.width.max | power.limit [W] | clocks.max.sm [MHz] | clocks.max.memory [MHz] |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | NVIDIA H100 80GB HBM3 | GPU-3b84c06a-611d-8e07-2726-44a868f646fe | 81559 MiB | 580.126.20 | 96.00.D0.00.02 | 5 | 16 | 700.00 W | 1980 MHz | 2619 MHz |
| 1 | NVIDIA H100 80GB HBM3 | GPU-8a39f0ec-5884-45d9-f056-53ace5e57df8 | 81559 MiB | 580.126.20 | 96.00.D0.00.02 | 5 | 16 | 700.00 W | 1980 MHz | 2619 MHz |
| 2 | NVIDIA H100 80GB HBM3 | GPU-3f7fea22-2c53-a28e-59a2-07cc6682cad0 | 81559 MiB | 580.126.20 | 96.00.D0.00.02 | 5 | 16 | 700.00 W | 1980 MHz | 2619 MHz |
| 3 | NVIDIA H100 80GB HBM3 | GPU-5cfe0e43-68b1-5c29-9082-c085abacf446 | 81559 MiB | 580.126.20 | 96.00.D0.00.02 | 5 | 16 | 700.00 W | 1980 MHz | 2619 MHz |
| 4 | NVIDIA H100 80GB HBM3 | GPU-43a78ae4-f89f-ca72-2ff2-27e8a8a27f74 | 81559 MiB | 580.126.20 | 96.00.D0.00.02 | 5 | 16 | 700.00 W | 1980 MHz | 2619 MHz |
| 5 | NVIDIA H100 80GB HBM3 | GPU-11f36bf0-2679-fa1a-3940-f938f0b6a05d | 81559 MiB | 580.126.20 | 96.00.D0.00.02 | 5 | 16 | 700.00 W | 1980 MHz | 2619 MHz |
| 6 | NVIDIA H100 80GB HBM3 | GPU-b73567b4-d441-96fe-2b36-39ff827022b2 | 81559 MiB | 580.126.20 | 96.00.D0.00.02 | 5 | 16 | 700.00 W | 1980 MHz | 2619 MHz |
| 7 | NVIDIA H100 80GB HBM3 | GPU-f57f9621-9754-17cd-b327-841e495b1b25 | 81559 MiB | 580.126.20 | 96.00.D0.00.02 | 5 | 16 | 700.00 W | 1980 MHz | 2619 MHz |

드라이버·아키텍처 (`nvidia-smi -q -i 0`, `/proc/driver/nvidia/version`):

```
Driver Version                                         : 580.126.20
CUDA Version                                           : 13.0
    Product Name                                       : NVIDIA H100 80GB HBM3
    Product Architecture                               : Hopper
    Max Clocks
NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.126.20  Wed Feb 18 05:56:34 UTC 2026
```

NVLink / PCIe 토폴로지 (`nvidia-smi topo -m`):

```
[4mGPU0	GPU1	GPU2	GPU3	GPU4	GPU5	GPU6	GPU7	NIC0	NIC1	NIC2	NIC3	NIC4	NIC5	NIC6	NIC7	NIC8	NIC9	NIC10	NIC11	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NV18	NV18	NV18	NV18	NV18	NV18	NV18	PIX	PXB	NODE	NODE	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS	0-55,112-167	0		N/A
GPU1	NV18	 X 	NV18	NV18	NV18	NV18	NV18	NV18	PXB	PIX	NODE	NODE	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS	0-55,112-167	0		N/A
GPU2	NV18	NV18	 X 	NV18	NV18	NV18	NV18	NV18	NODE	NODE	PIX	PXB	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS	0-55,112-167	0		N/A
GPU3	NV18	NV18	NV18	 X 	NV18	NV18	NV18	NV18	NODE	NODE	PXB	PIX	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS	0-55,112-167	0		N/A
GPU4	NV18	NV18	NV18	NV18	 X 	NV18	NV18	NV18	SYS	SYS	SYS	SYS	SYS	SYS	PIX	PXB	NODE	NODE	NODE	NODE	56-111,168-223	1		N/A
GPU5	NV18	NV18	NV18	NV18	NV18	 X 	NV18	NV18	SYS	SYS	SYS	SYS	SYS	SYS	PXB	PIX	NODE	NODE	NODE	NODE	56-111,168-223	1		N/A
GPU6	NV18	NV18	NV18	NV18	NV18	NV18	 X 	NV18	SYS	SYS	SYS	SYS	SYS	SYS	NODE	NODE	PIX	PXB	NODE	NODE	56-111,168-223	1		N/A
GPU7	NV18	NV18	NV18	NV18	NV18	NV18	NV18	 X 	SYS	SYS	SYS	SYS	SYS	SYS	NODE	NODE	PXB	PIX	NODE	NODE	56-111,168-223	1		N/A
NIC0	PIX	PXB	NODE	NODE	SYS	SYS	SYS	SYS	 X 	PXB	NODE	NODE	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS				
NIC1	PXB	PIX	NODE	NODE	SYS	SYS	SYS	SYS	PXB	 X 	NODE	NODE	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS				
NIC2	NODE	NODE	PIX	PXB	SYS	SYS	SYS	SYS	NODE	NODE	 X 	PXB	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS				
NIC3	NODE	NODE	PXB	PIX	SYS	SYS	SYS	SYS	NODE	NODE	PXB	 X 	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS				
NIC4	NODE	NODE	NODE	NODE	SYS	SYS	SYS	SYS	NODE	NODE	NODE	NODE	 X 	PIX	SYS	SYS	SYS	SYS	SYS	SYS				
NIC5	NODE	NODE	NODE	NODE	SYS	SYS	SYS	SYS	NODE	NODE	NODE	NODE	PIX	 X 	SYS	SYS	SYS	SYS	SYS	SYS				
NIC6	SYS	SYS	SYS	SYS	PIX	PXB	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS	 X 	PXB	NODE	NODE	NODE	NODE				
NIC7	SYS	SYS	SYS	SYS	PXB	PIX	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS	PXB	 X 	NODE	NODE	NODE	NODE				
NIC8	SYS	SYS	SYS	SYS	NODE	NODE	PIX	PXB	SYS	SYS	SYS	SYS	SYS	SYS	NODE	NODE	 X 	PXB	NODE	NODE				
NIC9	SYS	SYS	SYS	SYS	NODE	NODE	PXB	PIX	SYS	SYS	SYS	SYS	SYS	SYS	NODE	NODE	PXB	 X 	NODE	NODE				
NIC10	SYS	SYS	SYS	SYS	NODE	NODE	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS	NODE	NODE	NODE	NODE	 X 	PIX				
NIC11	SYS	SYS	SYS	SYS	NODE	NODE	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS	NODE	NODE	NODE	NODE	PIX	 X 				

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3
  NIC4: mlx5_4
  NIC5: mlx5_5
  NIC6: mlx5_6
  NIC7: mlx5_7
  NIC8: mlx5_8
  NIC9: mlx5_9
  NIC10: mlx5_10
  NIC11: mlx5_11
```

- GPU0~3 은 NUMA node 0 (CPU 0-55,112-167), GPU4~7 은 NUMA node 1 (CPU 56-111,168-223) 에 붙어 있다. 실험의 dual 인스턴스 분할(A = GPU0-3 + 소켓0, B = GPU4-7 + 소켓1)은 이 토폴로지를 따른다.

- 8장 전부 NV18 (NVLink 18 링크) 로 서로 연결되어 있다.

### 2.5 스토리지 / 네트워크

| 항목 | 값 |
|---|---|
| 블록 장치 | NAME     SIZE MODEL                      ROTA<br>sda      1.7T GBT3816-iMR                   0<br>sdb      500G VIRTUAL-DISK                  1<br>sdc        5G VIRTUAL-DISK                  1<br>sdd      200G VIRTUAL-DISK                  1<br>sde      100G VIRTUAL-DISK                  1<br>sdf       20G VIRTUAL-DISK                  1<br>nvme1n1  3.5T SAMSUNG MZQL23T8HCLS-00A07    0<br>nvme3n1  3.5T SAMSUNG MZQL23T8HCLS-00A07    0<br>nvme0n1  3.5T SAMSUNG MZQL23T8HCLS-00A07    0<br>nvme2n1  3.5T SAMSUNG MZQL23T8HCLS-00A07    0<br>nvme6n1  3.5T SAMSUNG MZQL23T8HCLS-00A07    0<br>nvme7n1  3.5T SAMSUNG MZQL23T8HCLS-00A07    0<br>nvme4n1  3.5T SAMSUNG MZQL23T8HCLS-00A07    0<br>nvme5n1  3.5T SAMSUNG MZQL23T8HCLS-00A07    0 |
| 파일시스템 | Filesystem             Size  Used Avail Use% Mounted on<br>/dev/mapper/rhel-root  1.8T  103G  1.6T   7% /<br>/dev/md5                25T  3.5T   22T  14% /data |
| RAID | Personalities : [raid6] [raid5] [raid4] <br>md5 : active raid5 nvme4n1p1[0] nvme6n1p1[2] nvme0n1p1[4] nvme3n1p1[8] nvme7n1p1[3] nvme1n1p1[5] nvme2n1p1[6] nvme5n1p1[1]<br>      26254233600 blocks super 1.2 level 5, 512k chunk, algorithm 2 [8/8] [UUUUUUUU]<br>      bitmap: 5/28 pages [20KB], 65536KB chunk |
| HF 캐시 위치 | `/data/mystous/.cache/huggingface` (= `~/.cache/huggingface`, `/data` = md5 xfs 24.5 TB) |
| NIC (lspci) | 6 개 — nvidia-smi topo 에 NIC0~11 |
| 인터넷 대역폭 (실측) | 단일 스트림 39.5 MB/s, 병렬 16 스트림 104~118 MB/s (Kimi-K2 1,030 GB 를 2 h 45 min 에 수신) |

## 3. 소프트웨어 구성 (실측 프로브)

### 3.1 컨테이너 런타임

| 항목 | 값 |
|---|---|
| nerdctl | nerdctl version 2.1.2 |
| containerd | containerd github.com/containerd/containerd/v2 v2.0.4 1a43cb6a1035441f9aca8f5666a9b3ef9e70ab20 |
| nvidia-container-toolkit | NVIDIA Container Toolkit CLI version 1.19.0 |

### 3.2 이미지

| 이미지 | digest | 크기 |
|---|---|---|
| `vllm/vllm-openai:nightly-a9a17e7095a66ef6c6685a1c7ddd657781a78d3c` | `sha256:3578c1fa6a9676e1de068b9d75c777cc865d251fadfbe6175ae82278739c6674` | 20.14GB |
| `lmsysorg/sglang:latest` | `sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1` | 33.1GB |
| `vllm/vllm-openai:latest` | `sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14` | 20.11GB |

### 3.3 컨테이너

| 컨테이너 | 역할 | cpuset (CPU / mems) | shm | ipc | 마운트 |
|---|---|---|---|---|---|
| `sgl-kt` | 서빙 본체 (단일 인스턴스 셀 전부, GPU-only 기준선) | - / - | 64 GiB | host | `['/home/mystous/.cache/huggingface->/models']` |
| `sgl-kt5` | dual 인스턴스 A (소켓0) | 0-55,112-167 / 0 | 64 GiB | private | `['/home/mystous/.cache/huggingface->/models']` |
| `sgl-kt2` | dual 인스턴스 B (소켓1) + 라우터 | 56-111,168-223 / 1 | 64 GiB | private | `['/home/mystous/.cache/huggingface->/models']` |
| `vllm-h100` | 벤치 클라이언트 (`vllm bench serve`) | - / - | 64 GiB | host | `['/home/mystous/hetero-exp->/exp', '/home/mystous/.cache/huggingface->/home/mystous/.cache/huggingface']` |

모든 컨테이너는 `--net host`. `sgl-kt5`/`sgl-kt2` 의 cpuset 은 8-29 dual 실험에서 확인된 kt-kernel 결함(스레드가 numactl 을 무시하고 절대 0번 코어부터 고정) 을 회피하기 위한 것이다.

### 3.4 서빙 스택 (`sgl-kt`, `sgl-kt5`, `sgl-kt2` 동일 — 3.5 참조)

```
Python 3.12.3
torch 2.13.0+cu130 cuda 13.0 cudnn 92000 nccl (2, 29, 7)
sglang=0.5.18
sgl-kernel=
kt-kernel=0.7.0.post1
flashinfer-python=0.6.17
triton=3.7.1
transformers=5.12.1
vllm=
sglang-router=0.3.2
sglang_commit=71de97b264b04dcd514cf904003028aefe9775c8
 python/sglang/srt/distributed/bootstrap.py         |   4 +-
 python/sglang/srt/layers/attention/dsa_backend.py  |   5 +-
 python/sglang/srt/layers/moe/kt_ep_wrapper.py      |  58 ++++++
 python/sglang/srt/layers/moe/topk.py               |  17 ++
 python/sglang/srt/managers/prefill_delayer.py      |   3 +-
 python/sglang/srt/model_executor/model_runner.py   | 222 ++++++++++++++++++++-
 .../runner/decode_cuda_graph_runner.py             |   7 +-
 .../runner_backend/full_cuda_graph_backend.py      |   9 +-
 python/sglang/srt/models/qwen3_moe.py              |   8 +-
 python/sglang/srt/speculative/eagle_worker_v2.py   |   4 +
 10 files changed, 326 insertions(+), 11 deletions(-)
Build cuda_13.0.r13.0/compiler.36424714_0
/usr/local/lib/python3.12/dist-packages/deep_gemm.disabled
```

- sglang 은 upstream commit `71de97b` 위에 로컬 수정 10파일(326+ / 11−)이 있다. 이 수정은 IDE_030~036 (hot expert 배치·deferral·callback-free 핸드오프 계열) 산출물이며 저장소 `shadow_assists/features/IDE_030/` 에 기록되어 있다.
- kt-kernel 0.7.0.post1 은 `/sgl-workspace/ktransformers/kt-kernel` 소스에서 빌드한 것이다 (9-07).
- `deep_gemm` 은 디렉터리 이름을 `deep_gemm.disabled` 로 바꿔 비활성화한 상태다.
- `sgl_kernel/moe.py` 의 `ignore_invalid_expert` 인자, `kt_ep_wrapper.py` 의 `gpu_experts_mask=None` — TSK_043 호환성 패치.

### 3.5 dual 컨테이너 동기화

`sgl-kt5`/`sgl-kt2` 는 원래 kt-kernel 0.7.0.post2 정식판 + 수정 2파일이었다 (`sgl-kt5_sw.txt` 는 동기화 후 상태). `eval/ide069/sync_containers.sh` 로 `sgl-kt` 의 kt_kernel 패키지·sglang 수정 10파일·deep_gemm 비활성 상태를 `/models` 마운트 경유로 복제했다 (nerdctl commit 은 snapshot mount 오류로 실패). 동기화 전후 단독 성능은 336.27 → 337.97 tok/s 로 같았다 — 즉 소켓 하나 cpuset 이 성능 차이의 원인이며 소프트웨어는 아니었다.

```
Version: 0.7.0.post1
 10 files changed, 326 insertions(+), 11 deletions(-)
/usr/local/lib/python3.12/dist-packages/deep_gemm.disabled
```

### 3.6 벤치 클라이언트 (`vllm-h100`)

| 항목 | 값 |
|---|---|
| 버전 | vllm 0.26.1rc1.dev1177+ga9a17e709 · Python 3.12.3 |
| 명령 | `vllm bench serve --backend openai --endpoint /v1/completions --dataset-name sonnet --dataset-path /tmp/sonnet.txt --sonnet-input-len 512 --sonnet-output-len 128 --sonnet-prefix-len 100 --request-rate inf --seed 42 --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95` + `--num-prompts 4×C --max-concurrency C` |

### 3.7 모델 · 가중치

| 모델 | snapshot | 크기 | safetensors |
|---|---|---|---|
| `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` | `003f183a92fbe5b9a8325aaa8b2ae797c91dd90f` | 450G | 50 |
| `moonshotai/Kimi-K2-Instruct` | `fd1984e2b7a3350dbf7305fe73a4ede25c14de50` | 959G | 62 |
| `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | 7.6G | 4 |
| `lmsys/SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI` | `b2a87592e335783be6c617c3be93ceeab99334a2` | 1.7G | 1 |

Qwen3-Coder-480B-A35B-Instruct-FP8 `config.json` 핵심:

```json
{
 "architectures": [
  "Qwen3MoeForCausalLM"
 ],
 "num_hidden_layers": 62,
 "hidden_size": 6144,
 "intermediate_size": 8192,
 "moe_intermediate_size": 2560,
 "num_experts": 160,
 "num_experts_per_tok": 8,
 "num_attention_heads": 96,
 "num_key_value_heads": 8,
 "head_dim": 128,
 "vocab_size": 151936,
 "max_position_embeddings": 262144,
 "torch_dtype": "bfloat16",
 "quantization_config": {
  "activation_scheme": "dynamic",
  "modules_to_not_convert": [
   "lm_head",
   "model.layers.0.input_layernorm",
   "model.layers.0.mlp.gate",
   "model.layers.0.post_attention_layernorm",
   "model.layers.1.input_layernorm",
   "model.layers.1.mlp.gate",
   "model.layers.1.post_attention_layernorm",
   "model.layers.2.input_layernorm",
   "model.layers.2.mlp.gate",
   "model.layers.2.post_attention_layernorm",
   "model.layers.3.input_layernorm",
   "model.layers.3.mlp.gate",
   "model.layers.3.post_attention_layernorm",
   "model.layers.4.input_layernorm",
   "model.layers.4.mlp.gate",
   "model.layers.4.post_attention_layernorm",
   "model.layers.5.input_layernorm",
   "model.layers.5.mlp.gate",
   "model.layers.5.post_attention_layernorm",
   "model.layers.6.input_layernorm",
   "model.layers.6.mlp.gate",
   "model.layers.6.post_attention_layernorm",
   "model.layers.7.input_layernorm",
   "model.layers.7.mlp.gate",
   "model.layers.7.post_attention_layernorm",
   "model.layers.8.input_layernorm",
   "model.layers.8.mlp.gate",
   "model.layers.8.post_attention_layernorm",
   "model.layers.9.input_layernorm",
   "model.layers.9.mlp.gate",
   "model.layers.9.post_attention_layernorm",
   "model.layers.10.input_layernorm",
   "model.layers.10.mlp.gate",
   "model.layers.10.post_attention_layernorm",
   "model.layers.11.input_layernorm",
   "model.layers.11.mlp.gate",
   "model.layers.11.post_attention_layernorm",
   "model.layers.12.input_layernorm",
   "model.layers.12.mlp.gate",
   "model.layers.12.post_attention_layernorm",
   "model.layers.13.input_layernorm",
   "model.layers.13.mlp.gate",
   "model.layers.13.post_attention_layernorm",
   "model.layers.14.input_layernorm",
   "model.layers.14.mlp.gate",
   "model.layers.14.post_attention_layernorm",
   "model.layers.15.input_layernorm",
   "model.layers.15.mlp.gate",
   "model.layers.15.post_attention_layernorm",
   "model.layers.16.input_layernorm",
   "model.layers.16.mlp.gate",
   "model.layers.16.post_attention_layernorm",
   "model.layers.17.input_layernorm",
   "model.layers.17.mlp.gate",
   "model.layers.17.post_attention_layernorm",
   "model.layers.18.input_layernorm",
   "model.layers.18.mlp.gate",
   "model.layers.18.post_attention_layernorm",
   "model.layers.19.input_layernorm",
   "model.layers.19.mlp.gate",
   "model.layers.19.post_attention_layernorm",
   "model.layers.20.input_layernorm",
   "model.layers.20.mlp.gate",
   "model.layers.20.post_attention_layernorm",
   "model.layers.21.input_layernorm",
   "model.layers.21.mlp.gate",
   "model.layers.21.post_attention_layernorm",
   "model.layers.22.input_layernorm",
   "model.layers.22.mlp.gate",
   "model.layers.22.post_attention_layernorm",
   "model.layers.23.input_layernorm",
   "model.layers.23.mlp.gate",
   "model.layers.23.post_attention_layernorm",
   "model.layers.24.input_layernorm",
   "model.layers.24.mlp.gate",
   "model.layers.24.post_attention_layernorm",
   "model.layers.25.input_layernorm",
   "model.layers.25.mlp.gate",
   "model.layers.25.post_attention_layernorm",
   "model.layers.26.input_layernorm",
   "model.layers.26.mlp.gate",
   "model.layers.26.post_attention_layernorm",
   "model.layers.27.input_layernorm",
   "model.layers.27.mlp.gate",
   "model.layers.27.post_attention_layernorm",
   "model.layers.28.input_layernorm",
   "model.layers.28.mlp.gate",
   "model.layers.28.post_attention_layernorm",
   "model.layers.29.input_layernorm",
   "model.layers.29.mlp.gate",
   "model.layers.29.post_attention_layernorm",
   "model.layers.30.input_layernorm",
   "model.layers.30.mlp.gate",
   "model.layers.30.post_attention_layernorm",
   "model.layers.31.input_layernorm",
   "model.layers.31.mlp.gate",
   "model.layers.31.post_attention_layernorm",
   "model.layers.32.input_layernorm",
   "model.layers.32.mlp.gate",
   "model.layers.32.post_attention_layernorm",
   "model.layers.33.input_layernorm",
   "model.layers.33.mlp.gate",
   "model.layers.33.post_attention_layernorm",
   "model.layers.34.input_layernorm",
   "model.layers.34.mlp.gate",
   "model.layers.34.post_attention_layernorm",
   "model.layers.35.input_layernorm",
   "model.layers.35.mlp.gate",
   "model.layers.35.post_attention_layernorm",
   "model.layers.36.input_layernorm",
   "model.layers.36.mlp.gate",
   "model.layers.36.post_attention_layernorm",
   "model.layers.37.input_layernorm",
   "model.layers.37.mlp.gate",
   "model.layers.37.post_attention_layernorm",
   "model.layers.38.input_layernorm",
   "model.layers.38.mlp.gate",
   "model.layers.38.post_attention_layernorm",
   "model.layers.39.input_layernorm",
   "model.layers.39.mlp.gate",
   "model.layers.39.post_attention_layernorm",
   "model.layers.40.input_layernorm",
   "model.layers.40.mlp.gate",
   "model.layers.40.post_attention_layernorm",
   "model.layers.41.input_layernorm",
   "model.layers.41.mlp.gate",
   "model.layers.41.post_attention_layernorm",
   "model.layers.42.input_layernorm",
   "model.layers.42.mlp.gate",
   "model.layers.42.post_attention_layernorm",
   "model.layers.43.input_layernorm",
   "model.layers.43.mlp.gate",
   "model.layers.43.post_attention_layernorm",
   "model.layers.44.input_layernorm",
   "model.layers.44.mlp.gate",
   "model.layers.44.post_attention_layernorm",
   "model.layers.45.input_layernorm",
   "model.layers.45.mlp.gate",
   "model.layers.45.post_attention_layernorm",
   "model.layers.46.input_layernorm",
   "model.layers.46.mlp.gate",
   "model.layers.46.post_attention_layernorm",
   "model.layers.47.input_layernorm",
   "model.layers.47.mlp.gate",
   "model.layers.47.post_attention_layernorm",
   "model.layers.48.input_layernorm",
   "model.layers.48.mlp.gate",
   "model.layers.48.post_attention_layernorm",
   "model.layers.49.input_layernorm",
   "model.layers.49.mlp.gate",
   "model.layers.49.post_attention_layernorm",
   "model.layers.50.input_layernorm",
   "model.layers.50.mlp.gate",
   "model.layers.50.post_attention_layernorm",
   "model.layers.51.input_layernorm",
   "model.layers.51.mlp.gate",
   "model.layers.51.post_attention_layernorm",
   "model.layers.52.input_layernorm",
   "model.layers.52.mlp.gate",
   "model.layers.52.post_attention_layernorm",
   "model.layers.53.input_layernorm",
   "model.layers.53.mlp.gate",
   "model.layers.53.post_attention_layernorm",
   "model.layers.54.input_layernorm",
   "model.layers.54.mlp.gate",
   "model.layers.54.post_attention_layernorm",
   "model.layers.55.input_layernorm",
   "model.layers.55.mlp.gate",
   "model.layers.55.post_attention_layernorm",
   "model.layers.56.input_layernorm",
   "model.layers.56.mlp.gate",
   "model.layers.56.post_attention_layernorm",
   "model.layers.57.input_layernorm",
   "model.layers.57.mlp.gate",
   "model.layers.57.post_attention_layernorm",
   "model.layers.58.input_layernorm",
   "model.layers.58.mlp.gate",
   "model.layers.58.post_attention_layernorm",
   "model.layers.59.input_layernorm",
   "model.layers.59.mlp.gate",
   "model.layers.59.post_attention_layernorm",
   "model.layers.60.input_layernorm",
   "model.layers.60.mlp.gate",
   "model.layers.60.post_attention_layernorm",
   "model.layers.61.input_layernorm",
   "model.layers.61.mlp.gate",
   "model.layers.61.post_attention_layernorm"
  ],
  "quant_method": "fp8",
  "weight_block_size": [
   128,
   128
  ]
 }
}
```

CPU expert 변환본 (`kt quant -m int4 -i fp8 --cpu-threads 96 --numa-nodes 2`):

```
== kt/qwen3-480b-int4 232G files=67
{'architectures': ['Qwen3MoeForCausalLM'], 'attention_bias': False, 'attention_dropout': 0.0, 'bos_token_id': 151643, 'decoder_sparse_step': 1, 'eos_token_id': 151645, 'head_dim': 128, 'hidden_act': 'silu', 'hidden_size': 6144, 'initializer_range': 0.02, 'intermediate_size': 8192, 'max_position_embe
== kt/kimi-k2-int4 488G files=64
{'architectures': ['DeepseekV3ForCausalLM'], 'attention_bias': False, 'attention_dropout': 0.0, 'auto_map': {'AutoConfig': 'configuration_deepseek.DeepseekV3Config', 'AutoModel': 'modeling_deepseek.DeepseekV3Model', 'AutoModelForCausalLM': 'modeling_deepseek.DeepseekV3ForCausalLM'}, 'aux_loss_alpha'
```

hotmap (`~/.cache/huggingface/kt/ide069/hotmap.json`, 컨테이너 `/models/kt/ide069/hotmap.json`) 통계:

```json
{
 "source": "expert_distribution_recorder_1789448276.8979676.pt",
 "layers": 62,
 "experts": 160,
 "total_calls": 71981504,
 "coverage_mean": {
  "16": 0.5979006495890944,
  "40": 0.8459506486555213,
  "64": 0.9411456587514483,
  "80": 0.9701907312189533,
  "96": 0.9863706098722249,
  "112": 0.9948588459613181,
  "128": 0.9986535985688768
 },
 "coverage_min": {
  "16": 0.31836567349301287,
  "40": 0.5825053058074474,
  "64": 0.7705479452054795,
  "80": 0.8543452495796697,
  "96": 0.9142870924175188,
  "112": 0.9541461095339159,
  "128": 0.9797380171439597
 },
 "coverage_max": {
  "16": 0.7332212452798985,
  "40": 0.9124024971748298,
  "64": 0.9774951076320939,
  "80": 0.9903289600617402,
  "96": 0.9957450180535266,
  "112": 0.9990490890548771,
  "128": 0.9998449601719909
 }
}
```

## 4. 실험 구성

### 4.1 공통

| 항목 | 값 |
|---|---|
| 모델 | Qwen3-Coder-480B-A35B-Instruct-FP8 (450 GB), CPU expert = AMXINT4 232 GB |
| 병렬 | TP4 (GPU 0-3). 기준선만 TP8+EP8 (GPU 0-7) |
| 공통 서버 플래그 | `--attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2` |
| hot expert | `--kt-num-gpu-experts N --init-expert-location /models/kt/ide069/hotmap.json` (N = 물리 id 0..N-1 이 GPU) |
| deferral | `--kt-max-deferred-experts-per-token N` |
| cuda graph | ON = `--cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64`, OFF = `--disable-cuda-graph` |
| KV | `--mem-fraction-static` + `--max-total-tokens` |
| 워크로드 | sonnet 입력 512 / 출력 128 / prefix 100, seed 42, 요청 수 = 4×C, request-rate inf |
| 품질 | greedy 4문항 (Paris / fibonacci / 5050 / reverse string) 전 셀, GSM8K 40문항 chat greedy (`eval/harness/gsm_eval.py`) |
| 계측 | `/proc/stat` 2초 CPU 샘플러, `nvidia-smi -l 5`, 부팅 전후 `free -g`·HBM |

### 4.2 셀 정의 파일 (원문)

`eval/ide069/cells_round1.txt`:

```
# name|gpu_experts|deferral|graph|mem_frac|max_total_tokens|concurrencies|extra
t1_hot96_def0_graph|96|0|on|0.92|24576|16,32|--ep-dispatch-algorithm dynamic
t2_hot96_def4_graph|96|4|on|0.92|24576|16,32,64|--ep-dispatch-algorithm dynamic
t3_hot112_def4_graph|112|4|on|0.95|16384|32|--ep-dispatch-algorithm dynamic
t4_hot96_def2_graph|96|2|on|0.92|24576|32|--ep-dispatch-algorithm dynamic
t5_hot96_def8_graph|96|8|on|0.92|24576|32|--ep-dispatch-algorithm dynamic
t6_hot64_def4_graph|64|4|on|0.92|32768|32|--ep-dispatch-algorithm dynamic
```

`eval/ide069/cells_round2.txt`:

```
# name|gpu_experts|deferral|graph|mem_frac|max_total_tokens|concurrencies|extra
t7_hot96_def4_kv40k|96|4|on|0.95|40960|32,64|--ep-dispatch-algorithm dynamic
t8_hot96_def4_eagle3|96|4|on|0.92|24576|32|--ep-dispatch-algorithm dynamic --speculative-algorithm EAGLE3 --speculative-draft-model-path /models/hub/models--lmsys--SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI/snapshots/b2a87592e335783be6c617c3be93ceeab99334a2 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
t9_hot96_def4_standalone4b|96|4|on|0.92|24576|32|--ep-dispatch-algorithm dynamic --speculative-algorithm STANDALONE --speculative-draft-model-path /models/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
t10_hot96_def4_cpu112|96|4|on|0.92|24576|32|--ep-dispatch-algorithm dynamic --kt-cpuinfer 112
t11_hot96_def4_static|96|4|on|0.92|24576|32|--ep-dispatch-algorithm static
t12_hot100_def4|100|4|on|0.95|16384|32|--ep-dispatch-algorithm dynamic
```

`eval/ide069/cells_round3.txt`:

```
# 3라운드 — 최고 구성(hot96 def4 KV40k)의 C sweep + 반복(jitter) + KV 상한 탐침
# name|gpu_experts|deferral|graph|mem_frac|max_total_tokens|concurrencies|extra
t13_hot96_def4_kv40k_rep1|96|4|on|0.95|40960|32,48,64,80|--ep-dispatch-algorithm dynamic
t14_hot96_def4_kv40k_rep2|96|4|on|0.95|40960|32,64|--ep-dispatch-algorithm dynamic
t15_hot96_def4_kv56k|96|4|on|0.96|57344|64,96|--ep-dispatch-algorithm dynamic
t16_hot96_def4_kv24k_rep|96|4|on|0.92|24576|32|--ep-dispatch-algorithm dynamic
```

### 4.3 하네스

| 파일 | 역할 |
|---|---|
| `eval/ide069/build_hotmap.py` | recorder .pt → hotmap.json + 커버리지 통계 |
| `eval/ide069/run_hotmap.sh` | recorder 켜고 대표 워크로드 → hotmap 생성 |
| `eval/ide069/run_sweep.sh` | 셀 정의 파일 기반 sweep (부팅→greedy→C별 벤치) |
| `eval/ide069/run_gsm_gate.sh` | GSM 40 품질 게이트 (deferral 0 vs N) |
| `eval/ide069/run_dual.sh` | TP4+오프로딩 ×2 동시 벤치 합산 + GPU-only TP8 기준선 |
| `eval/ide069/run_router.sh` | sglang_router 단일 엔드포인트 |
| `eval/ide069/sync_containers.sh` | sgl-kt 소프트웨어 상태를 cpuset 컨테이너에 복제 |
| `eval/ide069/make_full_report.py` | 이 문서 생성 |
| `eval/ide068/lib.sh`, `cpu_sample.sh`, `summarize.py` | 공용 함수·샘플러·집계 |

## 5. 결과 상세 (run 별)

표의 값은 각 셀의 `bench.log` 를 파싱한 것이다. CPU busy 는 그 벤치 구간의 `/proc/stat` 샘플 평균/최대. 부팅·HBM·DRAM·greedy 는 셀 루트의 `verdict.txt`, `hbm_after_boot.csv`, `free_after_boot.txt`, `smoke_texts.txt`.

### 5.1 `eval/results/20260915_135132_ide069_hotmap`

시작: `13:51:32 == IDE_069 hotmap start 20260915_135132 ==`

| 셀 / 벤치 | 완료 | 소요 s | 출력 tok/s | 전체 tok/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | CPU busy 평균/최대 % |
|---|---|---|---|---|---|---|---|---|---|
| `rec` | 64 | 186.35 | 43.96 | 216.37 | 8622.15 | 12197.27 | 292.44 | 295.58 | 44.2 / 49.7 |

RUN.log 핵심 줄:

```
Output token throughput (tok/s): 43.96 
```

### 5.2 `eval/results/20260915_135906_ide069_sweep_cells_round1`

시작: `13:59:06 == IDE_069 sweep 20260915_135906 spec=cells_round1.txt ==`

셀 정의:

```
# name|gpu_experts|deferral|graph|mem_frac|max_total_tokens|concurrencies|extra
t1_hot96_def0_graph|96|0|on|0.92|24576|16,32|--ep-dispatch-algorithm dynamic
t2_hot96_def4_graph|96|4|on|0.92|24576|16,32,64|--ep-dispatch-algorithm dynamic
t3_hot112_def4_graph|112|4|on|0.95|16384|32|--ep-dispatch-algorithm dynamic
t4_hot96_def2_graph|96|2|on|0.92|24576|32|--ep-dispatch-algorithm dynamic
t5_hot96_def8_graph|96|8|on|0.92|24576|32|--ep-dispatch-algorithm dynamic
t6_hot64_def4_graph|64|4|on|0.92|32768|32|--ep-dispatch-algorithm dynamic
```

| 셀 / 벤치 | 완료 | 소요 s | 출력 tok/s | 전체 tok/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | CPU busy 평균/최대 % |
|---|---|---|---|---|---|---|---|---|---|
| `t1_hot96_def0_graph/c16` | 64 | 25.44 | 321.98 | 1584.76 | 1223.62 | 1725.11 | 39.47 | 40.14 | 31.8 / 47.3 |
| `t1_hot96_def0_graph/c32` | 128 | 34.84 | 470.26 | 2316.90 | 1393.64 | 2201.17 | 55.31 | 60.33 | 34.2 / 46.3 |
| `t2_hot96_def4_graph/c16` | 64 | 24.89 | 329.13 | 1619.93 | 1222.73 | 1746.60 | 38.53 | 39.17 | 30.9 / 47.0 |
| `t2_hot96_def4_graph/c32` | 128 | 31.52 | 519.81 | 2560.99 | 1396.00 | 2121.38 | 49.21 | 50.66 | 34.2 / 46.6 |
| `t2_hot96_def4_graph/c64` | 256 | 77.07 | 425.15 | 2095.42 | 2828.02 | 15522.00 | 88.77 | 100.82 | 40.0 / 47.3 |
| `t4_hot96_def2_graph/c32` | 128 | 36.07 | 454.27 | 2238.12 | 2127.28 | 2822.71 | 53.81 | 56.29 | 35.0 / 47.5 |
| `t5_hot96_def8_graph/c32` | 128 | 32.76 | 500.17 | 2464.25 | 2048.52 | 2805.12 | 47.30 | 50.61 | 33.6 / 48.4 |
| `t6_hot64_def4_graph/c32` | 128 | 71.19 | 230.14 | 1133.88 | 2554.57 | 3665.10 | 119.32 | 125.91 | 39.4 / 47.5 |

**t1_hot96_def0_graph** — 부팅 `HEALTH_OK 110` · greedy 4/4 · HBM 74.2~74.6 GiB ×4 · DRAM used 307 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic
```

**t2_hot96_def4_graph** — 부팅 `HEALTH_OK 100` · greedy 4/4 · HBM 74.2~74.6 GiB ×4 · DRAM used 308 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic
```

**t3_hot112_def4_graph** — 부팅 `DIED 40` · greedy — · HBM — · DRAM used — GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 112 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 16384 --ep-dispatch-algorithm dynamic
```

오류:

```
orch-cuda-alloc-conf)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 840.00 MiB. GPU 1 has a total capacity of 79.18 GiB of which 473.31 MiB is free. Process 3245005 has 78.71 GiB memory in use. Of the allocated memory 76.92 GiB is allocated by PyTorch, and 2.62 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
```

**t4_hot96_def2_graph** — 부팅 `HEALTH_OK 100` · greedy 4/4 · HBM 74.2~74.6 GiB ×4 · DRAM used 308 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 2 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic
```

**t5_hot96_def8_graph** — 부팅 `HEALTH_OK 110` · greedy 4/4 · HBM 74.2~74.6 GiB ×4 · DRAM used 308 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 8 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic
```

**t6_hot64_def4_graph** — 부팅 `HEALTH_OK 100` · greedy 4/4 · HBM 52.9~53.3 GiB ×4 · DRAM used 308 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 64 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 32768 --ep-dispatch-algorithm dynamic
```

RUN.log 핵심 줄:

```
14:01:17 boot verdict=HEALTH_OK 110s gpus=0,1,2,3
Output token throughput (tok/s): 321.98 
Output token throughput (tok/s): 470.26 
14:04:57 boot verdict=HEALTH_OK 100s gpus=0,1,2,3
Output token throughput (tok/s): 329.13 
Output token throughput (tok/s): 519.81 
Output token throughput (tok/s): 425.15 
14:08:58 boot verdict=DIED 40s gpus=0,1,2,3
[rank1]:[W915 05:08:49.426956710 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 1 while trying to allocate 880803840 bytes (free: 496304128, total: 85017493504).
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 840.00 MiB. GPU 2 has a total capacity of 79.18 GiB of which 473.31 MiB is free. Process 3245097 has 78.71 GiB memory in use. Of the alloc
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 840.00 MiB. GPU 1 has a total capacity of 79.18 GiB of which 473.31 MiB is free. Process 3245005 has 78.71 GiB memory in use. Of the alloc
14:10:58 boot verdict=HEALTH_OK 100s gpus=0,1,2,3
Output token throughput (tok/s): 454.27 
14:14:11 boot verdict=HEALTH_OK 110s gpus=0,1,2,3
Output token throughput (tok/s): 500.17 
14:17:07 boot verdict=HEALTH_OK 100s gpus=0,1,2,3
Output token throughput (tok/s): 230.14 
```

### 5.3 `eval/results/20260915_141917_ide069_gsm40_hot96`

시작: `14:19:17 == IDE_069 GSM40 gate hot96 defs=0,4 ==`

| 셀 / 벤치 | 완료 | 소요 s | 출력 tok/s | 전체 tok/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | CPU busy 평균/최대 % |
|---|---|---|---|---|---|---|---|---|---|

**def0** — 부팅 `HEALTH_OK 100` · greedy 4/4 · HBM 74.2~74.6 GiB ×4 · DRAM used 308 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic
```

**def4** — 부팅 `HEALTH_OK 100` · greedy 4/4 · HBM 74.2~74.6 GiB ×4 · DRAM used 308 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic --kt-max-deferred-experts-per-token 4
```

GSM `eval/results/20260915_141917_ide069_gsm40_hot96/def0/gsm40.json`: **39/40** (acc 0.975)

GSM `eval/results/20260915_141917_ide069_gsm40_hot96/def4/gsm40.json`: **39/40** (acc 0.975)

RUN.log 핵심 줄:

```
14:21:17 boot verdict=HEALTH_OK 100s gpus=0,1,2,3
ACC 0.975 (39/40)
14:23:11 gsm 소요 103s
14:25:11 boot verdict=HEALTH_OK 100s gpus=0,1,2,3
ACC 0.975 (39/40)
14:27:07 gsm 소요 105s
```

### 5.4 `eval/results/20260915_142742_ide069_sweep_cells_round2`

시작: `14:27:42 == IDE_069 sweep 20260915_142742 spec=cells_round2.txt ==`

셀 정의:

```
# name|gpu_experts|deferral|graph|mem_frac|max_total_tokens|concurrencies|extra
t7_hot96_def4_kv40k|96|4|on|0.95|40960|32,64|--ep-dispatch-algorithm dynamic
t8_hot96_def4_eagle3|96|4|on|0.92|24576|32|--ep-dispatch-algorithm dynamic --speculative-algorithm EAGLE3 --speculative-draft-model-path /models/hub/models--lmsys--SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI/snapshots/b2a87592e335783be6c617c3be93ceeab99334a2 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
t9_hot96_def4_standalone4b|96|4|on|0.92|24576|32|--ep-dispatch-algorithm dynamic --speculative-algorithm STANDALONE --speculative-draft-model-path /models/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
t10_hot96_def4_cpu112|96|4|on|0.92|24576|32|--ep-dispatch-algorithm dynamic --kt-cpuinfer 112
t11_hot96_def4_static|96|4|on|0.92|24576|32|--ep-dispatch-algorithm static
t12_hot100_def4|100|4|on|0.95|16384|32|--ep-dispatch-algorithm dynamic
```

| 셀 / 벤치 | 완료 | 소요 s | 출력 tok/s | 전체 tok/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | CPU busy 평균/최대 % |
|---|---|---|---|---|---|---|---|---|---|
| `t10_hot96_def4_cpu112/c32` | 128 | 33.90 | 483.24 | 2380.81 | 2059.77 | 2740.65 | 49.65 | 51.77 | 40.4 / 57.4 |
| `t12_hot100_def4/c32` | 128 | 45.96 | 356.45 | 1756.17 | 2154.30 | 7220.87 | 65.75 | 71.83 | 36.6 / 47.3 |
| `t7_hot96_def4_kv40k/c32` | 128 | 35.67 | 459.34 | 2263.07 | 2073.20 | 2854.32 | 52.21 | 57.21 | 35.0 / 47.8 |
| `t7_hot96_def4_kv40k/c64` | 256 | 51.42 | 637.24 | 3140.71 | 2979.17 | 3706.93 | 77.45 | 86.63 | 38.7 / 51.8 |
| `t8_hot96_def4_eagle3/c32` | 128 | 55.83 | 293.46 | 1445.82 | 710.94 | 2853.35 | 95.64 | 133.95 | 38.7 / 48.2 |
| `t9_hot96_def4_standalone4b/c32` | 128 | 96.50 | 169.77 | 836.45 | 20170.67 | 22222.52 | 25.44 | 33.98 | 41.1 / 46.9 |

**t10_hot96_def4_cpu112** — 부팅 `HEALTH_OK 100` · greedy 4/4 · HBM 74.2~74.6 GiB ×4 · DRAM used 308 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic --kt-cpuinfer 112
```

**t11_hot96_def4_static** — 부팅 `DIED 20` · greedy — · HBM — · DRAM used — GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm static
```

오류:

```
raise ValueError(
ValueError: --ep-dispatch-algorithm static picks a different physical replica per rank, which only holds up when an a2a backend routes each token to a single rank. Use --ep-dispatch-algorithm dynamic with --moe-a2a-backend none.
```

**t12_hot100_def4** — 부팅 `HEALTH_OK 110` · greedy 4/4 · HBM 76.5~76.9 GiB ×4 · DRAM used 308 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 100 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 16384 --ep-dispatch-algorithm dynamic
```

**t7_hot96_def4_kv40k** — 부팅 `HEALTH_OK 100` · greedy 4/4 · HBM 75.1~75.6 GiB ×4 · DRAM used 308 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic
```

**t8_hot96_def4_eagle3** — 부팅 `HEALTH_OK 120` · greedy 4/4 · HBM 74.7~75.2 GiB ×4 · DRAM used 309 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic --speculative-algorithm EAGLE3 --speculative-draft-model-path /models/hub/models--lmsys--SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI/snapshots/b2a87592e335783be6c617c3be93ceeab99334a2 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

**t9_hot96_def4_standalone4b** — 부팅 `HEALTH_OK 130` · greedy 4/4 · HBM 75.1~75.5 GiB ×4 · DRAM used 310 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic --speculative-algorithm STANDALONE --speculative-draft-model-path /models/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

RUN.log 핵심 줄:

```
14:29:42 boot verdict=HEALTH_OK 100s gpus=0,1,2,3
Output token throughput (tok/s): 459.34 
Output token throughput (tok/s): 637.24 
14:34:11 boot verdict=HEALTH_OK 120s gpus=0,1,2,3
Output token throughput (tok/s): 293.46 
14:38:06 boot verdict=HEALTH_OK 130s gpus=0,1,2,3
Output token throughput (tok/s): 169.77 
14:42:08 boot verdict=HEALTH_OK 100s gpus=0,1,2,3
Output token throughput (tok/s): 483.24 
14:43:41 boot verdict=DIED 20s gpus=0,1,2,3
    raise ValueError(
ValueError: --ep-dispatch-algorithm static picks a different physical replica per rank, which only holds up when an a2a backend routes each token to a single rank. Use --ep-dispatch-algorithm dynamic 
14:45:52 boot verdict=HEALTH_OK 110s gpus=0,1,2,3
Output token throughput (tok/s): 356.45 
```

### 5.5 `eval/results/20260915_144748_ide069_dual_vs_tp8`

시작: `14:47:48 == IDE_069 dual vs tp8 start 20260915_144748 ==`

| 셀 / 벤치 | 완료 | 소요 s | 출력 tok/s | 전체 tok/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | CPU busy 평균/최대 % |
|---|---|---|---|---|---|---|---|---|---|
| `ref_gpu_tp8_ep8/c32` | 128 | 15.91 | 1029.75 | 5073.38 | 562.01 | 716.22 | 26.83 | 28.06 | 14.2 / 51.5 |
| `ref_gpu_tp8_ep8/c64` | 256 | 16.13 | 2031.11 | 10010.65 | 277.50 | 1107.62 | 26.34 | 31.79 | 14.2 / 51.5 |

**dual_cold_expert0** — 부팅 `A=1 B=1` · greedy — · HBM — · DRAM used — GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-threadpool-count 2 --kt-cpuinfer 48 --kt-num-gpu-experts 0 --disable-cuda-graph --mem-fraction-static 0.80 --max-total-tokens 65536
```

**dual_hot96_def4** — 부팅 `A=1 B=1` · greedy — · HBM — · DRAM used — GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-threadpool-count 2 --kt-cpuinfer 48 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic
```

**ref_gpu_tp8_ep8** — 부팅 `rc=0` · greedy 4/4 · HBM 71.9~72.4 GiB ×8 · DRAM used — GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 8 --ep-size 8 --attention-backend triton --trust-remote-code --mem-fraction-static 0.90 --max-total-tokens 131072
```

RUN.log 핵심 줄:

```
14:47:48 == IDE_069 dual vs tp8 start 20260915_144748 ==
15:04:17 ref C32: Successful requests: 128  Benchmark duration (s): 15.91  Output token throughput (tok/s): 1029.75  Total token throughput (tok/s): 5073.38  Median TTFT (ms): 562.01  P95 TTFT (ms): 716.22  Median TPOT (ms): 26.83  P95 TPOT (ms): 28.06  
15:04:46 ref C64: Successful requests: 256  Benchmark duration (s): 16.13  Output token throughput (tok/s): 2031.11  Total token throughput (tok/s): 10010.65  Median TTFT (ms): 277.50  P95 TTFT (ms): 1107.62  Median TPOT (ms): 26.34  P95 TPOT (ms): 31.79  
15:04:57 -- dual dual_cold_expert0 --
15:05:07 -- dual dual_hot96_def4 --
15:05:18 == dual done 15:05:18 ==
```

### 5.6 `eval/results/20260915_150557_ide069_sweep_cells_round3`

시작: `15:05:57 == IDE_069 sweep 20260915_150557 spec=cells_round3.txt ==`

셀 정의:

```
# 3라운드 — 최고 구성(hot96 def4 KV40k)의 C sweep + 반복(jitter) + KV 상한 탐침
# name|gpu_experts|deferral|graph|mem_frac|max_total_tokens|concurrencies|extra
t13_hot96_def4_kv40k_rep1|96|4|on|0.95|40960|32,48,64,80|--ep-dispatch-algorithm dynamic
t14_hot96_def4_kv40k_rep2|96|4|on|0.95|40960|32,64|--ep-dispatch-algorithm dynamic
t15_hot96_def4_kv56k|96|4|on|0.96|57344|64,96|--ep-dispatch-algorithm dynamic
t16_hot96_def4_kv24k_rep|96|4|on|0.92|24576|32|--ep-dispatch-algorithm dynamic
```

| 셀 / 벤치 | 완료 | 소요 s | 출력 tok/s | 전체 tok/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | CPU busy 평균/최대 % |
|---|---|---|---|---|---|---|---|---|---|
| `t13_hot96_def4_kv40k_rep1/c32` | 128 | 35.22 | 465.20 | 2291.96 | 2106.03 | 2816.35 | 52.14 | 54.02 | 35.6 / 50.0 |
| `t13_hot96_def4_kv40k_rep1/c48` | 192 | 43.53 | 564.57 | 2782.46 | 2661.83 | 3032.29 | 66.26 | 72.80 | 37.4 / 49.0 |
| `t13_hot96_def4_kv40k_rep1/c64` | 256 | 51.40 | 637.54 | 3142.23 | 2981.72 | 3763.93 | 77.02 | 86.36 | 38.4 / 49.3 |
| `t13_hot96_def4_kv40k_rep1/c80` | 320 | 170.83 | 239.77 | 1181.43 | 3166.89 | 13944.79 | 296.74 | 319.27 | 43.8 / 49.5 |
| `t14_hot96_def4_kv40k_rep2/c32` | 128 | 34.72 | 471.90 | 2324.99 | 2028.51 | 2953.37 | 51.42 | 55.75 | 34.5 / 47.7 |
| `t14_hot96_def4_kv40k_rep2/c64` | 256 | 50.35 | 650.75 | 3207.32 | 2981.22 | 3770.87 | 76.28 | 84.42 | 38.1 / 48.8 |
| `t15_hot96_def4_kv56k/c64` | 256 | 51.68 | 634.06 | 3125.07 | 3025.52 | 4306.71 | 77.55 | 87.03 | 38.7 / 50.2 |
| `t15_hot96_def4_kv56k/c96` | 384 | 149.52 | 328.72 | 1619.98 | 3868.40 | 5156.47 | 257.16 | 318.90 | 43.0 / 49.1 |
| `t16_hot96_def4_kv24k_rep/c32` | 128 | 33.65 | 486.90 | 2398.85 | 1965.58 | 2704.74 | 50.12 | 51.42 | 34.1 / 47.7 |

**t13_hot96_def4_kv40k_rep1** — 부팅 `HEALTH_OK 110` · greedy 4/4 · HBM 75.1~75.6 GiB ×4 · DRAM used 308 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic
```

**t14_hot96_def4_kv40k_rep2** — 부팅 `HEALTH_OK 100` · greedy 4/4 · HBM 75.1~75.6 GiB ×4 · DRAM used 308 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.95 --max-total-tokens 40960 --ep-dispatch-algorithm dynamic
```

**t15_hot96_def4_kv56k** — 부팅 `HEALTH_OK 100` · greedy 4/4 · HBM 76.1~76.6 GiB ×4 · DRAM used 308 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.96 --max-total-tokens 57344 --ep-dispatch-algorithm dynamic
```

**t16_hot96_def4_kv24k_rep** — 부팅 `HEALTH_OK 100` · greedy 4/4 · HBM 74.2~74.6 GiB ×4 · DRAM used 308 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic
```

RUN.log 핵심 줄:

```
15:08:08 boot verdict=HEALTH_OK 110s gpus=0,1,2,3
Output token throughput (tok/s): 465.20 
Output token throughput (tok/s): 564.57 
Output token throughput (tok/s): 637.54 
Output token throughput (tok/s): 239.77 
15:16:19 boot verdict=HEALTH_OK 100s gpus=0,1,2,3
Output token throughput (tok/s): 471.90 
Output token throughput (tok/s): 650.75 
15:20:23 boot verdict=HEALTH_OK 100s gpus=0,1,2,3
Output token throughput (tok/s): 634.06 
Output token throughput (tok/s): 328.72 
15:26:26 boot verdict=HEALTH_OK 100s gpus=0,1,2,3
Output token throughput (tok/s): 486.90 
```

### 5.7 `eval/results/20260915_152805_ide069_dual_vs_tp8`

시작: `15:28:05 == IDE_069 dual vs tp8 start 20260915_152805 ==`

| 셀 / 벤치 | 완료 | 소요 s | 출력 tok/s | 전체 tok/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | CPU busy 평균/최대 % |
|---|---|---|---|---|---|---|---|---|---|
| `dual_cold_expert0/c16/A` | 64 | 330.96 | 24.75 | 121.83 | 14676.94 | 20349.25 | 525.94 | 534.14 | — |
| `dual_cold_expert0/c16/B` | 64 | 335.17 | 24.44 | 120.30 | 15220.87 | 19407.31 | 534.38 | 539.08 | — |
| `dual_cold_expert0/c32/A` | 128 | 499.08 | 32.71 | 161.63 | 2605.27 | 32748.92 | 846.22 | 973.44 | — |
| `dual_cold_expert0/c32/B` | 128 | 492.04 | 33.30 | 164.06 | 10388.11 | 30437.84 | 850.57 | 927.11 | — |
| `dual_hot96_def4/c16/A` | 64 | 31.44 | 260.55 | 1282.40 | 1599.76 | 2203.10 | 48.14 | 51.85 | — |
| `dual_hot96_def4/c16/B` | 64 | 32.52 | 251.93 | 1239.98 | 1680.93 | 2240.36 | 49.88 | 53.46 | — |
| `dual_hot96_def4/c32/A` | 128 | 47.10 | 347.87 | 1713.90 | 1966.23 | 3047.82 | 74.13 | 83.13 | — |
| `dual_hot96_def4/c32/B` | 128 | 47.71 | 343.38 | 1691.75 | 2044.85 | 2825.68 | 75.65 | 86.44 | — |
| `single_s5_hot96_def4/c32` | 128 | 48.72 | 336.27 | 1656.76 | 2859.83 | 3617.25 | 72.63 | 81.63 | 36.9 / 48.0 |

**dual_cold_expert0** — 부팅 `A=0 B=0` · greedy — · HBM 11.0~11.5 GiB ×8 · DRAM used 536 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-threadpool-count 2 --kt-cpuinfer 48 --kt-num-gpu-experts 0 --disable-cuda-graph --mem-fraction-static 0.80 --max-total-tokens 65536
```

**dual_hot96_def4** — 부팅 `A=0 B=0` · greedy — · HBM 74.2~74.6 GiB ×8 · DRAM used 542 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-threadpool-count 2 --kt-cpuinfer 48 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic
```

**single_s5_hot96_def4** — 부팅 `rc=0` · greedy 4/4 · HBM — · DRAM used — GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-threadpool-count 2 --kt-cpuinfer 96 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic
```

RUN.log 핵심 줄:

```
15:28:05 == IDE_069 dual vs tp8 start 20260915_152805 ==
15:31:41 single C32: Successful requests: 128  Benchmark duration (s): 48.72  Output token throughput (tok/s): 336.27  Total token throughput (tok/s): 1656.76  Median TTFT (ms): 2859.83  P95 TTFT (ms): 3617.25  Median TPOT (ms): 72.63  P95 TPOT (ms): 81.63  
15:31:59 -- dual dual_cold_expert0 --
15:39:52 dual dual_cold_expert0 C16each: A=24.75  B=24.44  합산=49.19
15:48:24 dual dual_cold_expert0 C32each: A=32.71  B=33.30  합산=66.01
15:48:43 -- dual dual_hot96_def4 --
15:51:57 dual dual_hot96_def4 C16each: A=260.55  B=251.93  합산=512.48
15:52:58 dual dual_hot96_def4 C32each: A=347.87  B=343.38  합산=691.25
15:53:17 == dual done 15:53:17 ==
```

### 5.8 `eval/results/20260915_155341_ide069_router_dual`

시작: `15:53:41 == IDE_069 router-dual start 20260915_155341 ==`

| 셀 / 벤치 | 완료 | 소요 s | 출력 tok/s | 전체 tok/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | CPU busy 평균/최대 % |
|---|---|---|---|---|---|---|---|---|---|
| `router_cold_expert0/router_round_robin/c32` | 128 | 355.28 | 46.12 | 227.20 | 7972.48 | 21900.45 | 605.66 | 696.79 | 47.0 / 51.3 |
| `router_cold_expert0/router_round_robin/c64` | 256 | 563.30 | 58.13 | 286.66 | 8460.17 | 17832.73 | 939.03 | 1271.77 | 46.9 / 51.3 |

**router_cold_expert0** — 부팅 `A=0 B=0` · greedy — · HBM 11.0~11.5 GiB ×8 · DRAM used — GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-threadpool-count 2 --kt-cpuinfer 48 --kt-num-gpu-experts 0 --disable-cuda-graph --mem-fraction-static 0.80 --max-total-tokens 65536
```

RUN.log 핵심 줄:

```
15:53:41 == IDE_069 router-dual start 20260915_155341 ==
15:55:27 router(round_robin) up after 3s: OK
16:02:01 router(round_robin) C32: Successful requests: 128  Benchmark duration (s): 355.28  Output token throughput (tok/s): 46.12  Total token throughput (tok/s): 227.20  Median TTFT (ms): 7972.48  P95 TTFT (ms): 21900.45  Median TPOT (ms): 605.66  P95 TPOT (ms): 696.79  
16:11:37 router(round_robin) C64: Successful requests: 256  Benchmark duration (s): 563.30  Output token throughput (tok/s): 58.13  Total token throughput (tok/s): 286.66  Median TTFT (ms): 8460.17  P95 TTFT (ms): 17832.73  Median TPOT (ms): 939.03  P95 TPOT (ms): 1271.77  
```

### 5.9 `eval/results/20260915_171650_ide069_dual_vs_tp8_synced`

시작: `17:16:50 == IDE_069 dual vs tp8 start 20260915_171650 ==`

| 셀 / 벤치 | 완료 | 소요 s | 출력 tok/s | 전체 tok/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | CPU busy 평균/최대 % |
|---|---|---|---|---|---|---|---|---|---|
| `dual_hot96_def4/c16/A` | 64 | 32.02 | 255.81 | 1259.07 | 1638.41 | 2193.64 | 49.77 | 50.69 | — |
| `dual_hot96_def4/c16/B` | 64 | 31.09 | 263.52 | 1297.02 | 1588.78 | 2214.58 | 47.69 | 50.27 | — |
| `dual_hot96_def4/c32/A` | 128 | 46.78 | 350.26 | 1725.67 | 1886.91 | 2854.65 | 75.48 | 79.96 | — |
| `dual_hot96_def4/c32/B` | 128 | 46.51 | 352.25 | 1735.47 | 2035.15 | 3021.38 | 72.36 | 81.12 | — |
| `single_s5_hot96_def4/c32` | 128 | 48.48 | 337.97 | 1665.13 | 2923.84 | 3685.12 | 71.57 | 81.58 | 36.5 / 46.9 |

**dual_hot96_def4** — 부팅 `A=0 B=0` · greedy — · HBM 74.2~74.6 GiB ×8 · DRAM used 542 GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-threadpool-count 2 --kt-cpuinfer 48 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic
```

**single_s5_hot96_def4** — 부팅 `rc=0` · greedy 4/4 · HBM — · DRAM used — GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-threadpool-count 2 --kt-cpuinfer 96 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic
```

RUN.log 핵심 줄:

```
17:16:50 == IDE_069 dual vs tp8 start 20260915_171650 ==
17:20:04 single C32: Successful requests: 128  Benchmark duration (s): 48.48  Output token throughput (tok/s): 337.97  Total token throughput (tok/s): 1665.13  Median TTFT (ms): 2923.84  P95 TTFT (ms): 3685.12  Median TPOT (ms): 71.57  P95 TPOT (ms): 81.58  
17:20:22 -- dual dual_hot96_def4 --
17:23:25 dual dual_hot96_def4 C16each: A=255.81  B=263.52  합산=519.33
17:24:26 dual dual_hot96_def4 C32each: A=350.26  B=352.25  합산=702.51
17:24:44 == dual done 17:24:44 ==
```

### 5.10 `eval/results/20260915_172444_ide069_router_dual_synced`

시작: `17:24:44 == IDE_069 router-dual start 20260915_172444 ==`

| 셀 / 벤치 | 완료 | 소요 s | 출력 tok/s | 전체 tok/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | CPU busy 평균/최대 % |
|---|---|---|---|---|---|---|---|---|---|
| `router_hot96_def4/router_cache_aware/c32` | 128 | 33.59 | 487.81 | 2403.36 | 785.33 | 1804.03 | 55.67 | 69.92 | 36.5 / 48.8 |
| `router_hot96_def4/router_cache_aware/c64` | 256 | 50.49 | 649.02 | 3198.79 | 1710.79 | 2867.10 | 80.78 | 94.61 | 39.0 / 50.6 |
| `router_hot96_def4/router_round_robin/c32` | 128 | 34.27 | 478.13 | 2355.66 | 890.85 | 2255.37 | 56.27 | 70.68 | 35.6 / 50.4 |
| `router_hot96_def4/router_round_robin/c64` | 256 | 51.70 | 633.79 | 3123.72 | 1372.43 | 3073.19 | 89.45 | 100.71 | 39.2 / 50.0 |

**router_hot96_def4** — 부팅 `A=0 B=0` · greedy — · HBM 74.2~74.6 GiB ×8 · DRAM used — GB

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f --served-model-name q480 --host 127.0.0.1 --port 30000 --tp 4 --attention-backend triton --trust-remote-code --context-length 32768 --kt-weight-path /models/kt/qwen3-480b-int4 --kt-method AMXINT4 --kt-threadpool-count 2 --kt-cpuinfer 48 --kt-num-gpu-experts 96 --init-expert-location /models/kt/ide069/hotmap.json --kt-max-deferred-experts-per-token 4 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs 64 --mem-fraction-static 0.92 --max-total-tokens 24576 --ep-dispatch-algorithm dynamic
```

RUN.log 핵심 줄:

```
17:24:44 == IDE_069 router-dual start 20260915_172444 ==
17:27:08 router(round_robin) up after 0s: OK
17:28:02 router(round_robin) C32: Successful requests: 128  Benchmark duration (s): 34.27  Output token throughput (tok/s): 478.13  Total token throughput (tok/s): 2355.66  Median TTFT (ms): 890.85  P95 TTFT (ms): 2255.37  Median TPOT (ms): 56.27  P95 TPOT (ms): 70.68  
17:29:07 router(round_robin) C64: Successful requests: 256  Benchmark duration (s): 51.70  Output token throughput (tok/s): 633.79  Total token throughput (tok/s): 3123.72  Median TTFT (ms): 1372.43  P95 TTFT (ms): 3073.19  Median TPOT (ms): 89.45  P95 TPOT (ms): 100.71  
17:29:10 router(cache_aware) up after 0s: OK
17:30:04 router(cache_aware) C32: Successful requests: 128  Benchmark duration (s): 33.59  Output token throughput (tok/s): 487.81  Total token throughput (tok/s): 2403.36  Median TTFT (ms): 785.33  P95 TTFT (ms): 1804.03  Median TPOT (ms): 55.67  P95 TPOT (ms): 69.92  
17:31:08 router(cache_aware) C64: Successful requests: 256  Benchmark duration (s): 50.49  Output token throughput (tok/s): 649.02  Total token throughput (tok/s): 3198.79  Median TTFT (ms): 1710.79  P95 TTFT (ms): 2867.10  Median TPOT (ms): 80.78  P95 TPOT (ms): 94.61  
```

## 6. 해석 (RESULT.md 요약)

### 한눈에

| | 구성 (TP4, GPU 4장) | C32 tok/s | C64 tok/s | HBM/장 |
|---|---|---|---|---|
| 출발점 | expert 전량 CPU (IDE_068 b1) | 43.39 (C16) | — | 17 GiB |
| 물리 id 96 | hotmap 없이 앞 96개 GPU (IDE_068 b6) | 51.55 (C16) | — | 76 GiB |
| hotmap 96 | + 빈도 기반 배치 + cuda graph | 470.26 | — | 74.6 GiB |
| + deferral 4 | IDE_030 처방 | 519.81 / 486.90 | 425.15 | 74.6 GiB |
| **+ KV 40,960** | **최종 (GPU 4장)** | 465 ± 6 | **642 ± 7** | 75.6 GiB |
| dual ×2 (GPU 8장) | 위 구성 ×2, 소켓별 cpuinfer 48 | 519.33 | **702.51** | 74.6 GiB ×8 |
| GPU-only TP8+EP8 (GPU 8장) | 참조 | 1,029.75 | **2,031.11** | 72 GiB ×8 |

**출발점 대비 14.8× (C64), IDE_030 기록(C32 490.9 / C64 408.2) 대비 C64 +57 %.** GSM 40: deferral 0 = 39/40, deferral 4 = 39/40.

#### 2.1 hot expert 배치 — 지배 항

| hot 수 | 배치 | C32 tok/s | HBM/장 | 근거 |
|---|---|---|---|---|
| 0 | — | ≈43 (C16) | 17 GiB | IDE_068 b1 |
| 96 | 물리 id 0~95 (빈도 무관) | 51.55 (C16) | 76 GiB | IDE_068 b6 |
| 64 | hotmap | 230.14 | 53 GiB | t6 |
| **96** | **hotmap** | **470 ~ 520** | 74.6 GiB | t1/t2 |
| 100 | hotmap, KV 16k | 356.45 | 76.9 GiB | t12 — KV 압박이 상쇄 |
| 112 | hotmap | **OOM** | — | t3 |

- 같은 96개라도 빈도 기반 배치가 물리 id 배치의 **9배** 다. hotmap 이 이 실험의 핵심이다.
- **TP4 의 HBM 상한은 hot96 이다.** TP4 샤드 기준 expert 1개/층 ≈ 11.8 MB → 62층 = 0.73 GB/expert/GPU. 112개 = 82 GB > 79.2 GB. KV 를 0 으로 해도 안 들어간다 (t3 OOM 으로 확인). hot100 은 들어가지만 KV 가 16k 로 줄어 C32 에서 오히려 손해.

#### 2.2 deferral (`--kt-max-deferred-experts-per-token`)

| N | C32 tok/s | GSM 40 |
|---|---|---|
| 0 | 470.26 | 39/40 |
| 2 | 454.27 | — |
| **4** | **519.81** | **39/40** |
| 8 | 500.17 | — |

4 가 최적. 근사 연산이지만 GSM 40 에서 저하 0 — 채택.

#### 2.3 KV pool 크기 — 동시성 상한을 정한다

| max-total-tokens (mf) | C32 | C48 | C64 | C80 | C96 |
|---|---|---|---|---|---|
| 24,576 (0.92) | 519.8 / 486.9 | — | 425.2 | — | — |
| **40,960 (0.95)** | 459.3 / 465.2 / 471.9 | 564.6 | **637.2 / 637.5 / 650.8** | 239.8 (붕괴) | — |
| 57,344 (0.96) | — | — | 634.1 | — | 328.7 (붕괴) |

- 24k 는 C32 까지, 40k 는 C64 까지 감당한다. 그 위에서는 KV 부족으로 TPOT 이 3~4배 뛴다.
- KV 를 키우면 C32 는 약 10 % 낮아진다 (mem-fraction 0.92 → 0.95 에서 graph/workspace 여유가 줄어드는 것으로 추정, 원인 미검증).
- 56k 는 64 에서 40k 와 같고 96 은 붕괴 — 40k 가 C64 운전의 적정점.

#### 2.4 효과 없음 / 역효과

| 시도 | C32 tok/s | 판정 |
|---|---|---|
| cuda graph ON (expert 0·16, IDE_068 B3) | 43.5 / 46.1 | 효과 없음 — GPU 구간이 스텝의 소수라 launch overhead 가 안 보임 |
| cpuinfer 96 → 112 | 483.24 | 차이 없음 (jitter 범위) |
| EAGLE3 spec (480B 전용 draft) | 293.46 | **역효과** — draft 검증 토큰이 CPU expert 비용을 키움 |
| STANDALONE spec (Qwen3-4B) | 169.77 | **역효과**, TTFT 20 s |
| dispatch static | 엔진 거부 | a2a 백엔드 전제 (`ValueError`) |

#### 2.5 jitter

같은 구성 반복: C32 ±1.5 % (459/465/472), C64 ±1.1 % (637/638/651), KV24k C32 ±3.5 % (520/487). 이보다 작은 차이는 주장하지 않는다.

#### 4.3 읽는 법

1. **GPU 8장이 있으면 GPU-only 가 2.9배 빠르다** (2,031 vs 703 @총 64). 오프로딩의 가치는 GPU 가 모자랄 때(1~4장)이지, 8장을 다 쓸 수 있을 때가 아니다.
2. **dual 은 단일 4장 대비 +9 % 뿐이다** (703 vs 642). 하이브리드 스텝은 CPU expert 구간이 지배하고, 단일 인스턴스가 이미 두 소켓의 DRAM 대역폭을 다 쓴다. 인스턴스를 둘로 나누면 각자 소켓 하나(대역폭·물리코어 절반)를 받아 각 350 이 되고, 합이 703 이다 — GPU 를 4장 더 붙여도 CPU 가 그대로라 늘어날 여지가 작다.
3. **소켓 하나짜리 인스턴스 = 단일 양소켓 인스턴스의 65 %**: sgl-kt5 단독 (소켓0, cpuinfer 96) 337.97 @C32 vs sgl-kt (양 소켓) 487~520. 처음에는 소프트웨어 차이로 의심해 동기화했으나 동기화 후에도 338 로 같았다 → 원인은 cpuset (DRAM 대역폭·물리코어 절반) 이다.
4. **라우터 오버헤드**: 합산 대비 round_robin −7.9 % / −9.8 % (총 32 / 64), cache_aware −6.1 % / −7.6 %. expert 0 에서는 −6.2 % / −11.9 %. 단일 엔드포인트가 필요하면 cache_aware 가 낫다.
5. expert 0 dual (66) 은 8-30 교정판 (60.5) 과 같은 규모이고, 단일 양소켓 expert 0 (IDE_068 b1 43.4 @C16, 8-30 66.6 @C32) 을 넘지 못한다 — 동일한 CPU 대역폭 분할 논리.

전문은 `shadow_assists/features/IDE_069/RESULT.md`, 시간순 기록은 `PROGRESS.md`.

## 7. 파일 인벤토리

| 경로 | 바이트 | SHA256 (앞 16) |
|---|---|---|
| `eval/results/20260915_135132_ide069_hotmap/RUN.log` | 897 | `3f684806173f8077` |
| `eval/results/20260915_135132_ide069_hotmap/hotmap.json` | 42,933 | `7dc072dad8c5de8f` |
| `eval/results/20260915_135132_ide069_hotmap/hotmap_stats.json` | 787 | `81916c65ccf41140` |
| `eval/results/20260915_135132_ide069_hotmap/rec/.cpu_mon` | 8 | `6b7e9fbbf0e3e321` |
| `eval/results/20260915_135132_ide069_hotmap/rec/.gpu_mon` | 8 | `3447170c4dbdd4e2` |
| `eval/results/20260915_135132_ide069_hotmap/rec/bench.log` | 6,218 | `08898fbea1b163c3` |
| `eval/results/20260915_135132_ide069_hotmap/rec/bench_summary.txt` | 258 | `9580418d0f45e1a6` |
| `eval/results/20260915_135132_ide069_hotmap/rec/bench_t0.txt` | 11 | `afdc0240c593c379` |
| `eval/results/20260915_135132_ide069_hotmap/rec/bench_t1.txt` | 11 | `cf76ecacdf628a89` |
| `eval/results/20260915_135132_ide069_hotmap/rec/cpu_util.txt` | 2,172 | `b0359a657fc0c997` |
| `eval/results/20260915_135132_ide069_hotmap/rec/gpu_util.csv` | 8,382 | `412de2063ba19104` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/RUN.log` | 6,186 | `b7974f2dc6df60cc` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/cells.txt` | 535 | `a0539d7e49ff27b1` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/error_lines.log` | 940 | `41f58309ab012fa9` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/free_after_boot.txt` | 207 | `de9e4b9241c7489e` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/free_before.txt` | 207 | `d698ef486284c293` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/hbm_after_boot.csv` | 176 | `e126e08acc7141c8` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/hbm_before.csv` | 83 | `b9cc2bb75cbcd856` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/launch_cmd.txt` | 683 | `108a5267edee7146` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/server_tail.log` | 16,360 | `f2078b2e268754bf` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/smoke.json` | 1,523 | `c46c206cadca64db` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/smoke_chat.json` | 494 | `e7a997697b4f55a0` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/smoke_texts.txt` | 531 | `d3bb53d23df77b27` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/verdict.txt` | 14 | `e9996a2c4e0c4699` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c64/.cpu_mon` | 8 | `961324ae114cc782` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c64/.gpu_mon` | 8 | `0418bced6195337f` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c64/bench.log` | 7,340 | `22bccf5855c5e3c5` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c64/bench_summary.txt` | 259 | `2efdf768b017f04a` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c64/bench_t0.txt` | 11 | `5c1997bf54609cad` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c64/bench_t1.txt` | 11 | `bd18afba12d6b261` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c64/cpu_util.txt` | 962 | `6e6bb664cf702317` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c64/gpu_util.csv` | 3,982 | `3880183a3b6a6902` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c16/.cpu_mon` | 8 | `6e90514424aa73da` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c16/.gpu_mon` | 8 | `23756ce7773bd30e` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c16/bench.log` | 6,218 | `ccc097a42d67015f` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c16/bench_summary.txt` | 256 | `42ecb641a075d572` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c16/bench_t0.txt` | 11 | `a61a28d900a6a2dd` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c16/bench_t1.txt` | 11 | `2a96044129d16341` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c16/cpu_util.txt` | 390 | `42e80e49052c39ce` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c16/gpu_util.csv` | 1,665 | `56a9f289249a4231` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c32/.cpu_mon` | 8 | `72524b4cf65ba9f3` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c32/.gpu_mon` | 8 | `c704eb94ef484f89` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c32/bench.log` | 6,222 | `f4684eb0ae8ec932` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c32/bench_summary.txt` | 257 | `1c6cb56d0520b83d` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c32/bench_t0.txt` | 11 | `eee3153d8f061f91` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c32/bench_t1.txt` | 11 | `26ee2d7292623266` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c32/cpu_util.txt` | 478 | `25090fbe952bec6c` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t2_hot96_def4_graph/c32/gpu_util.csv` | 1,878 | `a228465c0244aeb9` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/error_lines.log` | 940 | `02df822b5a65455d` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/free_after_boot.txt` | 207 | `f98d60f35d36c7b3` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/free_before.txt` | 207 | `94cf5eb82046fd6e` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/hbm_after_boot.csv` | 176 | `e126e08acc7141c8` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/hbm_before.csv` | 72 | `a75a5b66b85cbe63` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/launch_cmd.txt` | 683 | `7ade54f449600e66` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/server_tail.log` | 15,543 | `3fbda52a48d5e935` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/smoke.json` | 1,616 | `574f6da903be3712` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/smoke_chat.json` | 494 | `d25dbfc1f7e86078` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/smoke_texts.txt` | 624 | `05c1d57d24f2ee5a` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/verdict.txt` | 14 | `e9996a2c4e0c4699` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/c32/.cpu_mon` | 8 | `69aabcbae54906f4` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/c32/.gpu_mon` | 8 | `e672d143c965f4b8` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/c32/bench.log` | 6,222 | `4c9946d66cb8e70e` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/c32/bench_summary.txt` | 257 | `32ef2ba6062c04e9` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/c32/bench_t0.txt` | 11 | `74b2b80f3f300174` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/c32/bench_t1.txt` | 11 | `481017e214e3016a` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/c32/cpu_util.txt` | 522 | `087056da8363f3ed` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t4_hot96_def2_graph/c32/gpu_util.csv` | 2,092 | `7f7604309e0c166f` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/error_lines.log` | 940 | `14470aea80bbdb0a` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/free_after_boot.txt` | 207 | `c7cf16927db36db5` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/free_before.txt` | 207 | `09449c92d7926c03` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/hbm_after_boot.csv` | 176 | `8303fc862911b704` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/hbm_before.csv` | 72 | `a75a5b66b85cbe63` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/launch_cmd.txt` | 645 | `c54aeed4e7c4c0a0` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/server_tail.log` | 15,339 | `20dd84f06c7c64a6` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/smoke.json` | 1,570 | `524faf3bcc187c30` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/smoke_chat.json` | 494 | `d1e53ba7b806cde2` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/smoke_texts.txt` | 578 | `031612b64f7c401d` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/verdict.txt` | 14 | `0e7437a6ec637db7` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/c16/.cpu_mon` | 8 | `15f5e6dd39841418` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/c16/.gpu_mon` | 8 | `9123c1dd9324946e` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/c16/bench.log` | 6,218 | `a645c58f9b8e21cc` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/c16/bench_summary.txt` | 256 | `f40d3886c4fb05bb` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/c16/bench_t0.txt` | 11 | `5ad10cf1701be2e4` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/c16/bench_t1.txt` | 11 | `a5adf55b475327b3` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/c16/cpu_util.txt` | 412 | `43badac46f55fbfc` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/c16/gpu_util.csv` | 1,667 | `d3fc11bf197c4418` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/c32/.cpu_mon` | 8 | `0af9ba10d820791e` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/c32/.gpu_mon` | 8 | `357eed7e0a217044` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/c32/bench.log` | 6,222 | `86660a118cba731e` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/c32/bench_summary.txt` | 257 | `f0faff635a11875d` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/c32/bench_t0.txt` | 11 | `ca6292e3db379576` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/c32/bench_t1.txt` | 11 | `e38c2d622d88e339` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/c32/cpu_util.txt` | 500 | `bb9383f2b865480a` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t1_hot96_def0_graph/c32/gpu_util.csv` | 2,089 | `4ec3c0e8e610e954` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/error_lines.log` | 940 | `ef7906b69f9b809d` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/free_after_boot.txt` | 207 | `f98d60f35d36c7b3` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/free_before.txt` | 207 | `a9cbe3d2028c49e2` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/hbm_after_boot.csv` | 176 | `afef553d1a9a2e62` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/hbm_before.csv` | 83 | `d07806552657b7df` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/launch_cmd.txt` | 683 | `55bcd7b01c0b7658` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/server_tail.log` | 15,416 | `fbf0e07a036736e1` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/smoke.json` | 1,577 | `6f58de986734acd4` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/smoke_chat.json` | 494 | `71df58407acf3cae` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/smoke_texts.txt` | 585 | `51c91fa4a0fa70dd` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/verdict.txt` | 14 | `e9996a2c4e0c4699` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/c32/.cpu_mon` | 8 | `cd860c92b8017ef2` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/c32/.gpu_mon` | 8 | `cdd65ffc21ce54dd` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/c32/bench.log` | 6,222 | `0571e3544bb9f8d4` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/c32/bench_summary.txt` | 259 | `d494df3a67dd811a` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/c32/bench_t0.txt` | 11 | `e7367dccf0fcde0b` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/c32/bench_t1.txt` | 11 | `5c8548413f0397e7` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/c32/cpu_util.txt` | 896 | `96e0366f9be8afe6` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t6_hot64_def4_graph/c32/gpu_util.csv` | 3,571 | `a39a6a7034db0d69` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/error_lines.log` | 940 | `7470f71215264dc8` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/free_after_boot.txt` | 207 | `de9e4b9241c7489e` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/free_before.txt` | 207 | `677a7e993f623e9a` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/hbm_after_boot.csv` | 176 | `8303fc862911b704` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/hbm_before.csv` | 83 | `85125491c604fadf` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/launch_cmd.txt` | 683 | `14a5ed1dff33dd51` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/server_tail.log` | 15,540 | `da416b250d59dca8` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/smoke.json` | 1,634 | `bad5df67d55edde2` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/smoke_chat.json` | 494 | `c7af1b0914f4df40` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/smoke_texts.txt` | 641 | `e6672e340a8e00e2` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/verdict.txt` | 14 | `0e7437a6ec637db7` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/c32/.cpu_mon` | 8 | `5cfc3fdd141da1c8` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/c32/.gpu_mon` | 8 | `1a546dd3badf2309` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/c32/bench.log` | 6,222 | `ffc3160f4aa58b75` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/c32/bench_summary.txt` | 257 | `fa0f23afb05de3e4` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/c32/bench_t0.txt` | 11 | `38053e0548d3f21e` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/c32/bench_t1.txt` | 11 | `dc1efecd51e323a5` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/c32/cpu_util.txt` | 478 | `52356a17eb7b253f` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t5_hot96_def8_graph/c32/gpu_util.csv` | 2,091 | `24381d7241a7dad1` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t3_hot112_def4_graph/error_lines.log` | 1,728 | `0da5fa09bd020269` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t3_hot112_def4_graph/free_before.txt` | 207 | `a9cbe3d2028c49e2` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t3_hot112_def4_graph/hbm_before.csv` | 83 | `4b6c05e490b1a2db` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t3_hot112_def4_graph/launch_cmd.txt` | 684 | `f8632e1cef566364` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t3_hot112_def4_graph/server_tail.log` | 14,611 | `d9f6f23a9c786999` |
| `eval/results/20260915_135906_ide069_sweep_cells_round1/t3_hot112_def4_graph/verdict.txt` | 8 | `f25eeb55dde43f31` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/RUN.log` | 982 | `c7e8482d91b89f37` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def0/error_lines.log` | 940 | `bb587b179ecbcf8f` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def0/free_after_boot.txt` | 207 | `f98d60f35d36c7b3` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def0/free_before.txt` | 207 | `c68de33c3e10c07c` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def0/gsm40.json` | 7,596 | `72f148b4dae077d4` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def0/hbm_after_boot.csv` | 176 | `8303fc862911b704` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def0/hbm_before.csv` | 72 | `a75a5b66b85cbe63` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def0/launch_cmd.txt` | 645 | `c54aeed4e7c4c0a0` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def0/server_tail.log` | 29,986 | `7e2661a4da1ffc72` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def0/smoke.json` | 1,570 | `e3cd784746db2a34` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def0/smoke_chat.json` | 494 | `05c73c4b8886f6a4` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def0/smoke_texts.txt` | 578 | `031612b64f7c401d` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def0/verdict.txt` | 14 | `e9996a2c4e0c4699` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def4/error_lines.log` | 940 | `4aee66d88fb5ffcb` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def4/free_after_boot.txt` | 207 | `f98d60f35d36c7b3` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def4/free_before.txt` | 207 | `1b5fc15da2151a73` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def4/gsm40.json` | 7,608 | `8114b0ff94f3a391` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def4/hbm_after_boot.csv` | 176 | `e126e08acc7141c8` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def4/hbm_before.csv` | 83 | `c2b8f112fb198480` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def4/launch_cmd.txt` | 683 | `65545744e3c19c5c` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def4/server_tail.log` | 29,523 | `21c9898754e5014d` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def4/smoke.json` | 1,523 | `00b0f016e4e685fc` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def4/smoke_chat.json` | 494 | `3b88415412dcf83c` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def4/smoke_texts.txt` | 531 | `d3bb53d23df77b27` |
| `eval/results/20260915_141917_ide069_gsm40_hot96/def4/verdict.txt` | 14 | `e9996a2c4e0c4699` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/RUN.log` | 5,608 | `5497e69dd7b764fd` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/cells.txt` | 1,080 | `8a81c8e91ac80299` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/error_lines.log` | 940 | `a281a84411bbf540` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/free_after_boot.txt` | 207 | `f44b5b788152c002` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/free_before.txt` | 207 | `4b0c87c54115f2e1` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/hbm_after_boot.csv` | 176 | `e27167fc73018915` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/hbm_before.csv` | 83 | `b02bcffc0cecc4b6` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/launch_cmd.txt` | 972 | `672c9af9125e0d6c` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/server_tail.log` | 21,470 | `a5498bccb8c8201a` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/smoke.json` | 1,572 | `087ffb758198a129` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/smoke_chat.json` | 494 | `b4f98e7b6b96df63` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/smoke_texts.txt` | 583 | `b898b432bb183836` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/verdict.txt` | 14 | `d62b5bfd1b7a4d1a` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/c32/.cpu_mon` | 8 | `778725bb79639679` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/c32/.gpu_mon` | 8 | `46783d9a7b8f9102` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/c32/bench.log` | 11,148 | `e12fa669b5fd1dcd` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/c32/bench_summary.txt` | 257 | `965f2c5b34fad31a` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/c32/bench_t0.txt` | 11 | `efede0e4430c8357` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/c32/bench_t1.txt` | 11 | `de03ea7c23c7f76a` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/c32/cpu_util.txt` | 742 | `364946ce6bc34e08` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t8_hot96_def4_eagle3/c32/gpu_util.csv` | 2,935 | `52e4660e0c0b52c0` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/error_lines.log` | 940 | `ae170fc8234d6d5f` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/free_after_boot.txt` | 207 | `4768963f34afa128` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/free_before.txt` | 207 | `9de5a025935d9cdd` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/hbm_after_boot.csv` | 176 | `e126e08acc7141c8` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/hbm_before.csv` | 84 | `e983a63e30272c29` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/launch_cmd.txt` | 701 | `dd76e5b65c3e0378` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/server_tail.log` | 15,542 | `49f1b9d42b8b3142` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/smoke.json` | 1,523 | `fcb3fd83b9e51096` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/smoke_chat.json` | 494 | `74420f81adb0201e` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/smoke_texts.txt` | 531 | `d3bb53d23df77b27` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/verdict.txt` | 14 | `e9996a2c4e0c4699` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/c32/.cpu_mon` | 8 | `8e95756fc0089d3e` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/c32/.gpu_mon` | 8 | `910827e0afef7942` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/c32/bench.log` | 6,222 | `ebfbcd71369f63fd` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/c32/bench_summary.txt` | 257 | `bdfc0f27e11ef0e6` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/c32/bench_t0.txt` | 11 | `dfd57ead0cb22c7c` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/c32/bench_t1.txt` | 11 | `670387274d701761` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/c32/cpu_util.txt` | 500 | `b7dc2959c8b8ca42` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t10_hot96_def4_cpu112/c32/gpu_util.csv` | 2,090 | `bc18454146c2c6b6` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/error_lines.log` | 940 | `e9f049c7c04fe10c` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/free_after_boot.txt` | 207 | `b92ed6e55bd4c537` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/free_before.txt` | 207 | `c68de33c3e10c07c` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/hbm_after_boot.csv` | 176 | `d1013cc1fb5153c6` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/hbm_before.csv` | 72 | `a75a5b66b85cbe63` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/launch_cmd.txt` | 683 | `8bee87aedbefe503` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/server_tail.log` | 15,290 | `4a0989da2bc7d530` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/smoke.json` | 1,523 | `7132daa596b60757` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/smoke_chat.json` | 494 | `287565c9a600df26` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/smoke_texts.txt` | 531 | `d3bb53d23df77b27` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/verdict.txt` | 14 | `e9996a2c4e0c4699` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/c64/.cpu_mon` | 8 | `5ffac8bc06d769e2` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/c64/.gpu_mon` | 8 | `4f59c0a3842a4812` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/c64/bench.log` | 6,224 | `426387e5ac8b4f2b` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/c64/bench_summary.txt` | 257 | `e8963c446895fd47` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/c64/bench_t0.txt` | 11 | `9fb4a18ee0a28926` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/c64/bench_t1.txt` | 11 | `b38b5b7ecfb0dbdf` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/c64/cpu_util.txt` | 698 | `0cda6c50db868a2f` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/c64/gpu_util.csv` | 2,722 | `f45450d3e4b3b614` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/c32/.cpu_mon` | 8 | `c9e6f7728f95e790` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/c32/.gpu_mon` | 8 | `cc7f239ad11f36ef` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/c32/bench.log` | 6,222 | `f3bd52bd4df6de53` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/c32/bench_summary.txt` | 257 | `b04b04cc834cd5b0` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/c32/bench_t0.txt` | 11 | `0a7baacd22bb499a` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/c32/bench_t1.txt` | 11 | `d3376803aef3202c` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/c32/cpu_util.txt` | 522 | `500191745b379d7d` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t7_hot96_def4_kv40k/c32/gpu_util.csv` | 2,094 | `04faf6b490fcec11` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/error_lines.log` | 940 | `eb67709bedc208be` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/free_after_boot.txt` | 207 | `0dd15bd87f6ef406` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/free_before.txt` | 207 | `5968a6ace9902c59` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/hbm_after_boot.csv` | 176 | `115975e42f58d270` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/hbm_before.csv` | 83 | `f43745639d7b2ca7` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/launch_cmd.txt` | 921 | `507f393887d8f50c` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/server_tail.log` | 23,147 | `ae477576ff56ce9c` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/smoke.json` | 1,626 | `c711527568e3e8ba` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/smoke_chat.json` | 494 | `04b75ca2786ee5e2` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/smoke_texts.txt` | 635 | `0edc26246ac44347` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/verdict.txt` | 14 | `20a5cd95789761db` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/c32/.cpu_mon` | 8 | `6b619620f5a7be70` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/c32/.gpu_mon` | 8 | `536b05512c15f05d` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/c32/bench.log` | 12,666 | `afdbfbb988fae3f5` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/c32/bench_summary.txt` | 258 | `caeccfcc2b4bd6e1` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/c32/bench_t0.txt` | 11 | `0df3dd4814ada2f5` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/c32/bench_t1.txt` | 11 | `12f91ea4c10cc7ba` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/c32/cpu_util.txt` | 1,182 | `34b18bd2268460b8` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t9_hot96_def4_standalone4b/c32/gpu_util.csv` | 4,615 | `bc548e7c9e0ea7c3` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t11_hot96_def4_static/error_lines.log` | 251 | `30d38163ce2ca43a` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t11_hot96_def4_static/free_before.txt` | 207 | `15b57c5a979a7ef3` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t11_hot96_def4_static/hbm_before.csv` | 83 | `85125491c604fadf` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t11_hot96_def4_static/launch_cmd.txt` | 682 | `809961a7ebccef30` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t11_hot96_def4_static/server_tail.log` | 1,884 | `f722fe58fd1d60d4` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t11_hot96_def4_static/verdict.txt` | 8 | `430937cb73dc8888` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/error_lines.log` | 940 | `32516b56afbd7ea5` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/free_after_boot.txt` | 207 | `f6333dc5b8d9f83b` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/free_before.txt` | 207 | `374833885d10bb75` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/hbm_after_boot.csv` | 176 | `888c70d4c32e5d7a` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/hbm_before.csv` | 72 | `a75a5b66b85cbe63` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/launch_cmd.txt` | 684 | `77cd7e128242ffb7` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/server_tail.log` | 17,323 | `914f70baf37c231b` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/smoke.json` | 1,369 | `bf5ce8f79add13e4` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/smoke_chat.json` | 494 | `a0a028ee08940c80` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/smoke_texts.txt` | 384 | `6f56500cedf8c85d` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/verdict.txt` | 14 | `0e7437a6ec637db7` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/c32/.cpu_mon` | 8 | `815ba6f49234aaf7` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/c32/.gpu_mon` | 8 | `46c7a7f082c9e91e` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/c32/bench.log` | 6,864 | `09c82664bcedf387` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/c32/bench_summary.txt` | 257 | `298e7917c01606b3` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/c32/bench_t0.txt` | 11 | `5a414b17103c7e2f` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/c32/bench_t1.txt` | 11 | `998f9be4a26ddf32` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/c32/cpu_util.txt` | 632 | `aac45ecb3626579c` |
| `eval/results/20260915_142742_ide069_sweep_cells_round2/t12_hot100_def4/c32/gpu_util.csv` | 2,513 | `5de004a5fb01d86d` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/RUN.log` | 1,807 | `c7269de3c888d206` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/dual_cold_expert0/free_before.txt` | 207 | `67afb702d571323b` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/dual_cold_expert0/verdict.txt` | 8 | `1f40e04b4c48a2fa` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/dual_cold_expert0/A/launch_cmd.txt` | 519 | `118b99bb658f590b` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/dual_cold_expert0/A/server_tail.log` | 0 | `e3b0c44298fc1c14` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/dual_cold_expert0/B/launch_cmd.txt` | 519 | `be821f121f43db3e` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/dual_cold_expert0/B/server_tail.log` | 0 | `e3b0c44298fc1c14` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/ref_gpu_tp8_ep8/hbm_after_boot.csv` | 104 | `19eb2f682f60a6f8` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/ref_gpu_tp8_ep8/launch_cmd.txt` | 368 | `fdb22aa0831372a3` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/ref_gpu_tp8_ep8/smoke.json` | 1,422 | `cb4cfd3e607f079c` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/ref_gpu_tp8_ep8/smoke_chat.json` | 494 | `f6fcb828083eafc6` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/ref_gpu_tp8_ep8/smoke_texts.txt` | 437 | `10d396b9ac326feb` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/ref_gpu_tp8_ep8/verdict.txt` | 5 | `93ff7811a209e2a8` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/ref_gpu_tp8_ep8/c64/bench.log` | 6,224 | `89776ab898090cc4` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/ref_gpu_tp8_ep8/c64/bench_summary.txt` | 238 | `0212c910ebf8e0ff` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/ref_gpu_tp8_ep8/c64/cpu_util.txt` | 124,079 | `f341fcc46868e7c4` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/ref_gpu_tp8_ep8/c32/bench.log` | 6,222 | `9e62062efb651b2a` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/ref_gpu_tp8_ep8/c32/bench_summary.txt` | 236 | `d2612266b9818d48` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/ref_gpu_tp8_ep8/c32/cpu_util.txt` | 124,380 | `94a49099a41a9180` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/dual_hot96_def4/free_before.txt` | 207 | `67afb702d571323b` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/dual_hot96_def4/verdict.txt` | 8 | `1f40e04b4c48a2fa` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/dual_hot96_def4/A/launch_cmd.txt` | 683 | `4c74de28db128652` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/dual_hot96_def4/A/server_tail.log` | 0 | `e3b0c44298fc1c14` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/dual_hot96_def4/B/launch_cmd.txt` | 683 | `32a295afda8391e7` |
| `eval/results/20260915_144748_ide069_dual_vs_tp8/dual_hot96_def4/B/server_tail.log` | 0 | `e3b0c44298fc1c14` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/RUN.log` | 5,143 | `f998e378db3f60f8` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/cells.txt` | 505 | `63132901a47ca364` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/error_lines.log` | 940 | `ee78cfd0d26f90b3` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/free_after_boot.txt` | 207 | `101eef5aabb89f04` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/free_before.txt` | 207 | `5dfdf41768d0f217` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/hbm_after_boot.csv` | 176 | `e126e08acc7141c8` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/hbm_before.csv` | 83 | `9c1f86f4e3505cea` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/launch_cmd.txt` | 683 | `108a5267edee7146` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/server_tail.log` | 15,541 | `0e5622a5ed24f902` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/smoke.json` | 1,523 | `01b7e0b099273748` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/smoke_chat.json` | 494 | `2960979eecd7a59f` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/smoke_texts.txt` | 531 | `d3bb53d23df77b27` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/verdict.txt` | 14 | `e9996a2c4e0c4699` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/c32/.cpu_mon` | 8 | `505dfde409661970` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/c32/.gpu_mon` | 8 | `3beb3fb26e3cdb3a` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/c32/bench.log` | 6,222 | `d6ba509938a4c6d4` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/c32/bench_summary.txt` | 257 | `7b9c651f6fa06fac` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/c32/bench_t0.txt` | 11 | `41b14016142c220b` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/c32/bench_t1.txt` | 11 | `394de217084bf887` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/c32/cpu_util.txt` | 500 | `52d973fc30e84fce` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t16_hot96_def4_kv24k_rep/c32/gpu_util.csv` | 2,093 | `dae4f66b40ff045a` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/error_lines.log` | 940 | `b23385f5a2ad7f10` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/free_after_boot.txt` | 207 | `ee6a27544b9637e0` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/free_before.txt` | 207 | `4e21e4a33c4ea2fa` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/hbm_after_boot.csv` | 176 | `d1013cc1fb5153c6` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/hbm_before.csv` | 72 | `a75a5b66b85cbe63` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/launch_cmd.txt` | 683 | `8bee87aedbefe503` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/server_tail.log` | 16,784 | `e73211b1e713d7cf` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/smoke.json` | 1,523 | `5dd1544b9abdc5f0` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/smoke_chat.json` | 494 | `3ffce801de616960` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/smoke_texts.txt` | 531 | `d3bb53d23df77b27` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/verdict.txt` | 14 | `0e7437a6ec637db7` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c64/.cpu_mon` | 8 | `ef1b20f073525cf7` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c64/.gpu_mon` | 8 | `d36c67f21e07d450` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c64/bench.log` | 6,224 | `0575366a4b2c380b` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c64/bench_summary.txt` | 257 | `03a87b95d27a3983` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c64/bench_t0.txt` | 11 | `2e2769943cbbbd66` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c64/bench_t1.txt` | 11 | `2b155ab099c41223` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c64/cpu_util.txt` | 698 | `c4ba8bd7d1fa0645` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c64/gpu_util.csv` | 2,722 | `fe4c5c65368fc04d` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c80/.cpu_mon` | 8 | `2a399f0ce14898ef` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c80/.gpu_mon` | 8 | `f617eb868f8d7b7f` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c80/bench.log` | 7,057 | `b11108dc1d8f7b46` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c80/bench_summary.txt` | 261 | `cf16598de3149bf2` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c80/bench_t0.txt` | 11 | `9543ad79923051e3` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c80/bench_t1.txt` | 11 | `bc1b18eb5de74862` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c80/cpu_util.txt` | 1,996 | `cb6df8dbd31d5899` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c80/gpu_util.csv` | 7,751 | `a2d45b9a064637f6` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c48/.cpu_mon` | 8 | `732ea9ed42e40991` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c48/.gpu_mon` | 8 | `12022bc61276f3cf` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c48/bench.log` | 6,223 | `d5a5f0825fd5fef5` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c48/bench_summary.txt` | 257 | `c65a9230e5a71239` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c48/bench_t0.txt` | 11 | `d28259d0a4eb6294` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c48/bench_t1.txt` | 11 | `fcb700464d43c40a` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c48/cpu_util.txt` | 610 | `436def2d09c45653` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c48/gpu_util.csv` | 2,510 | `63c464afb7cc6f23` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c32/.cpu_mon` | 8 | `0cad263998ad6cc7` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c32/.gpu_mon` | 8 | `d3baa792385d7a36` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c32/bench.log` | 6,222 | `64d2cb6b388a4d0a` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c32/bench_summary.txt` | 257 | `011f5c2c45a0eec1` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c32/bench_t0.txt` | 11 | `00a61ffd99dfb8bb` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c32/bench_t1.txt` | 11 | `7ef7ca66848b22cb` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c32/cpu_util.txt` | 522 | `a697e84fde38a9bd` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t13_hot96_def4_kv40k_rep1/c32/gpu_util.csv` | 2,093 | `51eb1328f33876c8` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/error_lines.log` | 940 | `41fb0fc55f85c37b` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/free_after_boot.txt` | 207 | `101eef5aabb89f04` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/free_before.txt` | 207 | `102c5a89c7b313b7` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/hbm_after_boot.csv` | 176 | `d1013cc1fb5153c6` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/hbm_before.csv` | 83 | `3d994125710cc3e8` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/launch_cmd.txt` | 683 | `8bee87aedbefe503` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/server_tail.log` | 15,290 | `285a7f737c807f1e` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/smoke.json` | 1,523 | `0e6ef6d53adfb2fd` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/smoke_chat.json` | 494 | `24aaadffe8269f30` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/smoke_texts.txt` | 531 | `d3bb53d23df77b27` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/verdict.txt` | 14 | `e9996a2c4e0c4699` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/c64/.cpu_mon` | 8 | `4014a6cc62c2fd01` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/c64/.gpu_mon` | 8 | `33ba212e25492e72` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/c64/bench.log` | 6,224 | `9a5c44d86e63a9de` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/c64/bench_summary.txt` | 257 | `00fec635a9192d1e` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/c64/bench_t0.txt` | 11 | `8a4c65284f43f462` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/c64/bench_t1.txt` | 11 | `67d9c010bf965c38` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/c64/cpu_util.txt` | 676 | `c4ad5118a50959d0` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/c64/gpu_util.csv` | 2,722 | `d85b3b9b1c6bec57` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/c32/.cpu_mon` | 8 | `186af553440bce3d` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/c32/.gpu_mon` | 8 | `ff2a41a546f3c61a` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/c32/bench.log` | 6,222 | `a98c25dbd4e13938` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/c32/bench_summary.txt` | 257 | `4c37220e9de90ce4` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/c32/bench_t0.txt` | 11 | `af972fa7344cbdcf` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/c32/bench_t1.txt` | 11 | `8a4c65284f43f462` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/c32/cpu_util.txt` | 500 | `7e3ac8f5efd15c6b` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t14_hot96_def4_kv40k_rep2/c32/gpu_util.csv` | 2,093 | `530e32f13fd779f2` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/error_lines.log` | 940 | `bc8d3853c141949b` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/free_after_boot.txt` | 207 | `101eef5aabb89f04` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/free_before.txt` | 207 | `d418176b780f6596` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/hbm_after_boot.csv` | 176 | `7b063a661ef5f8c5` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/hbm_before.csv` | 83 | `b02bcffc0cecc4b6` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/launch_cmd.txt` | 683 | `49fc605caa9f79fe` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/server_tail.log` | 14,832 | `eaed764b8f91b61b` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/smoke.json` | 1,523 | `59852da03114c75d` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/smoke_chat.json` | 494 | `82325a79a2e8a28e` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/smoke_texts.txt` | 531 | `d3bb53d23df77b27` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/verdict.txt` | 14 | `e9996a2c4e0c4699` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/c64/.cpu_mon` | 8 | `5dcb4f0f035857c5` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/c64/.gpu_mon` | 8 | `7f11d4ea7ab65974` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/c64/bench.log` | 6,224 | `bb61d9946ed6e43c` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/c64/bench_summary.txt` | 257 | `2b6f3e1e977c874e` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/c64/bench_t0.txt` | 11 | `d8b3264c23916b34` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/c64/bench_t1.txt` | 11 | `b56231d2681651e7` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/c64/cpu_util.txt` | 698 | `2b825cfb33b900cd` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/c64/gpu_util.csv` | 2,724 | `c69613b29ea3de59` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/c96/.cpu_mon` | 8 | `9af558b28fd4bb81` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/c96/.gpu_mon` | 8 | `0d1e5308c19b059b` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/c96/bench.log` | 6,335 | `2751a85f0baea935` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/c96/bench_summary.txt` | 260 | `da28e19b19619c74` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/c96/bench_t0.txt` | 11 | `e22f70dfeb016ed9` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/c96/bench_t1.txt` | 11 | `8c478a9d36ed9001` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/c96/cpu_util.txt` | 1,776 | `1538152e074d437e` |
| `eval/results/20260915_150557_ide069_sweep_cells_round3/t15_hot96_def4_kv56k/c96/gpu_util.csv` | 6,898 | `d5b16c1b696e37ba` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/RUN.log` | 2,201 | `7d80f4cbcd8b2a65` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/free_after_boot.txt` | 207 | `4c766788e4fa800e` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/free_before.txt` | 207 | `d12bd60bce173405` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/hbm_after_boot.csv` | 104 | `294ac79b7b729194` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/verdict.txt` | 8 | `aaee20256f45275d` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/A/launch_cmd.txt` | 519 | `118b99bb658f590b` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/A/server_tail.log` | 37,894 | `618366a78dbbda6e` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/A/smoke.json` | 1,580 | `e194019a581a3f79` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/A/smoke_chat.json` | 494 | `54e2f282f3f4d55a` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/A/smoke_texts.txt` | 586 | `77a4405e2bc1508a` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/B/launch_cmd.txt` | 519 | `be821f121f43db3e` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/B/server_tail.log` | 9,152 | `9da0c4b6dcf683e4` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/B/smoke.json` | 1,580 | `6f75c8f88ff78aca` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/B/smoke_chat.json` | 494 | `9fbf1084a6a342be` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/B/smoke_texts.txt` | 586 | `77a4405e2bc1508a` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/c16/.cpu_mon` | 8 | `4ba00a5c40a7590e` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/c16/.gpu_mon` | 8 | `151923f6a8ff2db8` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/c16/cpu_util.txt` | 3,802 | `00f947390fba2e00` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/c16/gpu_util.csv` | 16,477 | `9b3d343e3d2230fb` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/c16/A/bench.log` | 6,221 | `9e5cd988be50d295` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/c16/A/bench_summary.txt` | 239 | `b22e6ac0bc758782` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/c16/B/bench.log` | 6,221 | `c0539e613c46dbbf` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/c16/B/bench_summary.txt` | 239 | `f020a4e75b6658f7` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/c32/.cpu_mon` | 6 | `cade87d2f3688432` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/c32/.gpu_mon` | 6 | `91635c9a614fdbd4` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/c32/cpu_util.txt` | 5,583 | `d7f83159e25a931c` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/c32/gpu_util.csv` | 24,265 | `8dd7bf35d362471c` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/c32/A/bench.log` | 6,353 | `e5bb5da34eeda59c` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/c32/A/bench_summary.txt` | 239 | `48821ba988f1518b` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/c32/B/bench.log` | 6,227 | `74dd4d3829a39bd1` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_cold_expert0/c32/B/bench_summary.txt` | 240 | `e60c8a15e3b8d927` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/single_s5_hot96_def4/launch_cmd.txt` | 683 | `e61dd62e72a16d00` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/single_s5_hot96_def4/smoke.json` | 1,503 | `8741c82f7601685e` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/single_s5_hot96_def4/smoke_chat.json` | 494 | `2fcece1e506f43dd` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/single_s5_hot96_def4/smoke_texts.txt` | 512 | `cae6991885f4aa00` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/single_s5_hot96_def4/verdict.txt` | 5 | `93ff7811a209e2a8` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/single_s5_hot96_def4/c32/.cpu_mon` | 8 | `89cf2a5c263850d3` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/single_s5_hot96_def4/c32/.gpu_mon` | 8 | `3aee12152444db65` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/single_s5_hot96_def4/c32/bench.log` | 6,222 | `782a09993035b8f8` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/single_s5_hot96_def4/c32/bench_summary.txt` | 237 | `bab92907dd7dc9cb` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/single_s5_hot96_def4/c32/cpu_util.txt` | 654 | `10dad75d0a7dad2d` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/single_s5_hot96_def4/c32/gpu_util.csv` | 2,722 | `d81c0ca31159be07` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/free_after_boot.txt` | 207 | `90e1d83e27063e4e` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/free_before.txt` | 207 | `ed7bd0d6d8784f96` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/hbm_after_boot.csv` | 104 | `173e6731eaed1825` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/verdict.txt` | 8 | `aaee20256f45275d` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/A/launch_cmd.txt` | 683 | `4c74de28db128652` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/A/server_tail.log` | 14,103 | `43e469ae68880806` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/A/smoke.json` | 1,503 | `7e23616b3eba8330` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/A/smoke_chat.json` | 494 | `ffc0a8b73125ddb6` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/A/smoke_texts.txt` | 512 | `cae6991885f4aa00` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/B/launch_cmd.txt` | 683 | `32a295afda8391e7` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/B/server_tail.log` | 32,626 | `74f46542e9f2bce9` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/B/smoke.json` | 1,523 | `d499a97b41e023a2` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/B/smoke_chat.json` | 494 | `c4d7b60b9a74771f` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/B/smoke_texts.txt` | 531 | `d3bb53d23df77b27` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/c16/.cpu_mon` | 7 | `83997068954a2d22` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/c16/.gpu_mon` | 7 | `0ff36021f2d68c0f` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/c16/cpu_util.txt` | 479 | `a8ad1acf8f7881e2` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/c16/gpu_util.csv` | 2,335 | `4b39a3d6c755160d` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/c16/A/bench.log` | 6,218 | `17f8a6c3c8bd5d5f` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/c16/A/bench_summary.txt` | 236 | `e10eb08c3a6832d5` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/c16/B/bench.log` | 6,218 | `fdd3879890f6a0b8` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/c16/B/bench_summary.txt` | 236 | `ad04d500f1aeb0d4` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/c32/.cpu_mon` | 7 | `f4086095da28e47a` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/c32/.gpu_mon` | 7 | `40781dad5514c545` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/c32/cpu_util.txt` | 655 | `21d620bf7796e9a9` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/c32/gpu_util.csv` | 3,051 | `e674142f93a1e50f` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/c32/A/bench.log` | 6,222 | `6a29fd96b6714b3d` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/c32/A/bench_summary.txt` | 237 | `ab621b477eaa3faf` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/c32/B/bench.log` | 6,222 | `4994ce7c35e8c48e` |
| `eval/results/20260915_152805_ide069_dual_vs_tp8/dual_hot96_def4/c32/B/bench_summary.txt` | 237 | `dc44fc84cd773a07` |
| `eval/results/20260915_155341_ide069_router_dual/RUN.log` | 1,023 | `362efc97acf52900` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/hbm_after_boot.csv` | 104 | `294ac79b7b729194` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/verdict.txt` | 8 | `aaee20256f45275d` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/A/launch_cmd.txt` | 519 | `118b99bb658f590b` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/A/server_tail.log` | 6,927 | `892933efa7ae4220` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/B/launch_cmd.txt` | 519 | `be821f121f43db3e` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/B/server_tail.log` | 6,918 | `2d2bee7c856dfd9c` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/router_round_robin/router_tail.log` | 12,030 | `2ba07af3f6d8aada` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/router_round_robin/smoke.json` | 1,580 | `3774dae45d4dc2d2` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/router_round_robin/smoke_chat.json` | 494 | `e0be14194ae0d8c8` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/router_round_robin/smoke_texts.txt` | 586 | `77a4405e2bc1508a` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/router_round_robin/c64/.cpu_mon` | 7 | `7ba38f48dda062fb` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/router_round_robin/c64/.gpu_mon` | 7 | `55df12f34026b1e2` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/router_round_robin/c64/bench.log` | 8,091 | `04118b5415b49f43` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/router_round_robin/c64/bench_summary.txt` | 240 | `3a3db855f2671687` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/router_round_robin/c64/cpu_util.txt` | 6,286 | `42ad37349b21c801` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/router_round_robin/c64/gpu_util.csv` | 27,313 | `c54238372791e6b6` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/router_round_robin/c32/.cpu_mon` | 7 | `14d9202dcf9ed90d` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/router_round_robin/c32/.gpu_mon` | 7 | `023e64acbeb0086b` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/router_round_robin/c32/bench.log` | 7,637 | `a6dcf1addd112a36` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/router_round_robin/c32/bench_summary.txt` | 239 | `f3755a5d32c1ff62` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/router_round_robin/c32/cpu_util.txt` | 4,020 | `7090af95a6a593d1` |
| `eval/results/20260915_155341_ide069_router_dual/router_cold_expert0/router_round_robin/c32/gpu_util.csv` | 17,432 | `59c9f038410360b5` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/RUN.log` | 1,497 | `ea8361b2cf8d6d04` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/single_s5_hot96_def4/launch_cmd.txt` | 683 | `e61dd62e72a16d00` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/single_s5_hot96_def4/smoke.json` | 1,503 | `7dba63d36f962a17` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/single_s5_hot96_def4/smoke_chat.json` | 494 | `5c0cffc7c94e0711` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/single_s5_hot96_def4/smoke_texts.txt` | 512 | `cae6991885f4aa00` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/single_s5_hot96_def4/verdict.txt` | 5 | `93ff7811a209e2a8` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/single_s5_hot96_def4/c32/.cpu_mon` | 7 | `43b16039cbab9b54` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/single_s5_hot96_def4/c32/.gpu_mon` | 7 | `b357aa88bd3fe884` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/single_s5_hot96_def4/c32/bench.log` | 6,222 | `5bf10e34f3ef37cc` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/single_s5_hot96_def4/c32/bench_summary.txt` | 237 | `0ecb0b4e046fa6cb` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/single_s5_hot96_def4/c32/cpu_util.txt` | 654 | `e9e156f5021dfd0c` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/single_s5_hot96_def4/c32/gpu_util.csv` | 2,717 | `1c8532696b0f9144` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/free_after_boot.txt` | 207 | `d2502cc79a197289` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/free_before.txt` | 207 | `d5a8b734f9e08095` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/hbm_after_boot.csv` | 104 | `173e6731eaed1825` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/verdict.txt` | 8 | `aaee20256f45275d` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/A/launch_cmd.txt` | 683 | `4c74de28db128652` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/A/server_tail.log` | 8,719 | `4d1c65ab3b732308` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/A/smoke.json` | 1,503 | `e205b514906c866c` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/A/smoke_chat.json` | 494 | `ad9dd57100ba8755` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/A/smoke_texts.txt` | 512 | `cae6991885f4aa00` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/B/launch_cmd.txt` | 683 | `32a295afda8391e7` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/B/server_tail.log` | 8,783 | `496330358b925e34` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/B/smoke.json` | 1,523 | `57c119a3ff0a392d` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/B/smoke_chat.json` | 494 | `4a819e1d7a96df23` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/B/smoke_texts.txt` | 531 | `d3bb53d23df77b27` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/c16/.cpu_mon` | 7 | `ad70ae8baf12801b` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/c16/.gpu_mon` | 7 | `b4b60613451646dd` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/c16/cpu_util.txt` | 479 | `e1c1b5a4e1eb31cd` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/c16/gpu_util.csv` | 2,103 | `a589f8a33076bf82` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/c16/A/bench.log` | 6,218 | `0d0feae71a151490` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/c16/A/bench_summary.txt` | 236 | `24db5a564e9c824d` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/c16/B/bench.log` | 6,218 | `76fe40da36dacd00` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/c16/B/bench_summary.txt` | 236 | `f9f9438f93e85a6d` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/c32/.cpu_mon` | 7 | `e57653e44dc4e15e` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/c32/.gpu_mon` | 7 | `f0414835702d7b88` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/c32/cpu_util.txt` | 655 | `873e5370f1772789` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/c32/gpu_util.csv` | 3,042 | `d425589c01b7936d` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/c32/A/bench.log` | 6,222 | `027662235f598fe2` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/c32/A/bench_summary.txt` | 237 | `2455cc490f19b840` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/c32/B/bench.log` | 6,222 | `0bb3befab95501a4` |
| `eval/results/20260915_171650_ide069_dual_vs_tp8_synced/dual_hot96_def4/c32/B/bench_summary.txt` | 237 | `4d8834afe6a12c66` |
| `eval/results/20260915_172444_ide069_router_dual_synced/RUN.log` | 1,953 | `eb2753f0be0205c1` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/hbm_after_boot.csv` | 104 | `173e6731eaed1825` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/verdict.txt` | 8 | `aaee20256f45275d` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/A/launch_cmd.txt` | 683 | `4c74de28db128652` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/A/server_tail.log` | 6,337 | `eaa86ddc1e819d73` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_cache_aware/router_tail.log` | 12,282 | `0faa2a8ece9f8e12` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_cache_aware/smoke.json` | 1,523 | `44be0171f7630a76` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_cache_aware/smoke_chat.json` | 494 | `7be03caa8920e974` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_cache_aware/smoke_texts.txt` | 531 | `d3bb53d23df77b27` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_cache_aware/c64/.cpu_mon` | 7 | `097d2d05425bafc3` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_cache_aware/c64/.gpu_mon` | 7 | `93ad3b2241f79f18` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_cache_aware/c64/bench.log` | 6,802 | `f15942aeb3677ffa` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_cache_aware/c64/bench_summary.txt` | 237 | `5dad4b481f65c179` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_cache_aware/c64/cpu_util.txt` | 676 | `49a6a4eb4b907967` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_cache_aware/c64/gpu_util.csv` | 3,052 | `cbf2fbc31993d8a6` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_cache_aware/c32/.cpu_mon` | 7 | `bc74e5946b602f47` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_cache_aware/c32/.gpu_mon` | 7 | `402358fa34e106f3` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_cache_aware/c32/bench.log` | 7,238 | `c97e0bb2acabc7d8` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_cache_aware/c32/bench_summary.txt` | 236 | `9f970d8f935de434` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_cache_aware/c32/cpu_util.txt` | 501 | `75a1b011488c4526` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_cache_aware/c32/gpu_util.csv` | 2,345 | `b9d0b586a630fa13` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/B/launch_cmd.txt` | 683 | `32a295afda8391e7` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/B/server_tail.log` | 6,600 | `b03f9ea5efb3045b` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_round_robin/router_tail.log` | 12,340 | `3e4f212f387fe7dc` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_round_robin/smoke.json` | 1,503 | `f1a5a1fdc12076fb` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_round_robin/smoke_chat.json` | 494 | `c08849350fe53134` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_round_robin/smoke_texts.txt` | 512 | `cae6991885f4aa00` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_round_robin/c64/.cpu_mon` | 7 | `109ca059cde48166` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_round_robin/c64/.gpu_mon` | 7 | `e661d9a5b8f876cb` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_round_robin/c64/bench.log` | 7,324 | `8eff87cce634ffe6` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_round_robin/c64/bench_summary.txt` | 238 | `1ac10f847582eebf` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_round_robin/c64/cpu_util.txt` | 698 | `47442703afa9aad5` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_round_robin/c64/gpu_util.csv` | 3,044 | `d14d2308390cc9ca` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_round_robin/c32/.cpu_mon` | 7 | `734814ef7f8d5fc3` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_round_robin/c32/.gpu_mon` | 7 | `a5e17d1518e6caae` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_round_robin/c32/bench.log` | 7,114 | `d1bee570fc48e677` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_round_robin/c32/bench_summary.txt` | 236 | `1deacceb2bc20c2a` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_round_robin/c32/cpu_util.txt` | 501 | `caa656cea81c7c99` |
| `eval/results/20260915_172444_ide069_router_dual_synced/router_hot96_def4/router_round_robin/c32/gpu_util.csv` | 2,338 | `00c49c0daaedfd83` |
| `shadow_assists/features/IDE_069/hwsw/containers.txt` | 566 | `9a5b063f7dd9cbb6` |
| `shadow_assists/features/IDE_069/hwsw/df.txt` | 154 | `83d90ea3923b2de8` |
| `shadow_assists/features/IDE_069/hwsw/dmidecode_bios.txt` | 104 | `2a4659d3e097125f` |
| `shadow_assists/features/IDE_069/hwsw/dmidecode_cpu.txt` | 246 | `b33f9c4eb4a29ec0` |
| `shadow_assists/features/IDE_069/hwsw/dmidecode_mem_raw.txt` | 5,312 | `73ac3c88726e210f` |
| `shadow_assists/features/IDE_069/hwsw/dmidecode_mem_summary.txt` | 85 | `e300a09beb1cf9f0` |
| `shadow_assists/features/IDE_069/hwsw/dmidecode_system.txt` | 49 | `d429f6cf8e352749` |
| `shadow_assists/features/IDE_069/hwsw/free.txt` | 207 | `5123b61babf25316` |
| `shadow_assists/features/IDE_069/hwsw/host.txt` | 96 | `730ca56bd482eb73` |
| `shadow_assists/features/IDE_069/hwsw/hotmap_stats.json` | 787 | `81916c65ccf41140` |
| `shadow_assists/features/IDE_069/hwsw/images.txt` | 352 | `60f25eb789e6da54` |
| `shadow_assists/features/IDE_069/hwsw/isa.txt` | 173 | `5d6339364bc00eb1` |
| `shadow_assists/features/IDE_069/hwsw/kt_weights.txt` | 835 | `30c1122b31152721` |
| `shadow_assists/features/IDE_069/hwsw/lsblk.txt` | 690 | `582577e7ef164d3a` |
| `shadow_assists/features/IDE_069/hwsw/lscpu.txt` | 3,476 | `c71a65c21407759f` |
| `shadow_assists/features/IDE_069/hwsw/lspci.txt` | 1,070 | `0b81dae845d959db` |
| `shadow_assists/features/IDE_069/hwsw/mdstat.txt` | 320 | `6e21eb6c4a0a53fd` |
| `shadow_assists/features/IDE_069/hwsw/model_480b_config.json` | 7,709 | `6c1c6ab9193433a2` |
| `shadow_assists/features/IDE_069/hwsw/models.txt` | 453 | `2be8d1f2a701d005` |
| `shadow_assists/features/IDE_069/hwsw/net.txt` | 600 | `4e466be4d3405c61` |
| `shadow_assists/features/IDE_069/hwsw/numactl.txt` | 0 | `e3b0c44298fc1c14` |
| `shadow_assists/features/IDE_069/hwsw/nvidia_gpu0_q.txt` | 377 | `605d810c15040b52` |
| `shadow_assists/features/IDE_069/hwsw/nvidia_gpus.csv` | 1,316 | `38eed975ed304c7d` |
| `shadow_assists/features/IDE_069/hwsw/nvidia_topo.txt` | 2,969 | `0e475cadbd9fe643` |
| `shadow_assists/features/IDE_069/hwsw/runtime.txt` | 208 | `e8285ea219fa7d4d` |
| `shadow_assists/features/IDE_069/hwsw/sgl-kt5_sw.txt` | 134 | `54dc59e283743ce6` |
| `shadow_assists/features/IDE_069/hwsw/sgl-kt_sw.txt` | 1,045 | `61c2f455c213ad3e` |
| `shadow_assists/features/IDE_069/hwsw/turbo.txt` | 22 | `a4ba65808c26c142` |
| `shadow_assists/features/IDE_069/hwsw/vllm-h100_sw.txt` | 48 | `2a46463dbf3412b9` |

합계 593 파일 · 1,499,627 바이트 (결과 디렉터리 기준). 전문은 `RAW_DATA.md`.

