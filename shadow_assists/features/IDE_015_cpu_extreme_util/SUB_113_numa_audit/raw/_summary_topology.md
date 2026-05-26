# NUMA / GPU 매핑 요약

## 1. Hardware
- CPU: Intel Xeon Platinum 8480+ (Sapphire Rapids), 2 socket × 56 core × 2 HT = 224 logical
- Mem: 2 TB total (1 TB / NUMA node)
- GPU: NVIDIA H100 × 8 (80 GB / GPU)

## 2. NUMA topology
| NUMA | Physical cores | HT siblings | Mem (size / free) |
|---:|---|---|---|
| 0 | 0-55 | 112-167 | 1031 GB / 431 GB |
| 1 | 56-111 | 168-223 | 1032 GB / 97 GB |
- node distances: local 10, remote 21 (2.1× penalty)

## 3. GPU ↔ NUMA affinity (from nvidia-smi topo -m)
| GPU | PCIe BDF | NUMA | CPU Affinity |
|---:|---|---:|---|
| 0 | 0a:00.0 | 0 | 0-55, 112-167 |
| 1 | 18:00.0 | 0 | 0-55, 112-167 |
| 2 | 23:00.0 | 0 | 0-55, 112-167 |
| 3 | 2c:00.0 | 0 | 0-55, 112-167 |
| 4 | 87:00.0 | 1 | 56-111, 168-223 |
| 5 | 90:00.0 | 1 | 56-111, 168-223 |
| 6 | b8:00.0 | 1 | 56-111, 168-223 |
| 7 | c1:00.0 | 1 | 56-111, 168-223 |

## 4. GPU NVLink topology
- All GPU 쌍 NV18 (= 18 NVLink bonded) — full mesh
- NIC: GPU 0-3 ↔ NIC 0-5 (NUMA 0), GPU 4-7 ↔ NIC 6-10 (NUMA 1)

## 5. canonical AGSD (Qwen 32B TP=4×2) 배치
| backend | GPU | NUMA |
|---|---|---:|
| vanilla (port 8001) | 0-3 | 0 |
| trident (port 8002) | 4-7 | 1 |

## 6. SUB_112 CPU fill placement (CPU 80-111)
- 위치: NUMA 1 의 physical core 후반부 32개
- HT 시블링 (192-223) 미사용 ✓
- 이전 RESULTS.md 의 "cross-NUMA isolation" 표현 부정확 — **실제로는 trident backend (NUMA 1) 와 동일 NUMA**
- 실제 작동 원리: physical-core pinning + HT 회피 + vllm process 의 CPU 코어 침범 방지

## 7. OS-level isolation 현황 (IDE_020 target)
| 항목 | 현재 | 목표 |
|---|---|---|
| isolcpus | ❌ 미설정 (cmdline 에 없음) | `isolcpus=80-111` |
| hugepages | ❌ 0 페이지 (THP 일부만) | 2 GB × 128 페이지 사전 예약 |
| cgroup cpuset | ❌ 미사용 | `cpuset.cpus=80-111` for CPU fill workers |
| IRQ affinity | ❌ default | smp_affinity NUMA 0 으로 격리 |
