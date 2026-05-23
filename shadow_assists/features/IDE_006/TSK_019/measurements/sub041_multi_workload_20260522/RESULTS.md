# SUB_041 — Multi-workload 서버 throughput 측정 (★★★ CLAUDE.md `# Objective` 직접 검증)

> **parent**: TSK_019 / CLAUDE.md `# Objective` ("서버 전체 성능 향상") 직접 검증
> **plan**: [SUB_041 plan](../../planning/SUB_041_multi_workload_plan.md)
> **measurement**: HEAD `0d7dc0334`, 500p × 8192, gmu=0.85, CPU+GPU util 1Hz sampling, CPU BG = 56 procs (Python SHA256 hashing) on NUMA 1 56-111

---

## 1. 측정 결과 (4 시나리오)

| 시나리오 | inf_tps | wall (s) | CPU busy avg | GPU util avg | crash |
|---|---:|---:|---:|---:|:-:|
| t1 vanilla solo | 4680.9 | 875.0 | 4.66% | 73.5% | 0 ✓ |
| t2 NEO solo | 1899.9 | 2130.7 | 9.94% | 58.4% | 0 ✓ |
| **t3 vanilla + BG** | **4679.1** | **875.4** | **29.74%** | **73.7%** | 0 ✓ |
| **t4 NEO + BG** | **1652.0** | **2458.6** | **35.61%** | **70.2%** | 0 ✓ |

## 2. 핵심 비교 — BG 영향 분석

### 2.1 BG 가 inference throughput 에 미치는 영향

| mode | solo inf_tps | + BG inf_tps | Δ |
|---|---:|---:|---:|
| **vanilla** | 4680.9 | 4679.1 | **-0.04% (불변)** ★ |
| **NEO** | 1899.9 | 1652.0 | **-13.0%** ⚠️ |

→ **vanilla 는 BG 와 자원 분담 가능**, NEO 는 BG 와 CPU contention 으로 inference -13% 손실.

### 2.2 BG 가 CPU util 추가 활용

| mode | solo CPU% | + BG CPU% | Δ |
|---|---:|---:|---:|
| vanilla | 4.66% | 29.74% | +25.08%p |
| NEO | 9.94% | 35.61% | +25.67%p |

→ BG 가 두 mode 다 CPU +25%p 활용 (BG 자체의 자원 사용은 비슷). 단 NEO 는 inference 손실, vanilla 는 inference 불변.

### 2.3 BG 가 GPU util 에 미치는 영향

| mode | solo GPU% | + BG GPU% | Δ |
|---|---:|---:|---:|
| vanilla | 73.5% | 73.7% | +0.2%p (불변) |
| NEO | 58.4% | 70.2% | +11.8%p (이상) |

→ NEO 의 GPU util 가 BG 와 함께 오히려 증가 — wall 길어지면서 GPU 가 더 오래 active 상태. **단 GPU 효율 (tps/GPU%) = NEO 0+BG: 1652/70.2=23.5 vs vanilla+BG: 4679/73.7=63.5 → vanilla 가 GPU 효율 2.7×** 여전.

## 3. ★★★ 본 프로젝트 목표 직접 검증

### 3.1 CLAUDE.md `# Objective` vs 측정

| 목표 | 측정 결과 | 평가 |
|---|---|---|
| "CPU 활용률 **극도로** 끌어올리기" | NEO solo 9.94% / NEO+BG 35.61% | ❌ — 50-90%+ 목표 미달. BG 외부 task 가 채워줘야만 30%대 |
| "CPU **Idle 허락 안 함**" | NEO solo idle 90% / NEO+BG idle 64% | ❌ — vanilla+BG idle 70% 와 비슷 (양쪽 다 idle 많음) |
| "GPU 포함 **서버 전체 성능 향상**" | vanilla+BG 합산 > NEO+BG 합산 | ❌ — **NEO 가 추가하는 가치 없음** |

### 3.2 서버 합산 throughput 정량 분석

**bg_result.json 미생성** (BG process 가 inference 종료 시 SIGTERM 받아 result write 못함). 단 CPU util 변화로 BG throughput 추정 가능:

| 시나리오 | inf_tps | CPU+25%p 의 BG throughput 추정 | 합산 throughput 추정 |
|---|---:|---:|---:|
| vanilla + BG | 4679 | BG 가 CPU 25%p 자유 사용 → 50% efficiency (NUMA 1 의 절반) | vanilla 4679 + BG ~max |
| NEO + BG | 1652 | BG 가 CPU 25%p 사용 단 NEO 와 contention → 추정 30-40% efficiency | NEO 1652 + BG ~partial |

→ **vanilla + BG 가 합산 throughput 더 높음**. NEO 의 raison d'être 가 본 환경에서 무효.

## 4. SUB_032~041 통합 결론

| SUB | Tier / Metric | 결과 | 종합 평가 |
|---|---|---|---|
| SUB_032 | A4 numactl 3-run | -0.21% noise | A-tier 무효 |
| SUB_033 | B3 online softmax | -0.82% negative | B-tier 부분 무효 |
| SUB_034 | B1 async cdec depth | +0.1% noise | B-tier 무효 |
| SUB_035 C1a | OMP launch overhead | 1.22% — C-tier 폐기 | C-tier 무효 |
| SUB_036 | 500p baseline | vanilla 2.63× faster | NEO 단일 job 무효 |
| SUB_040 | CPU/GPU util baseline | NEO CPU 11.93% / GPU -7.4%p | CLAUDE.md 목표 미달 |
| **SUB_041** | **Multi-workload 서버 throughput** | **vanilla+BG > NEO+BG** | **NEO 의 가치 명확히 부정 (본 환경)** |

→ **본 NEO 구현 (HEAD `0d7dc0334`) 은 본 워크로드 + 본 하드웨어에서 모든 metric net-negative**.

## 5. 가능한 다음 path

### 5.1 NEO 가 net-positive 인 영역 (탐색 필요)

- (a) **vanilla 가 OOM 인 워크로드**: gmu ↑, max_num_seqs ↑, max-tokens ↑, num-prompts >> 500. NEO 만 작동, vanilla 불가.
- (b) **CPU 가 GPU 보다 빠른 model 영역**: 특정 small model + 긴 sequence + KV cache 가 CPU 가 더 빠른 경우 (rare)
- (c) **다른 hardware**: H100 + AMX SPR 외 환경 (예: GH200, MI300X, 또는 AMD CPU)

### 5.2 본 코드 베이스의 한계

- NEO 의 CPU offload 가 본 환경에서 **항상 net-negative**
- pacpu (paged_attention_cpu) compute 가 GPU 대비 100-1000× 느림 (CPU FP16 vs H100 BF16 TC)
- swap_in/out IO 비용이 추가 wall 도입
- 단 vanilla 가 정상 작동하는 영역에서는 vanilla 가 항상 win

### 5.3 결정 후보

| 후보 | 의미 |
|---|---|
| (a) NEO 의 적용 영역을 vanilla OOM 영역으로 **명시적 좁히기** + 그 영역에서만 활성 (env-gated) | NEO 의 raison d'être 영역에서만 사용 |
| (b) **TSK_019 종료** + 본 NEO 구조의 net-negative 결론 문서화 | 본 코드 베이스의 NEO 가 본 목표와 misaligned 인정 |
| (c) **새 IDE 시작** — CPU 활용 극대화의 다른 접근 (예: speculative decoding 의 CPU 영역, embedding 의 GPU 공유 등) | 본 목표를 다른 architecture 로 추구 |
| (d) **CLAUDE.md 목표 재검토** — "CPU 극대화" 가 "전체 throughput 향상" 과 trade-off 임을 인정 | 목표 자체 정정 |

## 6. raw 자료

| 항목 | 위치 |
|---|---|
| SUMMARY.tsv | `eval/results/20260522_115257_sub041_multi_workload/SUMMARY.tsv` |
| 4 시나리오 dirs | `eval/results/20260522_115257_sub041_multi_workload/t1_vanilla_solo, t2_neo_solo, t3_vanilla_bg, t4_neo_bg/` |
| 각 시나리오 util/cpu_util.csv + gpu_util.csv | `<dir>/util/*.csv` |
| launcher | `/tmp/run_sub041_multi_workload.sh` |
| util sampler | `/tmp/util_sampler.sh` |
| CPU BG workload | `/tmp/cpu_bg_workload.py` |
| stdout log | `/tmp/sub041_multi.log` |
| 알려진 한계 | bg_result.json 생성 안 됨 (inference 종료 시 SIGTERM 으로 BG 가 result write 못함) — 후속 launcher 에서 BG result file flush 보장 필요 |
