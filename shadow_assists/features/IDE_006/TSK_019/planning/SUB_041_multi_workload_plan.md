# SUB_041 — Multi-workload 서버 throughput 측정 plan

> **parent**: TSK_019 / CLAUDE.md `# Objective` 직접 정합
> **출처**: 사용자 명시 (turn N) — "프로젝트 목표 재확인" 후 (c) Path 선택 — util metric 통합 + multi-workload 병행.
> **선행**: SUB_040 (CPU/GPU util baseline) 완료 후 진입.

---

## 1. 본 프로젝트 목표 (CLAUDE.md `# Objective` 원문)

> - vLLM을 수정하여 **CPU의 활용률을 극도로 끌어 올려** GPU가 아닌 **GPU가 포함된 서버 또는 Cluster 전체의 성능을 향상** 시킨다.
> - 특히 **CPU의 활용률이 Idle 또는 낮은 Utilization을 허락하지 않는다.**

→ **단일 inference job throughput 이 아닌, 서버 전체의 CPU+GPU 합산 throughput**.

## 2. 측정 시나리오 (4-way)

| 시나리오 | inference workload | concurrent CPU workload | 측정 metric |
|---|---|---|---|
| **t1 vanilla solo** | vanilla LLM inference 500p | 없음 | inf_tps, CPU util, GPU util |
| **t2 NEO solo** | NEO LLM inference 500p | 없음 | inf_tps, CPU util, GPU util |
| **t3 vanilla + CPU BG** | vanilla LLM inference 500p | CPU BG task (embedding 또는 stress) | inf_tps, BG_tps, CPU util, GPU util, **합산 throughput** |
| **t4 NEO + CPU BG** | NEO LLM inference 500p | 동일 CPU BG task | inf_tps, BG_tps, CPU util, GPU util, **합산 throughput** |

**비교 영역**:
- t1 vs t2: 단일 job 비교 (SUB_036 와 동일, util 추가)
- **t3 vs t4: 서버 전체 (CPU+GPU 합산) 비교 — 본 목표의 직접 검증**
- t1 vs t3 / t2 vs t4: BG 영향 (interference)

## 3. CPU BG task 후보

### 3.1 옵션 A — Sentence embedding (실용적, 합리적)

```bash
# Hugging Face sentence-transformers 의 CPU-only 영역 inference
python -m sentence_transformers.SentenceTransformer \
    --model all-MiniLM-L6-v2 --device cpu \
    --input-file /tmp/embeddings_input.txt \
    --output-file /tmp/embeddings_output.json
```

- 측정: input lines / second
- 장점: real-world workload, NEO 와 CPU 자원 경합 패턴 정확
- 단점: 사전 install 필요

### 3.2 옵션 B — `stress-ng --cpu N --cpu-load PCT` (가장 통제됨)

```bash
stress-ng --cpu 80 --cpu-load 100 --timeout 1800s --metrics-brief
```

- 측정: bogo ops / second
- 장점: control 정확, install 쉬움
- 단점: throughput metric 이 synthetic

### 3.3 권장 — 옵션 B (stress-ng) 먼저, 그 다음 옵션 A (실용 검증)

이유: SUB_041 의 핵심은 "CPU 자원이 활용되는가" 확인 → synthetic 으로 충분. 옵션 A 는 SUB_042 (별도 follow-up) 로.

## 4. launcher 설계

```bash
# /tmp/run_sub041_multi_workload.sh
for SCENARIO in t1_vanilla_solo t2_neo_solo t3_vanilla_bg t4_neo_bg; do
  # 1) start util sampler (CPU + GPU)
  /tmp/util_sampler.sh "${RUN_DIR}/util" 2700 &
  SAMPLER_PID=$!

  # 2) for t3/t4: start CPU BG (stress-ng) on NUMA node 1 (NEO 가 NUMA 0 사용)
  if [[ "$SCENARIO" == *"bg"* ]]; then
    numactl --cpunodebind=1 --membind=1 stress-ng \
        --cpu 56 --cpu-load 100 --timeout 2700s \
        --metrics-brief > "${RUN_DIR}/stress_ng.out" 2>&1 &
    BG_PID=$!
  fi

  # 3) run inference (vanilla or NEO)
  ${PY} run_neo_baseline.py ... > "${RUN_DIR}/engine.log.stdout" 2>&1

  # 4) stop BG (if any)
  [[ -n "${BG_PID:-}" ]] && kill -TERM ${BG_PID}
  wait ${SAMPLER_PID}

  # 5) parse: inf_tps, BG bogo_ops/s, CPU util avg, GPU util avg
done
```

## 5. metric 합산 계산

```python
# t3 (vanilla + BG):
#   - inf_throughput = vanilla tps (= GPU 활용)
#   - bg_throughput  = stress-ng bogo_ops/s (= CPU 활용)
#   - server_perf    = normalize 후 합산 — 예: 정규화 0-1 scale

# t4 (NEO + BG):
#   - inf_throughput = NEO tps
#   - bg_throughput  = stress-ng bogo_ops/s (NEO 의 CPU 점유 후 남은 자원 사용)
#   - server_perf    = 정규화 합산

# 비교:
#   - vanilla + BG: vanilla 가 GPU full, BG 가 CPU full → 둘 다 max
#   - NEO + BG: NEO 가 CPU 일부 점유 + GPU 일부 활용 → BG 가 CPU 적게, NEO 가 GPU 적게 → 합산 ↓
#   - 본 목표 "CPU 활용률 극대화" 가 win 인지 → 본 비교가 결정
```

## 6. 가설

| 가설 | 예상 결과 |
|---|---|
| H1: vanilla + BG 가 NEO + BG 보다 합산 throughput 높음 | NEO 의 raison d'être 무효 — 본 목표 미달 |
| H2: NEO + BG 가 vanilla + BG 보다 합산 throughput 높음 | NEO 의 목표 달성 — CPU 자원 활용으로 서버 throughput ↑ |
| H3: 두 시나리오 비슷함 (±5%) | NEO 의 CPU 활용이 BG workload 와 trade-off — 별도 lever (NEO 우선순위 lower, BG 우선순위 higher 등) 필요 |

→ **결과에 따라 NEO 의 가치 영역 정의 변경**.

## 7. effort

- launcher 작성: 30 min (SUB_040 launcher 재사용 + BG 추가)
- 사전 install stress-ng: 5 min (apt 또는 conda)
- 측정: t1 ~15min + t2 ~40min + t3 ~15min + t4 ~40min = ~110min wall
- 분석 + RESULTS doc: 30 min
- **총 ~3 hr**

## 8. SUB_040 결과 의존성

| SUB_040 결과 | SUB_041 진입 결정 |
|---|---|
| NEO CPU util 가 vanilla 보다 높음 (예: NEO 40% vs vanilla 0%) | NEO 의 CPU 활용 입증 → SUB_041 진입 (BG workload 가 CPU 자원 어떻게 분담받는지) |
| NEO CPU util 가 낮음 (예: NEO 5% vs vanilla 0%) | NEO 의 CPU 활용 가설 깨짐 → SUB_041 보다 NEO CPU util 향상 lever 우선 |
| NEO GPU util 가 vanilla 보다 낮음 (예: NEO 50% vs vanilla 90%) | NEO 가 GPU idle 시킴 — CLAUDE.md "CPU idle 허락 안 함" 의 mirror 문제 |
