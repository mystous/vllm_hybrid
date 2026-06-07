# L7 — Model-Type Oracle Router (CPU 활용, multi-model serving)

## 위치
- parent: `SUB_201` / `IDE_022`
- sibling levers: `a1_vllm_integration`, `a2_e2e`, `b1_e2e`, `b2_*`, `b3_*`, `moe_offload*`
- 본 lever (L7): `poc/l7_oracle_router/`

## 한 줄
TSK_042 에서 측정된 **(model_family × workload_type × method) oracle table 215 cells** 를
**runtime 에 CPU 가 dict-get 으로 O(1) lookup** 하여 multi-model vLLM 클러스터에서
요청을 최적 instance(=method) 로 라우팅한다.

## TSK_044 와의 차이 (중요)
| 축 | TSK_044 (기각) | **L7** |
|---|---|---|
| 분류 단위 | per-request (prompt content) | **model-level (model_family + workload_type)** |
| 입력 | prompt 본문 (regex C0~C3 / ONNX) | **model_name (헤더), workload_type 헤더 or coarse-grained 추정** |
| CPU overhead | classify latency 측정 필요 | **lookup 124 ns / dispatch 1.0 μs** (negligible) |
| 결정 | 같은 prompt 가 다른 method 로 갈 수 있음 | **같은 (모델, corpus) 는 항상 동일 method** (oracle 결정론) |
| 데이터 출처 | classifier 학습/규칙 | TSK_042 measured cells (변경 없음) |
| 시간이 지나면? | distribution drift 위험 | TSK_042 재측정 시 oracle table 만 hot-swap |

## 산출물
- `oracle_table.py` — CSV → `Oracle` 객체 (`.lookup()`, `.to_dataframe()`, `.to_json()`)
- `router.py` — `OracleRouter` (in-proc) + lookup micro-bench CLI
- `bench.py` — cluster TPS simulation (default / static_best / oracle × uniform / realistic)
- `bench_summary.csv`, `bench_full.json` — 시뮬레이션 결과
- `oracle_table.csv` — 70 cells lookup table
- `extra_stats.json` — winner 분포, family 별 uplift 분포
- `MEASUREMENTS.md` — 측정·시뮬레이션 결과 + 본 task 결론

## 실행
```
cd .../poc/l7_oracle_router
export LD_LIBRARY_PATH=/workspace/vllm_dev_prj/lib/python3.12/site-packages/torch/lib
/workspace/vllm_dev_prj/bin/python oracle_table.py    # 70 row lookup table
/workspace/vllm_dev_prj/bin/python router.py --n 5000000   # dispatch micro-bench
/workspace/vllm_dev_prj/bin/python bench.py           # cluster TPS simulation
```

## 결론 (요약)
- oracle routed cluster vs default(vanilla 단일 cluster): **net +84.3% (realistic mix)** / +70.3% (uniform mix)
- vs static-best 단일 method (suffix-only cluster): **+3.8%** — 즉 단순한 suffix-only 도 큰 폭 개선
  이지만 oracle 이 추가로 R1-671B (suffix 0.45×) 같은 outlier 를 vanilla 로 보냄으로써 추가 3-4% 회수.
- CPU 활용: dispatch 당 **1.0 μs** → 단일 코어 0.97 M QPS. routing 으로 인한 host overhead 사실상 0.
- 자세한 표·해석은 `MEASUREMENTS.md` 참조.
