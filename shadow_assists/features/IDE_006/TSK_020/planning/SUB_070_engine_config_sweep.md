# SUB_070 — engine config sweep on top of SUB_047 best

> **parent**: TSK_020 / root lever 재정의 (자체 비판 후 — 2026-05-24)
> **status**: 활성 (2026-05-24)
> **effort**: 30-50분 (6 cells × ~7분/cell)
> **bottleneck (측정값 직접 backed)**: GPU SM 54.7% — **idle 45.3%**

## 왜 이것이 진짜 lever 인가

bottleneck-driven SUB_065~069 모두 기각 (5/5). 자체 비판 결과:
- 5 lever 모두 **ngram lookup 자체 안에서만** search.
- 그러나 측정값: ngram time ~1-2 ms / step time 70-90 ms = **1-2% only**.
- 결과: best case 도 +1-2%, 실제로는 step overhead 가 더 커서 회귀.

진짜 idle 영역: **GPU SM 45.3%**. step time 의 95% 를 차지하는 GPU forward 가 saturate 되어 있지 않음. 원인 추정:
- `max_num_seqs=256` 이 concurrency 상한 → 더 많은 in-flight request 영역 SM 영역 채움
- `gpu_memory_utilization=0.85` → KV pool 영역 제약 → max concurrent batch ↑ 영역 제한
- `max_num_batched_tokens=8192` → prefill chunk 영역 제약

## sweep 영역 (SUB_047 best config 위에서 — cap=8, div_tp=0, spec=7, lookup=5/2)

| cell | gmu | max_num_seqs | max_num_batched_tokens | 가설 |
|---|---:|---:|---:|---|
| baseline | 0.85 | 256 | 8,192 | SUB_047 best 재현 (~10,956 tps) |
| A1 (gmu+) | **0.90** | 256 | 8,192 | KV pool ↑ → in-flight ↑ |
| A2 (gmu++) | **0.92** | 256 | 8,192 | KV pool ↑↑ |
| B1 (seqs+) | 0.85 | **384** | 8,192 | concurrency ↑ |
| B2 (seqs++) | 0.85 | **512** | 8,192 | concurrency ↑↑ |
| C1 (bt+) | 0.85 | 256 | **16,384** | prefill chunk ↑ |

기대: best cell 이 +5-15% throughput 가능. OOM 가능성 (gmu++ 또는 seqs++) 검증 필요 — crash check 포함.

## 측정 방법

- 1-run @ 500p × 8192 (effect 확정 시 3-run 재측정)
- baseline + 5 lever cells, util sampler 동시 가동
- crash 발생 시 해당 cell 기각

## 후속

- best cell → 3-run canonical 재측정 → SUB_047 best 교체 또는 alternative best로 등록
- 2-axis 결합 sweep (예: gmu++ × seqs+) — 효과 큰 axis 발견 시 진행
