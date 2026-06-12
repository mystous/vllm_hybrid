# HWC Round 3 — KV fp8 baseline 추가 stack

## Context

- Round 1 winner: H8 (KV fp8) = +4.02% ± 0.61%
- Round 2 best: R2-5 (fp8 + enable_sp) = +4.25% ± 1.02% (std 안에 fp8 단독과 동등)
- Round 3 baseline: **KV fp8** (lib_common 가 항상 추가)

## R3 candidates (각 5-sweep)

| # | Lever | Stack | Mechanism |
|---|---|---|---|
| R3-1 | fp8 + FULL_DECODE_ONLY cudagraph | — | prefill 은 eager (compile 안함), decode 만 capture → 작은 setup overhead 줄임 |
| R3-2 | fp8 + max-num-batched-tokens=16384 | — | 더 큰 batch — KV cache 활용 ↑ |
| R3-3 | fp8 + --async-scheduling | — | host scheduler 와 GPU 의 overlap |
| R3-4 | fp8 + async + batched=8192 + maxseqs=256 | combo | 모두 결합 |
| R3-A | fp8 + enable_sp + async | sp+async | R2 winner + async |
| R3-B | fp8 + enable_sp + batched=16384 | sp+batched | R2 winner + 큰 batch |
| R3-C | fp8 + enable_sp + FULL_DECODE_ONLY | sp+full_decode | R2 winner + graph mode |
| R3-D | fp8 + enable_sp + maxseqs=256 | sp+maxseqs | R2 winner + seqs |

## Results

_(자동 채워짐)_
