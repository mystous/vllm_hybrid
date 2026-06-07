# L3 — CPU tree spec-decoding (K-branch verify)

본 디렉토리는 SUB_201 후속 lever **L3** 의 작업 자산을 담는다.

## 가설

현재 vLLM `suffix` spec-decode 는 한 step 당 **단일 path × K 토큰**만 GPU 로 보낸다.
같은 prompt context 가 여러 candidate continuation 을 갖는 경우 단일 path 는
"어느 한 branch 를 추측해서 한 번 잡으면 K 까지, 못 잡으면 0" 의 binary 결과를 가져,
실측된 평균 accept rate α 가 낮아지는 원인이 된다 (TSK_042 α=0.46 — single-path
upper bound).

K-branch tree spec 으로 확장하면:

1. CPU 가 K branch (각 depth ~K') tree 를 동시에 유지,
2. GPU 가 tree-attention 으로 batch verify,
3. **best matching branch** 가 accept 되어 α 가 본질적으로 ↑.

본 PoC 의 임무: **K-branch tree 의 accept rate 잠재 ↑ 폭을 실측**.

## 구현 요약

- patch: `vllm/v1/spec_decode/suffix_decoding.py` (in-tree, env-gated, no-op
  when `VLLM_L3_TREE_SPEC` 미설정).
- env vars:
  - `VLLM_L3_TREE_SPEC=1` — arctic `SuffixDecodingCache.speculate(use_tree_spec=True)`.
  - `VLLM_L3_TREE_BRANCHES=K` — soft cap on root branches (default 4).
  - `VLLM_L3_TREE_STATS_PATH=/path.jsonl` — per-step tree dump for offline α.
- linearization: tree → best root-to-leaf path (단일 path API 호환), 잠재 α 는
  `_tree_max_accept_len()` 로 동시 측정.

## 제약: 본 vLLM 빌드는 tree-attention verifier 가 아직 없다

GPU `RejectionSampler` 는 1-D `draft_token_ids` 만 받아 linear 검증을 한다.
따라서 patch 는 (a) **best path linearize → GPU verify (실측 wall-clock)**,
(b) **tree 잠재 α offline (upper bound)** 두 metric 을 동시에 산출한다.
(b) 가 (a) 대비 의미 있게 높으면, tree-attention verifier 가 들어왔을 때의
회수 가능 throughput uplift 의 ceiling 을 보여준다.

## 진행 / 산출

- `MEASUREMENTS.md` — 실측·시뮬 표 (final).
- `launcher.sh` — 단일-GPU(3) Qwen2.5-32B serve.
- `run_l3_bench.sh` — sharegpt 100p × conc=8 × max-tok=256 throughput runner.
- `analyze_stats.py` — patched stats jsonl → α 표 변환.
- `out/{baseline,tree}.json` — throughput_runner summary.
- `stats/{baseline,tree_bN}.jsonl` — per-step tree dump.
