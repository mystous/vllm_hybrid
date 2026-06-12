# SPDX-License-Identifier: Apache-2.0
# SUB_213 U1/U2 — uniform draft padding CPU 단위 테스트 (GPU 불필요)
#
# 실행:
#   cd /workspace/host_vllm_hybrid
#   PYTHONPATH=. /workspace/vllm_dev_prj/bin/python \
#       shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_213_fap_suffix_uniform/test_pad_uniform.py
#
# VllmConfig 전체 구성 없이 object.__new__ + mock 으로 propose() 경로만 검증.
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, "/workspace/host_vllm_hybrid")

PAD_FLAG = "/tmp/vllm_l3_VLLM_SUFFIX_PAD_UNIFORM.flag"

from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer  # noqa: E402


class MockDraft:
    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.parents = list(range(-1, len(token_ids) - 1))
        self.probs = [0.9] * len(token_ids)


class MockSuffixCache:
    """speculate() 가 요청별로 지정된 길이의 draft 를 반환."""

    def __init__(self, drafts_by_req):
        self.drafts_by_req = drafts_by_req
        self.active_requests = set()
        self.cached_requests = set()

    def start_request(self, req_id, prompt_token_ids):
        self.active_requests.add(req_id)

    def add_active_response(self, req_id, sampled_ids):
        pass

    def speculate(self, req_id, pattern, **kwargs):
        return MockDraft(list(self.drafts_by_req[req_id]))

    def stop_request(self, req_id):
        self.active_requests.discard(req_id)

    def evict_cached_response(self, req_id):
        pass


def make_proposer(pad_uniform, K=8, max_model_len=16384, drafts_by_req=None):
    p = object.__new__(SuffixDecodingProposer)
    p.num_speculative_tokens = K
    p.max_tree_depth = 64
    p.max_spec_factor = 1.0
    p.min_token_prob = 0.1
    p.max_model_len = max_model_len
    p._pad_uniform = pad_uniform
    p._l3_tree_enabled = False
    p._l3_branches = 4
    p._l3_stats_path = ""
    p._l3_flush_every = 200
    p._l3_prev_proposal = {}
    p.suffix_cache = MockSuffixCache(drafts_by_req or {})
    return p


def make_batch(req_specs):
    """req_specs: list of (req_id, num_tokens)."""
    n = len(req_specs)
    req_ids = [r for r, _ in req_specs]
    return SimpleNamespace(
        req_ids=req_ids,
        num_tokens_no_spec=[t for _, t in req_specs],
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        num_prompt_tokens=[4] * n,
        token_ids_cpu=np.arange(n * 32, dtype=np.int64).reshape(n, 32) % 1000,
    )


def run():
    K = 8
    failures = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name} {detail}")
        if not cond:
            failures.append(name)

    # ---------- U1a: pad ON — 짧은 draft(3) 가 정확히 K 로 패딩 ----------
    p = make_proposer(True, K=K, drafts_by_req={"a": [11, 12, 13]})
    out = p.propose(make_batch([("a", 100)]), [[7]])
    check("U1a pad short→K", len(out[0]) == K, f"len={len(out[0])} {out[0]}")
    check("U1a prefix preserved", out[0][:3] == [11, 12, 13], f"{out[0][:3]}")
    check("U1a pad_tok == last sampled", all(t == 7 for t in out[0][3:]), f"{out[0][3:]}")

    # ---------- U1b: pad ON — 빈 draft(0) 도 K 로 패딩 ----------
    p = make_proposer(True, K=K, drafts_by_req={"b": []})
    out = p.propose(make_batch([("b", 100)]), [[42]])
    check("U1b pad empty→K", out[0] == [42] * K, f"{out[0]}")

    # ---------- U1c: pad ON — 긴 draft(12) 는 K 로 truncate ----------
    p = make_proposer(True, K=K, drafts_by_req={"c": list(range(200, 212))})
    out = p.propose(make_batch([("c", 100)]), [[1]])
    check("U1c truncate long→K", out[0] == list(range(200, 208)), f"{out[0]}")

    # ---------- U1d: max_model_len 경계 — target = mml - num_tokens - 1 ----------
    p = make_proposer(True, K=K, max_model_len=105, drafts_by_req={"d": [5, 6]})
    out = p.propose(make_batch([("d", 100)]), [[9]])
    # target = min(8, 105-100-1) = 4
    check("U1d boundary target=4", len(out[0]) == 4, f"len={len(out[0])} {out[0]}")

    # ---------- U1e: num_tokens >= mml → [] 유지 (uniform 비대상) ----------
    p = make_proposer(True, K=K, max_model_len=100, drafts_by_req={"e": [5]})
    out = p.propose(make_batch([("e", 100)]), [[9]])
    check("U1e at-mml stays empty", out[0] == [], f"{out[0]}")

    # ---------- U1f: partial prefill (빈 sampled) → [] 유지 ----------
    p = make_proposer(True, K=K, drafts_by_req={"f": [5]})
    out = p.propose(make_batch([("f", 100)]), [[]])
    check("U1f partial-prefill stays empty", out[0] == [], f"{out[0]}")

    # ---------- U2a: pad OFF — upstream 동일 (no-op) ----------
    p = make_proposer(False, K=K, drafts_by_req={"g": [11, 12, 13]})
    out = p.propose(make_batch([("g", 100)]), [[7]])
    check("U2a OFF passthrough", out[0] == [11, 12, 13], f"{out[0]}")

    # ---------- U2b: env OFF + flag file ON → _env_bool True ----------
    from vllm.v1.spec_decode.suffix_decoding import _env_bool

    os.environ.pop("VLLM_SUFFIX_PAD_UNIFORM", None)
    had_flag = os.path.exists(PAD_FLAG)
    try:
        check("U2b-pre env/flag both off", not _env_bool("VLLM_SUFFIX_PAD_UNIFORM"))
        with open(PAD_FLAG, "w") as f:
            f.write("1")
        check("U2b flag-file fallback ON", _env_bool("VLLM_SUFFIX_PAD_UNIFORM"))
    finally:
        if not had_flag:
            os.unlink(PAD_FLAG)

    # ---------- U1g: 멀티 요청 batch — 전원 동일 길이 (uniform) ----------
    p = make_proposer(
        True, K=K,
        drafts_by_req={"r0": [], "r1": [1, 2], "r2": list(range(20))},
    )
    out = p.propose(
        make_batch([("r0", 50), ("r1", 60), ("r2", 70)]),
        [[3], [4], [5]],
    )
    check("U1g all uniform len==K", all(len(o) == K for o in out),
          f"lens={[len(o) for o in out]}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} — {failures}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    run()
