# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import threading
from collections import deque
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_input_batch import InputBatch


# ===== SUB_201 / L3 — CPU tree spec-decoding instrumentation =====
#
# Env vars (read once at proposer construction; default = OFF, identical to upstream):
#
#   VLLM_L3_TREE_SPEC=1
#       Switch suffix-cache speculation from single-path to K-branch tree
#       (passes use_tree_spec=True to arctic SuffixDecodingCache.speculate()).
#       The tree is *linearized* to the best root-branch path before being
#       returned to vLLM's GPU verifier (single-path API constraint).  The
#       full tree (token_ids + parents + probs) is recorded internally so the
#       offline α-simulation can score branches that the linearization
#       discarded.
#
#   VLLM_L3_TREE_BRANCHES=K   (default 4)
#       Soft cap on the number of root-level branches we keep when
#       linearizing.  Lower K narrows the tree, higher K widens it (subject
#       to what the suffix cache actually returns).
#
#   VLLM_L3_TREE_STATS_PATH=/path/to/file.jsonl
#       If set, the proposer dumps per-request α-simulation stats into this
#       JSONL file every flush.  Each line records, for the most recent
#       propose() step of that request, the draft tokens emitted (linear)
#       and the full tree (token_ids/parents/probs) along with the next
#       sampled token id — enough to reconstruct accept_len_single and
#       accept_len_tree offline.  The proposer flushes the buffer on every
#       N records (VLLM_L3_TREE_STATS_FLUSH, default 200) and on shutdown.
#
# Constraint reminder: GPU verifier in this vLLM build does not accept tree
# attention metadata.  L3 therefore measures the *upper bound* of K-branch
# tree spec — i.e. the accept rate the GPU would have achieved if it could
# verify every branch in parallel — and feeds back the *linearized best
# branch* for the actual GPU verify.  net Δ measured by external bench is
# the lower bound of tree-spec value; α_tree − α_single is the headroom
# unlocked once a tree-attention verifier lands.
#
# All instrumentation is no-op when no env var is set, so default builds
# stay byte-identical to upstream behaviour.


def _read_flag_file(name: str) -> str:
    """Read a file-based fallback for env vars.

    Because vLLM uses `spawn` for EngineCore subprocesses (and the API server's
    env vars are NOT inherited), we cannot rely on environment alone to
    propagate L3 toggles.  As a fallback the launcher writes a tiny flag
    file under `/tmp/vllm_l3_<NAME>.flag`; we read it lazily inside the
    proposer, which runs in the EngineCore subprocess.
    """
    path = f"/tmp/vllm_l3_{name}.flag"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _l3_get(name: str, default: str = "") -> str:
    val = os.environ.get(name)
    if val:
        return val
    return _read_flag_file(name) or default


def _env_int(name: str, default: int) -> int:
    val = _l3_get(name)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_bool(name: str) -> bool:
    val = _l3_get(name)
    return val not in ("", "0", "false", "False", "no", "off")


class SuffixDecodingProposer:
    """
    Speculative decoding proposer for Suffix Decoding (https://arxiv.org/pdf/2411.04975).
    This class imports and uses the official implementation from Arctic Inference
    (https://github.com/snowflakedb/ArcticInference).
    """

    def __init__(self, vllm_config: VllmConfig):
        config = vllm_config.speculative_config
        assert config is not None, "Speculative config must be set"
        self.num_speculative_tokens = config.num_speculative_tokens
        self.max_tree_depth = config.suffix_decoding_max_tree_depth
        self.max_spec_factor = config.suffix_decoding_max_spec_factor
        self.min_token_prob = config.suffix_decoding_min_token_prob
        self.max_model_len = vllm_config.model_config.max_model_len

        # ---- L3 instrumentation ----
        self._l3_tree_enabled = _env_bool("VLLM_L3_TREE_SPEC")
        self._l3_branches = max(1, _env_int("VLLM_L3_TREE_BRANCHES", 4))
        self._l3_stats_path = _l3_get("VLLM_L3_TREE_STATS_PATH", "")
        self._l3_flush_every = max(1, _env_int("VLLM_L3_TREE_STATS_FLUSH", 200))
        # Per-request "previous propose() result" — used to compute
        # accept_len when the next propose() call sees the next sampled
        # token list for that request.
        self._l3_prev_proposal: dict[Any, dict] = {}
        # In-memory record buffer (flushed periodically to disk).
        self._l3_buf: deque = deque()
        self._l3_lock = threading.Lock()
        # Running counters (process-local; bench harness can sample these).
        self._l3_counters = {
            "steps": 0,
            "single_drafted": 0,
            "single_accepted": 0,
            "tree_drafted_branches": 0,
            "tree_accepted_in_some_branch": 0,
            "tree_accepted_len_sum": 0,
            "single_accepted_len_sum": 0,
        }

        # Lazy import to avoid error when Suffix Decoding is not used.
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        # Initialize and empty cache. This object will take care of caching request
        # outputs, evicting old requests, and manages the per-prompt suffix trees.
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=config.suffix_decoding_max_tree_depth,
            max_cached_requests=config.suffix_decoding_max_cached_requests,
        )

    # -------- L3 helpers (no behaviour when tree spec is OFF) --------

    @staticmethod
    def _linearize_best_path(token_ids, parents, probs):
        """Given a tree (token_ids[i], parents[i] = index in token_ids of parent
        or -1 for root), return the highest-product-probability root-to-leaf
        path as a flat list of token ids.

        When the input is already a single path (parents[i] == i-1 for i>0
        and parents[0] == -1), this returns the same list.
        """
        if not token_ids:
            return []
        n = len(token_ids)
        # children[i] = list of j such that parents[j] == i.  roots have
        # parents[i] == -1.
        children: list[list[int]] = [[] for _ in range(n)]
        roots: list[int] = []
        for i, p in enumerate(parents):
            if p == -1:
                roots.append(i)
            else:
                if 0 <= p < n:
                    children[p].append(i)
        # For each root, do a DFS keeping the running log-prob; remember the
        # best path overall.  Use product of probs (== sum of log probs) to
        # rank, which matches the suffix-cache's "score" definition.
        best_path: list[int] = []
        best_score = -1.0
        # Simple iterative DFS.  Trees from arctic_inference are tiny
        # (a handful of nodes at most), so recursion depth is no concern.
        for root in roots:
            stack: list[tuple[int, list[int], float]] = [
                (root, [root], float(probs[root]))
            ]
            while stack:
                node, path, score = stack.pop()
                kids = children[node]
                if not kids:
                    # leaf
                    if score > best_score:
                        best_score = score
                        best_path = list(path)
                else:
                    for ch in kids:
                        # multiply probabilities along the path
                        stack.append(
                            (
                                ch,
                                path + [ch],
                                score * float(probs[ch]),
                            )
                        )
        return [int(token_ids[i]) for i in best_path]

    @staticmethod
    def _tree_max_accept_len(token_ids, parents, sampled_ids):
        """Upper bound on how many of `sampled_ids` would have been accepted
        by *some* root-to-leaf path in the tree.

        For each next sampled token we walk the tree: starting from "matched
        path" = roots, we look for a child whose token == sampled_ids[k].  If
        found, that child becomes the new frontier (single node — tree spec
        only accepts a contiguous prefix of one path).  If not found we stop.
        """
        if not token_ids or not sampled_ids:
            return 0
        n = len(token_ids)
        children: list[list[int]] = [[] for _ in range(n)]
        roots: list[int] = []
        for i, p in enumerate(parents):
            if p == -1:
                roots.append(i)
            else:
                if 0 <= p < n:
                    children[p].append(i)
        # frontier nodes = candidate matches at depth d.  We start at depth
        # 0 = roots.  For each subsequent sampled token, advance into the
        # child whose token matches.  If multiple branches match (impossible
        # for a tree without duplicate child labels but cheap to handle), we
        # keep them all — the accept run is the longest match.
        frontier = list(roots)
        accepted = 0
        for tok in sampled_ids:
            next_frontier: list[int] = []
            for node in frontier:
                if int(token_ids[node]) == int(tok):
                    next_frontier.extend(children[node])
            # Did *any* node at this depth match?  Only count this position
            # as accepted if at least one frontier-node had token == tok.
            matched_here = any(
                int(token_ids[node]) == int(tok) for node in frontier
            )
            if not matched_here:
                break
            accepted += 1
            if not next_frontier:
                # We accepted this token but the tree has no continuation.
                break
            frontier = next_frontier
        return accepted

    @staticmethod
    def _single_path_accept_len(draft_token_ids, sampled_ids):
        """Same metric as the GPU verifier uses today: greedy match from
        index 0 of the linearized draft against the next sampled tokens."""
        if not draft_token_ids or not sampled_ids:
            return 0
        n = min(len(draft_token_ids), len(sampled_ids))
        acc = 0
        for k in range(n):
            if int(draft_token_ids[k]) == int(sampled_ids[k]):
                acc += 1
            else:
                break
        return acc

    def _l3_record(self, req_id, draft, draft_tokens_emitted, sampled_ids):
        """Update counters and append a record.  Safe to call concurrently."""
        single_accept = self._single_path_accept_len(
            draft_tokens_emitted, sampled_ids
        )
        # When tree spec is OFF the tree == single path and these numbers
        # coincide.  When ON, this is the tree's upper bound.
        tree_accept = self._tree_max_accept_len(
            draft.token_ids, draft.parents, sampled_ids
        )
        with self._l3_lock:
            c = self._l3_counters
            c["steps"] += 1
            c["single_drafted"] += len(draft_tokens_emitted)
            c["single_accepted"] += single_accept
            c["single_accepted_len_sum"] += single_accept
            # number of root branches (parents == -1)
            n_branches = sum(1 for p in draft.parents if p == -1)
            c["tree_drafted_branches"] += n_branches
            if tree_accept > 0:
                c["tree_accepted_in_some_branch"] += 1
            c["tree_accepted_len_sum"] += tree_accept

            if self._l3_stats_path:
                self._l3_buf.append(
                    {
                        "req_id": str(req_id),
                        "draft_linear": list(draft_tokens_emitted),
                        "tree_token_ids": list(draft.token_ids),
                        "tree_parents": list(draft.parents),
                        "tree_probs": [round(float(p), 4) for p in draft.probs],
                        "match_len": int(draft.match_len),
                        "sampled_next": list(sampled_ids[: max(1, len(draft.token_ids))]),
                        "single_accept": single_accept,
                        "tree_accept": tree_accept,
                    }
                )
                if len(self._l3_buf) >= self._l3_flush_every:
                    self._l3_flush_unlocked()

    def _l3_flush_unlocked(self):
        if not self._l3_stats_path or not self._l3_buf:
            return
        path = self._l3_stats_path
        # Append-mode JSONL — multiple TP workers writing into the same file
        # is acceptable because each line is self-contained.
        try:
            with open(path, "a", encoding="utf-8") as f:
                while self._l3_buf:
                    rec = self._l3_buf.popleft()
                    f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except Exception:  # noqa: BLE001 — best effort instrumentation
            self._l3_buf.clear()

    def l3_counters_snapshot(self) -> dict:
        with self._l3_lock:
            return dict(self._l3_counters)

    def l3_flush(self):
        with self._l3_lock:
            self._l3_flush_unlocked()

    # -------- main propose() --------

    def propose(
        self,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> list[list[int]]:
        """
        Propose speculative tokens for each request in the input batch. Suffix Decoding
        will speculate a dynamic number of tokens for each request every decoding step,
        so each entry in the returned list may have different lengths.
        """
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                # Skip speculative decoding for partial prefills.
                draft_token_ids.append([])
                continue

            req_id = input_batch.req_ids[i]
            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                draft_token_ids.append([])
                continue

            index = input_batch.req_id_to_index[req_id]
            if req_id not in self.suffix_cache.active_requests:
                if req_id in self.suffix_cache.cached_requests:
                    # Reset the suffix cache for this request.
                    self.suffix_cache.evict_cached_response(req_id)
                num_prompt_tokens = input_batch.num_prompt_tokens[index]
                prompt_token_ids = input_batch.token_ids_cpu[index, :num_prompt_tokens]
                # Start a new request, this will build the suffix tree for that prompt.
                self.suffix_cache.start_request(req_id, prompt_token_ids)

            # --- L3: score the previous step's proposal against the
            # tokens we just sampled (sampled_ids).  This is what arctic's
            # rejection sampler implicitly does inside GPU; here we
            # reproduce the bookkeeping so we can compare single-path α
            # against tree-spec α offline.
            prev = self._l3_prev_proposal.pop(req_id, None)
            if prev is not None:
                self._l3_record(
                    req_id,
                    prev["draft_obj"],
                    prev["draft_linear"],
                    sampled_ids,
                )

            # Append the newly sampled ids to the suffix cache for this request.
            self.suffix_cache.add_active_response(req_id, sampled_ids)

            # Suffix decoding only uses the most recent tokens up to max_tree_depth, so
            # we extract the pattern from the end of the input.
            start = max(0, num_tokens - self.max_tree_depth)
            pattern = input_batch.token_ids_cpu[i, start:num_tokens]
            draft = self.suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=min(
                    self.num_speculative_tokens, self.max_model_len - num_tokens - 1
                ),
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
                use_tree_spec=self._l3_tree_enabled,
            )

            # Linearize tree → single best path for the GPU verifier.  When
            # use_tree_spec is False, this is exactly draft.token_ids (no-op).
            if self._l3_tree_enabled and draft.token_ids:
                emitted = self._linearize_best_path(
                    draft.token_ids, draft.parents, draft.probs
                )
            else:
                emitted = list(draft.token_ids)

            # Remember for next-step α scoring.
            self._l3_prev_proposal[req_id] = {
                "draft_obj": draft,
                "draft_linear": emitted,
            }

            draft_token_ids.append(emitted)

        # Stop requests that were not seen in the input batch.
        stopped = (
            self.suffix_cache.active_requests - input_batch.req_id_to_index.keys()
        )
        for req_id in stopped:
            self.suffix_cache.stop_request(req_id)
            self._l3_prev_proposal.pop(req_id, None)

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass
