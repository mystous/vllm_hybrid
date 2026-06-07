"""Offline α analysis for L3 tree-spec stats.

Input  : VLLM_L3_TREE_STATS_PATH JSONL dumped by the patched
         SuffixDecodingProposer (one record per propose() step *that had a
         next-step result*).
Output : printed table — for each input file —
            steps           # number of propose-and-scored steps
            single_drafted  # ∑ len(draft_linear)
            single_accepted # ∑ accept_len(draft_linear, sampled_next)
            tree_branches   # ∑ number of root branches in the tree
            tree_drafted    # ∑ total nodes in the tree
            tree_accepted   # ∑ tree's longest-path match against sampled_next
            α_single        # single_accepted / single_drafted
            α_tree          # tree_accepted   / tree_drafted   (upper bound)
            α_tree_per_path # tree_accepted / steps  (apples-to-apples vs α_single*K
                            where K = num_speculative_tokens — accept *rate per step*)

Usage:
    python analyze_stats.py [path1.jsonl path2.jsonl ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def analyze(path: Path) -> dict:
    steps = 0
    single_drafted = 0
    single_accepted = 0
    tree_branches = 0
    tree_drafted = 0
    tree_accepted = 0
    sampled_lens = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            steps += 1
            single_drafted += len(rec.get("draft_linear", []))
            single_accepted += int(rec.get("single_accept", 0))
            tids = rec.get("tree_token_ids", [])
            par = rec.get("tree_parents", [])
            tree_drafted += len(tids)
            tree_branches += sum(1 for p in par if p == -1)
            tree_accepted += int(rec.get("tree_accept", 0))
            sampled_lens += len(rec.get("sampled_next", []))
    out = {
        "file": str(path),
        "steps": steps,
        "single_drafted": single_drafted,
        "single_accepted": single_accepted,
        "alpha_single": (single_accepted / single_drafted) if single_drafted else None,
        "tree_drafted": tree_drafted,
        "tree_branches_total": tree_branches,
        "tree_accepted": tree_accepted,
        "alpha_tree": (tree_accepted / tree_drafted) if tree_drafted else None,
        "single_accept_per_step": (single_accepted / steps) if steps else None,
        "tree_accept_per_step": (tree_accepted / steps) if steps else None,
        "mean_branches_per_step": (tree_branches / steps) if steps else None,
        "mean_tree_nodes_per_step": (tree_drafted / steps) if steps else None,
    }
    return out


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: analyze_stats.py path1.jsonl [path2.jsonl ...]", file=sys.stderr)
        sys.exit(2)
    for p in paths:
        r = analyze(Path(p))
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
