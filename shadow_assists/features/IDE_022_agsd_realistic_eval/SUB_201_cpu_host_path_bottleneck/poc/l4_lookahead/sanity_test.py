"""SUB_201/L4 sanity — global ngram dict 동작 검증 (단위 수준).

env:
  VLLM_NGRAM_GLOBAL_DICT=1
시나리오:
  1) 첫 request: prompt = [1,2,3,4,5,6,7,8,9,10] + 1 sampled. dict insert 검증
  2) 두 번째 request: prompt 끝이 [..., 4,5] → suffix [4,5] 로 global hit 검증.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_NGRAM_GLOBAL_DICT", "1")
os.environ.setdefault("VLLM_NGRAM_GLOBAL_DICT_MAX", "1000")
# fake config + run propose() directly with synthetic arrays.

sys.path.insert(0, "/workspace/host_vllm_hybrid")

import numpy as np

# We construct a minimal fake VllmConfig (only the fields NgramProposer uses).
class _Spec:
    prompt_lookup_min = 2
    prompt_lookup_max = 4
    num_speculative_tokens = 3
class _Model:
    max_model_len = 128
class _Sched:
    max_num_seqs = 8
class _Par:
    tensor_parallel_size = 1
class _Cfg:
    speculative_config = _Spec()
    model_config = _Model()
    scheduler_config = _Sched()
    parallel_config = _Par()


from vllm.v1.spec_decode.ngram_proposer import NgramProposer

prop = NgramProposer(_Cfg())
print(f"global_dict_enabled={prop.global_dict_enabled}, max_n={prop.max_n}, min_n={prop.min_n}, k={prop.k}")

# ===== Step 1: simulate request idx=0 with tokens with unique pattern =====
# Use distinct token values so each ngram has a single mapping.
max_seqs = 8
max_len = 128
token_ids_cpu = np.zeros((max_seqs, max_len), dtype=np.int32)
num_tokens_no_spec = np.zeros(max_seqs, dtype=np.int32)
seq_a = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # all unique
token_ids_cpu[0, :len(seq_a)] = seq_a
num_tokens_no_spec[0] = len(seq_a)
sampled = [[1]] + [[]] * (max_seqs - 1)
draft1 = prop.propose(sampled, num_tokens_no_spec, token_ids_cpu)
print(f"draft1[0] = {draft1[0]}  (no local match — pattern is monotonic)")
print(f"  dict_size = {len(prop.global_ngram_dict)} ingests = {prop.global_dict_ingests}")

# ===== Step 2: new request idx=1 — tokens end with [15,16] →  global dict should return (17,18,19) =====
token_ids_cpu[1, :2] = [15, 16]
num_tokens_no_spec[1] = 2
sampled = [[]] + [[1]] + [[]] * (max_seqs - 2)
draft2 = prop.propose(sampled, num_tokens_no_spec, token_ids_cpu)
print(f"draft2[1] = {draft2[1]}  (expect [17,18,19] from global dict)")
print(f"  global_dict_hits={prop.global_dict_hits}, lookups={prop.global_dict_lookups}, local_hits={prop.global_dict_local_hits}")

assert prop.global_dict_hits >= 1, "expected at least one global dict hit"
assert draft2[1] == [17, 18, 19], f"expected [17,18,19], got {draft2[1]}"
print("PASS — global ngram dict working")
