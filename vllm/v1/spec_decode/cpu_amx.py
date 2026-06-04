# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CpuAmxProposer — PoC wire-up of CPU/AMX-side draft proposer.

This is the PoC (A1 from SUB_201 §5) **wire-up layer**: it mirrors the
SuffixDecodingProposer interface so gpu_model_runner.py can dispatch on
`speculative_config.method == "cpu_amx_draft"` without any further changes
to the verify-side code path.

Current status: **toy proposer** — produces a deterministic fixed-id
sequence per request so that the vllm engine can be booted end-to-end and
the dispatch wire-up exercised. The real AMX kernel call (Qwen 0.5B BF16
forward backed by `libamx_draft_qwen05b.so` from SUB_187) is NOT yet
plumbed here; see SUB_198 scaffold (`cpu_amx_proposer_scaffold.py`) for
the 2-3 week dev list to make this a real token producer.

Why a toy: the AMX kernel today is a latency microbench (synthetic
LM-head matmul). Wiring it as the actual token producer requires a real
Qwen 0.5B forward (24 layers attn+MLP), CPU-side KV cache, sampler, and
tokenizer bridge — all out of scope for the 3-4h PoC. The PoC validates
**only** that the spec-decode dispatch in gpu_model_runner.py accepts a
new method string and routes propose() calls correctly.
"""
from __future__ import annotations

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_input_batch import InputBatch


class CpuAmxProposer:
    """CPU/AMX-side draft proposer (PoC toy).

    Interface-compatible with SuffixDecodingProposer:
      - __init__(vllm_config)
      - propose(input_batch, sampled_token_ids, slot_mappings=None)
          -> list[list[int]]
      - load_model(*args, **kwargs)
    """

    def __init__(self, vllm_config: VllmConfig):
        config = vllm_config.speculative_config
        assert config is not None, "Speculative config must be set"
        self.num_speculative_tokens: int = config.num_speculative_tokens
        self.max_model_len: int = vllm_config.model_config.max_model_len
        # Reserve placeholder for SUB_198 follow-on integration:
        #   self._lib = ctypes.CDLL(LIB_PATH); self._lib.amx_draft_qwen05b_init()
        # Held intentionally None in the PoC.
        self._amx_lib = None

    def propose(
        self,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> list[list[int]]:
        """Toy proposer: emit a deterministic K-length draft per request.

        For each request that has at least one newly sampled token, emit
        `[last + 1, last + 2, ..., last + K]` clamped to a safe id range.
        Requests without a sampled token (e.g. partial prefills) get [].
        """
        K = self.num_speculative_tokens
        # Clamp guard: keep proposed ids strictly positive and within a
        # conservative range so the verify-side sampler never sees a
        # negative id. We do not have direct access to vocab_size here;
        # 32_000 is a safe lower bound across the models we benchmark.
        SAFE_MAX_ID = 32_000

        drafts: list[list[int]] = []
        for sampled_ids in sampled_token_ids:
            if not sampled_ids:
                drafts.append([])
                continue
            last = int(sampled_ids[-1])
            draft = [((last + i) % SAFE_MAX_ID) + 1 for i in range(1, K + 1)]
            drafts.append(draft)
        return drafts

    def load_model(self, *args, **kwargs):
        # No model to load in the PoC. The real implementation would
        # mmap Qwen 0.5B BF16 weights and pre-tile them for AMX.
        return None


__all__ = ["CpuAmxProposer"]
