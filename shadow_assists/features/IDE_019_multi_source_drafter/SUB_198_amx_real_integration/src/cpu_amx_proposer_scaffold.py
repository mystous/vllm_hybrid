"""SUB_198 — CpuAmxProposer scaffold (real integration spec, not yet wired).

This is the *interface scaffold* for a real CpuAmxProposer class that would
replace SuffixDecodingProposer when `VLLM_USE_AMX_DRAFT=1` is set AND a real
Qwen 0.5B AMX-compiled draft model is available.

Current status: **scaffold-only**. The methods raise NotImplementedError because
the AMX kernel (libamx_draft_qwen05b.so) is a latency microbench, not a real
token-producer (see amx_proposer_proxy.py docstring).

To make this real, the following pieces must be added (estimated 2-3 weeks):
  1. AMX-accelerated Qwen 0.5B forward (24 layers × attention + MLP) — separate
     C++ module beyond the SUB_187 microbench. ~3000 lines.
  2. CPU-side KV cache aligned to AMX tile shape.
  3. Sampler (greedy + temperature) over AMX-produced logits.
  4. Tokenizer bridge for context tokens.
  5. Patch in gpu_model_runner.py:
       elif self.speculative_config.method == "cpu_amx_draft":
           self.drafter = CpuAmxProposer(self.vllm_config)
     And in the propose dispatch (~line 5220):
       elif isinstance(self.drafter, CpuAmxProposer):
           draft_token_ids = self.drafter.propose(
               input_batch, sampled_token_ids, slot_mappings
           )
  6. CLI flag: --speculative-config '{"method":"cpu_amx_draft","num_speculative_tokens":7}'
  7. ENV: VLLM_USE_AMX_DRAFT=1 to enable; falls back to ngram on missing kernel.

For SUB_198 measurement, the proxy in amx_proposer_proxy.py is fired
concurrently with the *existing* SuffixDecodingProposer to demonstrate that
the AMX kernel can co-reside without throughput regression.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch  # noqa: F401 — interface compatibility

# This import path mirrors what would be added to vllm/v1/spec_decode/cpu_amx.py.
# Currently lives outside vllm tree to avoid touching core.

LIB_PATH = Path(
    "/workspace/vllm_hybrid/shadow_assists/features/IDE_019_multi_source_drafter"
    "/SUB_187_amx_draft_head/build/libamx_draft_qwen05b.so"
)


class CpuAmxProposer:
    """Real CpuAmxProposer scaffold — NotImplementedError on token production.

    Interface mirrors SuffixDecodingProposer so it would drop into the
    gpu_model_runner.py drafter dispatch without further changes.
    """

    def __init__(self, vllm_config):  # type: ignore[no-untyped-def]
        config = vllm_config.speculative_config
        assert config is not None
        self.num_speculative_tokens = config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enabled = os.environ.get("VLLM_USE_AMX_DRAFT", "0") == "1"
        self._lib = None
        if self.enabled and LIB_PATH.exists():
            try:
                self._lib = ctypes.CDLL(str(LIB_PATH))
                self._lib.amx_draft_qwen05b_init.restype = ctypes.c_int
                self._lib.amx_draft_qwen05b_step_ms.restype = ctypes.c_double
                self._lib.amx_draft_qwen05b_step_ms.argtypes = [
                    ctypes.c_int, ctypes.c_int,
                ]
                rc = self._lib.amx_draft_qwen05b_init()
                if rc != 0:
                    self._lib = None
            except OSError:
                self._lib = None

    def propose(
        self,
        input_batch,  # type: ignore[no-untyped-def]
        sampled_token_ids,  # type: ignore[no-untyped-def]
        slot_mappings=None,
    ):
        raise NotImplementedError(
            "CpuAmxProposer.propose requires real Qwen 0.5B AMX forward — "
            "current libamx_draft_qwen05b.so is a latency microbench only. "
            "See SUB_198 RESULTS.md §3 for the required follow-on work."
        )

    def load_model(self, *args, **kwargs):
        # AMX kernel has its own internal weight init (synthetic, deterministic).
        # A real implementation would load Qwen 0.5B BF16 weights here.
        return None


__all__ = ["CpuAmxProposer"]
