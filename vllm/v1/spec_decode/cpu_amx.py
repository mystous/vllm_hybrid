# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CpuAmxProposer — CPU-side draft proposer.

Two operating modes, gated by env vars:

  * **toy mode** (default): produces deterministic ids
    `[last+1, last+2, ..., last+K]` — preserves dispatch wire-up so the
    engine boots end-to-end and `gpu_model_runner.py` can route
    `method == "cpu_amx_draft"` without a real draft model.

  * **real PyTorch CPU mode** (env `VLLM_USE_AMX_DRAFT=1`): lazy-loads
    `Qwen/Qwen2.5-0.5B-Instruct` on CPU (BF16) and runs K greedy
    autoregressive forward steps per request to produce real draft
    tokens. Uses PyTorch's CPU GEMM (oneDNN — auto-AMX on Sapphire
    Rapids, AVX-512 BF16 on Alder Lake/dev box).

The AMX custom kernel (`libamx_draft_qwen05b.so` from SUB_187) is **not**
called from this path. That integration requires a real Qwen 0.5B
forward built on top of the AMX kernel — see ARCHITECTURE_MAP.md §3
(SUB_198 dir) for the 4 sub-task estimate.

Env vars:
  * `VLLM_USE_AMX_DRAFT=1`    enable real (non-toy) draft (else toy)
  * `VLLM_CPU_DRAFT_USE_AMX=1`  inside real path, prefer the AMX
        kernel (libamx_draft_qwen05b.so via cpu_amx_kernel.ctypes
        binding). Off (default) → PyTorch CPU forward path. Auto-falls
        back to PyTorch path when kernel is missing or CPU lacks AMX.
        See `SUB_198_amx_real_integration/AMX_INTEGRATION_DESIGN.md`.
  * `VLLM_CPU_DRAFT_KERNEL_PATH=…`  override .so path (default resolved
        to shadow_assists/.../SUB_187/build/libamx_draft_qwen05b.so).
  * `VLLM_CPU_DRAFT_MODEL=…`  override HF model id (default Qwen 0.5B)
  * `VLLM_CPU_DRAFT_THREADS=N`  set `torch.set_num_threads` (default 8)
  * `VLLM_CPU_DRAFT_MAX_CTX=N`  truncate context to last N ids (default 256)
"""
from __future__ import annotations

import os
import sys
import threading

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_input_batch import InputBatch

# ─────────────────────────────────────────────────────────────────────
# Module-level singleton for the heavy real-model handle. Multiple
# CpuAmxProposer instances (per worker / per restart) share a single
# loaded model — load cost (~1-2s) is paid once per process.
# ─────────────────────────────────────────────────────────────────────
_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, tuple[object, object]] = {}
_WARNED: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[CpuAmxProposer] {msg}", file=sys.stderr, flush=True)


def _try_load_real_model(
    model_id: str,
    threads: int,
) -> tuple[object, object] | None:
    """Lazy-load Qwen 0.5B on CPU. Returns (model, tokenizer) or None."""
    cache_key = f"{model_id}::{threads}"
    with _MODEL_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]
        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM, AutoTokenizer)
        except ImportError:
            _warn_once(
                "no_transformers",
                "transformers not installed — falling back to toy mode.",
            )
            return None
        try:
            torch.set_num_threads(max(1, threads))
            tok = AutoTokenizer.from_pretrained(model_id)
            mdl = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            mdl.eval()
        except Exception as e:  # pragma: no cover — depends on env
            _warn_once(
                f"load_fail::{model_id}",
                f"failed to load {model_id} on CPU ({type(e).__name__}: {e})"
                " — falling back to toy mode.",
            )
            return None
        _MODEL_CACHE[cache_key] = (mdl, tok)
        return _MODEL_CACHE[cache_key]


class CpuAmxProposer:
    """CPU-side draft proposer.

    Interface-compatible with SuffixDecodingProposer:
      - __init__(vllm_config)
      - propose(input_batch, sampled_token_ids, slot_mappings=None)
          -> list[list[int]]
      - load_model(*args, **kwargs)
    """

    # Toy-mode clamp: keep proposed ids strictly positive and within a
    # conservative range so the verify-side sampler never sees a
    # negative id. We do not have direct access to vocab_size here;
    # 32_000 is a safe lower bound across the models we benchmark.
    TOY_SAFE_MAX_ID = 32_000

    def __init__(self, vllm_config: VllmConfig):
        config = vllm_config.speculative_config
        assert config is not None, "Speculative config must be set"
        self.num_speculative_tokens: int = config.num_speculative_tokens
        self.max_model_len: int = vllm_config.model_config.max_model_len

        # Mode: real CPU PyTorch forward vs toy.
        self._real_enabled: bool = (
            os.environ.get("VLLM_USE_AMX_DRAFT", "0") == "1"
        )
        self._model_id: str = os.environ.get(
            "VLLM_CPU_DRAFT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
        )
        try:
            self._threads = int(
                os.environ.get("VLLM_CPU_DRAFT_THREADS", "8"))
        except ValueError:
            self._threads = 8
        try:
            self._max_ctx = int(
                os.environ.get("VLLM_CPU_DRAFT_MAX_CTX", "256"))
        except ValueError:
            self._max_ctx = 256

        # Lazy model handle (populated on first propose() in real mode).
        self._model = None
        self._tokenizer = None
        # AMX kernel backend (SUB_198 step — ctypes binding to
        # libamx_draft_qwen05b.so). Gated by VLLM_CPU_DRAFT_USE_AMX=1.
        # Auto-falls back to PyTorch path if kernel unavailable.
        self._amx_enabled: bool = (
            os.environ.get("VLLM_CPU_DRAFT_USE_AMX", "0") == "1"
        )
        self._amx_kernel = None  # lazy AmxDraftKernel handle

    # ─────────────────────────────────────────────────────────────
    # AMX kernel backend (SUB_198 binding)
    # ─────────────────────────────────────────────────────────────
    def _ensure_amx_kernel(self) -> bool:
        """Try to acquire libamx_draft_qwen05b.so via ctypes.

        Returns True iff the kernel is loaded AND the CPU reports AMX
        availability AND init succeeded. On any failure, flips
        `self._amx_enabled = False` so subsequent calls fall through to
        the PyTorch path.
        """
        if self._amx_kernel is not None:
            return True
        try:
            # Local import keeps the import cost off the toy/PyTorch
            # paths and avoids a hard dependency on the .so being
            # present at module import time.
            from vllm.v1.spec_decode.cpu_amx_kernel import (
                AmxDraftKernel,
            )
        except Exception as e:  # pragma: no cover — defensive
            _warn_once(
                "amx_import_fail",
                f"cpu_amx_kernel import failed ({type(e).__name__}: {e})"
                " — falling back to PyTorch CPU path.",
            )
            self._amx_enabled = False
            return False
        try:
            kern = AmxDraftKernel.get()
        except Exception as e:  # pragma: no cover — defensive
            _warn_once(
                "amx_get_fail",
                f"AmxDraftKernel.get() failed ({type(e).__name__}: {e})"
                " — falling back to PyTorch CPU path.",
            )
            self._amx_enabled = False
            return False
        if not kern.is_available():
            _warn_once(
                "amx_unavailable",
                f"AMX kernel unavailable (loaded={kern.loaded}, "
                f"hw_amx={kern.hw_amx}, err={kern.load_error}) — "
                "falling back to PyTorch CPU path. Set "
                "VLLM_CPU_DRAFT_USE_AMX=0 to silence.",
            )
            self._amx_enabled = False
            return False
        rc = kern.ensure_init()
        if rc != 0:
            _warn_once(
                "amx_init_fail",
                f"amx_draft_qwen05b_init returned {rc} — falling back "
                "to PyTorch CPU path.",
            )
            self._amx_enabled = False
            return False
        self._amx_kernel = kern
        return True

    # ─────────────────────────────────────────────────────────────
    # Real PyTorch CPU forward path
    # ─────────────────────────────────────────────────────────────
    def _ensure_real_model(self) -> bool:
        if self._model is not None:
            return True
        loaded = _try_load_real_model(self._model_id, self._threads)
        if loaded is None:
            # Disable real mode permanently for this instance after first
            # failure so we do not retry every call.
            self._real_enabled = False
            return False
        self._model, self._tokenizer = loaded
        return True

    @torch.no_grad()
    def _propose_real_single(self, context_ids: list[int], k: int) -> list[int]:
        """Run k greedy autoregressive forward steps on CPU.

        Returns up to k token ids. No KV cache (uses naive re-forward of
        the growing prefix) — fine for K=7 on a tiny model; trade-off vs
        cache management complexity. Per-step cost on dev box: ~33ms / 8t.
        """
        if not context_ids:
            return []
        # Truncate to last max_ctx ids — Qwen 0.5B handles 32k but we
        # only need the local context for next-token prediction, and a
        # long ctx blows up the no-cache re-forward cost.
        ctx = context_ids[-self._max_ctx:]
        ids = torch.tensor([ctx], dtype=torch.long)
        out: list[int] = []
        for _ in range(k):
            logits = self._model(ids).logits  # type: ignore[union-attr]
            next_id = int(logits[0, -1].argmax())
            out.append(next_id)
            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=torch.long)], dim=1
            )
        return out

    # ─────────────────────────────────────────────────────────────
    # Toy fallback path
    # ─────────────────────────────────────────────────────────────
    def _propose_toy_single(self, sampled_ids: list[int], k: int) -> list[int]:
        if not sampled_ids:
            return []
        last = int(sampled_ids[-1])
        return [((last + i) % self.TOY_SAFE_MAX_ID) + 1 for i in range(1, k + 1)]

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────
    def propose(
        self,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> list[list[int]]:
        """Emit K-length drafts per request.

        Backend selection (in order):
          1) AMX kernel (VLLM_USE_AMX_DRAFT=1 + VLLM_CPU_DRAFT_USE_AMX=1
             + libamx_draft_qwen05b.so loadable + CPU has AMX). Calls
             `step_ms(B, K)` for latency timing; draft *ids* are still
             produced by the PyTorch CPU forward as fallback because the
             current kernel ABI is a microbench (no token-id I/O). When
             SUB_198 §3 (d) lands, this path will return kernel-derived
             ids directly.
          2) PyTorch CPU forward (VLLM_USE_AMX_DRAFT=1, no AMX kernel).
          3) Toy (default, dispatch wire-up only).
        """
        K = self.num_speculative_tokens

        use_real = self._real_enabled and self._ensure_real_model()
        use_amx = (
            self._real_enabled
            and self._amx_enabled
            and self._ensure_amx_kernel()
        )

        # AMX kernel exercise (latency probe only — kernel does not
        # currently accept input ids; SUB_198 §3 (d) will extend the
        # ABI to forward(input_bf16, B, K) → ids).
        if use_amx and self._amx_kernel is not None:
            try:
                # B clamped to [1, KERNEL_B_MAX] = [1, 16]. Non-empty
                # requests only.
                B_eff = max(1, min(16, sum(
                    1 for s in sampled_token_ids if s)))
                ms = self._amx_kernel.step_ms(B_eff, K)
                _warn_once(
                    "amx_step_ok",
                    f"AMX kernel step_ms(B={B_eff}, K={K}) = {ms:.3f} ms "
                    "(latency probe only; ids still from PyTorch path "
                    "until SUB_198 §3 (d) extends ABI).",
                )
            except Exception as e:  # pragma: no cover
                _warn_once(
                    "amx_step_fail",
                    f"AMX step_ms failed ({type(e).__name__}: {e})"
                    " — disabling AMX path; PyTorch fallback active.",
                )
                self._amx_enabled = False

        drafts: list[list[int]] = []
        for sampled_ids in sampled_token_ids:
            if not sampled_ids:
                drafts.append([])
                continue
            if use_real:
                # InputBatch holds the prefix; for the PoC scaffold we
                # use only the most recent `sampled_ids` as context. A
                # full implementation would gather token_ids from
                # input_batch.req_id_to_index etc.
                try:
                    drafts.append(
                        self._propose_real_single(list(sampled_ids), K))
                    continue
                except Exception as e:  # pragma: no cover
                    _warn_once(
                        "real_step_fail",
                        f"real-mode propose failed ({type(e).__name__}: {e})"
                        " — falling back to toy for remainder of batch.",
                    )
                    use_real = False
            drafts.append(self._propose_toy_single(list(sampled_ids), K))
        return drafts

    def load_model(self, *args, **kwargs):
        """Eagerly load the real model + AMX kernel if enabled (else
        no-op).

        vllm calls this once during worker init. Eager-loading here
        avoids paying the ~1-2s PyTorch cost (or ~265 MB AMX kernel
        init alloc cost) on the first propose() call.
        """
        if self._real_enabled:
            self._ensure_real_model()
            if self._amx_enabled:
                # Best-effort eager AMX init. On dev hosts this
                # silently flips _amx_enabled off via warn_once.
                self._ensure_amx_kernel()
        return None


__all__ = ["CpuAmxProposer"]
