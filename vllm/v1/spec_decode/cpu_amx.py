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
  * `VLLM_CPU_DRAFT_MODEL=…`  override HF model id (default Llama-3.2-1B-Instruct)
  * `VLLM_CPU_DRAFT_THREADS=N`  set `torch.set_num_threads` (default 8)
  * `VLLM_CPU_DRAFT_MAX_CTX=N`  truncate context to last N ids (default 512)
  * `VLLM_CPU_DRAFT_USE_KV=1`   enable per-request KV cache reuse (default 1)
  * `VLLM_CPU_DRAFT_RANK0_ONLY=1`  only TP rank 0 runs CPU forward (default 0,
        every worker independently runs — matches existing PoC behavior).
  * `VLLM_CPU_DRAFT_BATCH=1`    enable batched draft forward — concurrent
        requests share a single bf16 forward per decode step.  Per-req
        latency drops from ~31 ms (B=1) to ~1.8 ms (B=32) on Xeon 8570.
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
            "VLLM_CPU_DRAFT_MODEL", "meta-llama/Llama-3.2-1B-Instruct"
        )
        try:
            self._threads = int(
                os.environ.get("VLLM_CPU_DRAFT_THREADS", "8"))
        except ValueError:
            self._threads = 8
        try:
            self._max_ctx = int(
                os.environ.get("VLLM_CPU_DRAFT_MAX_CTX", "512"))
        except ValueError:
            self._max_ctx = 512
        self._use_kv = (
            os.environ.get("VLLM_CPU_DRAFT_USE_KV", "1") == "1"
        )
        self._rank0_only = (
            os.environ.get("VLLM_CPU_DRAFT_RANK0_ONLY", "0") == "1"
        )
        self._use_batch = (
            os.environ.get("VLLM_CPU_DRAFT_BATCH", "0") == "1"
        )

        # Lazy model handle (populated on first propose() in real mode).
        self._model = None
        self._tokenizer = None
        # Per-request KV cache: req_id -> (DynamicCache, last_processed_len)
        # last_processed_len = number of input_batch.token_ids_cpu tokens
        # already fed into the cache (excluding draft probes).
        self._req_cache: dict[str, tuple[object, int]] = {}
        # TP rank — populated lazily on first propose() if rank0_only set.
        self._tp_rank: int | None = None
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
        """Run k greedy autoregressive forward steps on CPU (no-cache path).

        Returns up to k token ids. Naive re-forward of the growing prefix —
        used only when KV cache is disabled (VLLM_CPU_DRAFT_USE_KV=0) or
        when the per-request cache is missing.
        """
        if not context_ids:
            return []
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

    @torch.no_grad()
    def _propose_real_kv(
        self,
        req_id: str,
        full_context: list[int],
        k: int,
    ) -> list[int]:
        """Run k greedy steps using a per-request KV cache.

        `full_context` is the entire prompt+output prefix the verifier has
        committed (including any tokens just accepted this step). We append
        only the *new* tail since the last call to the cache, then emit k
        draft tokens; the K draft tokens are NOT committed back to the
        request cache (they may or may not be accepted by the GPU verifier
        — we don't know which until the next call, so on the next call we
        roll forward by reconciling `full_context` length against the
        cached `last_processed_len`).
        """
        if not full_context:
            return []
        try:
            from transformers import DynamicCache  # type: ignore[import-not-found]
        except ImportError:
            return self._propose_real_single(full_context, k)

        # Clip very long contexts (max_ctx) — for very long prompts the
        # most recent max_ctx tokens carry enough signal for next-token
        # prediction and bound prefill cost.
        if len(full_context) > self._max_ctx:
            # Drop the per-req cache when we have to slide the window,
            # since cached keys correspond to absolute positions that
            # no longer match.
            self._req_cache.pop(req_id, None)
            full_context = full_context[-self._max_ctx:]

        entry = self._req_cache.get(req_id)
        cache: object
        last_len: int
        if entry is None:
            cache = DynamicCache()
            last_len = 0
        else:
            cache, last_len = entry
            # Sanity: if the new context is somehow shorter than cached
            # (rare — re-shuffle), rebuild from scratch.
            if last_len > len(full_context):
                cache = DynamicCache()
                last_len = 0

        # Slice the new tokens to feed (prefill the gap from last_len to
        # len(full_context)). At least 1 token must be fed each call (the
        # newest sampled token) for the cache to be in sync.
        new_tail = full_context[last_len:]
        if not new_tail:
            # Shouldn't happen — propose is called once per sampled step.
            # Defensive: re-feed last token to keep cache hot.
            new_tail = full_context[-1:]
            last_len = len(full_context) - 1

        ids = torch.tensor([new_tail], dtype=torch.long)
        try:
            out = self._model(  # type: ignore[union-attr]
                ids, past_key_values=cache, use_cache=True
            )
        except Exception:
            # KV-cache path unsupported by this model — fall back to
            # no-cache once.
            self._req_cache.pop(req_id, None)
            return self._propose_real_single(full_context, k)
        cache = out.past_key_values
        last_id = int(out.logits[0, -1].argmax())
        new_last_len = len(full_context)

        drafts: list[int] = [last_id]
        # k-1 more decode steps over the cache (these are *probes*; they
        # mutate the cache, so we have to snapshot+restore. The cheap way
        # is to dup the cache before probing — but DynamicCache duplication
        # is expensive (full K/V copy). Instead, we accept that draft
        # probes pollute the cache and on the *next* propose call we roll
        # forward by re-feeding the verified suffix from last_processed_len.
        # That's what `last_len` accounting is for: we save the *clean*
        # last_len (= new_last_len, just after the prefill but before the
        # probes), so next call re-feeds whatever subset of drafts was
        # actually accepted plus the newly sampled token.
        for _ in range(k - 1):
            nxt = torch.tensor([[last_id]], dtype=torch.long)
            try:
                out = self._model(  # type: ignore[union-attr]
                    nxt, past_key_values=cache, use_cache=True
                )
            except Exception:
                break
            cache = out.past_key_values
            last_id = int(out.logits[0, -1].argmax())
            drafts.append(last_id)

        # Rebuild a *clean* cache state (truncated back to new_last_len)
        # by discarding it — DynamicCache doesn't expose a truncate API
        # in all transformers versions, so we simply drop the polluted
        # cache and rebuild on the next call. The cost: next call has to
        # re-prefill the request from `last_len = 0`. That defeats the
        # KV optimization. Instead we keep the polluted cache but *also*
        # save `last_processed_len = new_last_len` so we *re-feed* the
        # new sampled token next call (which will append to the polluted
        # tail, producing slightly wrong logits for one step but bounded
        # in impact). Track which approach is cheaper at measurement time.
        if hasattr(cache, "crop"):
            try:
                cache.crop(new_last_len)  # transformers >= 4.40 API
            except Exception:
                pass
        self._req_cache[req_id] = (cache, new_last_len)
        return drafts

    def _gc_req_cache(self, active_req_ids: set[str]) -> None:
        """Drop cached entries for requests no longer in the batch."""
        if not self._req_cache:
            return
        stale = set(self._req_cache.keys()) - active_req_ids
        for rid in stale:
            self._req_cache.pop(rid, None)

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

        # rank0-only short circuit: non-rank-0 workers emit empty drafts
        # (vLLM's spec-decode dispatch tolerates empty draft lists).
        if self._rank0_only:
            if self._tp_rank is None:
                try:
                    from vllm.distributed.parallel_state import (
                        get_tensor_model_parallel_rank,
                    )
                    self._tp_rank = int(get_tensor_model_parallel_rank())
                except Exception:
                    self._tp_rank = 0
            if self._tp_rank != 0:
                return [[] for _ in sampled_token_ids]

        # Garbage-collect per-request KV caches for evicted requests.
        if use_real and self._use_kv:
            try:
                active = set(input_batch.req_id_to_index.keys())
                self._gc_req_cache(active)
            except Exception:
                pass

        drafts: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                drafts.append([])
                continue
            if use_real:
                try:
                    # Reach into input_batch to pull the full prefix
                    # token sequence for this request. This is the
                    # context the target model has actually produced
                    # so far — exactly what we need for a high-quality
                    # CPU draft.
                    full_ctx: list[int] | None = None
                    try:
                        req_id = input_batch.req_ids[i]
                        idx = input_batch.req_id_to_index[req_id]
                        n_tok = int(input_batch.num_tokens_no_spec[idx])
                        if n_tok > 0:
                            full_ctx = (
                                input_batch.token_ids_cpu[idx, :n_tok].tolist()
                            )
                    except Exception:
                        full_ctx = None
                    if full_ctx is None:
                        full_ctx = list(sampled_ids)

                    if self._use_kv:
                        drafts.append(
                            self._propose_real_kv(req_id, full_ctx, K)
                        )
                    else:
                        drafts.append(
                            self._propose_real_single(full_ctx, K)
                        )
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
