# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that samples the next tokens from the model's outputs."""

import os
import sys
import time

import torch
import torch.nn as nn

from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words
from vllm.v1.sample.ops.logprobs import batched_count_greater_than
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

logger = init_logger(__name__)

_SAMPLING_EPS = 1e-5


# ─── IDE_016 / SUB_174 AVX-512 sampling integration ───────────────────────
# ENV `VLLM_USE_AVX512_SAMPLING=1` 시 AVX-512 fused_sample kernel 을 sample
# step 의 side-by-side telemetry 로 호출한다. 본 patch 는 actual sampling
# path 를 대체하지 않으며 (vllm 의 GPU-resident logits 흐름 유지 → accuracy
# bit-exact gate PASS), kernel latency / CPU offload 가능성을 캡처하기 위한
# telemetry 모드이다. silent disable on import fail.
_avx512_smp_enabled: bool = (
    os.environ.get("VLLM_USE_AVX512_SAMPLING", "0") == "1"
)
_avx512_smp_pkg = None
_avx512_smp_init_attempted: bool = False
# step-level telemetry (per-process totals)
_avx512_smp_step_count: int = 0
_avx512_smp_native_total_ns: int = 0
_avx512_smp_avx_total_ns: int = 0
_avx512_smp_d2h_total_ns: int = 0
_avx512_smp_token_match_count: int = 0
_avx512_smp_token_total_count: int = 0
_avx512_smp_logprob_max_abs_diff: float = 0.0
# probe cadence: don't run AVX kernel on every step (it requires d2h transfer
# that would dominate cost); sample every Nth step instead. tuned by ENV.
_avx512_smp_probe_every: int = max(
    1, int(os.environ.get("VLLM_AVX512_SAMPLING_PROBE_EVERY", "16"))
)
_avx512_smp_probe_counter: int = 0


def _avx512_smp_get_pkg():
    """Lazy import the IDE_016 avx512_amx_pool package; silent disable on fail."""
    global _avx512_smp_pkg, _avx512_smp_init_attempted, _avx512_smp_enabled
    if not _avx512_smp_enabled:
        return None
    if _avx512_smp_pkg is not None:
        return _avx512_smp_pkg
    if _avx512_smp_init_attempted:
        return None
    _avx512_smp_init_attempted = True
    try:
        ide016_root = (
            "/workspace/vllm_hybrid/shadow_assists/features/"
            "IDE_016_avx512_amx_pool"
        )
        if ide016_root not in sys.path:
            sys.path.insert(0, ide016_root)
        import avx512_amx_pool as _pkg  # noqa: E402
        if not bool(_pkg.sampling.cpu_has_avx512()):
            logger.warning(
                "IDE_016 avx512 sampling: cpu_has_avx512=False — disabled"
            )
            _avx512_smp_enabled = False
            return None
        if not hasattr(_pkg, "sampling") or not hasattr(
            _pkg.sampling, "fused_sample"
        ):
            logger.warning(
                "IDE_016 avx512 sampling: fused_sample missing — disabled"
            )
            _avx512_smp_enabled = False
            return None
        _avx512_smp_pkg = _pkg
        logger.info(
            "IDE_016 avx512 sampling: lazy-init OK "
            "(fused_sample probe_every=%d)",
            _avx512_smp_probe_every,
        )
        return _pkg
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "IDE_016 avx512 sampling: lazy-init failed (%s) — disabled", exc
        )
        _avx512_smp_enabled = False
        return None


def avx512_smp_snapshot() -> dict:
    """Snapshot per-process AVX-512 sampling telemetry (used by SUB_174 RESULTS)."""
    return {
        "enabled": bool(_avx512_smp_enabled),
        "probe_every": _avx512_smp_probe_every,
        "step_count": _avx512_smp_step_count,
        "native_total_ns": _avx512_smp_native_total_ns,
        "avx_total_ns": _avx512_smp_avx_total_ns,
        "d2h_total_ns": _avx512_smp_d2h_total_ns,
        "token_match_count": _avx512_smp_token_match_count,
        "token_total_count": _avx512_smp_token_total_count,
        "logprob_max_abs_diff": _avx512_smp_logprob_max_abs_diff,
    }
# ────────────────────────────────────────────────────────────────────────


class Sampler(nn.Module):
    """
    A layer that samples the next tokens from the model's outputs
    with the following steps in order:

    1. If logprobs are requested:
        a) If `logprobs_mode` is `raw_logprobs`, compute logprobs
           as the final logprobs to return.
        b) If `logprobs_mode` is `raw_logits`, clone the logits
           as the final logprobs to return.
    2. Convert logits to float32.
    3. Apply allowed token ids whitelist.
    4. Apply bad words exclusion.
    5. Apply logit processors which are not argmax-invariant,
       i.e. that can impact greedy sampling.
        a) Min tokens processor
        b) Logit bias processor
    6. Apply penalties
        a) Repetition penalty
        b) Frequency penalty
        c) Presence penalty
    7. Sample the next tokens. `sample` method performs the following steps:
        a) If not `all_random`, perform greedy sampling. If `all_greedy`,
           return the greedily sampled tokens and final logprobs if requested.
        b) Apply temperature.
        c) Apply logit processors which are argmax-invariant, by default
           the min_p processor.
        d) Apply top_k and/or top_p.
        e) Sample the next tokens with the probability distribution.
        f) If `all_random` or temperature >= epsilon (1e-5), return the
           randomly sampled tokens and final logprobs if requested. Else,
           return the greedily sampled tokens and logprobs if requested.
    8. Gather the logprobs of the top `max_num_logprobs` and sampled token
       (if requested). Note that if the sampled token is within the top
       `max_num_logprobs`, the logprob will be eventually merged in
       `LogprobsProcessor` during output processing. Therefore, the
       final output may contain either `max_num_logprobs + 1` or
       `max_num_logprobs` logprobs.
    9. Return the final `SamplerOutput`.
    """

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs"):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler(logprobs_mode)
        self.pin_memory = is_pin_memory_available()
        self.logprobs_mode = logprobs_mode

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> SamplerOutput:
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            if logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif logprobs_mode == "raw_logits":
                if logits.dtype == torch.float32:
                    raw_logprobs = logits.clone()
                else:
                    raw_logprobs = logits.to(torch.float32)

        # Use float32 for the logits.
        logits = logits.to(torch.float32)

        logits = self.apply_logits_processors(
            logits, sampling_metadata, predict_bonus_token
        )
        # Sample the next token.
        sampled, processed_logprobs = self.sample(logits, sampling_metadata)
        if processed_logprobs is not None:
            raw_logprobs = processed_logprobs
        # Convert sampled token ids to int64 (long) type to ensure compatibility
        # with subsequent operations that may use these values as indices.
        # This conversion is necessary because FlashInfer sampling operations
        # return int32 (while PyTorch argmax and topk return int64).
        sampled = sampled.long()

        # Handle logprob_token_ids if specified (more efficient than full vocab)
        # This is used by generative_scoring API to get logprobs for specific tokens
        logprob_token_ids_tensors = None
        if sampling_metadata.logprob_token_ids:
            logprob_token_ids_tensors = self.gather_specific_token_logprobs(
                logits, sampling_metadata.logprob_token_ids, sampled
            )

        if num_logprobs is None:
            logprobs_tensors = logprob_token_ids_tensors
        elif num_logprobs == -1:
            # Return the full unsorted and unranked logprobs.
            logprobs_tensors = LogprobsTensors(
                torch.empty(0), raw_logprobs, torch.empty(0)
            )
        else:
            # Gather the logprobs and ranks of the topk and sampled token.
            logprobs_tensors = self.gather_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled
            )

        # If we have both num_logprobs and logprob_token_ids, prefer
        # logprob_token_ids as it's more specific
        if logprob_token_ids_tensors is not None and num_logprobs is not None:
            logprobs_tensors = logprob_token_ids_tensors

        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    def gather_specific_token_logprobs(
        self,
        logits: torch.Tensor,
        logprob_token_ids: dict[int, list[int]],
        sampled: torch.Tensor,
    ) -> LogprobsTensors | None:
        """Compute logprobs for specific token IDs using Triton kernel.

        This method handles heterogeneous token ID lists across requests by
        padding shorter lists to max length and using a fused Triton kernel
        for efficient log_softmax + gather computation.

        Benchmarks show the Triton kernel approach is ~1.4x faster than sparse
        gather for batch sizes > 1 due to the fused kernel reducing memory
        bandwidth requirements.

        Args:
            logits: [batch_size, vocab_size] tensor of logits
            logprob_token_ids: dict mapping req_index -> list of token IDs
            sampled: [batch_size] tensor of sampled token IDs

        Returns:
            LogprobsTensors with logprobs for the specified tokens, or None
            if no requests have logprob_token_ids.
        """
        if not logprob_token_ids:
            return None

        batch_size = logits.shape[0]
        device = logits.device

        # Find max number of tokens across all requests
        max_num_tokens = max(len(tids) for tids in logprob_token_ids.values())

        # Create padded token_ids tensor: [batch_size, max_num_tokens + 1]
        # +1 for sampled token in first position
        token_ids_tensor = torch.zeros(
            batch_size, max_num_tokens + 1, dtype=torch.int64, device=device
        )
        token_ids_tensor[:, 0] = sampled  # First column is sampled token

        # Create mask for valid positions (True = valid, False = padded)
        valid_mask = torch.zeros(
            batch_size, max_num_tokens + 1, dtype=torch.bool, device=device
        )
        valid_mask[:, 0] = True  # Sampled token is always valid

        # Fill in token IDs for each request
        for req_idx, token_ids in logprob_token_ids.items():
            num_tokens = len(token_ids)
            token_ids_tensor[req_idx, 1 : num_tokens + 1] = torch.tensor(
                token_ids, dtype=torch.int64, device=device
            )
            valid_mask[req_idx, 1 : num_tokens + 1] = True

        # Compute logprobs using the fused Triton kernel (log_softmax + gather)
        logprobs = compute_token_logprobs(logits, token_ids_tensor)

        # Mask invalid (padded) positions with -inf
        logprobs = logprobs.masked_fill(~valid_mask, float("-inf"))

        # Compute ranks for the sampled token
        sampled_logits = logits.gather(-1, sampled.unsqueeze(-1))
        token_ranks = (logits > sampled_logits).sum(dim=-1)

        return LogprobsTensors(
            logprob_token_ids=token_ids_tensor.to(torch.int32),
            logprobs=logprobs,
            selected_token_ranks=token_ranks,
        )

    @staticmethod
    def apply_temperature(
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        # Use in-place division to avoid creating a new tensor.
        # Avoid division by zero if there are greedy requests.
        if not all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        return logits.div_(temp.unsqueeze(dim=1))

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                processed_logprobs = None
                if sampling_metadata.max_num_logprobs is not None:
                    if logprobs_mode == "processed_logits":
                        processed_logprobs = logits
                    elif logprobs_mode == "processed_logprobs":
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        assert sampling_metadata.temperature is not None

        # Apply temperature.
        logits = self.apply_temperature(
            logits, sampling_metadata.temperature, sampling_metadata.all_random
        )

        # Apply logits processors that only apply to random sampling
        # (argmax invariant)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        # SUB_174: AVX-512 fused_sample telemetry (probe cadence guarded).
        # native sampling path on GPU remains source-of-truth — accuracy gate
        # bit-exact (token-level) wrt OFF baseline.
        global _avx512_smp_probe_counter
        do_probe = False
        if _avx512_smp_enabled:
            _avx512_smp_probe_counter += 1
            if _avx512_smp_probe_counter % _avx512_smp_probe_every == 0:
                do_probe = True

        native_t0 = time.perf_counter_ns() if _avx512_smp_enabled else 0

        # Apply top_k and/or top_p.
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        if _avx512_smp_enabled:
            native_t1 = time.perf_counter_ns()
            global _avx512_smp_step_count, _avx512_smp_native_total_ns
            _avx512_smp_step_count += 1
            _avx512_smp_native_total_ns += (native_t1 - native_t0)

            if do_probe:
                self._avx512_smp_probe(
                    logits, sampling_metadata, random_sampled
                )

        if greedy_sampled is None:
            return random_sampled, processed_logprobs

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled, processed_logprobs

    @staticmethod
    def _avx512_smp_probe(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        random_sampled: torch.Tensor,
    ) -> None:
        """SUB_174: side-by-side AVX-512 fused_sample probe.

        Copies a small slice of logits to CPU, runs AVX-512 fused_sample on it,
        and compares argmax (informational accuracy metric). Does not alter
        sampling output. Failures are silenced (telemetry path).
        """
        global _avx512_smp_avx_total_ns, _avx512_smp_d2h_total_ns
        global _avx512_smp_token_match_count, _avx512_smp_token_total_count
        global _avx512_smp_logprob_max_abs_diff
        pkg = _avx512_smp_get_pkg()
        if pkg is None:
            return
        try:
            # Determine scalar top_k / top_p / temperature for the kernel
            # (use first row's values as representative; kernels accept scalar).
            top_k_t = sampling_metadata.top_k
            top_p_t = sampling_metadata.top_p
            temp_t = sampling_metadata.temperature
            if top_k_t is None:
                k_val = int(logits.size(-1))
            else:
                k_val = int(top_k_t[0].item())
            if top_p_t is None:
                p_val = 1.0
            else:
                p_val = float(top_p_t[0].item())
            if temp_t is None:
                T_val = 1.0
            else:
                T_val = float(temp_t[0].item())
            # Cap k at vocab size, ensure positive.
            k_val = max(1, min(k_val, int(logits.size(-1))))

            # Limit probe batch to first 4 rows (kernel exercise — keep d2h cheap).
            B_probe = min(4, int(logits.size(0)))
            d2h_t0 = time.perf_counter_ns()
            sub = logits[:B_probe].detach().to(
                device="cpu", dtype=torch.float32, copy=True
            ).contiguous()
            d2h_t1 = time.perf_counter_ns()
            _avx512_smp_d2h_total_ns += (d2h_t1 - d2h_t0)

            avx_t0 = time.perf_counter_ns()
            avx_token_ids = pkg.sampling.fused_sample(
                sub, k_val, p_val, T_val, 0
            )
            avx_t1 = time.perf_counter_ns()
            _avx512_smp_avx_total_ns += (avx_t1 - avx_t0)

            # Token-level argmax informational match (kernel uses its own rng,
            # so this is not bit-exact even at temp=1 — just regression tracker).
            try:
                ref = random_sampled[:B_probe].detach().to(
                    device="cpu", dtype=torch.int64
                )
                avx_ids = avx_token_ids.to(torch.int64)
                _avx512_smp_token_total_count += B_probe
                _avx512_smp_token_match_count += int(
                    (avx_ids == ref).sum().item()
                )
            except Exception:
                pass

            # logprob max abs diff probe (CPU softmax-of-temperatured logits
            # vs AVX softmax). Computed on the first row only to bound cost.
            try:
                ref_lp = torch.log_softmax(sub[0], dim=-1)
                avx_probs = pkg.sampling.softmax(sub[0].clone())
                avx_lp = torch.log(avx_probs.clamp_min(1e-30))
                diff = float((ref_lp - avx_lp).abs().max().item())
                if diff > _avx512_smp_logprob_max_abs_diff:
                    _avx512_smp_logprob_max_abs_diff = diff
            except Exception:
                pass
        except Exception:
            # Telemetry must never break sampling.
            pass

    @staticmethod
    def compute_logprobs(logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    @staticmethod
    def gather_logprobs(
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: maximum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == torch.int64
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        # Avoid 0/1 specialization recompile on the batch dimension
        # of the compiled batched_count_greater_than. mark_unbacked makes
        # the size fully symbolic so dynamo doesn't specialize when
        # batch_size transitions from 1 to >=2.
        torch._dynamo.decorators.mark_unbacked(logprobs, 0)
        torch._dynamo.decorators.mark_unbacked(token_logprobs, 0)
        token_ranks = batched_count_greater_than(logprobs, token_logprobs)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)

    @staticmethod
    def _combine_outputs_with_spec_tokens(
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None = None,
    ) -> list[list[int]]:
        if spec_token_ids is None:
            return output_token_ids

        return [
            [*out, *spec] if spec else out
            for out, spec in zip(output_token_ids, spec_token_ids)
        ]

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool,
    ) -> torch.Tensor:
        bad_words_token_ids = sampling_metadata.bad_words_token_ids
        any_penalties_or_bad_words = (
            bool(bad_words_token_ids) or not sampling_metadata.no_penalties
        )

        output_token_ids = sampling_metadata.output_token_ids
        if predict_bonus_token and any_penalties_or_bad_words:
            # Combine base outputs with spec tokens when speculative decoding
            # is enabled.
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # Apply allowed token ids.
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))

        # Apply bad words exclusion.
        if bad_words_token_ids:
            apply_bad_words(logits, bad_words_token_ids, output_token_ids)

        # Apply logits processors which can impact greedy sampling.
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)

        # Apply penalties (e.g., freq_penalties).
        logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)
        return logits

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None
        return apply_all_penalties(
            logits,
            sampling_metadata.prompt_token_ids,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
            output_token_ids,
        )
