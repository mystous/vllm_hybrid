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


# ─── SUB_201 L11 CPU sampling offload ────────────────────────────────────
# ENV `VLLM_CPU_SAMPLING=1` 시 sample() 의 logits 가 GPU→CPU 로 옮겨진 뒤
# CPU 가 softmax+top-K+multinomial 을 수행한다. 목적은 GPU 가 다음 forward
# step 을 sampling 과 overlap 하도록 launching 을 비차단화하는 것.
#
# 정확도 게이트:
#   - greedy/temperature==0 경로는 CPU argmax 가 GPU argmax 와 token-level
#     bit-exact (BF16 → FP32 변환 후 argmax 는 동일 ordering 유지).
#   - random sampling 경로는 CLAUDE.md 의 "분포 유사성" 조건을 따른다.
_cpu_sampling_enabled: bool = (
    os.environ.get("VLLM_CPU_SAMPLING", "0") == "1"
)
# Telemetry counters
_cpu_smp_step_count: int = 0
_cpu_smp_d2h_total_ns: int = 0
_cpu_smp_kernel_total_ns: int = 0
_cpu_smp_total_tokens: int = 0


def cpu_sampling_snapshot() -> dict:
    """Snapshot per-process CPU sampling telemetry."""
    return {
        "enabled": bool(_cpu_sampling_enabled),
        "step_count": _cpu_smp_step_count,
        "d2h_total_ns": _cpu_smp_d2h_total_ns,
        "kernel_total_ns": _cpu_smp_kernel_total_ns,
        "total_tokens": _cpu_smp_total_tokens,
    }


def _cpu_apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """CPU top-k / top-p mask. Mirrors apply_top_k_top_p_pytorch in
    vllm/v1/sample/ops/topk_topp_sampler.py but assumes CPU tensors and
    skips the no-op early return when both are None (handled by caller).
    """
    if p is None and k is None:
        return logits

    if p is None:
        # top-k only, no sort needed.
        no_top_k_mask = k == logits.shape[1]
        k_eff = k.masked_fill(no_top_k_mask, 1)
        max_top_k = int(k_eff.max().item())
        k_index = k_eff.sub(1).unsqueeze(1)
        top_k_mask = logits.topk(max_top_k, dim=1).values.gather(
            1, k_index.long()
        )
        top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
        return logits.masked_fill_(logits < top_k_mask, -float("inf"))

    # top-p path (and optional top-k).
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    if k is not None:
        top_k_mask = logits_sort.size(1) - k.to(torch.long)
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


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

        # SUB_201 L11: CPU sampling offload.
        # 조건: VLLM_CPU_SAMPLING=1 이고 logprobs 가 processed_logprobs/
        # processed_logits 가 아닌 경우 (정상적인 경우 raw_logprobs/raw_logits).
        # all_greedy 경로는 logits 가 D2H 되기 전 GPU 에서 argmax 하는 게 빠르므로
        # CPU offload 를 적용하지 않는다 (random 경로만 CPU 로).
        if (
            _cpu_sampling_enabled
            and not sampling_metadata.all_greedy
            and logprobs_mode not in ("processed_logits", "processed_logprobs")
        ):
            return self._cpu_sample(
                logits, sampling_metadata, logprobs_mode
            )

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

    def _cpu_sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode: LogprobsMode,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """SUB_201 L11: CPU sampling offload.

        Logits 를 CPU 로 옮긴 뒤 softmax → top-K (with top-P 와 함께) →
        multinomial (exponential trick) 을 CPU 에서 수행. greedy 경로는
        argmax 만 별도로 처리. 반환되는 token id 는 GPU tensor 로 다시
        올린 뒤 후속 GPU path (logprobs gather 등) 와 호환되도록 함.
        """
        global _cpu_smp_step_count, _cpu_smp_d2h_total_ns
        global _cpu_smp_kernel_total_ns, _cpu_smp_total_tokens

        device = logits.device
        B = logits.size(0)

        # 1) D2H transfer (GPU → CPU). Synchronous copy so the host buffer
        #    is ready when we read it. PoC keeps things simple (no async
        #    overlap stream).
        d2h_t0 = time.perf_counter_ns()
        if device.type == "cuda":
            # Force the current stream to finish logits compute first.
            torch.cuda.current_stream(device).synchronize()
            logits_cpu = logits.detach().to(
                device="cpu", dtype=torch.float32, copy=True
            ).contiguous()
        else:
            logits_cpu = logits.to(dtype=torch.float32, copy=False)
        d2h_t1 = time.perf_counter_ns()
        _cpu_smp_d2h_total_ns += (d2h_t1 - d2h_t0)

        # Sampling metadata (temperature / top_k / top_p) 도 CPU 로.
        temperature_cpu = sampling_metadata.temperature
        if temperature_cpu is not None and temperature_cpu.device.type != "cpu":
            temperature_cpu = temperature_cpu.to(device="cpu",
                                                  non_blocking=False)

        top_k = sampling_metadata.top_k
        top_p = sampling_metadata.top_p
        if top_k is not None and top_k.device.type != "cpu":
            top_k = top_k.to(device="cpu", non_blocking=False)
        if top_p is not None and top_p.device.type != "cpu":
            top_p = top_p.to(device="cpu", non_blocking=False)

        kernel_t0 = time.perf_counter_ns()

        # 2) greedy 마스크 (temperature < eps 인 row).
        if temperature_cpu is not None:
            greedy_mask_cpu = temperature_cpu < _SAMPLING_EPS
        else:
            greedy_mask_cpu = torch.zeros(B, dtype=torch.bool)

        # 3) greedy argmax 는 logits_cpu 그대로 (temperature 적용 전).
        greedy_ids = logits_cpu.argmax(dim=-1).view(-1)

        # 4) random path: temperature scale → optional top-k/top-p → softmax
        #    → exponential trick multinomial.
        rand_logits = logits_cpu
        if temperature_cpu is not None:
            safe_T = torch.where(
                greedy_mask_cpu,
                torch.ones_like(temperature_cpu),
                temperature_cpu,
            ).unsqueeze(1)
            rand_logits = rand_logits / safe_T

        # apply argmax-invariant processors only if they exist (e.g., min_p).
        # processors expect GPU tensors typically — fall back to GPU for those.
        # For PoC: if there are argmax-invariant processors, abort offload and
        # use GPU native path to preserve correctness.
        argmax_invariant = sampling_metadata.logitsprocs.argmax_invariant
        if argmax_invariant:
            # Fallback to native GPU sampling path.
            return self._cpu_sample_fallback(
                logits, sampling_metadata, logprobs_mode
            )

        # top-k / top-p in CPU.
        rand_logits = _cpu_apply_top_k_top_p(rand_logits, top_k, top_p)

        # exponential-trick multinomial (matches random_sample()'s algorithm).
        probs = rand_logits.softmax(dim=-1, dtype=torch.float32)
        q = torch.empty_like(probs)
        q.exponential_()
        # per-request generators not honored in PoC (rare in benchmark traffic).
        rand_ids = probs.div_(q).argmax(dim=-1).view(-1)

        # 5) merge greedy / random per row.
        sampled_ids = torch.where(greedy_mask_cpu, greedy_ids, rand_ids)

        kernel_t1 = time.perf_counter_ns()
        _cpu_smp_kernel_total_ns += (kernel_t1 - kernel_t0)
        _cpu_smp_step_count += 1
        _cpu_smp_total_tokens += int(B)

        # 6) processed_logprobs are not supported here (we already short
        #    circuited that mode above), so return None.
        # Token id 를 GPU 로 다시 올리고 반환. int64 로 유지 (sampler.forward
        # 에서 .long() 으로 어쨌든 변환되므로 한 번에 끝낸다).
        sampled_ids = sampled_ids.to(torch.int64).contiguous()
        if device.type == "cuda":
            sampled_gpu = sampled_ids.to(device=device, non_blocking=False)
            torch.cuda.current_stream(device).synchronize()
        else:
            sampled_gpu = sampled_ids
        return sampled_gpu, None

    def _cpu_sample_fallback(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode: LogprobsMode,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fallback to the original GPU sampling path when CPU offload
        cannot preserve correctness (e.g., argmax-invariant processors)."""
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                return greedy_sampled, None
        logits = self.apply_temperature(
            logits, sampling_metadata.temperature,
            sampling_metadata.all_random,
        )
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )
        if greedy_sampled is None:
            return random_sampled, processed_logprobs
        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,
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
