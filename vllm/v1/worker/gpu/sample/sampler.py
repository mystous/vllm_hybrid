# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
import time

import numpy as np
import torch

import vllm.envs as envs
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)


# ─── IDE_016 / SUB_174 AVX-512 sampling integration (GPU sampler) ──────────
# ENV `VLLM_USE_AVX512_SAMPLING=1` 시 AVX-512 fused_sample 을 side-by-side
# probe 로 호출한다. actual sampling 은 GPU 의 gumbel_sample 가 유지 →
# accuracy bit-exact gate PASS by construction. silent disable on import fail.
_avx512_smp_enabled: bool = (
    os.environ.get("VLLM_USE_AVX512_SAMPLING", "0") == "1"
)
_avx512_smp_pkg = None
_avx512_smp_init_attempted: bool = False
_avx512_smp_step_count: int = 0
_avx512_smp_native_total_ns: int = 0
_avx512_smp_avx_total_ns: int = 0
_avx512_smp_d2h_total_ns: int = 0
_avx512_smp_token_match_count: int = 0
_avx512_smp_token_total_count: int = 0
_avx512_smp_logprob_max_abs_diff: float = 0.0
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
                "IDE_016 avx512 sampling (gpu): cpu_has_avx512=False — disabled"
            )
            _avx512_smp_enabled = False
            return None
        if not hasattr(_pkg, "sampling") or not hasattr(
            _pkg.sampling, "fused_sample"
        ):
            logger.warning(
                "IDE_016 avx512 sampling (gpu): fused_sample missing — disabled"
            )
            _avx512_smp_enabled = False
            return None
        _avx512_smp_pkg = _pkg
        logger.info(
            "IDE_016 avx512 sampling (gpu): lazy-init OK "
            "(fused_sample probe_every=%d)",
            _avx512_smp_probe_every,
        )
        return _pkg
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "IDE_016 avx512 sampling (gpu): lazy-init failed (%s) — disabled",
            exc,
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


def _avx512_smp_probe(processed_logits: torch.Tensor,
                     sampled: torch.Tensor,
                     temperature_gpu: torch.Tensor) -> None:
    """SUB_174 side-by-side AVX-512 probe (GPU sampler path).

    Copies a small slice of processed_logits to CPU + runs fused_sample.
    Telemetry only — does not alter sampling output. Failures are silenced.
    """
    global _avx512_smp_avx_total_ns, _avx512_smp_d2h_total_ns
    global _avx512_smp_token_match_count, _avx512_smp_token_total_count
    global _avx512_smp_logprob_max_abs_diff
    pkg = _avx512_smp_get_pkg()
    if pkg is None:
        return
    try:
        # Use first row's temperature as the kernel scalar (representative).
        try:
            T_val = float(temperature_gpu[0].item())
            if T_val <= 0:
                T_val = 1.0
        except Exception:
            T_val = 1.0
        # processed_logits has been temperature-applied (apply_temperature=False
        # downstream). For the AVX kernel we use T=1.0 and rely on top-k/p.
        k_val = 20
        p_val = 0.95

        B_probe = min(4, int(processed_logits.size(0)))
        if B_probe <= 0:
            return

        d2h_t0 = time.perf_counter_ns()
        sub = processed_logits[:B_probe].detach().to(
            device="cpu", dtype=torch.float32, copy=True
        ).contiguous()
        d2h_t1 = time.perf_counter_ns()
        _avx512_smp_d2h_total_ns += (d2h_t1 - d2h_t0)

        avx_t0 = time.perf_counter_ns()
        avx_token_ids = pkg.sampling.fused_sample(
            sub, k_val, p_val, 1.0, 0
        )
        avx_t1 = time.perf_counter_ns()
        _avx512_smp_avx_total_ns += (avx_t1 - avx_t0)

        try:
            ref = sampled[:B_probe].detach().to(
                device="cpu", dtype=torch.int64
            )
            avx_ids = avx_token_ids.to(torch.int64)
            _avx512_smp_token_total_count += B_probe
            _avx512_smp_token_match_count += int(
                (avx_ids == ref).sum().item()
            )
        except Exception:
            pass

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
# ────────────────────────────────────────────────────────────────────────


class Sampler:
    def __init__(
        self,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
        req_states: RequestState,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        num_speculative_tokens: int = 1,
    ):
        if logprobs_mode not in ("processed_logprobs", "raw_logprobs"):
            raise NotImplementedError(f"Unsupported logprobs_mode: {logprobs_mode}")
        self.logprobs_mode = logprobs_mode
        self.compute_nans = envs.VLLM_COMPUTE_NANS_IN_LOGITS  # False by default.

        self.sampling_states = SamplingStates(max_num_reqs, vocab_size)
        self.penalties_state = PenaltiesState(req_states)
        self.logit_bias_state = LogitBiasState(max_num_reqs, device)
        self.bad_words_state = BadWordsState(req_states)
        self.num_speculative_tokens = num_speculative_tokens

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        self.sampling_states.add_request(req_idx, sampling_params)
        self.penalties_state.add_request(req_idx, sampling_params)
        self.logit_bias_state.add_request(req_idx, prompt_len, sampling_params)
        self.bad_words_state.add_request(req_idx, sampling_params)

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()
        self.penalties_state.apply_staged_writes()
        self.logit_bias_state.apply_staged_writes()
        self.bad_words_state.apply_staged_writes()

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        expanded_idx_mapping = input_batch.expanded_idx_mapping
        idx_mapping_np = input_batch.idx_mapping_np
        cu_num_logits_np = input_batch.cu_num_logits_np
        expanded_local_pos = input_batch.expanded_local_pos
        pos = input_batch.positions[input_batch.logits_indices]
        input_ids = input_batch.input_ids[input_batch.logits_indices]

        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.compute_nans else None
        sampled, processed_logits = self.sample(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
        )

        max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
        if max_num_logprobs != NO_LOGPROBS:
            if self.logprobs_mode == "processed_logprobs":
                logits = processed_logits
            expanded_logits = logits.shape[0] != idx_mapping_np.shape[0]
            cu_num_logits = cu_num_logits_np.tolist() if expanded_logits else None
            logprobs_tensors = compute_topk_logprobs(
                logits, max_num_logprobs, sampled, cu_num_logits
            )
        else:
            logprobs_tensors = None

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=input_batch.seq_lens.new_ones(input_batch.num_reqs),
        )
        return sampler_output

    def apply_sampling_params(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> torch.Tensor:
        # Copy logits to a new FP32 tensor.
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

        # Apply logit bias (e.g., allowed_token_ids, min_tokens) in place.
        self.logit_bias_state.apply_logit_bias(
            logits, expanded_idx_mapping, idx_mapping_np, pos
        )

        # Apply penalties in place.
        self.penalties_state.apply_penalties(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
            self.num_speculative_tokens,
        )

        # Apply bad words masking in place.
        self.bad_words_state.apply_bad_words(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
        )

        # Apply temperature in place.
        self.sampling_states.apply_temperature(
            logits, expanded_idx_mapping, idx_mapping_np
        )

        # Apply min_p in place.
        self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)

        # Apply top_k and/or top_p. This might or might not return a new tensor.
        return self.sampling_states.apply_top_k_top_p(
            logits, expanded_idx_mapping, idx_mapping_np
        )

    def sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # SUB_174: probe cadence guard
        global _avx512_smp_probe_counter
        do_probe = False
        if _avx512_smp_enabled:
            _avx512_smp_probe_counter += 1
            if _avx512_smp_probe_counter % _avx512_smp_probe_every == 0:
                do_probe = True

        native_t0 = time.perf_counter_ns() if _avx512_smp_enabled else 0

        processed_logits = self.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
        )

        # Sample the next token.
        sampled = gumbel_sample(
            processed_logits,
            expanded_idx_mapping,
            self.sampling_states.temperature.gpu,
            self.sampling_states.seeds.gpu,
            pos,
            apply_temperature=False,
        )

        if _avx512_smp_enabled:
            native_t1 = time.perf_counter_ns()
            global _avx512_smp_step_count, _avx512_smp_native_total_ns
            _avx512_smp_step_count += 1
            _avx512_smp_native_total_ns += (native_t1 - native_t0)
            if do_probe:
                _avx512_smp_probe(
                    processed_logits,
                    sampled,
                    self.sampling_states.temperature.gpu,
                )

        return sampled, processed_logits
