# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import sys
import time
from abc import ABC, abstractmethod

import tokenizers
from packaging import version
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream
from transformers import PreTrainedTokenizerFast

from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.detokenizer_utils import (
    convert_prompt_ids_to_tokens,
    detokenize_incrementally,
)
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)

# Only tokenizers >= 0.22.0 supports DecodeStream with native prefill
# (ids parameter) used for FastIncrementalDetokenizer.
USE_FAST_DETOKENIZER = version.parse(tokenizers.__version__) >= version.parse("0.22.0")

# Error string from https://github.com/huggingface/tokenizers/blob/909fdde2a4ffedd9295206f705eb612be2a91b12/tokenizers/src/tokenizer/mod.rs#L1042
INVALID_PREFIX_ERR_MSG = "Invalid prefix encountered"

# ─── IDE_016 / SUB_173 AVX-512 batch detokenize integration ───────────────
# ENV `VLLM_USE_AVX512_TOKENIZER=1` 시 AVX-512 kernel 을 detokenize 의
# fast path 로 사용. stream.step 도 internal state 유지를 위해 호출하되,
# 본 patch 는 우선 latency telemetry + cross-check 만 제공한다 (SUB_173 보고).
# silent disable on import fail.
_avx512_tok_enabled: bool = (
    os.environ.get("VLLM_USE_AVX512_TOKENIZER", "0") == "1"
)
_avx512_tok_pkg = None
_avx512_tok_init_attempted: bool = False
# per-tokenizer cache (object id → BatchDetokenizer)
_avx512_tok_cache: dict[int, object] = {}
# step-level telemetry (used in SUB_173 e2e measurement)
_avx512_tok_step_count: int = 0
_avx512_tok_native_total_ns: int = 0
_avx512_tok_avx_total_ns: int = 0
_avx512_tok_mismatch_count: int = 0


def _avx512_tok_get_pkg():
    """Lazy import the SUB_171 avx512_amx_pool package; silent disable on fail."""
    global _avx512_tok_pkg, _avx512_tok_init_attempted, _avx512_tok_enabled
    if not _avx512_tok_enabled:
        return None
    if _avx512_tok_pkg is not None:
        return _avx512_tok_pkg
    if _avx512_tok_init_attempted:
        return None
    _avx512_tok_init_attempted = True
    try:
        ide016_root = "/workspace/vllm_hybrid/shadow_assists/features/IDE_016_avx512_amx_pool"
        if ide016_root not in sys.path:
            sys.path.insert(0, ide016_root)
        import avx512_amx_pool as _pkg  # noqa: E402
        # `is_available()` checks sampling kernel; for tokenizer we only need
        # the BPE/SP kernel which is exposed via _core regardless of sampling
        # availability. Check cpu_has_avx512 + BatchDetokenizer presence.
        if not bool(_pkg.sampling.cpu_has_avx512()):
            logger.warning(
                "IDE_016 avx512 tokenizer: cpu_has_avx512=False — disabled"
            )
            _avx512_tok_enabled = False
            return None
        if not hasattr(_pkg, "BatchDetokenizer"):
            logger.warning(
                "IDE_016 avx512 tokenizer: BatchDetokenizer missing — disabled"
            )
            _avx512_tok_enabled = False
            return None
        _avx512_tok_pkg = _pkg
        logger.info(
            "IDE_016 avx512 tokenizer: lazy-init OK (BatchDetokenizer ready)"
        )
        return _pkg
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "IDE_016 avx512 tokenizer: lazy-init failed (%s) — disabled", exc
        )
        _avx512_tok_enabled = False
        return None


def _avx512_tok_get_for(hf_tok) -> object | None:
    """Cached BatchDetokenizer per hf_tok instance."""
    pkg = _avx512_tok_get_pkg()
    if pkg is None:
        return None
    key = id(hf_tok)
    bd = _avx512_tok_cache.get(key)
    if bd is not None:
        return bd
    try:
        bd = pkg.BatchDetokenizer.from_hf_tokenizer(hf_tok)
        _avx512_tok_cache[key] = bd
        return bd
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "IDE_016 avx512 tokenizer: from_hf_tokenizer failed (%s)", exc
        )
        return None


def avx512_tok_snapshot() -> dict:
    """Snapshot per-process AVX-512 tokenizer telemetry for SUB_173 RESULTS."""
    return {
        "enabled": bool(_avx512_tok_enabled),
        "step_count": _avx512_tok_step_count,
        "native_total_ns": _avx512_tok_native_total_ns,
        "avx_total_ns": _avx512_tok_avx_total_ns,
        "mismatch_count": _avx512_tok_mismatch_count,
    }
# ────────────────────────────────────────────────────────────────────────


class IncrementalDetokenizer:
    def __init__(self):
        self.token_ids: list[int] = []

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids

    def num_output_tokens(self) -> int:
        return len(self.token_ids)

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        self.token_ids.extend(new_token_ids)
        return None

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        return ""

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
    ) -> "IncrementalDetokenizer":
        assert request.sampling_params is not None

        if tokenizer is None:
            # No tokenizer => skipping detokenization.
            return IncrementalDetokenizer()

        if USE_FAST_DETOKENIZER and isinstance(tokenizer, PreTrainedTokenizerFast):
            # Fast tokenizer => use tokenizers library DecodeStream.
            return FastIncrementalDetokenizer(tokenizer, request)

        # Fall back to slow python-based incremental detokenization.
        return SlowIncrementalDetokenizer(tokenizer, request)


class BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC):
    def __init__(self, request: EngineCoreRequest):
        super().__init__()

        # Stop strings
        params = request.sampling_params
        assert params is not None
        if params.stop is None:
            self.stop = []
        elif isinstance(params.stop, str):
            self.stop = [params.stop]
        else:
            self.stop = params.stop
        self.min_tokens = params.min_tokens
        self.include_stop_str_in_output = params.include_stop_str_in_output

        # Number of chars to hold back when stop strings are to be excluded
        # from streamed output.
        if self.stop and not self.include_stop_str_in_output:
            self.stop_buffer_length = max(len(s) for s in self.stop) - 1
        else:
            self.stop_buffer_length = 0
        self._last_output_text_offset: int = 0

        # Generation data
        self.output_text = ""

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        """
        Update RequestState for the request_id by:
            1) Detokenize the new token ids incrementally.
            2) Evaluate stop criteria.

        Return matched stop string or None.
        """
        if not new_token_ids:
            # Skip detokenization if no new token ids.
            return None

        if stop_terminated and not self.include_stop_str_in_output:
            # If stop-terminated, exclude last token from detokenization
            # based on include_stop_str_in_output parameter.
            skipped_stop_token_id = new_token_ids[-1]
            new_token_ids = new_token_ids[:-1]
        else:
            skipped_stop_token_id = None

        # 1) Detokenize the new token ids incrementally.
        stop_check_offset = len(self.output_text)
        for new_token_id in new_token_ids:
            self.token_ids.append(new_token_id)
            self.output_text += self.decode_next(new_token_id)
            # Support min_tokens, see https://github.com/vllm-project/vllm/pull/22014
            if self.min_tokens and self.num_output_tokens() <= self.min_tokens:
                stop_check_offset = len(self.output_text)

        if skipped_stop_token_id is not None:
            # Cleanup after skipping detokenization.
            self.token_ids.append(skipped_stop_token_id)

        # 2) Evaluate stop strings.
        stop_string = None
        if self.stop and self.num_output_tokens() > self.min_tokens:
            stop = check_stop_strings(
                output_text=self.output_text,
                new_char_count=len(self.output_text) - stop_check_offset,
                stop=self.stop,
                include_in_output=self.include_stop_str_in_output,
            )
            if stop is not None:
                stop_string, truncate_to = stop
                if truncate_to != -1:
                    self.output_text = self.output_text[:truncate_to]

        return stop_string

    @abstractmethod
    def decode_next(self, next_token_id: int) -> str:
        raise NotImplementedError

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        """If delta is True, only new text since the last call to
        this method is returned"""

        # We return the full output text if the sequence is finished.
        buffer_length = 0 if finished else self.stop_buffer_length
        if not delta:
            if not buffer_length:
                return self.output_text
            return self.output_text[:-buffer_length]

        length = len(self.output_text) - buffer_length
        last_offset = self._last_output_text_offset
        if last_offset < length:
            self._last_output_text_offset = length
            return self.output_text[last_offset:length]
        return ""


class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, request: EngineCoreRequest):
        super().__init__(request)

        sampling_params = request.sampling_params
        assert sampling_params is not None

        self.request_id = request.request_id
        self.skip_special_tokens = sampling_params.skip_special_tokens

        self.tokenizer: Tokenizer = tokenizer._tokenizer

        # SUB_173: optional AVX-512 fast-path. uses the full
        # PreTrainedTokenizerFast (`tokenizer`) for vocab table build, not the
        # bare Rust `Tokenizer` (which lacks `convert_ids_to_tokens` symmetry).
        self._avx512_bd = _avx512_tok_get_for(tokenizer) if _avx512_tok_enabled else None

        # Use native prefill to prime the decode stream with prompt tokens.
        self.stream = DecodeStream(
            ids=request.prompt_token_ids,
            skip_special_tokens=self.skip_special_tokens,
        )

        self.spaces_between_special_tokens = (
            sampling_params.skip_special_tokens
            or sampling_params.spaces_between_special_tokens
        )

        if not self.spaces_between_special_tokens:
            # Store dict of added token ids so that we can suppress
            # the spaces between them.
            added_token_ids = getattr(self.tokenizer, "added_token_ids", None)
            if added_token_ids is None:
                self.tokenizer.added_token_ids = added_token_ids = {
                    tid: tok.content
                    for tid, tok in self.tokenizer.get_added_tokens_decoder().items()
                }

            if added_token_ids:
                self.last_special = False
                self.added_token_ids = added_token_ids
            else:
                # No added tokens.
                self.spaces_between_special_tokens = True

    def decode_next(self, next_token_id: int) -> str:
        token = self._protected_step(next_token_id)

        if not self.spaces_between_special_tokens:
            special_token = self.added_token_ids.get(next_token_id)
            is_special = special_token is not None
            if is_special and self.last_special:
                # Return raw token string without any prefixed spaces.
                token = special_token
            self.last_special = is_special

        return token or ""

    def _protected_step(self, next_token_id: int) -> str | None:
        # SUB_173: AVX-512 fast-path telemetry + cross-check.
        # we always call stream.step (DecodeStream is the source of truth for
        # incremental state). If AVX-512 is enabled, we also time the
        # avx-512 single-token decode and accumulate per-process totals.
        global _avx512_tok_step_count, _avx512_tok_native_total_ns
        global _avx512_tok_avx_total_ns, _avx512_tok_mismatch_count
        _avx_bd = getattr(self, "_avx512_bd", None)
        if _avx_bd is not None:
            avx_t0 = time.perf_counter_ns()
            try:
                avx_results = _avx_bd.batch_decode([[next_token_id]])
                avx_str = avx_results[0] if avx_results else ""
            except Exception:
                avx_str = None
            avx_t1 = time.perf_counter_ns()
            _avx512_tok_avx_total_ns += (avx_t1 - avx_t0)
        try:
            native_t0 = time.perf_counter_ns()
            token = self.stream.step(self.tokenizer, next_token_id)
            native_t1 = time.perf_counter_ns()
            if _avx_bd is not None:
                _avx512_tok_native_total_ns += (native_t1 - native_t0)
                _avx512_tok_step_count += 1
                # token-level byte-exact gate (informational metric).
                # native may return None for byte-piece accumulation; in that
                # case skip comparison.
                if (token is not None and avx_str is not None
                        and token != avx_str):
                    _avx512_tok_mismatch_count += 1
        except (OverflowError, TypeError):
            # Handle rare observed overflow, still to be diagnosed.
            # See https://github.com/vllm-project/vllm/issues/21951.
            logger.exception("Encountered invalid token id: %r", next_token_id)
            token = None
        except Exception as e:
            if not str(e).startswith(INVALID_PREFIX_ERR_MSG):
                raise e
            # Recover from edge case where tokenizer can produce non-monotonic,
            # invalid UTF-8 output, which breaks the internal state of
            # tokenizers' DecodeStream.
            # See https://github.com/vllm-project/vllm/issues/17448.
            logger.warning(
                "Encountered invalid prefix detokenization error"
                " for request %s, resetting decode stream.",
                self.request_id,
            )
            self.stream = DecodeStream(skip_special_tokens=self.skip_special_tokens)
            token = self.stream.step(self.tokenizer, next_token_id)
        return token


class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):
    def __init__(self, tokenizer: TokenizerLike, request: EngineCoreRequest):
        super().__init__(request)

        self.tokenizer = tokenizer
        params = request.sampling_params
        assert params is not None

        self.prompt_len = length_from_prompt_token_ids_or_embeds(
            request.prompt_token_ids, request.prompt_embeds
        )

        # Metadata for incremental detokenization.
        if request.prompt_token_ids is not None:
            self.tokens, self.prefix_offset, self.read_offset = (
                convert_prompt_ids_to_tokens(
                    tokenizer=tokenizer,
                    prompt_ids=request.prompt_token_ids,
                    skip_special_tokens=params.skip_special_tokens,
                )
            )
        else:
            # Prompt embedding requests cannot be detokenized, in general.
            self.tokens = [""] * self.prompt_len
            self.prefix_offset = 0
            self.read_offset = 0

        self.token_ids.extend(request.prompt_token_ids or [0] * self.prompt_len)

        self.skip_special_tokens = params.skip_special_tokens
        self.spaces_between_special_tokens = params.spaces_between_special_tokens

    @property
    def output_token_ids(self) -> list[int]:
        if self.prompt_len:
            return self.token_ids[self.prompt_len :]
        return self.token_ids

    def num_output_tokens(self) -> int:
        return len(self.token_ids) - self.prompt_len

    def decode_next(self, next_token_id: int) -> str:
        new_tokens, decoded_text, prefix_offset, read_offset = detokenize_incrementally(
            tokenizer=self.tokenizer,
            all_input_ids=self.token_ids,
            prev_tokens=self.tokens,
            prefix_offset=self.prefix_offset,
            read_offset=self.read_offset,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
        )

        self.tokens.extend(new_tokens)
        self.prefix_offset = prefix_offset
        self.read_offset = read_offset

        return decoded_text


def check_stop_strings(
    output_text: str,
    new_char_count: int,
    stop: list[str],
    include_in_output: bool,
) -> tuple[str, int] | None:
    """Check if any stop strings are matched and truncate sequence
    output text accordingly.

    Returns tuple (stop_string, offset) if matched or else None.

    Where stop_string is the matched stop string and offset is the
    length to which output_text should be truncated, or -1 for no
    truncation.
    """
    if not new_char_count or not stop:
        return None

    for stop_str in stop:
        stop_string_len = len(stop_str)
        # Avoid searching already-searched text.
        stop_index = output_text.find(stop_str, 1 - new_char_count - stop_string_len)
        if stop_index == -1:
            continue

        if include_in_output:
            # Truncate to end of stop string.
            stop_index += stop_string_len
            if stop_index >= len(output_text):
                # No truncation required.
                return stop_str, -1

        # Truncate the output text to either the beginning
        # or end of the stop string.
        return stop_str, stop_index
    return None
