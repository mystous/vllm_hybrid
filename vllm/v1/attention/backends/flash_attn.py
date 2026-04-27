# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""

import copy
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

# IDE_006 / TSK_002 §4.4 — module-level imports for hot_cold_attention.
# These were previously imported lazily inside the function body, which
# triggered a sys.modules dict lookup on every cold-path-bearing layer
# call (80 layers × N decode steps × 8 workers).
from vllm.v1.attention.ops.cpu_partial_attention import (
    _ASYNC_OVERLAP_DISABLED as _PARTIAL_ASYNC_DISABLED,
    _PROFILE_ENABLED as _PARTIAL_PROFILE_ENABLED,
    _profile_should_emit as _partial_profile_should_emit,
    forward_partial_with_lse,
    forward_partial_with_lse_async,
)
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_supports_fp8,
    flash_attn_supports_quant_query_input,
    get_flash_attn_version,
    is_fa_version_supported,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.worker.workspace import current_workspace_manager

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        get_scheduler_metadata,
        reshape_and_cache_flash,
    )
import vllm.envs as envs
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
    get_current_vllm_config_or_none,
    get_layers_from_vllm_config,
)
from vllm.config.cache import CacheDType
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    get_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class FlashAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        if (
            model_config
            and model_config.is_hybrid
            and (
                cache_config.mamba_ssm_cache_dtype == "float32"
                or cache_config.mamba_cache_dtype == "float32"
            )
        ):
            # NOTE(tdoublep): while in principle, FA supports
            # MultipleOf(16), these are the block sizes that do not
            # suffer from the NaN propagation problem described here:
            # https://github.com/Dao-AILab/flash-attention/issues/1974
            return [16, 32, 64]
        return [MultipleOf(16)]

    forward_includes_kv_cache_update: bool = False

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        if current_platform.is_xpu():
            return max(default_block_size, 64)
        return super().get_preferred_block_size(default_block_size)

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """FlashAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        fa_version = get_flash_attn_version()
        return fa_version is not None and fa_version >= 3

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers, 2, block_size, head_size)
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        if head_size % 8 != 0:
            return False
        if head_size <= 256:
            return True
        if is_fa_version_supported(4):
            return head_size <= 512
        return False

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if is_quantized_kv_cache(kv_cache_dtype):
            return flash_attn_supports_fp8()
        return kv_cache_dtype in ["auto", "float16", "bfloat16"]

    @classmethod
    def supports_sink(cls) -> bool:
        if not is_flash_attn_varlen_func_available():
            return False
        return flash_attn_supports_sinks()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if has_sink and device_capability < DeviceCapability(9, 0):
            return "sink not supported on compute capability < 9.0"
        return None


@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # For GQA DCP
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    causal: bool = True

    # Cold-KV CPU partial attention (IDE_006 / TSK_002). When enabled, the
    # model_runner routes the cold prefix of each request's block_table
    # through the TSK_001 CPU partial-attention kernel and the hot suffix
    # through the standard GPU flash_attn path, merging via
    # merge_attn_states. Defaults keep the schema bit-identical to the
    # pre-IDE_006 path.
    enable_hot_cold_split: bool = False
    num_cold_blocks: torch.Tensor | None = None
    # Phase 4c additions — populated by FlashAttentionMetadataBuilder
    # only when enable_hot_cold_split is True. cold_cpu_block_ids
    # carries each request's CPU canonical-buffer block IDs for its
    # cold prefix (padded to max_cold_blocks_per_req), aligned with
    # the same per-request order as block_table / num_cold_blocks.
    # query_positions is the absolute sequence position of each query
    # token, used by the CPU partial-attention kernel's causal mask.
    # max_num_cold_blocks_host is the host-side max of num_cold_blocks
    # captured once at build time; the per-layer dispatcher uses this
    # to skip a per-call GPU→CPU sync.
    cold_cpu_block_ids: torch.Tensor | None = None
    query_positions: torch.Tensor | None = None
    max_num_cold_blocks_host: int = 0


def _get_sliding_window_configs(
    vllm_config: VllmConfig,
) -> set[tuple[int, int] | None]:
    """Get the set of all sliding window configs used in the model."""
    sliding_window_configs: set[tuple[int, int] | None] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        assert isinstance(layer.impl, FlashAttentionImpl)
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    # FA3:
    # Supports full cudagraphs for all cases.
    #
    # FA2:
    # For FA2, a graph is captured with max_query_len=1, (which is what we
    # capture by default for num_tokens <= max_num_seqs when there is no
    # spec-decode) then these graphs will not work for mixed prefill-decode
    # (unlike FA3). This is due to special max_query_len=1 packed-GQA handling
    # in FA2.
    # In summary if we are running with spec decodes the graphs would
    # work for mixed prefill-decode and uniform-decode. But for non-spec decodes
    # the graphs would not work for mixed prefill-decode; sorta the inverse
    # of UNIFORM_SINGLE_TOKEN_DECODE.
    # There's probably a better way to describe this using `AttentionCGSupport`
    # but for now just set it to `UNIFORM_BATCH` to get use to drop down
    # to FULL_AND_PIECEWISE.
    # TODO(luka, lucas): audit FA2 as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )
    supports_update_block_table: bool = True

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.attention_config = vllm_config.attention_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.aot_schedule = get_flash_attn_version() == 3

        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0

        self.cp_kv_cache_interleave_size = (
            self.parallel_config.cp_kv_cache_interleave_size
        )

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        if self.use_full_cuda_graph and self.aot_schedule:
            # FA3 scheduler_metadata size: 1 + round_up(batch_size, 4) * 4
            # The +1 is for the tile_count_semaphore (synchronization).
            # The 4 slots per batch element (num_prepare_batch_vectors) are:
            #   prepare_varlen + dynamic_split + sort_batches + head_swizzle
            # See: https://github.com/vllm-project/flash-attention/blob/5824e6e/hopper/flash_api.cpp#L664-L671  # noqa: E501
            max_batch_size = max(
                vllm_config.scheduler_config.max_num_seqs,
                self.max_cudagraph_size or 0,
            )
            self.scheduler_metadata = torch.zeros(
                1 + round_up(max_batch_size, 4) * 4,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = (
                self.attention_config.flash_attn_max_num_splits_for_cuda_graph
            )

        if self.dcp_world_size > 1:
            max_num_reqs = vllm_config.scheduler_config.max_num_seqs
            self._dcp_context_kv_lens = torch.zeros(
                max_num_reqs,
                dtype=torch.int32,
                device=self.device,
            )

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        num_cold_blocks: torch.Tensor | None = None,
        enable_hot_cold_split: bool = False,
        cold_cpu_block_ids: torch.Tensor | None = None,
        query_positions: torch.Tensor | None = None,
        max_num_cold_blocks_host: int = 0,
    ) -> FlashAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode

        num_cold_blocks / enable_hot_cold_split / cold_cpu_block_ids /
        query_positions are the IDE_006 / TSK_002 Cold-KV CPU partial
        attention inputs; all default-off so existing callers are
        unchanged. They are forwarded directly to FlashAttentionMetadata.
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # Disable AOT schedule for spec-decode proposer (not worth the overhead)
        # and for batch invariance (schedule varies with max_seqlen_q/k).
        aot_schedule = (
            self.aot_schedule and not fast_build and not envs.VLLM_BATCH_INVARIANT
        )

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        max_num_splits = 0  # 0 means use FA3's heuristics, not CG compatible
        if (
            self.use_full_cuda_graph
            and self.max_cudagraph_size is not None
            and num_actual_tokens <= self.max_cudagraph_size
        ):
            # NOTE(woosuk): Setting num_splits > 1 may increase the memory
            # usage, because the intermediate buffers of size [num_splits,
            # num_heads, num_tokens, head_size] are allocated. Therefore,
            # we only set num_splits when using cuda graphs.
            max_num_splits = self.max_num_splits

        if envs.VLLM_BATCH_INVARIANT:
            max_num_splits = 1

        def schedule(
            batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
        ):
            cache_dtype = self.cache_config.cache_dtype
            if is_quantized_kv_cache(cache_dtype):
                qkv_dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                    cache_dtype
                )
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q * self.dcp_world_size,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=self.aot_sliding_window,
                    num_splits=max_num_splits,
                )
            return None

        use_cascade = common_prefix_len > 0
        max_dcp_context_kv_len = 0
        dcp_context_kv_lens = None

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None

        if self.dcp_world_size > 1:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            context_kv_lens = seq_lens - query_lens
            local_context_kv_lens = get_dcp_local_seq_lens(
                context_kv_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            self._dcp_context_kv_lens[:num_reqs] = local_context_kv_lens
            self._dcp_context_kv_lens[num_reqs:] = 0
            dcp_context_kv_lens = self._dcp_context_kv_lens[:num_reqs]

            # After DCP distribution, the maximum number of tokens for any rank is
            # ceil(L / (N * I)) * I, where L is max_seq_len, N is dcp_world_size,
            # and I is cp_kv_cache_interleave_size.
            # This eliminates GPU->CPU sync while minimizing workspace over-allocation.
            num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
            max_dcp_context_kv_len = (
                (max_seq_len + num_partitions - 1) // num_partitions
            ) * self.cp_kv_cache_interleave_size

            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=dcp_context_kv_lens,
                max_seq_len=max_dcp_context_kv_len,
                causal=False,
            )
        elif use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            # Use GPU tensor directly - no CPU sync needed
            suffix_kv_lens = seq_lens[:num_reqs] - common_prefix_len
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False,
            )
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=suffix_kv_lens,
                max_seq_len=max_seq_len - common_prefix_len,
                causal=True,
            )
        else:
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=seq_lens,
                max_seq_len=max_seq_len,
                causal=causal,
            )
        # For FA3 + full cudagraph
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
            enable_hot_cold_split=enable_hot_cold_split,
            num_cold_blocks=num_cold_blocks,
            cold_cpu_block_ids=cold_cpu_block_ids,
            query_positions=query_positions,
            max_num_cold_blocks_host=max_num_cold_blocks_host,
        )
        return attn_metadata

    def update_block_table(
        self,
        metadata: FlashAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> FlashAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return use_cascade_attention(*args, **kwargs)


class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=alibi_slopes is not None,
            head_size=head_size,
        )
        # head_size > 256 requires FA4 on SM90+; force upgrade from FA3
        if (
            head_size > 256
            and self.vllm_flash_attn_version == 3
            and current_platform.is_cuda()
            and current_platform.is_device_capability_family(90)
        ):
            self.vllm_flash_attn_version = 4
        logger.info_once(
            "Using FlashAttention version %s",
            self.vllm_flash_attn_version,
            scope="local",
        )
        # Cache the batch invariant result for use in forward passes
        self.batch_invariant_enabled = envs.VLLM_BATCH_INVARIANT

        if is_quantized_kv_cache(self.kv_cache_dtype) and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device."
            )

        self.sinks = sinks
        if self.sinks is not None:
            assert flash_attn_supports_sinks(), (
                "Sinks are only supported in FlashAttention 3"
            )
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

        self.supports_quant_query_input = flash_attn_supports_quant_query_input()

        vllm_config = get_current_vllm_config_or_none()
        dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        self.dcp_combine = dcp_a2a_lse_reduce if dcp_a2a else cp_lse_ag_out_rs

        self._dcp_dtype: torch.dtype | None = None
        if vllm_config is not None and self.dcp_world_size > 1:
            self._dcp_dtype = vllm_config.model_config.dtype

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(0)

        if is_quantized_kv_cache(self.kv_cache_dtype):
            # queries are quantized in the attention layer
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        # Cold-KV CPU partial attention dispatcher (IDE_006 / TSK_002
        # Phase 4c). Active only when (a) the metadata builder set
        # `enable_hot_cold_split=True` (requires
        # `KVTransferConfig.enable_cpu_partial_attention` opt-in plus a
        # populated OffloadingConnectorMetadata) AND (b) at least one
        # request in the batch actually has a cold prefix. When the
        # batch happens to have no cold blocks, fall through to the
        # standard hot path so we avoid the extra dispatch overhead and
        # the hot_cold_attention internal `output.copy_(hot_output)`
        # short-circuit.
        if (
            attn_metadata.enable_hot_cold_split
            and attn_metadata.max_num_cold_blocks_host > 0
        ):
            from vllm.v1.attention.ops.kv_view_adapter import KVPageLayout

            if not has_kv_transfer_group():
                raise RuntimeError(
                    "enable_hot_cold_split is True but no KV transfer "
                    "group is registered; check kv_transfer_config / "
                    "KVConnector setup."
                )
            connector = get_kv_transfer_group()
            cpu_kv_cache = connector.get_cpu_kv_buffer_for_layer(layer.layer_name)
            if cpu_kv_cache is None:
                raise RuntimeError(
                    "Connector did not surface CPU KV buffer for layer "
                    f"{layer.layer_name!r}; cold path cannot proceed. "
                    "Either disable enable_cpu_partial_attention or use "
                    "a connector that overrides "
                    "get_cpu_kv_buffer_for_layer (e.g. OffloadingConnector)."
                )

            block_size_value = key_cache.shape[1]
            cold_kv_layout = KVPageLayout(
                head_dim=self.head_size,
                num_kv_heads=self.num_kv_heads,
                block_size=block_size_value,
                dtype=key_cache.dtype,
            )

            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            num_cold_blocks_t = attn_metadata.num_cold_blocks
            # Use the host-side max captured by the metadata builder so
            # per-layer dispatch avoids a GPU→CPU sync. The builder
            # records this in `max_num_cold_blocks_host` from the same
            # numpy array it builds `num_cold_blocks` from. When the
            # tensor is None / empty, fall back to 0 — the dispatcher
            # below short-circuits to the standard hot path then.
            max_num_cold_blocks = attn_metadata.max_num_cold_blocks_host

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)
            q_descale = (
                layer._q_scale.expand(descale_shape)
                if self.supports_quant_query_input
                else None
            )
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)

            hot_cold_attention(
                output=output[:num_actual_tokens],
                query=query[:num_actual_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                cu_query_lens=cu_seqlens_q,
                max_query_len=attn_metadata.max_query_len,
                seqused_k=seqused_k,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                sliding_window=self.sliding_window,
                logits_soft_cap=self.logits_soft_cap,
                block_table=attn_metadata.block_table,
                block_size=block_size_value,
                num_cold_blocks=num_cold_blocks_t,
                max_num_cold_blocks=max_num_cold_blocks,
                fa_version=self.vllm_flash_attn_version,
                causal=attn_metadata.causal,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                cpu_kv_cache=cpu_kv_cache,
                cold_kv_layout=cold_kv_layout,
                cold_block_ids=attn_metadata.cold_cpu_block_ids,
                query_positions=attn_metadata.query_positions,
            )
            return output

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            q_descale = (
                layer._q_scale.expand(descale_shape)
                if self.supports_quant_query_input
                else None
            )
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)

            if self.dcp_world_size > 1:
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                return output
            else:
                sliding_window_size = (
                    list(self.sliding_window)
                    if self.sliding_window is not None
                    else None
                )
                flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=sliding_window_size,
                    block_table=block_table,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=scheduler_metadata,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    num_splits=attn_metadata.max_num_splits,
                    s_aux=self.sinks,
                )
                return output

        # Cascade attention (rare case).
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return

        key_cache, value_cache = kv_cache.unbind(0)

        # Reshape the input keys and values and store them in the cache.
        # Skip this if sharing KV cache with an earlier attention layer.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def _forward_with_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        q_descale: torch.Tensor | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        block_table = attn_metadata.block_table

        query = query.contiguous()
        query_across_dcp = get_dcp_group().all_gather(query, dim=1)
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        n = query_across_dcp.shape[0]
        (dcp_context_out,) = current_workspace_manager().get_simultaneous(
            (
                (n, self.num_heads * self.dcp_world_size, self.head_size),
                self._dcp_dtype,
            ),
        )
        context_attn_out, context_lse = flash_attn_varlen_func(
            q=query_across_dcp,
            k=key_cache,
            v=value_cache,
            out=dcp_context_out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=attn_metadata.dcp_context_kv_lens,
            max_seqlen_k=attn_metadata.max_dcp_context_kv_len,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
        )
        # FA returns LSE in shape [ H, B ] but DCP combine wants [ B, H ]
        context_attn_out_cor, context_lse_cor = self.dcp_combine(
            context_attn_out,
            context_lse.transpose(0, 1),
            get_dcp_group(),
            return_lse=True,
        )
        context_lse_cor = context_lse_cor.transpose(0, 1).contiguous()

        (dcp_query_out,) = current_workspace_manager().get_simultaneous(
            ((query.shape[0], self.num_heads, self.head_size), self._dcp_dtype),
        )
        query_attn_out, query_lse = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=dcp_query_out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
        )
        assert context_attn_out_cor.shape == query_attn_out.shape
        assert context_lse_cor.shape == query_lse.shape
        merge_attn_states(
            output,
            context_attn_out_cor,
            context_lse_cor,
            query_attn_out,
            query_lse,
        )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        # For encoder attention, process FP8 quantization if needed
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,  # type: ignore[union-attr]
            self.num_kv_heads,
        )

        # Call flash attention directly on Q, K, V tensors
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,  # Encoder attention is bidirectional
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape)
            if self.supports_quant_query_input
            else None,
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=1 if self.batch_invariant_enabled else 0,
        )

        return output


def use_cascade_attention(
    common_prefix_len: int,
    query_lens: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi: bool,
    use_sliding_window: bool,
    use_local_attention: bool,
    num_sms: int,
    dcp_world_size: int,
) -> bool:
    """Decide whether to use cascade attention.

    This function 1) checks whether cascade attention is supported with the
    given configuration, and 2) heuristically decides whether using cascade
    attention can improve performance.
    """
    # Too short common prefix. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
    # NOTE(woosuk): This is the common case. We should return False as soon as
    # possible to avoid any unnecessary computation.
    if common_prefix_len < 256:
        return False
    # Cascade attention is currently not supported with these variants.
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    # Too few queries. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False
    # disable cascade attention for DCP
    if dcp_world_size > 1:
        return False

    # Heuristics to decide whether using cascade attention is beneficial.
    # 1. When FlashDecoding is not used for normal attention, cascade attention
    #    is likely to be faster since it saves memory bandwidth.
    num_queries_per_kv = num_query_heads // num_kv_heads
    # The criteria for using FlashDecoding can be found in the following link:
    # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
    use_flash_decoding = (
        num_queries_per_kv > 1
        and not use_sliding_window
        and not use_alibi
        and np.all(query_lens == 1)
    )
    if not use_flash_decoding:
        # Use cascade attention.
        return True

    # 2. When FlashDecoding is used for normal attention, it is not clear
    #    whether cascade attention is beneficial, because FlashDecoding can
    #    launch more CTAs than cascade attention.
    #    We use a simple performance model to compare the two methods.
    #    NOTE(woosuk): The performance model is very rough and may not be
    #    accurate.
    num_tokens = num_reqs
    # NOTE(woosuk): These are default tile sizes. flash-attn might use
    # different tile sizes (e.g., 64 or 256) depending on the configuration.
    q_tile_size = 128
    kv_tile_size = 128
    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = (
        num_reqs * num_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
    )
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    # Use cascade attention if it is faster than FlashDecoding.
    return cascade_time < flash_decoding_time


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    max_num_splits: int,
    fa_version: int,
    prefix_scheduler_metadata: torch.Tensor | None = None,
    suffix_scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    assert alibi_slopes is None, "Cascade attention does not support ALiBi."
    # TODO: Support sliding window.
    assert sliding_window == (-1, -1), (
        "Cascade attention does not support sliding window."
    )

    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0
    descale_shape = (cu_prefix_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process shared prefix.
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        seqused_k=prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=list(sliding_window),
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=prefix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
        # s_aux is incorporated into prefix_lse inside the GPU kernel,
        # enabling its effect during the final attention merge.
        s_aux=s_aux,
        num_splits=1 if envs.VLLM_BATCH_INVARIANT else max_num_splits,
    )

    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process suffix per query.
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=list(sliding_window),
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=suffix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
        num_splits=1 if envs.VLLM_BATCH_INVARIANT else max_num_splits,
    )

    # Merge prefix and suffix outputs, and store the result in output.
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)


# IDE_006 / TSK_004 — cold-path firing breadcrumb. Per-process counter
# that emits a stderr line on the first few firings only so an operator
# can confirm in the run log that cold path actually executed without
# burying the rest of the output. After the limit we set a "done" flag
# so subsequent calls do not even touch the global int — the per-call
# cost collapses to a single bool read.
_COLD_PATH_FIRING_COUNT = 0
_COLD_PATH_FIRING_LOG_LIMIT = 5
_COLD_PATH_FIRING_LOG_DONE = False


# IDE_006 / TSK_004 §4.6 — dedicated CUDA stream for cold-path GPU
# operations (D2H of reduced query rows, H2D of reduced cold result,
# scatter into full-size). Hot path (flash_attn_varlen_func) stays on
# the default stream. With these on different streams the GPU can
# execute the hot kernel concurrently while CPU does partial-attn work
# on the host side, satisfying PLN_001 §4.3 's overlap design intent
# (T_Q_transfer + T_CPU_partial + T_partial_transfer ≤ T_GPU_hot_attn).
# Lazy + cached per device index. Operator opt-out via
# ``VLLM_COLD_KV_DISABLE_OVERLAP=1`` falls back to single-stream
# (sequential) execution — useful for A/B comparison or debugging.
_COLD_STREAM_CACHE: dict[int, "torch.cuda.Stream | None"] = {}
_COLD_STREAM_OVERLAP_DISABLED = bool(
    os.environ.get("VLLM_COLD_KV_DISABLE_OVERLAP", "")
) and os.environ["VLLM_COLD_KV_DISABLE_OVERLAP"] not in ("0", "")


# IDE_006 / TSK_004 — reusable cold scatter buffers (deferred item 4).
# Previously each ``hot_cold_attention`` call allocated 32 MB
# ``cold_output_gpu`` + 512 KB ``cold_lse_gpu`` and ran zero/full
# kernels to pre-fill them. With heavy workload (80 layer × decode
# steps × 8 worker), the per-call alloc + fill kernels add up. Since
# the row dimension (``num_tokens``) is bounded by the engine's max
# in-flight token count, we cache reusable buffers per (device,
# dtype) and reset only the rows we wrote on the previous call. The
# rest stay at -inf for cold_lse and arbitrary for cold_output (merge
# only reads cold rows where cold_lse > -inf).
_COLD_SCATTER_BUFS: dict[
    tuple[int, int, int, torch.dtype, torch.dtype],
    "tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]",
] = {}


def _get_cold_scatter_buffers(
    device: torch.device,
    num_tokens: int,
    num_q_heads: int,
    head_dim: int,
    output_dtype: torch.dtype,
    lse_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, "torch.Tensor | None"]:
    """Return ``(cold_output_buf, cold_lse_buf, last_dirty_idx)``.

    The buffers are keyed by (device, num_q_heads, head_dim,
    output_dtype, lse_dtype) — those dimensions are fixed by the
    model config so the worker only allocates them once. ``num_tokens``
    is the row count and varies per call; the buffer grows in place
    when needed but never shrinks. ``last_dirty_idx`` is the index
    tensor written on the previous call so the caller can restore
    -inf for those rows before writing new values, keeping the rest
    of cold_lse_buf at -inf without a full-tensor fill kernel.
    """
    dev_idx = device.index if device.index is not None else 0
    key = (dev_idx, num_q_heads, head_dim, output_dtype, lse_dtype)
    entry = _COLD_SCATTER_BUFS.get(key)
    if entry is None or entry[0].size(0) < num_tokens:
        cold_output_buf = torch.empty(
            (num_tokens, num_q_heads, head_dim),
            dtype=output_dtype,
            device=device,
        )
        cold_lse_buf = torch.full(
            (num_q_heads, num_tokens),
            float("-inf"),
            dtype=lse_dtype,
            device=device,
        )
        entry = (cold_output_buf, cold_lse_buf, None)
        _COLD_SCATTER_BUFS[key] = entry
    return entry


def _set_cold_scatter_dirty(
    device: torch.device,
    num_q_heads: int,
    head_dim: int,
    output_dtype: torch.dtype,
    lse_dtype: torch.dtype,
    dirty_idx: torch.Tensor,
) -> None:
    """Record the row indices written on this call so the next call
    can restore -inf for them before writing new values."""
    dev_idx = device.index if device.index is not None else 0
    key = (dev_idx, num_q_heads, head_dim, output_dtype, lse_dtype)
    entry = _COLD_SCATTER_BUFS[key]
    _COLD_SCATTER_BUFS[key] = (entry[0], entry[1], dirty_idx)


def _get_cold_path_stream(device: torch.device) -> "torch.cuda.Stream | None":
    """Return a dedicated CUDA stream for cold-path GPU ops on ``device``.

    Returns ``None`` if the device is CPU or overlap is disabled — caller
    falls back to running cold-path GPU ops on the default stream
    (same behaviour as before §4.6)."""
    if _COLD_STREAM_OVERLAP_DISABLED or device.type != "cuda":
        return None
    idx = device.index if device.index is not None else 0
    s = _COLD_STREAM_CACHE.get(idx)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _COLD_STREAM_CACHE[idx] = s
    return s


def hot_cold_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    block_size: int,
    num_cold_blocks: torch.Tensor,
    max_num_cold_blocks: int,
    fa_version: int,
    causal: bool = True,
    suffix_scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    cpu_kv_cache: list[torch.Tensor] | None = None,
    cold_kv_layout: "KVPageLayout | None" = None,  # noqa: F821
    cold_block_ids: torch.Tensor | None = None,
    query_positions: torch.Tensor | None = None,
) -> None:
    """Per-sequence variable-length cold-prefix attention (IDE_006 / TSK_002 §4.4).

    The "cold" prefix of each request's block_table has been evicted to
    CPU by the OffloadingConnector; the corresponding GPU block_table
    rows are no longer valid. The "hot" suffix stays on GPU. This
    function runs the hot part via flash_attn_varlen_func with the
    block_table sliced to drop the cold columns and seqused_k clipped
    to drop the cold KV tokens, and (Phase 3b) runs the cold part via
    TSK_001's forward_partial_with_lse on CPU. The two outputs are
    merged via merge_attn_states.

    Phase 3a (current): GPU hot path only. When max_num_cold_blocks
    is 0, this is bit-identical to a plain flash_attn_varlen_func
    call (no merge needed). When max_num_cold_blocks > 0, the cold
    path is not yet wired and the function raises
    NotImplementedError — Phase 3b will fill that in.

    The caller is expected to compute max_num_cold_blocks on the host
    side (e.g. from the OffloadingConnectorMetadata dict before
    building the device tensor) so this function does not introduce
    a GPU→CPU sync.
    """
    assert max_num_cold_blocks >= 0
    assert sliding_window == (-1, -1), (
        "hot_cold_attention does not support sliding window yet."
    )

    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Hot block_table: per-sequence shift to drop the cold prefix.
    # A naive column slice `block_table[:, max_num_cold_blocks:]` only works
    # when every sequence in the batch has the same cold count; if seq i
    # has num_cold_blocks[i] < max_num_cold_blocks, the slice would drop
    # `max_num_cold_blocks - num_cold_blocks[i]` valid hot blocks for that
    # sequence. The per-row gather below shifts each row by exactly its own
    # num_cold_blocks[i] and pads the tail with the original block_table's
    # padding (NULL block IDs) — flash_attn never reads past seqused_k, so
    # that pad is harmless.
    if max_num_cold_blocks == 0:
        hot_block_table = block_table
    else:
        # We allocate the full original block_table width as the hot column
        # budget. A naive `max_blocks - max_num_cold_blocks` width fails when
        # one sequence is 100% cold (max_num_cold_blocks == max_blocks) but
        # another sequence has 0 cold and therefore needs the full width of
        # hot blocks; that case would compute width 0 and lose every
        # sequence's hot KV. Using the full width is safe because
        # hot_seqused_k clips the kernel's reach per-sequence and any slot
        # that ends up beyond the row's real hot region is harmless.
        num_seqs_bt, max_blocks_bt = block_table.shape
        max_hot_blocks_bt = max_blocks_bt
        row_idx = torch.arange(
            num_seqs_bt, device=block_table.device
        ).unsqueeze(1)
        col_offsets = torch.arange(
            max_hot_blocks_bt, device=block_table.device
        ).unsqueeze(0)
        # Cast num_cold_blocks to the same dtype as the index arange so the
        # broadcast add doesn't materialise a different (larger) dtype.
        col_idx = num_cold_blocks.to(col_offsets.dtype).unsqueeze(1) + col_offsets
        col_idx_clamped = col_idx.clamp_max(max_blocks_bt - 1)
        hot_block_table = block_table[row_idx, col_idx_clamped]

    # Per-sequence hot KV length = total seq KV length − cold KV tokens.
    # clamp to 0 so a sequence with all-hot (num_cold_blocks[i]==0) is unaffected.
    cold_kv_tokens = num_cold_blocks * block_size
    hot_seqused_k = (seqused_k - cold_kv_tokens).clamp_(min=0)

    # Hot path's max KV length must be ≥ the maximum hot_seqused_k across the
    # batch, which is `max_seqlen_k - min(num_cold_blocks) * block_size`.
    # Computing `min(num_cold_blocks)` on-device would force a sync; instead
    # we conservatively use the original `max_seqlen_k` as the upper bound.
    # The flash_attn kernel respects per-sequence seqused_k so the only cost
    # of the looser bound is a slightly larger kernel workspace allocation.
    hot_max_seqlen_k = max_seqlen_k

    hot_output, hot_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=hot_seqused_k,
        max_seqlen_q=max_query_len,
        max_seqlen_k=hot_max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=list(sliding_window),
        block_table=hot_block_table,
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=suffix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
    )

    if max_num_cold_blocks == 0:
        # Degenerate: no cold blocks across the batch. The hot path produced
        # the full attention; copy into output and return. merge_attn_states
        # is not needed because there is no second tensor to merge.
        output.copy_(hot_output)
        return

    # ----- Phase 3b: cold path on CPU + LSE merge on GPU ---------------
    # Required cold inputs (caller — Phase 4 — supplies them from the
    # OffloadingConnector worker state and the kv_cache_spec).
    if (
        cpu_kv_cache is None
        or cold_kv_layout is None
        or cold_block_ids is None
        or query_positions is None
    ):
        raise ValueError(
            "hot_cold_attention requires cpu_kv_cache, cold_kv_layout, "
            "cold_block_ids, and query_positions when max_num_cold_blocks "
            f"> 0 (got max_num_cold_blocks={max_num_cold_blocks}). These "
            "are populated by the model_runner dispatcher (Phase 4) from "
            "the OffloadingConnector worker state and kv_cache_spec."
        )

    # cpu_kv_cache is a list of canonical int8 buffers per layer.
    # len == 1 → combined K+V layout (e.g. mamba). len == 2 → split K
    # and V (FlashAttention's OffloadingConnector mirror — first entry
    # is K-only, second is V-only). Any other length is a bug in the
    # caller's plumbing.
    if len(cpu_kv_cache) == 1:
        cold_kv_combined = cpu_kv_cache[0]
        cold_kv_v_split = None
    elif len(cpu_kv_cache) == 2:
        cold_kv_combined = cpu_kv_cache[0]
        cold_kv_v_split = cpu_kv_cache[1]
    else:
        raise ValueError(
            "cpu_kv_cache must be a list of 1 (combined K+V) or 2 (split "
            "K, V) int8 canonical buffers; got list of length "
            f"{len(cpu_kv_cache)}."
        )

    # Diagnostic instrumentation. Gated by VLLM_PARTIAL_ATTN_PROFILE; takes
    # GPU sync hits to make the wall-time numbers honest. Off by default.
    if _PARTIAL_PROFILE_ENABLED:
        import time as _time
        if query.device.type == "cuda":
            torch.cuda.synchronize(query.device)
        _t_d2h0 = _time.perf_counter()

    # ----- IDE_006 / TSK_004 — per-seq filter ----------------------------
    # Only rows belonging to seqs with ``num_cold_blocks > 0`` need the CPU
    # cold-prefix attention; rows of cold-less seqs are full-attention
    # already covered by the GPU hot path. Without this filter we D2H the
    # entire batch query (e.g. 16384 × 8 × 128 × 2 = 32 MB), allocate a
    # 32 MB output tensor in the C++ kernel for the kernel's outer loop to
    # immediately skip, and H2D 32 MB back. With the filter we forward
    # only the rows that actually need cold attention; the kernel's inputs
    # and outputs scale to the cold-needing token count instead of the
    # whole batch.
    #
    # Step 1: small D2H of the per-seq metadata that decides which seqs
    # need cold processing. These are O(num_seqs) tensors, ≤ 1 KB.
    # ``.cpu()`` is a no-op when the tensor is already on CPU.
    num_cold_blocks_cpu = num_cold_blocks.cpu()
    cu_query_lens_cpu = cu_query_lens.cpu()
    seq_lens_total_cpu = seqused_k.cpu()

    cu_q_list = cu_query_lens_cpu.tolist()
    n_cold_list = num_cold_blocks_cpu.tolist()
    need_cold_seq_ids = [i for i, n in enumerate(n_cold_list) if n > 0]

    # Should not happen — ``max_num_cold_blocks > 0`` was verified above —
    # but guard so we don't D2H an empty batch into the kernel.
    if not need_cold_seq_ids:
        output.copy_(hot_output)
        return

    # Step 2: build a row-index tensor on the host that maps the
    # need-cold tokens back to their positions in the full batch. We
    # keep it on CPU first; the H2D copy is 4 bytes × reduced_n_tokens,
    # which is much smaller than the query D2H it replaces.
    reduced_token_ids: list[int] = []
    reduced_cu_list: list[int] = [0]
    for i in need_cold_seq_ids:
        qs, qe = cu_q_list[i], cu_q_list[i + 1]
        reduced_token_ids.extend(range(qs, qe))
        reduced_cu_list.append(reduced_cu_list[-1] + (qe - qs))

    # Pure-decode small batches (1 token per cold seq) are the common
    # case after this filter.
    reduced_token_idx_cpu = torch.tensor(reduced_token_ids, dtype=torch.long)

    # Cold-path firing breadcrumb — first few firings per worker only.
    # After the limit ``_COLD_PATH_FIRING_LOG_DONE`` short-circuits the
    # branch to a single global bool read, no string format, no stderr
    # write. Per-call overhead collapses to ~10 ns once limit reached.
    global _COLD_PATH_FIRING_COUNT, _COLD_PATH_FIRING_LOG_DONE
    if not _COLD_PATH_FIRING_LOG_DONE:
        _COLD_PATH_FIRING_COUNT += 1
        import sys as _sys
        print(
            f"[IDE_006/TSK_004 cold-path fired pid={os.getpid()}] "
            f"#{_COLD_PATH_FIRING_COUNT}/{_COLD_PATH_FIRING_LOG_LIMIT} "
            f"need_cold_seqs={len(need_cold_seq_ids)}/{len(n_cold_list)} "
            f"reduced_rows={len(reduced_token_ids)}/{cu_q_list[-1]} "
            f"max_cold_blocks={max_num_cold_blocks}",
            file=_sys.stderr,
            flush=True,
        )
        if _COLD_PATH_FIRING_COUNT >= _COLD_PATH_FIRING_LOG_LIMIT:
            _COLD_PATH_FIRING_LOG_DONE = True

    # ----- §4.6 GPU/CPU overlap setup ------------------------------------
    # Cold-path GPU operations (index_select-then-D2H, H2D-then-scatter)
    # are issued on a dedicated CUDA stream so they run concurrently
    # with the hot-path FA kernel on the default stream. CPU partial-
    # attention then overlaps with the FA kernel via Python's natural
    # pause during the synchronous .cpu() / forward_partial_with_lse
    # calls — while Python is blocked, default stream's FA continues.
    # Without this overlap (single-stream sequential), prod measured
    # 5.6× slowdown vs baseline (cold_verify run @ 0a9d313288). With
    # overlap, wall-clock collapses to max(T_FA, T_d2h+T_cpu+T_h2d).
    device = hot_output.device
    cold_stream = _get_cold_path_stream(device)
    cold_stream_ctx = (
        torch.cuda.stream(cold_stream)
        if cold_stream is not None
        else nullcontext()
    )

    # Step 3: targeted index_select on the cold stream. Each input
    # tensor may be on CPU or GPU independently. ``.cpu()`` is no-op on
    # CPU tensors, so an unconditional pass through is safe.
    need_cold_seq_idx_cpu = torch.tensor(need_cold_seq_ids, dtype=torch.long)

    with cold_stream_ctx:
        if query.device.type == "cuda":
            row_idx_dev = reduced_token_idx_cpu.to(query.device, non_blocking=True)
            reduced_query_cpu = query.index_select(0, row_idx_dev).cpu()
        else:
            reduced_query_cpu = query.index_select(0, reduced_token_idx_cpu)
        if query_positions.device.type == "cuda":
            qp_idx_dev = reduced_token_idx_cpu.to(query_positions.device, non_blocking=True)
            reduced_qpos_cpu = query_positions.index_select(0, qp_idx_dev).cpu()
        else:
            reduced_qpos_cpu = query_positions.index_select(0, reduced_token_idx_cpu)
        if cold_block_ids.device.type == "cuda":
            cbi_idx_dev = need_cold_seq_idx_cpu.to(cold_block_ids.device, non_blocking=True)
            reduced_cbi = cold_block_ids.index_select(0, cbi_idx_dev).cpu()
        else:
            reduced_cbi = cold_block_ids.index_select(0, need_cold_seq_idx_cpu)

    # num_cold_blocks_cpu / seq_lens_total_cpu were already mirrored to
    # CPU at the top of the function (small tensors). Indexing on CPU
    # here is consistent with that — no GPU op needed.
    reduced_cbl = num_cold_blocks_cpu.index_select(0, need_cold_seq_idx_cpu)
    reduced_sl = seq_lens_total_cpu.index_select(0, need_cold_seq_idx_cpu)
    reduced_cu_cpu = torch.tensor(reduced_cu_list, dtype=torch.int32)

    if _PARTIAL_PROFILE_ENABLED:
        _t_d2h1 = _time.perf_counter()
        _t_kernel0 = _t_d2h1

    # Step 5: NEO-style async issue of CPU partial-attention. The
    # background thread starts the C++ kernel immediately; control
    # returns to Python on the main thread *without* blocking. This
    # frees the main thread to (a) wait on hot-path FA via
    # cold_stream sync, (b) build full-size scatter buffers on GPU,
    # (c) — at the model_runner level — start the next layer's
    # GPU work. The future is awaited just before merge below.
    cold_future = forward_partial_with_lse_async(
        query=reduced_query_cpu,
        cold_kv_cache=cold_kv_combined,
        cold_kv_layout=cold_kv_layout,
        cold_block_ids=reduced_cbi,
        cold_block_lens=reduced_cbl,
        cu_seqlens_q=reduced_cu_cpu,
        seq_lens_total=reduced_sl,
        query_positions=reduced_qpos_cpu,
        softmax_scale=softmax_scale,
        causal=causal,
        cold_kv_cache_v=cold_kv_v_split,
    )

    # Block here only at result-needed time. While we're waiting the
    # background thread is doing useful CPU work and the default CUDA
    # stream is finishing its hot-path FA kernel concurrently.
    cold_output_reduced_cpu, cold_lse_reduced_cpu = cold_future.result()

    if _PARTIAL_PROFILE_ENABLED:
        _t_kernel1 = _time.perf_counter()
        _t_h2d0 = _t_kernel1

    # Step 6: H2D the reduced result + scatter into full-size tensors,
    # also on the cold stream so it overlaps with any remaining hot
    # path work. ``cold_lse`` for skipped rows is -inf so
    # merge_attn_states naturally drops them and the hot-path output
    # flows through unchanged for those rows.
    num_tokens = hot_output.size(0)
    num_q_heads = hot_output.size(1)
    head_dim = hot_output.size(2)

    with cold_stream_ctx:
        cold_output_reduced_gpu = cold_output_reduced_cpu.to(
            device=device, non_blocking=True
        )
        cold_lse_reduced_gpu = cold_lse_reduced_cpu.to(
            device=device, non_blocking=True
        )
        # Bring the row-index tensor to ``device`` for the scatter.
        # cheap; tensor is reduced_n int64 entries.
        if reduced_token_idx_cpu.device != device:
            reduced_token_idx_dev = reduced_token_idx_cpu.to(
                device=device, non_blocking=True
            )
        else:
            reduced_token_idx_dev = reduced_token_idx_cpu

        # Re-use module-cached cold scatter buffers (deferred item 4).
        # cold_output_buf: arbitrary (only need_cold rows are read by
        #   merge thanks to cold_lse=-inf gating elsewhere).
        # cold_lse_buf: kept at -inf except on the rows we wrote last
        #   call, which we now restore to -inf before writing new
        #   values. Total per-call work is O(reduced rows), not
        #   O(num_tokens).
        cold_output_buf, cold_lse_buf, last_dirty_idx = _get_cold_scatter_buffers(
            device,
            num_tokens,
            num_q_heads,
            head_dim,
            hot_output.dtype,
            hot_lse.dtype,
        )
        if last_dirty_idx is not None:
            # Reset previous call's dirty rows back to -inf (only those
            # rows; rest of the buffer was never overwritten).
            cold_lse_buf.index_fill_(1, last_dirty_idx, float("-inf"))
        cold_output_buf.index_copy_(
            0, reduced_token_idx_dev, cold_output_reduced_gpu
        )
        cold_lse_buf.index_copy_(
            1, reduced_token_idx_dev, cold_lse_reduced_gpu
        )
        _set_cold_scatter_dirty(
            device,
            num_q_heads,
            head_dim,
            hot_output.dtype,
            hot_lse.dtype,
            reduced_token_idx_dev,
        )
        # merge_attn_states needs same-shape inputs; if the cached buf
        # is larger than current num_tokens, take a [:num_tokens] view.
        if cold_output_buf.size(0) == num_tokens:
            cold_output_gpu = cold_output_buf
            cold_lse_gpu = cold_lse_buf
        else:
            cold_output_gpu = cold_output_buf[:num_tokens]
            cold_lse_gpu = cold_lse_buf[:, :num_tokens]

    # Sync the default stream with cold_stream so merge_attn_states (on
    # default) sees finalized cold tensors. wait_stream is event-based
    # and doesn't block Python — it just inserts a stream-side wait.
    if cold_stream is not None:
        torch.cuda.current_stream(device).wait_stream(cold_stream)

    if _PARTIAL_PROFILE_ENABLED:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        _t_h2d1 = _time.perf_counter()
        _t_merge0 = _t_h2d1

    merge_attn_states(
        output, hot_output, hot_lse, cold_output_gpu, cold_lse_gpu
    )

    if _PARTIAL_PROFILE_ENABLED:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        _t_merge1 = _time.perf_counter()
        if _partial_profile_should_emit():
            import sys as _sys
            print(
                f"[IDE_006/TSK_004 profile pid={os.getpid()} stage=hot_cold] "
                f"d2h_ms={(_t_d2h1 - _t_d2h0) * 1000:.2f} "
                f"kernel_ms={(_t_kernel1 - _t_kernel0) * 1000:.2f} "
                f"h2d_ms={(_t_h2d1 - _t_h2d0) * 1000:.2f} "
                f"merge_ms={(_t_merge1 - _t_merge0) * 1000:.2f} "
                f"total_ms={(_t_merge1 - _t_d2h0) * 1000:.2f} "
                f"q.shape={tuple(query.shape)} "
                f"reduced_q={tuple(reduced_query_cpu.shape)} "
                f"max_cold_blocks={max_num_cold_blocks}",
                file=_sys.stderr,
                flush=True,
            )
