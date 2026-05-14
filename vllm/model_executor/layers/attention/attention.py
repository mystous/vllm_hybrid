# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.config.vllm import VllmConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionType,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
    get_kv_quant_mode,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention import MLAAttention

logger = init_logger(__name__)


def validate_kv_sharing_target(
    current_layer_name, target_layer_name, static_forward_context
):
    error_msg = (
        f"Specified KV sharing target layer for {current_layer_name} "
        f"is not valid: target layer {target_layer_name} "
    )

    if current_layer_name == target_layer_name:
        raise ValueError(error_msg + "cannot be the same as the current layer.")

    if target_layer_name not in static_forward_context:
        from vllm.model_executor.models.utils import extract_layer_index

        # If target layer name is not in the static fwd context, it means either
        # a) the target layer does not come BEFORE the current layer, or
        # b) the target layer is not an Attention layer that exists in the model
        current_layer_idx = extract_layer_index(current_layer_name)
        target_layer_idx = extract_layer_index(target_layer_name)
        if current_layer_idx <= target_layer_idx:
            raise ValueError(error_msg + "must come before the current layer.")
        else:
            raise ValueError(error_msg + "is not a valid Attention layer in the model.")

    # Currently KV sharing is only supported between layers of the same type
    target_layer_attn_type = static_forward_context[target_layer_name].attn_type
    expected = static_forward_context[current_layer_name].attn_type
    if target_layer_attn_type != expected:
        raise ValueError(
            error_msg + f"must be the same type as the current layer ({expected})."
        )


def should_load_quant_weights(quant_method: QuantizeMethodBase | None) -> bool:
    """Returns whether the quantization method should load quantized weights."""
    return quant_method is not None and not isinstance(
        quant_method, UnquantizedLinearMethod
    )


def set_default_quant_scales(layer: nn.Module, register_buffer: bool = False) -> None:
    """Sets default quantization scales for the layer."""
    if register_buffer:
        layer.register_buffer("_k_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_v_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_q_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_prob_scale", torch.tensor(1.0, dtype=torch.float32))
    else:
        layer._k_scale.fill_(1.0)
        layer._v_scale.fill_(1.0)
        layer._q_scale.fill_(1.0)
        layer._prob_scale.fill_(1.0)

    # We also keep q/k/v_scale on host (cpu) memory for attention
    # backends that require the scales to be on host instead of on device.
    # e.g. Flashinfer
    layer._q_scale_float = 1.0
    layer._k_scale_float = 1.0
    layer._v_scale_float = 1.0
    layer._prob_scale_float = 1.0

    # Initialize q/k/v range constants used by calc_kv_scales
    layer.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
    layer.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
    layer.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)


def _init_kv_cache_quant(
    layer: nn.Module,
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> None:
    """Initializes KV cache scaling factors and quantization method.

    This helper function sets up the KV cache quantization attributes that are
    shared between Attention and MLAAttention layers. It initializes scale
    tensors for query, key, value, and probability, and configures the
    quantization method if applicable.

    Args:
        layer: The attention layer instance to initialize.
        quant_config: Optional quantization configuration.
        prefix: Layer name prefix for quantization method lookup.
    """

    # Note [Register q/k/v/prob scales in state dict]
    # When calling model.to(device), only parameters/buffers in state dict are
    # moved. If not registering q/k/v/prob scales in state dict, there would
    # be an IMA error when a cuda kernel (e.g., quant_fp8) accesses the tensor
    # on cpu.
    # Registering in state dict means it interacts with weight loading. One edge
    # case is when quant_method is None, or quant_method is UnquantizedLinearMethod
    # (i.e., should_load_quant_weights(quant_method) == False).
    # In this case, the checkpoint does not have the scales. We need to
    # initialize the scales to 1.0 and update the scales after weight loading.
    # This is espectially important when we load dummy weights first (providing
    # wrong scales) and then load real weights (which misses scales and keeps the
    # wrong scales from dummy load).
    set_default_quant_scales(layer, register_buffer=True)

    # The output scale on host memory. This should be the input scale of
    # the quant op after this attention layer.
    layer._o_scale_float = None

    quant_method = (
        quant_config.get_quant_method(layer, prefix=prefix) if quant_config else None
    )

    # See [Note: Register q/k/v/prob scales in state dict]
    if should_load_quant_weights(quant_method):
        assert isinstance(quant_method, BaseKVCacheMethod)
        # TODO (mgoin): kv cache dtype should be specified in the FP8
        # checkpoint config and become the "auto" behavior
        if layer.kv_cache_dtype == "fp8_e5m2":
            raise ValueError("fp8_e5m2 kv-cache is not supported with fp8 checkpoints.")
        # If quantization is enabled, we make "k_scale" and "v_scale"
        # parameters so that it can be loaded from the model checkpoint.
        # The k/v_scale will then be converted back to native float32
        # values after weight loading.
        layer.quant_method = quant_method
        layer.quant_method.create_weights(layer)


class Attention(nn.Module, AttentionLayerBase):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        use_alibi_sqrt: bool | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        attn_backend: type[AttentionBackend] | None = None,
        head_size_v: int | None = None,
        **extra_impl_args,
    ) -> None:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.
        """
        super().__init__()
        if per_layer_sliding_window is not None:
            # per-layer sliding window
            sliding_window = per_layer_sliding_window
        elif cache_config is not None:
            # model-level sliding window
            sliding_window = cache_config.sliding_window
        else:
            sliding_window = None

        vllm_config = get_current_vllm_config()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            calculate_kv_scales = False

        # llm-compressor mdls need to set cache_dtype to "fp8" manually.
        kv_cache_scheme = getattr(quant_config, "kv_cache_scheme", None)
        if kv_cache_scheme is not None:
            kv_cache_dtype = "fp8"
            calculate_kv_scales = False
            if cache_config is not None:
                cache_config.cache_dtype = "fp8"
                cache_config.calculate_kv_scales = False

        # Check if per-head quant scales are required based on kv_cache_scheme
        use_per_head_quant_scales = (
            kv_cache_scheme is not None
            and kv_cache_scheme.get("strategy") == "attn_head"
        )

        # Skip quantization for specified layers
        if cache_config is not None and cache_config.kv_cache_dtype_skip_layers:
            from vllm.model_executor.models.utils import extract_layer_index

            skip = False
            # Check attention type
            if (
                sliding_window is not None
                and "sliding_window" in cache_config.kv_cache_dtype_skip_layers
            ):
                skip = True
            # Check layer index
            layer_idx = extract_layer_index(prefix)
            if str(layer_idx) in cache_config.kv_cache_dtype_skip_layers:
                skip = True
            if skip:
                kv_cache_dtype = "auto"
                calculate_kv_scales = False
            logger.info(
                "Layer %s: kv_cache_dtype=%s, sliding_window=%s",
                prefix,
                kv_cache_dtype,
                sliding_window,
            )

        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            kv_cache_dtype, vllm_config.model_config
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.calculate_kv_scales = calculate_kv_scales
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) is not divisible by num_kv_heads ({num_kv_heads})"
        )
        self.quant_config = quant_config
        self.layer_name = prefix

        self.num_heads = num_heads
        self.head_size = head_size
        self.head_size_v = self.head_size if head_size_v is None else head_size_v
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.has_sink = extra_impl_args.get("sinks") is not None

        # NOTE: model_config may be None during certain tests
        model_config = vllm_config.model_config
        self.use_mm_prefix = model_config is not None and model_config.is_mm_prefix_lm

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()
        if attn_backend is None:
            self.attn_backend = get_attn_backend(
                head_size,
                dtype,
                kv_cache_dtype,
                use_mla=False,
                has_sink=self.has_sink,
                use_mm_prefix=self.use_mm_prefix,
                use_per_head_quant_scales=use_per_head_quant_scales,
                attn_type=attn_type,
            )
        else:
            self.attn_backend = attn_backend
        backend_supports_alibi_sqrt = self.attn_backend.supports_alibi_sqrt()
        use_alibi_sqrt = use_alibi_sqrt if use_alibi_sqrt else False
        if use_alibi_sqrt and not backend_supports_alibi_sqrt:
            raise ValueError(
                f"use_alibi_sqrt is not supported by backend "
                f"{self.attn_backend.get_name()}."
            )
        self.use_alibi_sqrt = bool(use_alibi_sqrt)
        if backend_supports_alibi_sqrt:
            extra_impl_args["use_alibi_sqrt"] = self.use_alibi_sqrt
        # prefix caching + batch invariance is currently not supported for
        # FLASHINFER and TRITON_MLA.
        if (
            cache_config is not None
            and cache_config.enable_prefix_caching
            and envs.VLLM_BATCH_INVARIANT
            and (
                self.attn_backend.get_name() == "FLASHINFER"
                or self.attn_backend.get_name() == "TRITON_MLA"
            )
        ):
            logger.warning_once(
                "Disabling prefix caching for FLASHINFER/TRITON_MLA "
                "with batch invariance, as it is not yet supported.",
                scope="local",
            )
            cache_config.enable_prefix_caching = False

        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **extra_impl_args,
        )
        self.backend = AttentionBackendEnum[self.attn_backend.get_name()]
        self.dtype = dtype

        # For cuda-alike (CUDA and ROCM) and cpu platforms, we control how
        # torch.compile works by registering the attention as one giant
        # opaque custom op. For other platforms, we directly call them
        # and let torch.compile handle them.
        self.use_direct_call = not current_platform.opaque_attention_op()

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.attn_type = attn_type

        if kv_sharing_target_layer_name is not None:
            validate_kv_sharing_target(
                prefix,
                kv_sharing_target_layer_name,
                compilation_config.static_forward_context,
            )
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        # use a placeholder kv cache tensor during init, which will be replaced
        # by bind_kv_cache
        # this variable will not be accessed if use_direct_call is True
        self.kv_cache = torch.tensor([])

        # Initialize KV cache quantization attributes
        _init_kv_cache_quant(self, quant_config, prefix)

        # Initialize TurboQuant buffers (Pi, S, centroids) if tq cache dtype
        if kv_cache_dtype.startswith("turboquant_"):
            self._init_turboquant_buffers(kv_cache_dtype, head_size, prefix)

        # for attn backends supporting query quantization
        self.query_quant = None
        if (
            self.impl.supports_quant_query_input
            and (
                self.kv_cache_dtype.startswith("fp8") or self.kv_cache_dtype == "nvfp4"
            )
            and not self.kv_cache_dtype.endswith("per_token_head")
        ):
            is_per_head = (
                hasattr(self, "q_scale") and self.q_scale.numel() == self.num_kv_heads
            )
            block_size = self.head_size * self.num_heads // self.num_kv_heads
            self.query_quant = QuantFP8(
                static=True,
                group_shape=GroupShape(-1, block_size)
                if is_per_head
                else GroupShape.PER_TENSOR,
            )

    def _init_turboquant_buffers(
        self, cache_dtype: str, head_size: int, prefix: str
    ) -> None:
        """Initialize TurboQuant centroids for Lloyd-Max quantization."""
        from vllm.model_executor.layers.quantization.turboquant.centroids import (
            get_centroids,
        )
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype, head_size)

        self.register_buffer(
            "_tq_centroids",
            get_centroids(head_size, tq_config.centroid_bits),
        )
        self._tq_config = tq_config

        # Pre-allocate decode intermediate buffers so model.to(device) moves
        # them to GPU *before* the memory profiler runs.  Without this the
        # profiler gives all free memory to KV cache blocks and the first
        # decode OOMs when these buffers are lazily allocated.
        _vllm_cfg = get_current_vllm_config()
        B = _vllm_cfg.scheduler_config.max_num_seqs
        Hq = self.num_heads
        S = _vllm_cfg.attention_config.tq_max_kv_splits_for_cuda_graph
        D = head_size
        self.register_buffer(
            "_tq_mid_o_buf",
            torch.empty(B, Hq, S, D + 1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_tq_output_buf",
            torch.empty(B, Hq, D, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_tq_lse_buf",
            torch.empty(B, Hq, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        # For some alternate attention backends like MLA the attention output
        # shape does not match the query shape, so we optionally let the model
        # definition specify the output tensor shape.
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.

        Attention metadata (`attn_metadata`) is set using a context manager in
        the model runner's `execute_model` method. It is accessed via forward
        context using
        `vllm.forward_context.get_forward_context().attn_metadata`.
        """
        if self.calculate_kv_scales:
            torch.ops.vllm.maybe_calc_kv_scales(
                query, key, value, _encode_layer_name(self.layer_name)
            )
        output_dtype = query.dtype
        if self.query_quant is not None:
            # quantizing with a simple torch operation enables
            # torch.compile to fuse this into previous ops
            # which reduces overheads during decoding.
            # Otherwise queries are quantized using custom ops
            # which causes decoding overheads
            assert self.kv_cache_dtype in {"fp8", "fp8_e4m3", "nvfp4"}

            # check if query quantization is supported
            if self.impl.supports_quant_query_input:
                query, _ = self.query_quant(query, self._q_scale)

        if output_shape is None:
            # Handle both 2D [num_tokens, hidden] and
            # 3D [num_tokens, heads, head_dim] query
            num_tokens = query.shape[0]
            output_shape = torch.Size((num_tokens, self.num_heads * self.head_size_v))
        output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
        hidden_size = output_shape[-1]
        # Reshape the query, key, and value tensors.
        # NOTE(woosuk): We do this outside the custom op to minimize the
        # CPU overheads from the non-CUDA-graph regions.
        query = query.view(-1, self.num_heads, self.head_size)
        output = output.view(-1, self.num_heads, self.head_size_v)
        if key is not None:
            key = key.view(-1, self.num_kv_heads, self.head_size)
        if value is not None:
            value = value.view(-1, self.num_kv_heads, self.head_size_v)
        kv_cache_dummy_dep = None
        if self.use_direct_call:
            # Skip this if sharing KV cache with an earlier attention layer.
            if (
                not self.attn_backend.forward_includes_kv_cache_update
                and self.kv_sharing_target_layer_name is None
                and key is not None
                and value is not None
            ):
                kv_cache_dummy_dep = unified_kv_cache_update(
                    key, value, self.layer_name
                )
            unified_attention_with_output(
                query,
                key,
                value,
                output,
                self.layer_name,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
        else:
            # Skip this if sharing KV cache with an earlier attention layer.
            encoded = _encode_layer_name(self.layer_name)
            if (
                not self.attn_backend.forward_includes_kv_cache_update
                and self.kv_sharing_target_layer_name is None
                and key is not None
                and value is not None
            ):
                kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
                    key, value, encoded
                )
            torch.ops.vllm.unified_attention_with_output(
                query,
                key,
                value,
                output,
                encoded,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
        return output.view(-1, hidden_size)

    def calc_kv_scales(self, query, key, value):
        self._q_scale.copy_(torch.abs(query).max() / self.q_range)
        self._k_scale.copy_(torch.abs(key).max() / self.k_range)
        self._v_scale.copy_(torch.abs(value).max() / self.v_range)
        self._q_scale_float = self._q_scale.item()
        self._k_scale_float = self._k_scale.item()
        self._v_scale_float = self._v_scale.item()
        # We only calculate the scales once
        self.calculate_kv_scales = False

    def extra_repr(self) -> str:
        s = f"head_size={self.impl.head_size}"  # type: ignore
        s += f", num_heads={self.impl.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.impl.num_kv_heads}"  # type: ignore
        s += f", scale={self.impl.scale}"  # type: ignore
        s += f", backend={self.impl.__class__.__name__}"
        return s

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        self.impl.process_weights_after_loading(act_dtype)

        # If we should not load quant weights, we initialize the scales to 1.0
        # as the default value. See [Note: Register q/k/v/prob scales in state dict]
        # for more details.
        quant_method = (
            self.quant_config.get_quant_method(self, prefix=self.layer_name)
            if self.quant_config
            else None
        )
        if not should_load_quant_weights(quant_method):
            set_default_quant_scales(self, register_buffer=False)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Block size may get updated after model loading, refresh it
        block_size = vllm_config.cache_config.block_size
        # Should not be called for enc-dec or encoder-only attention.
        assert self.attn_type == AttentionType.DECODER
        quant_mode = get_kv_quant_mode(self.kv_cache_dtype)
        if self.sliding_window is not None:
            assert not vllm_config.model_config.use_mla, (
                "MLA is not supported for slidingwindow"
            )
            return SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=quant_mode,
                sliding_window=self.sliding_window,
            )
        elif self.kv_cache_dtype.startswith("turboquant_"):
            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )
            from vllm.v1.kv_cache_interface import TQFullAttentionSpec

            tq_config = TurboQuantConfig.from_cache_dtype(
                self.kv_cache_dtype, self.head_size
            )
            return TQFullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size,
                dtype=self.kv_cache_torch_dtype,
                tq_slot_size=tq_config.slot_size_aligned,
            )
        else:
            return FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=quant_mode,
            )


def maybe_calc_kv_scales(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]

    # Only calculate if the layer's calculate_kv_scales flag is True
    # This flag gets set to False after the first forward pass
    if not self.calculate_kv_scales:
        return

    self.calc_kv_scales(query, key, value)


def maybe_calc_kv_scales_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="maybe_calc_kv_scales",
    op_func=maybe_calc_kv_scales,
    mutates_args=["query", "key", "value"],
    fake_impl=maybe_calc_kv_scales_fake,
)


def get_attention_context(
    layer_name: str,
) -> tuple[Any, "Attention | MLAAttention", torch.Tensor, torch.Tensor]:
    """Extract attention context for a given layer.

    This helper function extracts the attention metadata, attention layer
    instance, KV cache tensor, and slot mapping for a specific layer.

    Args:
        layer_name: The name/identifier of the attention layer.

    Returns:
        A tuple containing:
        - attn_metadata: Attention metadata for this specific layer, or None if
            no metadata available
        - attn_layer: The attention layer instance (Attention or MLAAttention)
        - kv_cache: The KV cache tensor for current forward pass
        - slot_mapping: The slot mapping for this specific layer

        Note: attn_metadata may be None, but attn_layer and kv_cache are always
        extracted from the forward context.
    """
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    attn_layer: Attention | MLAAttention = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache
    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict), (
        f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
    )
    layer_slot_mapping = slot_mapping.get(layer_name)
    return attn_metadata, attn_layer, kv_cache, layer_slot_mapping


def unified_kv_cache_update(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """
    Returns a dummy that is passed to unified_attention to signal a side effect and
    the data dependency between them to ensure torch.compile preserves ordering.
    """
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        assert hasattr(attn_layer.impl, "do_kv_cache_update"), (
            f"{attn_layer.impl.__class__.__name__} does not support kv cache update"
        )
        attn_layer.impl.do_kv_cache_update(
            attn_layer,
            key,
            value,
            kv_cache,
            layer_slot_mapping,
        )

    return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)


def unified_kv_cache_update_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return torch.empty(0, device=key.device, dtype=key.dtype)


direct_register_custom_op(
    op_name="unified_kv_cache_update",
    op_func=unified_kv_cache_update,
    fake_impl=unified_kv_cache_update_fake,
    mutates_args=[],
)


@maybe_transfer_kv_layer
def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    # kv_cache_dummy_dep is not used but accepting it creates a data dependency
    # that ensures torch.compile preserves ordering between KV cache update and
    # attention forward.
    del kv_cache_dummy_dep
    layer_name = _resolve_layer_name(layer_name)
    attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)

    # IDE_006 4.5.2.c v33 architectural — CUDA stream 분리: NEO cdec
    # dispatch 의 *CPU computation* (``_pa.forward_attention``) 을
    # background thread 로 submit *전에* vanilla GPU forward 호출 →
    # 두 작업 진짜 병렬 실행. C extension 호출 시 GIL release 가
    # main thread (GPU forward) 의 진행 보장.
    #
    # 이전: 직렬 — vanilla GPU + CPU pacpu = 합산 latency
    # 본 fix: 병렬 — max(GPU, CPU) latency. NEO advantage 의 *진짜* 발휘.
    cdec_future = None
    cdec_t0 = cdec_t1 = 0
    try:
        from vllm.forward_context import get_forward_context as _get_fc
        _fc = _get_fc()
        _tok = getattr(_fc, "neo_cdec_token_slice", None)
        _seq = getattr(_fc, "neo_cdec_seq_slice", None)
        _req_ids = getattr(_fc, "neo_cdec_req_ids", None)
        # [Cleanup 2026-05-14] D-cdec-trace dev-only diagnostic 제거.
        # 진단 작업 완료 (SUB_023/025/026/027 chain) — 분석 path 추적
        # 카운터/로그 hot-path 에서 환경 변수 lookup 매 cdec 호출 비용 제거.
        if (_tok is None or _tok[1] <= _tok[0]
                or _seq is None or _seq[1] <= _seq[0]
                or not _req_ids):
            cdec_future = None
        elif not hasattr(torch.ops, "pacpu"):
            cdec_future = None
        else:
            from vllm.v1.core.sched import neo_cpu_kv_buffer as _ncb
            buf = _ncb.get_active_buffer()
            if buf is None:
                cdec_future = None
            else:
                _t0, _t1 = _tok
                _s0, _s1 = _seq
                cdec_count = _t1 - _t0
                nh = self.impl.num_heads
                nkh = self.impl.num_kv_heads
                hd = self.impl.head_size
                # Setup (main thread) — GIL safe state copy.
                q_cdec = query[_t0:_t1].view(cdec_count, nh, hd)
                k_cdec = key[_t0:_t1].view(cdec_count, nkh, hd)
                v_cdec = value[_t0:_t1].view(cdec_count, nkh, hd)
                import re
                m = re.search(r"layers\.(\d+)", layer_name)
                if m is not None:
                    layer_idx = int(m.group(1))
                    # [Plan v4 Option L] NEO 정통 정합 — cdec dispatch 직전
                    # 증분 alloc. ``seq_len`` 의 nblock 부족 시 buf.extend
                    # → block_table[block_pos] 항상 valid.
                    # NEO ``BlockManager.prepare()`` 의 ``cpu_block_manager
                    # .alloc(cdec_reqs, omit_last=False)`` 호출 등가.
                    # env VLLM_NEO_OPTION_L (default 1) 로 비활성 가능.
                    import os as _os_optL
                    _option_l_enabled = (_os_optL.environ.get(
                        "VLLM_NEO_OPTION_L", "1") == "1")
                    if _option_l_enabled and hasattr(
                            buf, "ensure_capacity"):
                        # [v1.5.2 root fix] FlashAttentionMetadata 는
                        # ``seq_lens_cpu`` / ``_seq_lens_cpu`` attribute 미존재.
                        # v1.5 의 "이미 CPU" 가정 fail → seq_lens_attr = None →
                        # cdec dispatch silent skip. fallback: ``seq_lens.cpu()``.
                        # 1회 cache on forward_context — 80 layers × N sub-batches
                        # 의 GPU→CPU sync 80x → 1x 축소.
                        _seq_lens_attr_optL = getattr(
                            attn_metadata, "seq_lens_cpu", None,
                        )
                        if _seq_lens_attr_optL is None:
                            _seq_lens_attr_optL = getattr(
                                attn_metadata, "_seq_lens_cpu", None,
                            )
                        if _seq_lens_attr_optL is None:
                            _seq_lens_attr_optL = getattr(
                                _fc, "_neo_cdec_seq_lens_cpu_cache", None,
                            )
                        if _seq_lens_attr_optL is None:
                            _seq_lens_gpu_optL = getattr(
                                attn_metadata, "seq_lens", None,
                            )
                            if _seq_lens_gpu_optL is not None:
                                _seq_lens_attr_optL = _seq_lens_gpu_optL.cpu()
                                try:
                                    _fc._neo_cdec_seq_lens_cpu_cache = (
                                        _seq_lens_attr_optL
                                    )
                                except Exception:
                                    pass
                        if _seq_lens_attr_optL is not None:
                            # 이미 CPU tensor — .cpu() 호출 X, .tolist() 만.
                            _seq_lens_optL = (
                                _seq_lens_attr_optL[_s0:_s1].tolist()
                            )
                            _BLOCK_SIZE_L = 16
                            for _i_l, _rid_l in enumerate(_req_ids):
                                _seq_len_l = _seq_lens_optL[_i_l]
                                _target_nblk_l = (
                                    (_seq_len_l + _BLOCK_SIZE_L - 1)
                                    // _BLOCK_SIZE_L
                                )
                                _r = buf.ensure_capacity(
                                    _rid_l, _target_nblk_l,
                                )
                                if _r is None:
                                    if not getattr(
                                            self,
                                            "_neo_option_l_fail_logged",
                                            False):
                                        from vllm.logger import (
                                            init_logger as _il_l,
                                        )
                                        _il_l(
                                            "vllm.attention.neo_optL"
                                        ).warning(
                                            "[Option L] ensure_capacity "
                                            "fail rid=%s target_nblk=%d",
                                            _rid_l, _target_nblk_l,
                                        )
                                        self._neo_option_l_fail_logged = True
                    max_blocks_per_seq = max(
                        len(buf.get_block_ids(rid) or [])
                        for rid in _req_ids
                    ) or 1
                    block_table_rows = []
                    valid = True
                    for rid in _req_ids:
                        ids = buf.get_block_ids(rid)
                        if ids is None:
                            valid = False
                            break
                        row = list(ids) + [0] * (
                            max_blocks_per_seq - len(ids)
                        )
                        block_table_rows.append(row)
                    if valid:
                        block_table_cpu = torch.tensor(
                            block_table_rows, dtype=torch.int32,
                        )
                        # [v1.5.2 root fix] FlashAttentionMetadata 는
                        # ``seq_lens_cpu`` / ``_seq_lens_cpu`` attribute 미존재.
                        # v1.5 의 "이미 CPU" 가정 fail → seq_lens_attr = None →
                        # cdec dispatch silent skip. fallback: ``seq_lens.cpu()``.
                        # 1회 cache on forward_context — 80 layers 의 sync 1회로.
                        seq_lens_attr = getattr(
                            attn_metadata, "seq_lens_cpu", None
                        )
                        if seq_lens_attr is None:
                            seq_lens_attr = getattr(
                                attn_metadata, "_seq_lens_cpu", None
                            )
                        if seq_lens_attr is None:
                            seq_lens_attr = getattr(
                                _fc, "_neo_cdec_seq_lens_cpu_cache", None,
                            )
                        if seq_lens_attr is None:
                            _seq_lens_gpu = getattr(
                                attn_metadata, "seq_lens", None,
                            )
                            if _seq_lens_gpu is not None:
                                seq_lens_attr = _seq_lens_gpu.cpu()
                                try:
                                    _fc._neo_cdec_seq_lens_cpu_cache = seq_lens_attr
                                except Exception:
                                    pass
                        if seq_lens_attr is not None:
                            seq_ids_list = list(range(cdec_count))
                            seq_lengths = (
                                seq_lens_attr[_s0:_s1].tolist()
                            )
                            # [Plan v4 D11 dynamic-debug] pacpu kernel
                            # input 의 *동적* 정합성 검증. SEGV stack 이
                            # 매 회차 brute::store_kv 의 memcpy 인데
                            # 정적 코드 분석으로는 *어떤 req* 의 *어떤
                            # 값* 이 invalid 인지 식별 불가. 본 precheck
                            # 가 dispatch 직전 OOB 영역 진입 reqs 를
                            # log + skip → SEGV 회피 + root data
                            # 추출. BLOCK_SIZE=16 (Llama default).
                            _BLOCK_SIZE_D11 = 16
                            _oob_reqs_d11 = []
                            for _i_d11, _rid_d11 in enumerate(_req_ids):
                                _seq_len_d11 = seq_lengths[_i_d11]
                                _row_d11 = block_table_rows[_i_d11]
                                _nblk_d11 = len(buf.get_block_ids(_rid_d11) or [])
                                _block_pos_d11 = (_seq_len_d11 - 1) // _BLOCK_SIZE_D11
                                if (_block_pos_d11 < 0
                                        or _block_pos_d11 >= _nblk_d11
                                        or _block_pos_d11 >= len(_row_d11)):
                                    _oob_reqs_d11.append((
                                        _i_d11, _rid_d11, _seq_len_d11,
                                        _nblk_d11, _block_pos_d11,
                                        len(_row_d11),
                                    ))
                            if _oob_reqs_d11:
                                from vllm.logger import init_logger as _einit_d11
                                _d11_logger = _einit_d11(
                                    "vllm.attention.neo_cdec_d11",
                                )
                                _d11_cnt = getattr(
                                    self, "_neo_d11_oob_cnt", 0,
                                ) + len(_oob_reqs_d11)
                                self._neo_d11_oob_cnt = _d11_cnt
                                if (_d11_cnt <= 30
                                        or _d11_cnt % 100 == 0):
                                    _d11_logger.error(
                                        "[NEO CDEC D11 OOB PRECHECK] "
                                        "found %d OOB reqs (total=%d): %s",
                                        len(_oob_reqs_d11), _d11_cnt,
                                        _oob_reqs_d11[:5],
                                    )
                                # OOB reqs 가 하나라도 있으면 cdec
                                # dispatch 자체 skip — pacpu kernel
                                # SEGV 회피. data 손실은 vanilla GPU
                                # attention 으로 fallback (cdec_future
                                # = None 으로 흘러감).
                                cdec_future = None
                                raise RuntimeError(
                                    f"D11 OOB precheck: skipping cdec "
                                    f"dispatch ({len(_oob_reqs_d11)} OOB)"
                                )
                            # TSK_019 SUB_022 (R4) — pinned q/k/v 1 회
                            # alloc 재사용 + cross-dtype copy_ 로 cast +
                            # transfer 1 step. 매 layer alloc 폐기.
                            # SUB_017 (R4) — 별 CUDA stream 위 비동기
                            # H2D, main stream GPU compute 와 자동
                            # overlap. main thread 가 submit 직전
                            # event.synchronize() 호출 (GIL 안전 — worker
                            # thread sync 는 SUB_005 회귀 root).
                            q_cpu, k_cpu, v_cpu = _get_neo_pinned_qkv(
                                cdec_count, nh, nkh, hd,
                            )
                            _xfer_stream = _get_neo_communication_stream()
                            _xfer_stream.wait_stream(
                                torch.cuda.current_stream()
                            )
                            # TSK_019 SUB_024 (R6 변형) — GPU 측 명시
                            # cast (BF16 → FP16) 후 same-dtype copy.
                            # NEO 는 GPU/CPU 모두 FP16 이라 cast 자체
                            # 없음. vllm_hybrid 는 GPU BF16 / CPU FP16
                            # 불가피 — implicit (SUB_022) vs explicit
                            # (본 회차) 위치 측정.
                            with torch.cuda.stream(_xfer_stream):
                                q_cpu.copy_(
                                    q_cdec.to(torch.float16),
                                    non_blocking=True,
                                )
                                k_cpu.copy_(
                                    k_cdec.to(torch.float16),
                                    non_blocking=True,
                                )
                                v_cpu.copy_(
                                    v_cdec.to(torch.float16),
                                    non_blocking=True,
                                )
                            _xfer_event = _xfer_stream.record_event()
                            # TSK_019 root #1 fix — main thread sync 제거.
                            # worker thread (_neo_cdec_compute_cpu 의 첫
                            # 줄) 가 host-side wait. main thread 가
                            # GPU forward launch + backlog 형성 가능.
                            # Submit CPU pacpu kernel to background
                            # thread — 진짜 병렬 시작.
                            cdec_future = (
                                _get_neo_cdec_executor().submit(
                                    _neo_cdec_compute_cpu,
                                    _xfer_event,
                                    layer_idx,
                                    float(self.impl.scale),
                                    seq_ids_list,
                                    seq_lengths,
                                    q_cpu, k_cpu, v_cpu,
                                    buf.k_cpu, buf.v_cpu,
                                    block_table_cpu,
                                    cdec_count, nh, hd,
                                )
                            )
                            # 22-item monitor 추적 — env-gated cdec dispatch
                            # counter (VLLM_NEO_PROFILE=1 시만 emit).
                            global _neo_cdec_call_count
                            try:
                                _neo_cdec_call_count += 1
                            except NameError:
                                _neo_cdec_call_count = 1
                            import os as _os_cdec_call
                            if (_os_cdec_call.environ.get(
                                    "VLLM_NEO_PROFILE", "0") == "1"
                                and _neo_cdec_call_count % 100 == 0):
                                logger.info(
                                    "[NEO CDEC CALL] count=%d cdec_rows=%d "
                                    "layer=%d",
                                    _neo_cdec_call_count, cdec_count,
                                    layer_idx,
                                )
                            cdec_t0 = _t0
                            cdec_t1 = _t1
    except Exception as _cdec_setup_e:  # noqa: BLE001
        cdec_future = None
        try:
            from vllm.logger import init_logger as _einit
            _ne_logger = _einit("vllm.attention.neo_cdec_error")
            _stat_attr = "_neo_cdec_fail_count"
            _cnt = getattr(self, _stat_attr, 0) + 1
            setattr(self, _stat_attr, _cnt)
            if _cnt <= 5 or _cnt % 1000 == 0:
                import traceback as _tb
                _ne_logger.error(
                    "[NEO CDEC SETUP FAIL] count=%d type=%s msg=%s\n%s",
                    _cnt, type(_cdec_setup_e).__name__, _cdec_setup_e,
                    _tb.format_exc(),
                )
        except Exception:
            pass

    # IDE_006 — drain pending cdec from the previous attention call.
    # Placed AFTER this call's own cdec submit (above) so the new
    # future is already running on the CDEC_WORKERS pool when we wait
    # on the previous one. Two cdec futures concurrent. In sync mode
    # _neo_pending_cdec_state is always None; this call is fast no-op.
    _neo_drain_pending_cdec()

    # IDE_006 4.5.2.c v38 — cdec-only sub-batch GPU attention skip.
    # b1 sub-batch 의 모든 row 가 cdec 인 경우 GPU attention 결과는
    # CPU pacpu 결과로 100% overwrite → GPU compute 완전 낭비.
    # swiftllm 원본도 cdec rows 를 GPU attention path 에서 제외.
    # 조건: cdec_future 활성 + cdec slice 가 query 전체 cover (b1).
    skip_gpu_attn = (
        cdec_future is not None
        and cdec_t0 == 0
        and cdec_t1 == query.size(0)
    )
    # [PROFILE per-layer] env VLLM_NEO_PROFILE=1 활성. GPU forward + cdec
    # wait elapsed 측정. batch[0]/batch[1] count 도 cumulative stats.
    # [try92 fix] 모듈 전역 stats — 각 layer instance 별 self attribute 가
    # 아니라 모든 worker 의 모든 layer 합산. log_freq=500 도달 가능.
    import os as _os_pl_prof
    _pl_prof_on = (_os_pl_prof.environ.get("VLLM_NEO_PROFILE", "0") == "1")
    _t_gpu_start = 0.0
    if _pl_prof_on:
        import time as _t_pl_prof
        _t_gpu_start = _t_pl_prof.perf_counter()
        global _NEO_PL_GLOBAL_STATS
        if "_NEO_PL_GLOBAL_STATS" not in globals():
            _NEO_PL_GLOBAL_STATS = {
                "gpu_ms_sum": 0.0, "gpu_count": 0, "gpu_ms_max": 0.0,
                "cdec_ms_sum": 0.0, "cdec_count": 0, "cdec_ms_max": 0.0,
                "b0_count_sum": 0, "b1_count_sum": 0,
                "skip_gpu_count": 0, "log_freq": 50,  # [Phase 6.2] 50 로 줄임
            }
    if not skip_gpu_attn:
        # Vanilla GPU forward (main thread, parallel with CPU pacpu).
        self.impl.forward(
            self,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )
    if _pl_prof_on and not skip_gpu_attn:
        _gpu_ms = (_t_pl_prof.perf_counter() - _t_gpu_start) * 1000
        _st = _NEO_PL_GLOBAL_STATS
        _st["gpu_ms_sum"] += _gpu_ms
        _st["gpu_count"] += 1
        if _gpu_ms > _st["gpu_ms_max"]:
            _st["gpu_ms_max"] = _gpu_ms
        _st["b0_count_sum"] += query.size(0) - (cdec_t1 - cdec_t0)
        _st["b1_count_sum"] += (cdec_t1 - cdec_t0) if cdec_future is not None else 0

    # Wait for CPU pacpu task + apply result to output[cdec rows].
    # TSK_019 plan B3 — host fence 위치 layer-end (NEO 원본
    # `transformer_layer.py:336` 의 inline call 위치 동등). cdec_future
    # submit (line 853) 후 GPU forward launch (line 913) 직렬 — backlog
    # 형성. 그 후 layer-end 에서 host wait. KV cache sequential
    # dependency (layer i output → layer i+1 input) 로 layer-pipeline
    # X. 본 위치가 NEO 패턴.
    # IDE_006 — async cdec branch. When async mode is enabled by the
    # forward_pipeline orchestrator, save cdec_future as pending and
    # skip the blocking wait. The next attention call (or an explicit
    # drain by the orchestrator) will resolve it.
    if cdec_future is not None and _neo_async_cdec_mode:
        # IDE_006 Phase 3 — push to deque (depth controlled by
        # VLLM_NEO_CDEC_PIPELINE_DEPTH). If queue is full, drain oldest
        # synchronously before appending — keeps in-flight bound.
        import os as _os_q
        _max_depth = int(
            _os_q.environ.get("VLLM_NEO_CDEC_PIPELINE_DEPTH", "1")
        )
        if len(_neo_pending_cdec_queue) >= _max_depth:
            _neo_drain_pending_cdec()  # FIFO drain oldest
        _neo_pending_cdec_queue.append(
            (cdec_future, output, cdec_t0, cdec_t1)
        )
        if _pl_prof_on:
            _st = _NEO_PL_GLOBAL_STATS
            _st["cdec_count"] += 1
            if skip_gpu_attn:
                _st["skip_gpu_count"] += 1
        cdec_future = None
    if cdec_future is not None:
        _t_cdec_wait_start = 0.0
        if _pl_prof_on:
            _t_cdec_wait_start = _t_pl_prof.perf_counter()
        try:
            out_buf = cdec_future.result()
            if _pl_prof_on:
                _cdec_wait_ms = (
                    _t_pl_prof.perf_counter() - _t_cdec_wait_start
                ) * 1000
                _st = _NEO_PL_GLOBAL_STATS
                _st["cdec_ms_sum"] += _cdec_wait_ms
                _st["cdec_count"] += 1
                if _cdec_wait_ms > _st["cdec_ms_max"]:
                    _st["cdec_ms_max"] = _cdec_wait_ms
                if skip_gpu_attn:
                    _st["skip_gpu_count"] += 1
                # log every N
                if _st["cdec_count"] % _st["log_freq"] == 0:
                    from vllm.logger import init_logger as _init_pl
                    _pl_log = _init_pl("vllm.attention.neo_pl_prof")
                    _gpu_avg = (_st["gpu_ms_sum"] / max(1, _st["gpu_count"]))
                    _cdec_avg = (_st["cdec_ms_sum"]
                                 / max(1, _st["cdec_count"]))
                    _b0_avg = _st["b0_count_sum"] / max(1, _st["gpu_count"])
                    _b1_avg = _st["b1_count_sum"] / max(1, _st["cdec_count"])
                    _pl_log.info(
                        "[PROFILE PER-LAYER] cdec_count=%d "
                        "gpu_avg=%.2fms gpu_max=%.2fms "
                        "cdec_wait_avg=%.2fms cdec_wait_max=%.2fms "
                        "ratio=%.2fx b0_avg=%.0f b1_avg=%.0f "
                        "skip_gpu=%d (gpu_count=%d)",
                        _st["cdec_count"],
                        _gpu_avg, _st["gpu_ms_max"],
                        _cdec_avg, _st["cdec_ms_max"],
                        _cdec_avg / max(_gpu_avg, 0.01),
                        _b0_avg, _b1_avg,
                        _st["skip_gpu_count"], _st["gpu_count"],
                    )
            out_gpu = out_buf.to(output.device).to(output.dtype)
            # TSK_019 fix — out_buf shape = (cdec_count, nh * hd).
            # output 의 cdec slice 가 (cdec_count, nh, hd) 3D 또는
            # (cdec_count, nh*hd) 2D 가능. ``.reshape`` 사용 — view 가
            # contiguous 보장 안 될 때 copy fallback. shape mismatch 회피.
            _n_rows = cdec_t1 - cdec_t0
            output[cdec_t0:cdec_t1].reshape(_n_rows, -1).copy_(
                out_gpu.reshape(_n_rows, -1)
            )
        except Exception as _cdec_apply_e:  # noqa: BLE001
            try:
                from vllm.logger import init_logger as _einit
                _ne_logger = _einit("vllm.attention.neo_cdec_error")
                _stat_attr = "_neo_cdec_fail_count"
                _cnt = getattr(self, _stat_attr, 0) + 1
                setattr(self, _stat_attr, _cnt)
                if _cnt <= 5 or _cnt % 1000 == 0:
                    import traceback as _tb
                    _ne_logger.error(
                        "[NEO CDEC APPLY FAIL] count=%d type=%s msg=%s\n%s",
                        _cnt, type(_cdec_apply_e).__name__,
                        _cdec_apply_e, _tb.format_exc(),
                    )
            except Exception:
                pass


# _neo_cdec_call_count removed (cleanup 2026-05-14)


# IDE_006 — async cdec mode. When enabled by an outer pipeline
# orchestrator (forward_pipeline), unified_attention_with_output does
# NOT block on cdec_future at layer-end. Instead the (future, output,
# cdec slice) is saved as pending; the *next* attention call drains
# it (waits + applies H2D). Two cdec futures can be in flight at once
# on the CDEC_WORKERS pool. The drain inside attention() happens
# *after* this call's own cdec submit, so concurrency is naturally
# pipelined. forward_pipeline / forward_first_stage / forward_last_
# stage call _neo_drain_pending_cdec() explicitly before any postproj
# that consumes the attention output, to keep the buffer fresh.
_neo_async_cdec_mode = False
# IDE_006 Phase 3 — deeper pipeline: deque allows multiple cdec futures
# in flight (depth controlled by VLLM_NEO_CDEC_PIPELINE_DEPTH, default 1).
# Each entry: (future, output, t0, t1). FIFO drain order.
import collections as _collections_neo  # noqa: E402
_neo_pending_cdec_queue: _collections_neo.deque = _collections_neo.deque()
# 22-item monitor — cdec dispatch counter (env-gated emit by VLLM_NEO_PROFILE).
_neo_cdec_call_count: int = 0


def _neo_drain_pending_cdec() -> None:
    """Drain ONE oldest pending cdec future (FIFO). No-op when queue
    empty. Phase 3 — supports deque depth > 1.
    """
    if not _neo_pending_cdec_queue:
        return
    cdec_future, out_tensor, cdec_t0, cdec_t1 = (
        _neo_pending_cdec_queue.popleft()
    )
    try:
        out_buf = cdec_future.result()
        import os as _os_drain
        _use_xfer = (
            _os_drain.environ.get("VLLM_NEO_CDEC_DRAIN_XFER_STREAM", "1")
            == "1"
        )
        if _use_xfer and torch.cuda.is_available():
            _xfer_stream = _get_neo_communication_stream()
            # Order the comm stream behind the producer of out_tensor
            # (default stream up to this point) so we never read torn
            # data.
            _cur_stream = torch.cuda.current_stream()
            _xfer_stream.wait_stream(_cur_stream)
            with torch.cuda.stream(_xfer_stream):
                out_gpu = out_buf.to(
                    out_tensor.device, non_blocking=True
                ).to(out_tensor.dtype)
                _n_rows = cdec_t1 - cdec_t0
                out_tensor[cdec_t0:cdec_t1].reshape(_n_rows, -1).copy_(
                    out_gpu.reshape(_n_rows, -1), non_blocking=True
                )
            # Make sure subsequent default-stream ops see the H2D copy.
            _cur_stream.wait_stream(_xfer_stream)
        else:
            out_gpu = out_buf.to(out_tensor.device).to(out_tensor.dtype)
            _n_rows = cdec_t1 - cdec_t0
            out_tensor[cdec_t0:cdec_t1].reshape(_n_rows, -1).copy_(
                out_gpu.reshape(_n_rows, -1)
            )
    except Exception as _drain_e:  # noqa: BLE001
        try:
            from vllm.logger import init_logger as _einit
            _einit("vllm.attention.neo_cdec_error").error(
                "[NEO PENDING CDEC DRAIN FAIL] %s: %s",
                type(_drain_e).__name__, _drain_e,
            )
        except Exception:
            pass


def _neo_set_async_cdec_mode(enabled: bool) -> None:
    """Toggle async cdec mode."""
    global _neo_async_cdec_mode
    _neo_async_cdec_mode = bool(enabled)


import contextlib as _contextlib_neo  # noqa: E402


def _neo_drain_all_pending_cdec() -> None:
    """Drain the entire pending queue (FIFO). Phase 3."""
    while _neo_pending_cdec_queue:
        _neo_drain_pending_cdec()


@_contextlib_neo.contextmanager
def neo_async_cdec_scope():
    """Context manager: enables async cdec mode for the enclosed block.

    Entry drains any leftover pending queue (safety) and enables async
    mode. Exit drains *all* remaining queued futures.
    """
    _neo_drain_all_pending_cdec()
    _neo_set_async_cdec_mode(True)
    try:
        yield
    finally:
        _neo_drain_all_pending_cdec()
        _neo_set_async_cdec_mode(False)


def _neo_cdec_compute_cpu(
    xfer_event,
    layer_idx: int,
    softmax_scale: float,
    seq_ids: list[int],
    seq_lengths: list[int],
    q_cpu: torch.Tensor,
    k_cpu: torch.Tensor,
    v_cpu: torch.Tensor,
    k_cache_layer: torch.Tensor,
    v_cache_layer: torch.Tensor,
    block_table_cpu: torch.Tensor,
    cdec_count: int,
    nh: int,
    hd: int,
) -> torch.Tensor:
    """IDE_006 4.5.2.c v33 — NEO cdec dispatch 의 *CPU computation* 단계
    만 별 thread 에서 실행. C extension (``_pa.forward_attention``) 호출
    시 GIL release → main thread 의 GPU forward 와 진짜 병렬 진행.

    TSK_019 root #1 fix — main thread 에서 ``_xfer_event.synchronize()``
    호출 X. worker thread 가 첫 줄에서 host-side wait. main thread block
    제거 → GPU stream backlog 형성 가능.
    """
    if xfer_event is not None:
        xfer_event.synchronize()
    from vllm.v1.attention.ops import neo_pacpu as _pa
    out_buf = torch.empty(cdec_count, nh * hd, dtype=torch.float32)
    _pa.forward_attention(
        cur_layer=layer_idx,
        softmax_scale=softmax_scale,
        seq_ids=seq_ids,
        seq_lengths=seq_lengths,
        q=q_cpu,
        k_new=k_cpu,
        v_new=v_cpu,
        k_cache_layer=k_cache_layer,
        v_cache_layer=v_cache_layer,
        block_table=block_table_cpu,
        output=out_buf,
    )
    return out_buf


_neo_cdec_executor = None
_neo_pinned_qkv = None  # (q, k, v) pinned tensors — TSK_019 SUB_022 (R4)
_neo_pinned_qkv_capacity = 0  # current capacity (num_tokens)
_neo_communication_stream = None  # TSK_019 SUB_017 — 별 stream (R4)


def _get_neo_communication_stream():
    """TSK_019 SUB_017 (R4) — module-level lazy CUDA stream for GPU→CPU
    async H2D. NEO `swiftllm/worker/model.py:153` `cpu_communication_stream`
    동등 패턴. main stream 의 GPU compute 와 자동 overlap.
    """
    global _neo_communication_stream
    if _neo_communication_stream is None:
        _neo_communication_stream = torch.cuda.Stream()
    return _neo_communication_stream


def _get_neo_pinned_qkv(num_tokens, num_q_heads, num_kv_heads, head_dim,
                        dtype=torch.float16):
    """TSK_019 SUB_022 (R4) — module-level pinned q/k/v cache.

    Single max-size alloc; view-overwrite per layer. cdec dispatch 의
    매 layer 매 step `q_cdec.to(torch.float16).cpu()` 새 alloc + cast 를
    한 번 alloc + cross-dtype `copy_` 로 대체. NEO `block_swapper.py:57-63`
    동등 패턴.
    """
    global _neo_pinned_qkv, _neo_pinned_qkv_capacity
    if _neo_pinned_qkv is None or num_tokens > _neo_pinned_qkv_capacity:
        capacity = max(num_tokens, 256)
        q = torch.empty(
            (capacity, num_q_heads, head_dim),
            dtype=dtype, pin_memory=True,
        )
        k = torch.empty(
            (capacity, num_kv_heads, head_dim),
            dtype=dtype, pin_memory=True,
        )
        v = torch.empty(
            (capacity, num_kv_heads, head_dim),
            dtype=dtype, pin_memory=True,
        )
        _neo_pinned_qkv = (q, k, v)
        _neo_pinned_qkv_capacity = capacity
    q, k, v = _neo_pinned_qkv
    return q[:num_tokens], k[:num_tokens], v[:num_tokens]


def _get_neo_cdec_executor():
    """TSK_019 SUB_023 (R7) — ThreadPoolExecutor max_workers 단계적
    증가. 1 → 2 → 4 + py-spy GIL profile 검증 후 적정 값 결정. 본
    회차 max_workers=2: layer 간 cdec 직렬화 일부 해제. NEO
    `transformer_layer.py:336` 은 OMP-free 패턴 — vllm_hybrid 의
    ThreadPool 은 GIL 환경, contention 측정 필수.
    Override 가능 — env ``VLLM_NEO_CDEC_WORKERS``.
    """
    global _neo_cdec_executor
    if _neo_cdec_executor is None:
        import concurrent.futures
        import os as _os_env
        _max_workers = int(
            _os_env.environ.get("VLLM_NEO_CDEC_WORKERS", "2")
        )
        _neo_cdec_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_max_workers,
            thread_name_prefix="neo-cdec",
        )
    return _neo_cdec_executor


def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_attention_with_output",
    op_func=unified_attention_with_output,
    mutates_args=["output", "output_block_scale"],
    fake_impl=unified_attention_with_output_fake,
)
