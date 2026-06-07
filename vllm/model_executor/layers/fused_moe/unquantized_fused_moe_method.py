# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.nn import Module

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEQuantConfig,
    biased_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    convert_to_unquantized_kernel_format,
    make_unquantized_moe_kernel,
    select_unquantized_moe_backend,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum

logger = init_logger(__name__)


# --8<-- [start:unquantized_fused_moe]
@CustomOp.register("unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    # --8<-- [end:unquantized_fused_moe]

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.unquantized_backend, self.experts_cls = select_unquantized_moe_backend(
            moe_config=self.moe,
        )

    @property
    def is_monolithic(self) -> bool:
        # Escape hatch for CPU, which stays on the old monolithic path.
        if self.unquantized_backend == UnquantizedMoeBackend.CPU:
            return True
        return super().is_monolithic

    @property
    def supports_eplb(self) -> bool:
        return True

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ):
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic for all but the CPU backend. CPU backend is monolithic. "
            "So this function should not be called."
        )

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> FusedMoEExpertsModular:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if self.moe.is_act_and_mul:
            w13_up_dim = 2 * intermediate_size_per_partition
        else:
            w13_up_dim = intermediate_size_per_partition
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_up_dim,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(num_experts, w13_up_dim, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Pad the weight tensor. This is an optimization on ROCm platform, which
        # can benefit from tensors located far enough from one another in memory
        if (
            envs.VLLM_ROCM_MOE_PADDING
            and current_platform.is_rocm()
            and weight.stride(-1) == 1
            and (weight.stride(-2) * weight.element_size()) % 512 == 0
        ):
            num_pad = 256 // weight.element_size()
            weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
            torch.accelerator.empty_cache()

        return weight

    def _setup_kernel(
        self,
        layer: Module,
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        # Shuffle weights to runtime format.
        w13, w2 = convert_to_unquantized_kernel_format(
            self.unquantized_backend,
            layer=layer,
            w13_weight=w13,
            w2_weight=w2,
        )
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)

        # Setup moe kernel.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None
        assert self.experts_cls is not None
        self.moe_kernel = make_unquantized_moe_kernel(
            quant_config=self.moe_quant_config,
            moe_config=self.moe,
            backend=self.unquantized_backend,
            experts_cls=self.experts_cls,
            routing_tables=layer._maybe_init_expert_routing_tables(),
            shared_experts=layer.shared_experts,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        # Padding the weight for better performance on ROCm
        layer.w13_weight.data = self._maybe_pad_weight(layer.w13_weight.data)
        layer.w2_weight.data = self._maybe_pad_weight(layer.w2_weight.data)

        if self.unquantized_backend in [
            UnquantizedMoeBackend.TPU,
            UnquantizedMoeBackend.OOT,
        ]:
            # OOT handles internally.
            return

        elif self.unquantized_backend == UnquantizedMoeBackend.CPU:
            # CPU stays on the old path — no oracle, no moe_kernel.
            from vllm.model_executor.layers.fused_moe import cpu_fused_moe

            if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                from vllm.model_executor.layers.utils import check_cpu_sgl_kernel

                dtype_w13 = layer.w13_weight.dtype
                _, n_w13, k_w13 = layer.w13_weight.size()
                dtype_w2 = layer.w2_weight.dtype
                _, n_w2, k_w2 = layer.w2_weight.size()
                if (
                    envs.VLLM_CPU_SGL_KERNEL
                    and check_cpu_sgl_kernel(n_w13, k_w13, dtype_w13)
                    and check_cpu_sgl_kernel(n_w2, k_w2, dtype_w2)
                ):
                    packed_w13_weight = torch.ops._C.convert_weight_packed(
                        layer.w13_weight
                    )
                    assert packed_w13_weight.size() == layer.w13_weight.size()
                    layer.w13_weight.copy_(packed_w13_weight)
                    del packed_w13_weight
                    packed_w2_weight = torch.ops._C.convert_weight_packed(
                        layer.w2_weight
                    )
                    assert packed_w2_weight.size() == layer.w2_weight.size()
                    layer.w2_weight.copy_(packed_w2_weight)
                    self.cpu_fused_moe: Callable = cpu_fused_moe.SGLFusedMOE(layer)
                else:
                    self.cpu_fused_moe = cpu_fused_moe.CPUFusedMOE(layer)
            else:
                self.cpu_fused_moe = cpu_fused_moe.CPUFusedMOE(layer)
        elif self.unquantized_backend == UnquantizedMoeBackend.XPU:
            w13 = layer.w13_weight
            w2 = layer.w2_weight

            w13.data = w13.transpose(-1, -2).contiguous()
            w2.data = w2.transpose(-1, -2).contiguous()

            self._setup_kernel(
                layer=layer,
                w13=w13,
                w2=w2,
            )
        else:
            self._setup_kernel(
                layer=layer,
                w13=layer.w13_weight,
                w2=layer.w2_weight,
            )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        if self.moe.has_bias:
            return biased_moe_quant_config(
                layer.w13_bias,
                layer.w2_bias,
            )
        else:
            return FUSED_MOE_UNQUANTIZED_CONFIG

    def apply(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.forward(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts_input=shared_experts_input,
        )

    def forward_native(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            shared_experts_input=shared_experts_input,
        )

    def forward_cuda(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        # PoC (SUB_201): MoE expert CPU offload via kt_kernel.
        # Activated by VLLM_MOE_CPU_OFFLOAD=1 + a per-layer wrapper attached
        # under attribute `_kt_layer_wrapper` (set by the model loader, e.g.
        # qwen3_moe.py). Falls through to the standard triton path when off.
        import os as _os
        kt_wrapper = getattr(self, "_kt_layer_wrapper", None)
        # Guard: kt-kernel's CPU dispatch is undefined on empty token batches
        # (warmup) and small uncommon shapes. Fall through to the vLLM path in
        # those cases.
        num_tokens = int(x.shape[0]) if x.dim() > 0 else 0
        if (
            kt_wrapper is not None
            and kt_wrapper.is_ready()
            and num_tokens > 0
            and x.dim() == 2
        ):
            if _os.environ.get("VLLM_MOE_KT_DEBUG") == "1":
                from vllm.logger import init_logger as _init
                _l = _init(__name__)
                _l.warning(
                    "[kt-forward] layer=%d num_tokens=%d hidden=%d x.dtype=%s "
                    "topk_ids.dtype=%s topk_w.dtype=%s",
                    kt_wrapper.layer_idx, num_tokens, int(x.shape[1]),
                    str(x.dtype), str(topk_ids.dtype), str(topk_weights.dtype),
                )
            return self._kt_forward(
                layer, x, topk_weights, topk_ids, shared_experts_input, kt_wrapper
            )
        return self.forward_native(
            layer, x, topk_weights, topk_ids, shared_experts_input
        )

    def _kt_forward(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        kt_wrapper,
    ) -> torch.Tensor:
        """Hybrid CPU+GPU MoE forward using kt-kernel for CPU experts.

        Mirrors SGLang's KTEPWrapperMethod.apply:
          (main)       staging = SharedStagingBuffer.get_slice(num_tokens)
                       staging.copy_(x, non_blocking=True)
          (cpu_stream) wrapper.submit_forward(staging, topk_ids_full, topk_w, stream)
          (main)       gpu_out = fused_experts(x, w13, w2, masked_topk_ids, ...)
          (cpu_stream) cpu_out = wrapper.sync_forward(staging, stream)
          (main)       wait_event(cpu_done); return gpu_out + cpu_out

        Protection mechanisms (#34 fix):
          * ``SharedStagingBuffer`` (process lifetime) avoids the caching
            allocator freeing ``x.clone()`` while the C++ task still holds
            the pointer.
          * ``ensure_batch_size_captured(num_tokens)`` promotes the per-token
            CPU buffer tuple to ``KExpertsCPUBuffer.capture_buffers`` (process
            lifetime) so cross-layer slot rotation cannot point at a
            ``temp_buffer`` that has been overwritten by another call site.
        """
        import os
        from vllm.model_executor.layers.fused_moe.kt_kernel_binding import (
            ensure_batch_size_captured,
            get_or_create_shared_staging_buffer,
        )

        device = x.device
        main_stream = torch.cuda.current_stream(device)

        num_tokens = int(x.shape[0])
        hidden_size = int(x.shape[1])

        # Debug: forward step-by-step sync (set VLLM_MOE_KT_DEBUG_SYNC=1).
        _debug_sync = os.environ.get("VLLM_MOE_KT_DEBUG_SYNC") == "1"

        def _dsync(stage: str) -> None:
            if _debug_sync:
                torch.cuda.synchronize(device)
                from vllm.logger import init_logger as _il
                _il(__name__).warning(
                    "[kt-fwd] layer=%d stage=%s OK (n=%d)",
                    kt_wrapper.layer_idx, stage, num_tokens,
                )

        # Process-wide GPU staging buffer (SGLang-compatible).
        staging_max = max(kt_wrapper.chunked_prefill, num_tokens)
        staging = get_or_create_shared_staging_buffer(
            max_tokens=staging_max,
            hidden_size=hidden_size,
            dtype=torch.bfloat16,
            device=device,
        ).get_slice(num_tokens)
        _dsync("after_get_staging")
        # Copy on main_stream first; this guarantees the staging slot is
        # populated before the cpu_stream picks it up via wait_stream below.
        # ``copy_`` does the dtype cast in-place into staging's BF16 storage,
        # so no transient tensor outlives this call.
        staging.copy_(x, non_blocking=True)
        _dsync("after_staging_copy")

        # Pin this batch_size into kt_kernel's capture_buffers dict so the
        # CPU pinned-mem buffer survives the next layer's get_buffer() call.
        ensure_batch_size_captured(num_tokens)

        # SGLANG_KT_HYBRID_NO_CPU_STREAM=1-equivalent: submit+sync on main
        # stream. Default ON for stability; setting VLLM_MOE_NO_CPU_STREAM=0
        # forks to a dedicated cpu_stream (SGLang default) once boot is OK.
        use_cpu_stream = os.environ.get("VLLM_MOE_NO_CPU_STREAM", "1") == "0"

        if use_cpu_stream:
            cpu_stream = kt_wrapper.cpu_stream
            sync_event = kt_wrapper.sync_done_event
            cpu_stream.wait_stream(main_stream)
            with torch.cuda.stream(cpu_stream):
                kt_wrapper.kt_wrapper.submit_forward(
                    staging, topk_ids, topk_weights, cpu_stream.cuda_stream
                )
        else:
            kt_wrapper.kt_wrapper.submit_forward(
                staging,
                topk_ids,
                topk_weights,
                main_stream.cuda_stream,
            )
        _dsync("after_submit_forward")

        # Step 2: GPU expert path uses vLLM's native ``expert_map`` semantics:
        # pass the original ``topk_ids`` (0..num_total_experts-1) along with
        # an ``expert_map`` that resolves CPU experts to -1 and GPU experts
        # to their GPU-local index. vLLM's moe_align_block_size_kernel's
        # ``has_expert_map`` branch handles the -1 correctly; we sidestep
        # the OOB-write hazard of the no-map path (root cause #34).
        _dsync("before_gpu_path")
        if kt_wrapper.num_gpu_experts > 0:
            assert self.moe_kernel is not None
            gpu_out = self.moe_kernel.apply(
                hidden_states=x,
                w1=layer.w13_weight[:kt_wrapper.num_gpu_experts],
                w2=layer.w2_weight[:kt_wrapper.num_gpu_experts],
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=layer.global_num_experts,
                expert_map=kt_wrapper.expert_map_cuda,
                shared_experts_input=shared_experts_input,
            )
        else:
            gpu_out = torch.zeros_like(x)
        _dsync("after_gpu_path")

        # Step 4: sync CPU result → merge
        if use_cpu_stream:
            with torch.cuda.stream(cpu_stream):
                cpu_out = kt_wrapper.kt_wrapper.sync_forward(
                    staging, cpu_stream.cuda_stream
                )
                sync_event.record(cpu_stream)
            main_stream.wait_event(sync_event)
        else:
            cpu_out = kt_wrapper.kt_wrapper.sync_forward(
                staging, main_stream.cuda_stream
            )
        _dsync("after_sync_forward")

        if cpu_out.shape != gpu_out.shape:
            cpu_out = cpu_out.view_as(gpu_out)
        return gpu_out + cpu_out

    def apply_monolithic(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        assert self.is_monolithic
        if self.unquantized_backend == UnquantizedMoeBackend.CPU:
            assert self.moe_kernel is None
            return self.cpu_fused_moe(
                layer,
                x,
                layer.use_grouped_topk,
                layer.top_k,
                router_logits,
                layer.renormalize,
                layer.topk_group,
                layer.num_expert_group,
                layer.global_num_experts,
                layer.expert_map,
                layer.custom_routing_function,
                layer.scoring_func,
                layer.routed_scaling_factor,
                layer.e_score_correction_bias,
                layer.apply_router_weight_on_input,
                layer.activation,
            )
        else:
            assert self.moe_kernel is not None
            return self.moe_kernel.apply_monolithic(
                x,
                layer.w13_weight,
                layer.w2_weight,
                router_logits,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                num_expert_group=layer.num_expert_group,
                topk_group=layer.topk_group,
                e_score_correction_bias=layer.e_score_correction_bias,
                routed_scaling_factor=layer.routed_scaling_factor,
            )
