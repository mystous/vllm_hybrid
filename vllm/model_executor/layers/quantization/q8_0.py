# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Q8_0 Quantization for CPU AVX-512 VNNI environments.

Q8_0 format (compatible with llama.cpp):
  Block size = 32 elements
  Each block: FP16 scale (2 bytes) + int8 quants[32] (32 bytes) = 34 bytes

This quantization is CPU-only and requires AVX-512 VNNI instructions.
"""

from typing import Any, Optional

import torch

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)

Q8_0_BLOCK_SIZE = 32
Q8_0_BLOCK_BYTES = 34  # 2 (FP16 scale) + 32 (int8 quants)


class Q8_0Config(QuantizationConfig):
    """Configuration for Q8_0 quantization (llama.cpp compatible)."""

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "Q8_0Config()"

    @classmethod
    def get_name(cls) -> str:
        return "q8_0"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float32, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # CPU-only, no GPU capability needed
        return -1

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Q8_0Config":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            return Q8_0LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []


class Q8_0LinearMethod(QuantizeMethodBase):
    """Linear method for Q8_0 quantization."""

    def __init__(self, quant_config: Q8_0Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: dict[str, Any],
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        n_blocks_per_row = (
            input_size_per_partition + Q8_0_BLOCK_SIZE - 1
        ) // Q8_0_BLOCK_SIZE

        # Q8_0 packed weight: stored as raw bytes
        # Layout: [N, n_blocks_per_row * Q8_0_BLOCK_BYTES]
        qweight = torch.nn.Parameter(
            torch.empty(
                output_size_per_partition,
                n_blocks_per_row * Q8_0_BLOCK_BYTES,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("qweight", qweight)

        # Store original shapes for runtime
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        set_weight_attrs = extra_weight_attrs.get("set_weight_attrs")
        if set_weight_attrs is not None:
            set_weight_attrs(qweight, {"input_dim": 1, "output_dim": 0})

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # If weight was loaded as FP32/BF16, quantize to Q8_0
        if hasattr(layer, "weight") and layer.weight is not None:
            weight = layer.weight.data
            if weight.dtype in (torch.float32, torch.bfloat16):
                N = weight.size(0)
                K = weight.size(1)
                n_blocks = (K + Q8_0_BLOCK_SIZE - 1) // Q8_0_BLOCK_SIZE

                # Quantize to Q8_0 format
                qweight = torch.empty(
                    N, n_blocks * Q8_0_BLOCK_BYTES, dtype=torch.uint8
                )

                try:
                    from vllm._custom_ops import cpu_ops

                    ops = cpu_ops()
                    ops.q8_0_quantize_weight(qweight, weight)
                except (ImportError, AttributeError):
                    # Fallback: Python quantization
                    _quantize_q8_0_python(weight, qweight, N, K)

                layer.qweight = torch.nn.Parameter(
                    qweight, requires_grad=False
                )
                # Remove original weight to save memory
                del layer.weight

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        N = layer.output_size_per_partition
        M = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1]

        # Reshape input to 2D
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        M = x.shape[0]

        output = torch.empty(M, N, dtype=x.dtype, device=x.device)

        try:
            from vllm._custom_ops import cpu_ops

            ops = cpu_ops()
            ops.q8_0_linear(output, x, qweight, bias)
        except (ImportError, AttributeError):
            # Fallback: dequantize and use torch.mm
            K = x.shape[1]
            w_fp32 = _dequantize_q8_0_python(qweight, N, K)
            output_f32 = torch.mm(x.float(), w_fp32.t())
            if bias is not None:
                output_f32 += bias.float()
            output.copy_(output_f32.to(x.dtype))

        # Reshape output to match input dims
        if len(orig_shape) > 2:
            output = output.reshape(*orig_shape[:-1], N)

        return output


def _quantize_q8_0_python(
    weight: torch.Tensor,
    qweight: torch.Tensor,
    N: int,
    K: int,
) -> None:
    """Python fallback for Q8_0 quantization."""
    import struct

    n_blocks = (K + Q8_0_BLOCK_SIZE - 1) // Q8_0_BLOCK_SIZE
    w_float = weight.float()
    qw_bytes = qweight.numpy() if qweight.device.type == "cpu" else qweight.cpu().numpy()

    for n in range(N):
        for b in range(n_blocks):
            k_start = b * Q8_0_BLOCK_SIZE
            k_end = min(k_start + Q8_0_BLOCK_SIZE, K)
            block_vals = w_float[n, k_start:k_end]

            max_abs = block_vals.abs().max().item()
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            inv_scale = 1.0 / scale

            # FP16 scale
            scale_fp16 = struct.pack("<e", scale)
            offset = b * Q8_0_BLOCK_BYTES
            qw_bytes[n, offset] = scale_fp16[0]
            qw_bytes[n, offset + 1] = scale_fp16[1]

            # INT8 quants
            quants = (block_vals * inv_scale).clamp(-127, 127).round().to(
                torch.int8
            )
            for k in range(k_end - k_start):
                qw_bytes[n, offset + 2 + k] = quants[k].item() & 0xFF

    if qweight.device.type == "cpu":
        qweight.copy_(torch.from_numpy(qw_bytes))


def _dequantize_q8_0_python(
    qweight: torch.Tensor, N: int, K: int
) -> torch.Tensor:
    """Python fallback for Q8_0 dequantization."""
    import struct

    n_blocks = (K + Q8_0_BLOCK_SIZE - 1) // Q8_0_BLOCK_SIZE
    result = torch.zeros(N, K, dtype=torch.float32)
    qw = qweight.numpy() if qweight.device.type == "cpu" else qweight.cpu().numpy()

    for n in range(N):
        for b in range(n_blocks):
            k_start = b * Q8_0_BLOCK_SIZE
            k_end = min(k_start + Q8_0_BLOCK_SIZE, K)
            offset = b * Q8_0_BLOCK_BYTES

            # Read FP16 scale
            scale_bytes = bytes([qw[n, offset], qw[n, offset + 1]])
            scale = struct.unpack("<e", scale_bytes)[0]

            # Read INT8 quants and dequantize
            for k in range(k_end - k_start):
                val = qw[n, offset + 2 + k]
                if val > 127:
                    val -= 256  # Convert to signed
                result[n, k_start + k] = float(val) * scale

    return result
