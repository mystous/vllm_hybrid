# [kt-kernel] Inference mode ignores num_gpu_experts (SFT-only) — docs/API make double-compute easy

**Version**: kt-kernel 0.7.0.post2 + SGLang 0.5.18 kt_ep_wrapper
**Symptom**: In inference mode, `num_gpu_experts=N` with `gpu_experts_mask=None` results in the CPU computing ALL experts while the GPU also computes experts 0..N-1; outputs are the sum (subtly distorted, hard to notice). Only `gpu_experts_mask` is honored in inference mode.

**Note**: SGLang's kt_ep_wrapper currently passes `gpu_experts_mask=None` unconditionally (see companion SGLang report), so `--kt-num-gpu-experts > 0` double-computes on the current integration.

**Suggested fix**: in inference mode, either derive the mask from num_gpu_experts when mask is None, or raise if num_gpu_experts>0 and mask is None.
