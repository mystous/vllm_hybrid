# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NEO-style CPU-resident KV cache buffer (TSK_015 Phase 4.1).

When ``SchedulerConfig.kv_cache_policy == "exclusive"``, requests that
NEO scheduler decides to swap out of GPU need a place to keep their KV
on the CPU side. This module provides ``NeoCpuKvBuffer`` — a pinned
CPU tensor + per-request block index, sized once at engine startup
based on model dimensions and ``max_num_seqs``.

The buffer's layout matches NEO's ``pacpu`` expectation
(see ``csrc/cpu/pacpu/dtype.h``)::

    (num_layers, num_blocks, num_kv_heads, block_size, head_dim)

so that ``torch.ops.pacpu.paged_attention_cpu`` can read it directly
(via ``num_layers=1`` per-layer view at call time — see
``vllm/v1/attention/ops/neo_pacpu.py``).

Phase 4.1 only owns *allocation* and *bookkeeping*. Actual GPU↔CPU
movement is Phase 4.2 / 4.3 (next turns).

See ``shadow_assists/features/IDE_006/NEO_code_deepdive.md`` §5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class NeoCpuKvBufferSpec:
    """Static dimensions captured at startup. Drives buffer allocation."""

    num_layers: int
    num_kv_heads: int
    block_size: int
    head_dim: int
    # Total CPU-side blocks. Sized to fit ``max_cpu_resident_reqs`` ×
    # ``ceil(max_model_len / block_size)``. Caller should round up so
    # the worst-case fully-filled CPU residence is supported.
    num_cpu_blocks: int
    dtype: torch.dtype = torch.float16   # NEO pacpu expects FP16

    @property
    def per_block_elems(self) -> int:
        return self.num_kv_heads * self.block_size * self.head_dim

    def cpu_buffer_bytes(self) -> int:
        # K + V each: layers * blocks * per_block_elems * dtype_bytes
        elt = torch.tensor([], dtype=self.dtype).element_size()
        per = (
            self.num_layers
            * self.num_cpu_blocks
            * self.per_block_elems
            * elt
        )
        return per * 2     # K + V


@dataclass
class _PerReqAllocation:
    """Bookkeeping for a single CPU-resident request: which CPU
    block_ids hold its KV. Length matches the number of full blocks."""

    block_ids: list[int] = field(default_factory=list)
    # ``num_tokens`` 가 늘어나면 마지막 block 의 partial 채움도 추적해야
    # 하지만, NEO 의 swap-out 은 *완료된 prefill + decode 진행 중* 의 req
    # 만 대상이라 block 단위로 정렬되어 있다고 가정. 추후 chunked 로
    # 변하면 ``last_block_offset`` 추가.


class NeoCpuKvBuffer:
    """CPU-resident KV cache buffer + per-request block index.

    Allocates ``K_cpu`` and ``V_cpu`` once at construction, both shaped
    ``(num_layers, num_cpu_blocks, num_kv_heads, block_size, head_dim)``
    in pinned CPU memory. The free list ``_free_block_ids`` tracks
    available CPU block slots; per-request allocations come out of
    this pool via :py:meth:`alloc` and return on :py:meth:`free`.

    The buffer does **not** move data — that is Phase 4.2 / 4.3. This
    class only owns the allocation accounting.
    """

    def __init__(self, spec: NeoCpuKvBufferSpec) -> None:
        self.spec = spec
        shape = (
            spec.num_layers,
            spec.num_cpu_blocks,
            spec.num_kv_heads,
            spec.block_size,
            spec.head_dim,
        )
        # Pinned-memory tensors so PCIe transfer (Phase 4.2) is fast.
        # ``pin_memory=True`` requires CUDA available; on CPU-only
        # systems fall back to regular (cpu_only paths exist for
        # tests).
        try:
            self.k_cpu = torch.empty(
                shape, dtype=spec.dtype, pin_memory=True
            )
            self.v_cpu = torch.empty(
                shape, dtype=spec.dtype, pin_memory=True
            )
        except RuntimeError as e:
            logger.warning(
                "NeoCpuKvBuffer: pinned allocation failed (%s) — "
                "fallback to non-pinned. Phase 4.2 PCIe DMA will be "
                "synchronous.", e,
            )
            self.k_cpu = torch.empty(shape, dtype=spec.dtype)
            self.v_cpu = torch.empty(shape, dtype=spec.dtype)

        self._free_block_ids: list[int] = list(range(spec.num_cpu_blocks))
        self._req_alloc: dict[str, _PerReqAllocation] = {}

        logger.info(
            "NeoCpuKvBuffer allocated: shape=%s dtype=%s pinned=%s "
            "size=%.2f MiB",
            shape,
            spec.dtype,
            getattr(self.k_cpu, "is_pinned", lambda: False)(),
            spec.cpu_buffer_bytes() / (1024 ** 2),
        )

    # ------------------------------------------------------------------
    # Per-request allocation API (Phase 4.1 — bookkeeping only)
    # ------------------------------------------------------------------
    def alloc(self, req_id: str, num_blocks: int) -> list[int] | None:
        """Reserve ``num_blocks`` CPU block_ids for ``req_id``. Returns
        the assigned block_ids, or ``None`` if the free pool is
        insufficient (caller should treat as a swap-out failure).
        Re-alloc for the same req_id raises (caller should ``free``
        first)."""
        if req_id in self._req_alloc:
            raise ValueError(
                f"NeoCpuKvBuffer.alloc: req_id {req_id!r} already allocated"
            )
        if num_blocks > len(self._free_block_ids):
            return None
        ids = self._free_block_ids[-num_blocks:]
        del self._free_block_ids[-num_blocks:]
        self._req_alloc[req_id] = _PerReqAllocation(block_ids=ids)
        return ids

    def free(self, req_id: str) -> list[int] | None:
        """Release the block_ids belonging to ``req_id``. Returns the
        freed block_ids (so callers can wipe them). Returns ``None`` if
        the req was not allocated."""
        alloc = self._req_alloc.pop(req_id, None)
        if alloc is None:
            return None
        self._free_block_ids.extend(alloc.block_ids)
        return alloc.block_ids

    def get_block_ids(self, req_id: str) -> list[int] | None:
        alloc = self._req_alloc.get(req_id)
        return alloc.block_ids if alloc is not None else None

    # ------------------------------------------------------------------
    # Phase 4.2 — per-layer K/V move primitives
    # ------------------------------------------------------------------
    # vLLM HND per-layer KV: (num_blocks, num_kv_heads, block_size, head_dim)
    # caller must slice by *the req's GPU block_ids* before calling. The
    # primitive does CPU side bookkeeping + cast + memcpy.
    #
    # ``copy_layer_in``  : GPU→CPU (swap-out direction). Caller passes K/V
    #                     slice for *one layer*, sliced by req's GPU
    #                     blocks. Buffer stores at req's CPU block_ids.
    # ``copy_layer_out`` : CPU→Tensor (swap-in direction). Returns the K/V
    #                     slice for one layer; caller copies back to GPU.
    #
    # vLLM 의 KV 가 BF16 인 경우가 대다수, NEO format 은 FP16 — cast 자동
    # 처리. dtype mismatch 의 정밀도 손실은 attention 수준에서 vanilla
    # 와 *분포 동등* (CLAUDE.md Constraint 운영 해석).

    def copy_layer_in(
        self,
        req_id: str,
        layer_idx: int,
        # k_src/v_src: (num_blocks_for_req, num_kv_heads, block_size, head_dim)
        k_src: torch.Tensor,
        v_src: torch.Tensor,
    ) -> None:
        """GPU→CPU per-layer copy. Source tensor 는 req 의 GPU blocks 만
        포함 (caller responsible for slicing by GPU block_ids).
        """
        block_ids = self.get_block_ids(req_id)
        if block_ids is None:
            raise ValueError(
                f"copy_layer_in: req_id {req_id!r} not allocated. Call "
                "``alloc`` first."
            )
        nbk = len(block_ids)
        if k_src.shape[0] != nbk or v_src.shape[0] != nbk:
            raise ValueError(
                f"copy_layer_in: src tensors expect {nbk} blocks, got "
                f"k={k_src.shape[0]}, v={v_src.shape[0]}"
            )
        if not (0 <= layer_idx < self.spec.num_layers):
            raise ValueError(
                f"copy_layer_in: layer_idx {layer_idx} out of range "
                f"[0, {self.spec.num_layers})"
            )

        # Cast to NEO's FP16 if needed; move to CPU. Pinned dst makes
        # cudaMemcpyAsync the natural fast path when source is GPU.
        cast = self.spec.dtype
        if k_src.dtype != cast:
            k_src = k_src.to(cast)
        if v_src.dtype != cast:
            v_src = v_src.to(cast)
        # ``self.k_cpu[layer_idx, idx]`` (advanced indexing) returns a
        # *copy*, not a view — ``.copy_()`` on it would not write back.
        # Use ``index_put_`` or direct indexed assignment instead.
        # Move src to CPU first if needed (it may be on GPU).
        idx = torch.tensor(block_ids, dtype=torch.long)
        k_cpu_src = k_src.detach().to("cpu", non_blocking=False)
        v_cpu_src = v_src.detach().to("cpu", non_blocking=False)
        # ``Tensor[layer, idx] = value`` performs scatter into the
        # contiguous storage. For pinned dst, this is a CPU-side memcpy
        # (after the H2D pulled by ``.to("cpu")`` above).
        self.k_cpu[layer_idx, idx] = k_cpu_src
        self.v_cpu[layer_idx, idx] = v_cpu_src

    def copy_layer_out(
        self,
        req_id: str,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CPU→host tensor. Returns ``(k, v)`` of shape
        ``(num_blocks_for_req, num_kv_heads, block_size, head_dim)`` for
        the requested layer. Caller copies these to GPU (or reads them
        for CPU compute via neo_pacpu).
        """
        block_ids = self.get_block_ids(req_id)
        if block_ids is None:
            raise ValueError(
                f"copy_layer_out: req_id {req_id!r} not allocated"
            )
        if not (0 <= layer_idx < self.spec.num_layers):
            raise ValueError(
                f"copy_layer_out: layer_idx {layer_idx} out of range"
            )
        idx = torch.tensor(block_ids, dtype=torch.long)
        # Returned tensors are *views* into the buffer (no copy). Caller
        # responsible for ``.contiguous()`` if a contiguous block is
        # needed (most kernels accept strided).
        k = self.k_cpu[layer_idx, idx]
        v = self.v_cpu[layer_idx, idx]
        return k, v

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        return len(self._free_block_ids)

    @property
    def num_resident_reqs(self) -> int:
        return len(self._req_alloc)


# ----------------------------------------------------------------------
# IDE_006 / TSK_015.Step3.2.c.6 — module-level singleton register/get
#
# ``unified_attention_with_output`` 의 NEO cdec dispatch hook 은
# attention layer 영역에서 worker-side ``NeoCpuKvBuffer`` 에 접근해야
# 한다. forward_context 나 attn_metadata 에 buffer reference 를 attach
# 하려면 backend 별 metadata 에까지 schema 가 전파되어야 하므로,
# process-local module singleton 을 1 차 path 로 사용한다.
#
# 한 worker process 안에 하나의 active buffer 만 의미가 있으므로 (TP=N
# 인 경우 worker 별 process 가 자기 buffer 를 보유), 단일 slot
# ``_active_buffer`` 로 충분. 호출 흐름:
#   1. ``GPUModelRunner._ensure_neo_cpu_kv_buffer`` 가 buffer alloc 직후
#      ``set_active_buffer(buffer)`` 호출.
#   2. ``unified_attention_with_output`` 이 dispatch hook 안에서
#      ``get_active_buffer()`` 로 lookup. ``None`` 이면 NEO 비활성 →
#      vanilla path.
# ----------------------------------------------------------------------
_active_buffer: NeoCpuKvBuffer | None = None


def set_active_buffer(buffer: NeoCpuKvBuffer | None) -> None:
    """Register / clear the worker-process-local active CPU KV buffer.

    Pass ``None`` to deregister (e.g. on engine shutdown). Idempotent —
    overwrites any prior registration.
    """
    global _active_buffer
    _active_buffer = buffer


def get_active_buffer() -> NeoCpuKvBuffer | None:
    """Return the currently registered CPU KV buffer, or ``None`` when
    NEO is inactive in this worker process."""
    return _active_buffer

    def __len__(self) -> int:
        return self.num_resident_reqs


def make_spec_from_config(vllm_config) -> NeoCpuKvBufferSpec | None:
    """Best-effort constructor from a ``VllmConfig``. Returns ``None``
    if any required hyperparameter is missing — caller falls back to
    *not* allocating the CPU buffer (NEO scheduler then refuses
    swap-out, which is safe).
    """
    try:
        model_cfg = vllm_config.model_config
        cache_cfg = vllm_config.cache_config
        sched_cfg = vllm_config.scheduler_config
        parallel_cfg = vllm_config.parallel_config

        hf_cfg = model_cfg.hf_config
        num_layers = int(hf_cfg.num_hidden_layers)
        num_kv_heads_total = int(
            getattr(
                hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads
            )
        )
        # TP shards num_kv_heads if it divides cleanly (else replication).
        tp = max(parallel_cfg.tensor_parallel_size, 1)
        if num_kv_heads_total % tp == 0:
            num_kv_heads = num_kv_heads_total // tp
        else:
            num_kv_heads = num_kv_heads_total
        head_dim = int(
            getattr(hf_cfg, "head_dim",
                    hf_cfg.hidden_size // hf_cfg.num_attention_heads)
        )
        block_size = max(int(cache_cfg.block_size), 1)
        # CPU pool sizing — vLLM 의 ``CacheConfig.swap_space`` (GB) 가 명시된
        # 경우 그 값을 cap 으로 사용. 안 명시되면 max_seqs/8 × max_model_len
        # 의 보수적 default (예: 256 max_seqs × 16384 max_len + 80 layer +
        # FP16 = ~80 GB 까지 자라므로 1/8 cap 으로 ~10 GB 영역).
        max_seqs = max(int(sched_cfg.max_num_seqs), 1)
        max_model_len = max(int(model_cfg.max_model_len), 1)
        blocks_per_req = (max_model_len + block_size - 1) // block_size

        # Per-block bytes for K + V combined. Use FP16 (NEO format).
        elt_bytes = torch.tensor([], dtype=torch.float16).element_size()
        per_block_bytes = (
            num_kv_heads * block_size * head_dim * elt_bytes * 2  # K+V
        )
        per_layer_per_req_bytes = per_block_bytes * blocks_per_req
        all_layers_per_req_bytes = per_layer_per_req_bytes * num_layers

        # vLLM's swap_space (GB) — if user set it, treat as our cap.
        swap_space_gb = float(getattr(cache_cfg, "swap_space", 0) or 0)
        if swap_space_gb > 0:
            cap_bytes = int(swap_space_gb * (1024 ** 3))
            max_cpu_resident_reqs = max(1, cap_bytes // all_layers_per_req_bytes)
        else:
            # No explicit cap — pick a reasonable default (1/8 of max_seqs).
            max_cpu_resident_reqs = max(1, max_seqs // 8)
        num_cpu_blocks = max_cpu_resident_reqs * blocks_per_req

        logger.info(
            "NeoCpuKvBuffer sizing: max_cpu_resident_reqs=%d × "
            "blocks_per_req=%d = %d cpu_blocks (%.2f GiB total)",
            max_cpu_resident_reqs,
            blocks_per_req,
            num_cpu_blocks,
            (num_cpu_blocks * per_block_bytes * num_layers) / (1024 ** 3),
        )

        # NEO's pacpu uses FP16. vLLM models are typically BF16 — Phase
        # 4.2 will need a cast. For now allocate FP16 buffer; cast at
        # move time.
        return NeoCpuKvBufferSpec(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            head_dim=head_dim,
            num_cpu_blocks=num_cpu_blocks,
            dtype=torch.float16,
        )
    except (AttributeError, ValueError, ZeroDivisionError) as e:
        logger.warning(
            "NeoCpuKvBuffer: cannot build spec from VllmConfig (%s). "
            "exclusive policy will refuse swap-out.", e,
        )
        return None
