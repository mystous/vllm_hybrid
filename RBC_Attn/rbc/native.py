"""FlashInfer native paged-attention 대조.

**LSE 단위 규약**: 이 flashinfer 버전은 LSE 를 log2 밑으로 돌려준다 (측정으로 확인:
native/ours 비율 1.442695 = 1/ln2, 환산 후 최대차 9.5e-07). `run()` 은 자연로그로
환산해 돌려주므로 우리 커널·FP64 참조와 같은 정의가 된다. 값을 맞추려고 바꾼 것이
아니라 두 구현의 LSE 정의를 일치시킨 것이다.

설계 §6 의 매핑 계약:
  - 원래 KV block 을 page 로 그대로 쓴다 (page_size = BK). K/V 를 query 마다 복제하지 않는다.
  - `(KV head, query position)` 을 논리 request 로 취급한다. 한 request 의 page 목록은
    그 query 가 선택한 KV block 목록이다 (CSR 을 그대로 indptr/indices 로 쓴다).
  - 절대 causal 경계는 마지막 page 의 유효 길이 `last_page_len` 으로 표현한다:
        last_page_len = min(BK, q_pos - last_block*BK + 1)
    선택된 block 중 최대 번호가 q_pos 를 담는 block 이면 정확히 부분 block 이 되고,
    그보다 앞이면 BK 가 된다.
  - 빈 선택 행은 native 매핑이 없다 → 명시적으로 미지원 처리한다 (설계 §6).

이것은 동등 연산의 native control 이며, MSA 저자 커널 전체를 대체하는 비교가 아니다.
"""

from __future__ import annotations

import numpy as np

try:
    import flashinfer
    import torch
    HAVE_FLASHINFER = True
    FLASHINFER_VERSION = flashinfer.__version__
    FLASHINFER_ERROR = None
except Exception as _e:                                     # pragma: no cover
    HAVE_FLASHINFER = False
    FLASHINFER_VERSION = None
    FLASHINFER_ERROR = f"{type(_e).__name__}: {_e}"


LN2 = 0.6931471805599453      # flashinfer LSE 가 log2 밑이므로 환산에 쓴다


class NativeUnsupported(RuntimeError):
    """native 경로가 이 입력을 표현할 수 없다 (우회하지 않고 그대로 올린다)."""


def build_metadata(ptr, block_ids, q_positions, bk, causal):
    """CSR + 절대 위치로부터 native page 메타데이터를 만든다.

    반환 (indptr[int32], indices[int32], last_page_len[int32]).
    빈 행이 있으면 NativeUnsupported.
    """
    ptr = np.asarray(ptr, dtype=np.int64)
    block_ids = np.asarray(block_ids, dtype=np.int32)
    nq = len(ptr) - 1
    rows = np.diff(ptr)
    if np.any(rows == 0):
        empty = int((rows == 0).sum())
        raise NativeUnsupported(
            f"빈 선택 행 {empty}개 — native paged 매핑에 대응하는 표현이 없다 (설계 §6)")

    last_page_len = np.empty(nq, dtype=np.int32)
    for i in range(nq):
        row = block_ids[ptr[i]:ptr[i + 1]]
        if np.any(np.diff(row) <= 0):
            raise NativeUnsupported(f"query {i} 행이 정렬/중복 계약을 위반")
        last_b = int(row[-1])
        if causal:
            valid = int(q_positions[i]) - last_b * bk + 1
            if valid <= 0:
                raise NativeUnsupported(
                    f"query {i}: 마지막 선택 block {last_b} 이 q_pos {q_positions[i]} 보다 뒤다")
            last_page_len[i] = min(bk, valid)
        else:
            last_page_len[i] = bk
    return ptr.astype(np.int32), block_ids, last_page_len


class NativeRunner:
    """FlashInfer BatchDecodeWithPagedKVCacheWrapper 기반 대조 실행기."""

    def __init__(self, device="cuda", workspace_mb=256):
        if not HAVE_FLASHINFER:
            raise NativeUnsupported(f"flashinfer 없음: {FLASHINFER_ERROR}")
        self.device = device
        self.ws = torch.empty(workspace_mb * 1024 * 1024, dtype=torch.uint8, device=device)
        self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self.ws, kv_layout="NHD")
        self._planned = None

    def plan(self, indptr, indices, last_page_len, *, g, d, bk, scale=None):
        """native 계획 (CPU plan 상당). fresh 측정에 이 시간이 들어간다.

        이 flashinfer 버전(0.6.x)은 `sm_scale` 을 run() 이 아니라 plan() 에서 받는다.
        지시서 §1-4 가 허용하는 버전 API 대응이며 알고리즘·dtype·mask 는 바꾸지 않는다.
        """
        dev = self.device
        self._ind = (torch.from_numpy(indptr).to(dev),
                     torch.from_numpy(indices).to(dev),
                     torch.from_numpy(last_page_len).to(dev))
        kw = dict(num_qo_heads=g, num_kv_heads=1, head_dim=d, page_size=bk,
                  q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
        if scale is not None:
            kw["sm_scale"] = float(scale)
        self.wrapper.plan(self._ind[0], self._ind[1], self._ind[2], **kw)
        self._planned = (g, d, bk)

    def run(self, q, k, v, *, bk, scale, return_lse=True):
        """q: [Nq, G, D], k/v: [Nk, D] (CUDA). 반환 (out[Nq,G,D], lse[Nq,G])."""
        if self._planned is None:
            raise RuntimeError("plan 을 먼저 불러야 한다")
        nq, g, d = q.shape
        nk = k.shape[0]
        n_pages = nk // bk
        if n_pages * bk != nk:
            raise NativeUnsupported(
                f"native paged 는 Nk={nk} 가 page_size={bk} 의 배수여야 한다 (tail block 미지원)")
        # [n_pages, page_size, num_kv_heads=1, head_dim] — 복사 없이 view
        kc = k.view(n_pages, bk, 1, d)
        vc = v.view(n_pages, bk, 1, d)
        res = self.wrapper.run(q, (kc, vc), return_lse=return_lse)
        if return_lse:
            out, lse = res
            return out, lse * LN2          # log2 -> 자연로그
        return res, None


def versions() -> dict:
    out = dict(flashinfer=FLASHINFER_VERSION, available=HAVE_FLASHINFER,
               error=FLASHINFER_ERROR)
    try:
        import torch as _t
        out["torch"] = _t.__version__
        out["cuda"] = _t.version.cuda
        if _t.cuda.is_available():
            out["device"] = _t.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import triton as _tr
        out["triton"] = _tr.__version__
    except Exception:
        pass
    return out
