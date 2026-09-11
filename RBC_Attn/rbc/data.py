"""합성 mask 생성과 실제 capture 저장/적재.

계약 (설계 §2, §5):
  Q: [Hkv, Nq, G, D]     K, V: [Hkv, Nk, D]
  mask: query-major CSR (Hkv-major). ptr[h] 길이 Nq+1, block_ids 는 KV block 인덱스.
        같은 query 행 안에서 중복 없이 오름차순 정렬.
  q_positions: 절대 query 위치 (causal 경계 판정용).
  source_dtype: 원 표현 ('bfloat16' | 'float16' | 'float32').
"""

from __future__ import annotations

import numpy as np

VALID_DTYPES = ("bfloat16", "float16", "float32")


# --------------------------------------------------------------------------
# CSR 유틸
# --------------------------------------------------------------------------
def validate_csr(ptr: np.ndarray, block_ids: np.ndarray, nq: int, n_blocks: int) -> None:
    """query-major CSR 의 계약을 검사한다. 위반하면 ValueError."""
    if ptr.ndim != 1 or ptr.shape[0] != nq + 1:
        raise ValueError(f"ptr 길이는 Nq+1={nq+1} 이어야 한다 (받음 {ptr.shape})")
    if ptr[0] != 0:
        raise ValueError("ptr[0] 은 0 이어야 한다")
    if np.any(np.diff(ptr) < 0):
        raise ValueError("ptr 은 비감소여야 한다")
    if ptr[-1] != block_ids.shape[0]:
        raise ValueError(f"ptr[-1]={ptr[-1]} 과 block_ids 길이 {block_ids.shape[0]} 불일치")
    if block_ids.size and (block_ids.min() < 0 or block_ids.max() >= n_blocks):
        raise ValueError(f"block id 범위 밖 (0..{n_blocks-1})")
    for i in range(nq):
        row = block_ids[ptr[i]:ptr[i + 1]]
        if row.size > 1 and np.any(np.diff(row) <= 0):
            raise ValueError(f"query {i} 행이 정렬돼 있지 않거나 중복이 있다")


def causal_filter(ptr, block_ids, q_positions, bk):
    """절대 causal 경계를 적용해 q_pos 보다 완전히 뒤인 KV block 을 제거한다.

    block b 는 [b*bk, (b+1)*bk) 를 담는다. 그 시작 위치가 q_pos 보다 크면 전부 미래다.
    부분 겹침 block 은 남기고 커널 안에서 원소 단위로 마스킹한다.
    """
    nq = len(ptr) - 1
    out_ptr = np.zeros_like(ptr)
    keep = []
    for i in range(nq):
        row = block_ids[ptr[i]:ptr[i + 1]]
        if row.size:
            row = row[row * bk <= q_positions[i]]
        keep.append(row)
        out_ptr[i + 1] = out_ptr[i] + row.size
    return out_ptr, (np.concatenate(keep) if keep else np.zeros(0, dtype=block_ids.dtype))


# --------------------------------------------------------------------------
# 합성 패턴
# --------------------------------------------------------------------------
def make_pattern(pattern: str, hkv: int, nq: int, n_blocks: int, sel: int,
                 rng: np.random.Generator):
    """query-major CSR 를 만든다. 반환 (ptr[hkv, nq+1], block_ids_list[hkv])."""
    ptrs, ids = [], []
    for h in range(hkv):
        rows = []
        if pattern == "common_private":
            # 설계 §3: 공통 sel/2 개 + 자기만 쓰는 sel/2 개 (겹치지 않음)
            n_common = max(1, sel // 2)
            n_priv = sel - n_common
            common = rng.choice(n_blocks, size=n_common, replace=False)
            pool = np.setdiff1d(np.arange(n_blocks), common)
            if pool.size < n_priv * nq:
                raise ValueError("private block 을 겹치지 않게 배정할 블록이 부족하다")
            perm = rng.permutation(pool)
            for i in range(nq):
                priv = perm[i * n_priv:(i + 1) * n_priv]
                rows.append(np.unique(np.concatenate([common, priv])))
        elif pattern == "random":
            for _ in range(nq):
                rows.append(np.sort(rng.choice(n_blocks, size=sel, replace=False)))
        elif pattern == "clustered":
            # 연속 구간 + 최근 window. Q-outer 가 잘하는 반례.
            for i in range(nq):
                start = int(rng.integers(0, max(1, n_blocks - sel)))
                rows.append(np.arange(start, start + sel) % n_blocks)
                rows[-1] = np.unique(rows[-1])
        elif pattern == "disjoint":
            # query 마다 완전히 분리된 구간. 묶을 것이 없는 반례.
            per = n_blocks // max(1, nq)
            if per < sel:
                raise ValueError("disjoint 패턴에 블록이 부족하다")
            for i in range(nq):
                base = i * per
                rows.append(np.arange(base, base + sel))
        else:
            raise ValueError(f"알 수 없는 패턴: {pattern}")
        p = np.zeros(nq + 1, dtype=np.int64)
        for i, r in enumerate(rows):
            p[i + 1] = p[i] + r.size
        ptrs.append(p)
        ids.append(np.concatenate(rows).astype(np.int32) if rows else np.zeros(0, np.int32))
    return np.stack(ptrs), ids


def make_synthetic(pattern="common_private", hkv=2, nq=256, nk=65536, g=8, d=128,
                   bk=128, sel=16, seed=0, dtype="bfloat16", causal=True):
    """합성 입력 한 벌을 만든다. 반환 dict (numpy, FP32 컨테이너)."""
    rng = np.random.default_rng(seed)
    n_blocks = nk // bk
    ptr, ids = make_pattern(pattern, hkv, nq, n_blocks, sel, rng)

    # 절대 query 위치: 문맥 끝에 붙어 있는 prefill 묶음으로 둔다.
    q_positions = np.arange(nk - nq, nk, dtype=np.int64)
    if causal:
        new_ptr, new_ids = [], []
        for h in range(hkv):
            p, b = causal_filter(ptr[h], ids[h], q_positions, bk)
            new_ptr.append(p); new_ids.append(b)
        ptr, ids = np.stack(new_ptr), new_ids

    scale = 1.0 / np.sqrt(d)
    q = rng.standard_normal((hkv, nq, g, d), dtype=np.float32) * 0.5
    k = rng.standard_normal((hkv, nk, d), dtype=np.float32) * 0.5
    v = rng.standard_normal((hkv, nk, d), dtype=np.float32) * 0.5
    return dict(q=q, k=k, v=v, ptr=ptr, block_ids=ids, bk=bk,
                q_positions=q_positions, causal=causal, source_dtype=dtype,
                scale=np.float32(scale))


# --------------------------------------------------------------------------
# capture 저장/적재 (설계 §7)
# --------------------------------------------------------------------------
def save_capture(path, q, k, v, ptr, block_ids, bk, q_positions, causal=True,
                 source_dtype="bfloat16"):
    """실제 모델 capture 를 NPZ 로 저장한다.

    q: [Hkv,Nq,G,D], k/v: [Hkv,Nk,D] (FP32 컨테이너. BF16 값은 정확히 담긴다)
    ptr: [Hkv, Nq+1], block_ids: [Hkv] 의 리스트 또는 이어붙인 배열 + 길이
    """
    if source_dtype not in VALID_DTYPES:
        raise ValueError(f"source_dtype 은 {VALID_DTYPES} 중 하나여야 한다")
    ptr = np.asarray(ptr, dtype=np.int64)
    hkv, nq = q.shape[0], q.shape[1]
    if isinstance(block_ids, (list, tuple)):
        lens = np.array([b.size for b in block_ids], dtype=np.int64)
        flat = np.concatenate([np.asarray(b, dtype=np.int32) for b in block_ids])
    else:
        flat = np.asarray(block_ids, dtype=np.int32)
        lens = np.array([ptr[h][-1] for h in range(hkv)], dtype=np.int64)
    n_blocks = (k.shape[1] + bk - 1) // bk
    off = 0
    for h in range(hkv):
        validate_csr(ptr[h], flat[off:off + lens[h]], nq, n_blocks)
        off += lens[h]
    np.savez(path, q=q.astype(np.float32), k=k.astype(np.float32), v=v.astype(np.float32),
             ptr=ptr, block_ids=flat, block_lens=lens, bk=np.int64(bk),
             q_positions=np.asarray(q_positions, dtype=np.int64),
             causal=np.bool_(causal), source_dtype=np.str_(source_dtype))


def load_capture(path):
    z = np.load(path, allow_pickle=False)
    lens = z["block_lens"]
    flat = z["block_ids"]
    ids, off = [], 0
    for L in lens:
        ids.append(flat[off:off + L]); off += int(L)
    d = z["q"].shape[-1]
    return dict(q=z["q"], k=z["k"], v=z["v"], ptr=z["ptr"], block_ids=ids,
                bk=int(z["bk"]), q_positions=z["q_positions"], causal=bool(z["causal"]),
                source_dtype=str(z["source_dtype"]), scale=np.float32(1.0 / np.sqrt(d)))


# --------------------------------------------------------------------------
# 참조 구현 (numpy FP64) — 정확성 판정의 기준
# --------------------------------------------------------------------------
def reference_attention(sample, head=None):
    """선택된 KV edge 만으로 attention 을 FP64 로 계산한다.

    반환 (out[hkv,nq,g,d] float64, lse[hkv,nq,g] float64).
    빈 행은 out=0, lse=-inf.
    """
    q, k, v = sample["q"].astype(np.float64), sample["k"].astype(np.float64), sample["v"].astype(np.float64)
    ptr, ids, bk = sample["ptr"], sample["block_ids"], sample["bk"]
    qpos, causal, scale = sample["q_positions"], sample["causal"], float(sample["scale"])
    hkv, nq, g, d = q.shape
    heads = range(hkv) if head is None else [head]
    out = np.zeros((hkv, nq, g, d), dtype=np.float64)
    lse = np.full((hkv, nq, g), -np.inf, dtype=np.float64)
    nk = k.shape[1]
    for h in heads:
        for i in range(nq):
            blocks = ids[h][ptr[h][i]:ptr[h][i + 1]]
            if blocks.size == 0:
                continue
            cols = []
            for b in blocks:
                lo, hi = int(b) * bk, min(int(b) * bk + bk, nk)
                idx = np.arange(lo, hi)
                if causal:
                    idx = idx[idx <= qpos[i]]
                cols.append(idx)
            cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
            if cols.size == 0:
                continue
            kk, vv = k[h, cols], v[h, cols]              # [S,D]
            s = (q[h, i] @ kk.T) * scale                  # [G,S]
            m = s.max(axis=1, keepdims=True)
            p = np.exp(s - m)
            l = p.sum(axis=1, keepdims=True)
            out[h, i] = (p @ vv) / l
            lse[h, i] = (m[:, 0] + np.log(l[:, 0]))
    return out, lse
