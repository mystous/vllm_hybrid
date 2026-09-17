#!/usr/bin/env python3
"""IDE_074 M1 — GPU interval 집계 (지시서 §4.2 정의).
장치별로 실제 GPU operation(kernel/gpu_memcpy/gpu_memset)만 측정창 W 에 clip 한 뒤
duration 합 / interval union / idle / compute·memcpy union·overlap / MoE 경계 사이 유휴를 계산한다.
annotation(gpu_user_annotation 등)은 busy 합계에 넣지 않는다. TP rank 를 하나로 합치지 않는다.
단위: 입력 interval 은 [start, end) (float, µs). 결과 dict 의 키는 §4.2 이름을 그대로 쓴다."""
from typing import Iterable, List, Tuple, Dict, Optional

Interval = Tuple[float, float]


def clip(ivs: Iterable[Interval], W: Interval) -> List[Interval]:
    """W=[s,e) 로 clip. 길이 0 이하 구간은 버린다."""
    s, e = W
    out = []
    for a, b in ivs:
        a2, b2 = max(a, s), min(b, e)
        if b2 > a2: out.append((a2, b2))
    return out


def union(ivs: Iterable[Interval]) -> List[Interval]:
    """정렬 후 병합. 포함된 interval 은 흡수 (최종 end 는 max)."""
    ivs = sorted(ivs)
    out: List[List[float]] = []
    for a, b in ivs:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return [(a, b) for a, b in out]


def length(ivs: Iterable[Interval]) -> float:
    return sum(b - a for a, b in ivs)


def intersect(u1: List[Interval], u2: List[Interval]) -> List[Interval]:
    """두 union 목록의 교집합 (둘 다 병합·정렬 상태여야 함)."""
    i = j = 0; out = []
    while i < len(u1) and j < len(u2):
        a = max(u1[i][0], u2[j][0]); b = min(u1[i][1], u2[j][1])
        if b > a: out.append((a, b))
        if u1[i][1] < u2[j][1]: i += 1
        else: j += 1
    return out


def device_window_metrics(kernels: Iterable[Interval], memcpys: Iterable[Interval], memsets: Iterable[Interval], W: Interval, annotations: Iterable[Interval] = ()) -> Dict[str, float]:
    """한 장치·한 측정창의 §4.2 집계. annotations 는 받아도 busy 에 더하지 않는다 (별도 필드로만 보고)."""
    k = clip(kernels, W); c = clip(memcpys, W); m = clip(memsets, W)
    ops = k + c + m
    busy_u = union(ops); comp_u = union(k); copy_u = union(c)
    wlen = W[1] - W[0]
    busy = length(busy_u)
    r = {
        "window_start_us": W[0], "window_end_us": W[1], "window_len_us": wlen,
        "gpu_op_count": len(ops), "kernel_count": len(k), "memcpy_count": len(c), "memset_count": len(m),
        "gpu_op_duration_sum_us": length(ops),
        "gpu_busy_union_us": busy,
        "gpu_idle_union_us": wlen - busy,
        "gpu_busy_fraction": (busy / wlen) if wlen > 0 else None,
        "compute_union_us": length(comp_u),
        "memcpy_union_us": length(copy_u),
        "compute_memcpy_overlap_us": length(intersect(comp_u, copy_u)),
        "annotation_duration_sum_us": length(clip(annotations, W)),   # 참고값. busy 에 미포함
        "busy_union_last_end_us": busy_u[-1][1] if busy_u else None,
    }
    assert r["gpu_busy_union_us"] <= wlen + 1e-6, "busy union 이 측정창보다 큼 → INVALID_AGGREGATION"
    return r


def gap_between(G: Interval, all_ops: Iterable[Interval]) -> Dict[str, float]:
    """두 검증된 MoE 경계 사이 G=[prev_end,next_start) 의 유휴 = |G| - |union(all ops clipped to G)|.
    G 이전에 시작해 내부로 걸쳐 들어온 작업도 clip 으로 포함된다. 마지막 tail 도 포함된다."""
    inside = union(clip(all_ops, G))
    glen = G[1] - G[0]
    busy = length(inside)
    return {"gap_start_us": G[0], "gap_end_us": G[1], "gap_len_us": glen, "busy_in_gap_us": busy, "idle_in_gap_us": glen - busy, "ops_in_gap": len(inside)}


class UnionIndex:
    """all_ops 의 union 을 한 번 만들고 임의 구간의 busy 길이를 O(log n) 으로 구한다 (gap_between 과 같은 정의)."""
    def __init__(self, all_ops: Iterable[Interval]):
        import bisect
        self.u = union(all_ops); self.starts = [a for a, b in self.u]; self.ends = [b for a, b in self.u]
        self.pref = [0.0]
        for a, b in self.u: self.pref.append(self.pref[-1] + (b - a))
        self._bisect = bisect
    def busy(self, G: Interval) -> Tuple[float, int]:
        s, e = G
        if e <= s or not self.u: return 0.0, 0
        i = self._bisect.bisect_right(self.ends, s)            # 첫 interval with end > s
        j = self._bisect.bisect_left(self.starts, e)           # intervals with start < e → [i, j)
        if j <= i: return 0.0, 0
        tot = self.pref[j] - self.pref[i]
        tot -= max(0.0, s - self.starts[i]); tot -= max(0.0, self.ends[j - 1] - e)
        return tot, j - i


def inter_boundary_gaps(boundaries: List[Interval], all_ops: List[Interval]) -> List[Dict[str, float]]:
    """boundaries: 검증된 MoE 경계 interval 목록 (정렬). 인접 쌍 (prev_end, next_start) 마다 gap_between 과 동일 정의 (UnionIndex 로 계산)."""
    idx = UnionIndex(all_ops); out = []
    for (a0, a1), (b0, b1) in zip(boundaries, boundaries[1:]):
        if b0 > a1:
            busy, n = idx.busy((a1, b0)); out.append({"gap_start_us": a1, "gap_end_us": b0, "gap_len_us": b0 - a1, "busy_in_gap_us": busy, "idle_in_gap_us": (b0 - a1) - busy, "ops_in_gap": n})
        else: out.append({"gap_start_us": a1, "gap_end_us": b0, "gap_len_us": 0.0, "busy_in_gap_us": 0.0, "idle_in_gap_us": 0.0, "ops_in_gap": 0})
    return out


# ---------------- 합성 단위시험 (지시서 §4.3, 임의 µs) ----------------
def _selftest() -> List[str]:
    fails = []
    def chk(name, got, exp):
        if abs(got - exp) > 1e-9: fails.append(f"{name}: got {got} expected {exp}")
    # 1 중첩·annotation
    r = device_window_metrics([(1, 5), (3, 7)], [(6, 9)], [], (0, 10), annotations=[(0, 10)])
    chk("op_sum", r["gpu_op_duration_sum_us"], 11); chk("compute_union", r["compute_union_us"], 6); chk("copy_union", r["memcpy_union_us"], 3)
    chk("overlap", r["compute_memcpy_overlap_us"], 1); chk("busy", r["gpu_busy_union_us"], 8); chk("idle", r["gpu_idle_union_us"], 2)
    # 2 연속 커널
    r = device_window_metrics([(1, 3), (3, 5)], [], [], (0, 6)); chk("contig_busy", r["gpu_busy_union_us"], 4)
    g = inter_boundary_gaps([(1, 3), (3, 5)], [(1, 3), (3, 5)]); chk("contig_gap", g[0]["gap_len_us"], 0)
    # 3 중간 작업 없음
    chk("empty_gap", gap_between((2, 8), [])["idle_in_gap_us"], 6)
    # 4 마지막 tail
    chk("tail_gap", gap_between((2, 8), [(3, 5)])["idle_in_gap_us"], 4)
    # 5 경계 밖에서 진입
    g = gap_between((2, 8), [(0, 3), (7, 10)]); chk("enter_busy", g["busy_in_gap_us"], 2); chk("enter_gap", g["idle_in_gap_us"], 4)
    # 6 측정창 clip
    r = device_window_metrics([(-2, 2), (8, 12)], [], [], (0, 10)); chk("clip_busy", r["gpu_busy_union_us"], 4); chk("clip_idle", r["gpu_idle_union_us"], 6)
    # 7 포함된 interval
    r = device_window_metrics([(1, 9), (2, 3)], [], [], (0, 10)); chk("contained_busy", r["gpu_busy_union_us"], 8); chk("contained_end", r["busy_union_last_end_us"], 9)
    # 8 장치 분리
    r0 = device_window_metrics([(0, 10)], [], [], (0, 10)); r1 = device_window_metrics([], [], [], (0, 10))
    chk("dev0_frac", r0["gpu_busy_fraction"], 1.0); chk("dev1_frac", r1["gpu_busy_fraction"], 0.0)
    # UnionIndex ≡ gap_between 검증 (§4.3 3~5 번 재사용)
    for G, ops_, exp in (((2, 8), [], 6), ((2, 8), [(3, 5)], 4), ((2, 8), [(0, 3), (7, 10)], 4)):
        b, n = UnionIndex(ops_).busy(G); chk(f"unionindex{G}{ops_}", (G[1] - G[0]) - b, exp)
    return fails


if __name__ == "__main__":
    f = _selftest()
    print("SELFTEST", "PASS" if not f else "FAIL", f)
    raise SystemExit(1 if f else 0)
