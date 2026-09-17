#!/usr/bin/env python3
"""IDE_075 A01 — 안전한 시각 파싱·시간창 도구 (지시서 부록 B.1/B.5). 정수 ns, 명시적 시간대, 파싱 실패는 예외(호출자가 INVALID_TIMESTAMP 기록).
Interval 집계: 완전 포함 표본의 시간 가중 평균(mean_full) 과 경계 비례 추정(mean_overlap_est, 균일 가정 표시) 분리. 중복 표본 coverage 중복 계수 금지."""
from __future__ import annotations
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import List, Tuple, Dict, Optional

TIME_RE = re.compile(r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})(?:\.(?P<frac>\d{1,9}))?$")
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def local_timestamp_ns(date_text: str, time_text: str, timezone_name: Optional[str]) -> int:
    """현지 날짜·시각 → UTC epoch ns (정수). 시간대 미확정·모호 시각·형식 오류는 ValueError."""
    if not timezone_name: raise ValueError("UNRESOLVED_TIMEZONE")
    m = TIME_RE.fullmatch((time_text or "").strip())
    if m is None: raise ValueError(f"INVALID_TIME: {time_text!r}")
    try: tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc: raise ValueError(f"UNKNOWN_TIMEZONE: {timezone_name}") from exc
    day = datetime.strptime((date_text or "").strip(), "%Y-%m-%d")   # 9월 31일 등은 여기서 ValueError
    naive = day.replace(hour=int(m["h"]), minute=int(m["m"]), second=int(m["s"]))
    a = naive.replace(tzinfo=tz, fold=0); b = naive.replace(tzinfo=tz, fold=1)
    if a.utcoffset() != b.utcoffset(): raise ValueError("AMBIGUOUS_OR_NONEXISTENT_LOCAL_TIME")
    utc = a.astimezone(timezone.utc)
    if utc.astimezone(tz).replace(tzinfo=None) != naive: raise ValueError("NONEXISTENT_LOCAL_TIME")
    d = utc - EPOCH; seconds = d.days * 86400 + d.seconds
    return seconds * 1_000_000_000 + int((m["frac"] or "").ljust(9, "0"))


def iso_timestamp_ns(text: str) -> int:
    """ISO8601(오프셋 필수) → epoch ns 정수. 소수초 9자리까지 보존 (fromisoformat 은 µs 까지만이라 직접 파싱)."""
    t = text.strip(); mm = re.fullmatch(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})", t)
    if not mm: raise ValueError(f"INVALID_ISO: {text!r}")
    base = datetime.fromisoformat(f"{mm.group(1)}T{mm.group(2)}{'+00:00' if mm.group(4) == 'Z' else mm.group(4)}")
    d = base.astimezone(timezone.utc) - EPOCH; return (d.days * 86400 + d.seconds) * 1_000_000_000 + int((mm.group(3) or "").ljust(9, "0"))


def epoch_float_to_ns(x: float) -> int:
    """float epoch(초) → ns. float 정밀도 한계(µs 급) 는 호출자가 precision 으로 기록."""
    return int(round(x * 1e9))


Interval = Tuple[int, int]   # [start_ns, end_ns)


def classify_samples(samples: List[Tuple[Interval, float]], W: Interval) -> Dict:
    """samples: [((a,b), value)], W=(s,e). 완전 포함 / 경계 / 창 밖 / 중복 을 계수하고 mean_full·mean_overlap_est 를 계산."""
    s, e = W; full = []; boundary = []; outside = 0; seen = set(); dup = 0
    for (a, b), v in samples:
        if (a, b) in seen: dup += 1; continue
        seen.add((a, b))
        if b <= s or a >= e: outside += 1; continue
        if a >= s and b <= e: full.append(((a, b), v))
        else: boundary.append(((a, b), v))
    def union_len(ivs):
        ivs = sorted(ivs); tot = 0; cur = None
        for a, b in ivs:
            if cur is None or a > cur[1]:
                if cur: tot += cur[1] - cur[0]
                cur = [a, b]
            else: cur[1] = max(cur[1], b)
        if cur: tot += cur[1] - cur[0]
        return tot
    W_len = e - s; full_cov = union_len([iv for iv, _ in full]) / W_len if W_len else None
    mean_full = (sum(v * (b - a) for (a, b), v in full) / sum(b - a for (a, b), _ in full)) if full else None
    ov = [((max(a, s), min(b, e)), v) for (a, b), v in full + boundary]
    mean_ov = (sum(v * (b - a) for (a, b), v in ov) / sum(b - a for (a, b), _ in ov)) if ov else None
    return {"n_full": len(full), "n_boundary": len(boundary), "n_outside": outside, "n_duplicate": dup, "full_sample_coverage": full_cov, "mean_full": mean_full,
            "mean_overlap_est": mean_ov, "mean_overlap_assumption": "each value represents a constant rate over its sample interval", "overlap_coverage": (union_len([iv for iv, _ in ov]) / W_len) if (ov and W_len) else None}


def _selftest():
    fails = []
    def chk(n, c):
        if not c: fails.append(n)
    x = local_timestamp_ns("2026-09-17", "13:37:59.527", "Asia/Seoul"); chk("T01", x % 1_000_000_000 == 527_000_000)
    chk("T02", x == local_timestamp_ns("2026-09-17", "04:37:59.527", "UTC") == iso_timestamp_ns("2026-09-17T04:37:59.527+00:00"))
    chk("T03", local_timestamp_ns("2026-09-17", "13:37:59", "Asia/Seoul") % 1_000_000_000 == 0)
    for n, args in (("T04", ("2026-09-31", "13:37:59", "Asia/Seoul")), ("T05", ("2026-09-17", "13:37:59", None)), ("T06a", ("2026-09-17", "", "Asia/Seoul")), ("T06b", ("2026-09-17", "13:3x:59", "Asia/Seoul"))):
        try: local_timestamp_ns(*args); fails.append(n)
        except ValueError: pass
    chk("T07", local_timestamp_ns("2026-09-17", "13:37:59.527000200", "Asia/Seoul") - x == 200)
    # T08 clock jump: 단일 offset 금지 → anchors 별 offset 목록을 그대로 두는 것은 호출자 계약 (여기선 두 anchor offset 차이 검출 함수 없음) → 표시만
    r = classify_samples([((11_000, 12_000), 1.0)], (10_000, 20_000)); chk("W01", r["n_full"] == 1 and abs(r["full_sample_coverage"] - 0.1) < 1e-9)
    r = classify_samples([((9_500, 10_500), 2.0)], (10_000, 20_000)); chk("W02", r["n_boundary"] == 1 and r["n_full"] == 0 and r["mean_full"] is None and r["mean_overlap_est"] == 2.0)
    r = classify_samples([((20_000, 21_000), 1.0)], (10_000, 20_000)); chk("W03", r["n_outside"] == 1 and r["mean_overlap_est"] is None)
    r = classify_samples([((11_000, 12_000), 1.0), ((11_000, 12_000), 1.0)], (10_000, 20_000)); chk("W05", r["n_duplicate"] == 1 and abs(r["full_sample_coverage"] - 0.1) < 1e-9)
    r = classify_samples([((i * 1000, (i + 1) * 1000), (5.0 if 10_000 <= i * 1000 < 20_000 else 1.0)) for i in range(30)], (10_000, 20_000)); chk("W06", r["mean_full"] == 5.0 and r["n_outside"] == 20)
    return fails


if __name__ == "__main__":
    f = _selftest(); print("SELFTEST", "PASS" if not f else "FAIL", f); raise SystemExit(1 if f else 0)
