#!/usr/bin/env python3
"""SUB_201 L1 KV quant — runs/*.json + *_gpu_post.csv 집계.

Output: 모델 별 baseline(auto) vs fp8 KV 비교 표 (markdown).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS = Path(__file__).parent / "runs"


def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        return {"_error": repr(e)}


def load_gpu_mem(p: Path) -> tuple[int, int] | tuple[None, None]:
    """returns (sum_used_MiB, n_gpus)."""
    if not p.exists():
        return None, None
    used_total = 0
    n = 0
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = [p.strip() for p in ln.split(",")]
        try:
            used = int(parts[1].split()[0])  # "9232 MiB" → 9232
        except Exception:  # noqa: BLE001
            continue
        used_total += used
        n += 1
    return used_total, n


def row(tag: str) -> dict:
    summ = RUNS / f"{tag}.json"
    gpu_post = RUNS / f"{tag}_gpu_post.csv"
    gpu_boot = RUNS / f"{tag}_gpu_boot.csv"
    out = {
        "tag": tag,
        "exists": summ.exists(),
    }
    if summ.exists():
        d = load_json(summ)
        out.update(
            {
                "output_tps": d.get("output_tps"),
                "ttft_p50": d.get("ttft_ms_p50"),
                "ttft_p99": d.get("ttft_ms_p99"),
                "tpot_p50": d.get("tpot_ms_p50"),
                "tpot_p99": d.get("tpot_ms_p99"),
                "n": d.get("n"),
                "n_ok": d.get("n_ok"),
                "gpu_util_mean": d.get("gpu_util_mean"),
                "gpu_mem_mib_mean": d.get("gpu_mem_mib_mean"),
                "cpu_util_mean": d.get("cpu_util_mean"),
            }
        )
    used_post, n_post = load_gpu_mem(gpu_post)
    used_boot, n_boot = load_gpu_mem(gpu_boot)
    if used_post is not None:
        out["gpu_used_post_MiB_sum"] = used_post
        out["gpu_used_post_MiB_per"] = round(used_post / max(n_post, 1), 0)
    if used_boot is not None:
        out["gpu_used_boot_MiB_sum"] = used_boot
        out["gpu_used_boot_MiB_per"] = round(used_boot / max(n_boot, 1), 0)
    return out


def pct(new, base) -> str:
    if new is None or base is None or base == 0:
        return "-"
    return f"{(new - base) / base * 100:+.1f}%"


def cmp_table(model_label: str, base_tag: str, fp8_tag: str) -> str:
    base = row(base_tag)
    new = row(fp8_tag)
    lines = [
        f"### {model_label}",
        "",
        "| metric | baseline (auto) | fp8 KV | Δ% |",
        "|---|---:|---:|---:|",
    ]
    for k, label in [
        ("output_tps", "output_tps (gen tok/s, aggregate)"),
        ("ttft_p50", "TTFT p50 (ms)"),
        ("ttft_p99", "TTFT p99 (ms)"),
        ("tpot_p50", "TPOT p50 (ms/tok)"),
        ("tpot_p99", "TPOT p99 (ms/tok)"),
        ("gpu_mem_mib_mean", "GPU mem mean (MiB)"),
        ("gpu_used_boot_MiB_per", "GPU used @ boot (MiB/dev)"),
        ("gpu_used_post_MiB_per", "GPU used @ post (MiB/dev)"),
        ("n_ok", "n_ok / n"),
    ]:
        b = base.get(k)
        n = new.get(k)
        if k == "n_ok":
            b_str = f"{base.get('n_ok')} / {base.get('n')}"
            n_str = f"{new.get('n_ok')} / {new.get('n')}"
            delta = "-"
        else:
            b_str = "-" if b is None else f"{b:.1f}" if isinstance(b, (int, float)) else str(b)
            n_str = "-" if n is None else f"{n:.1f}" if isinstance(n, (int, float)) else str(n)
            # 메트릭별 방향: TPS 위로, latency 아래로, mem 아래로
            delta = pct(n, b)
            if k.startswith("ttft") or k.startswith("tpot") or "mem" in k or "used" in k:
                # latency·mem 은 음수 % 가 개선
                pass
        lines.append(f"| {label} | {b_str} | {n_str} | {delta} |")
    lines.append("")
    return "\n".join(lines)


def main():
    pairs = [
        ("M1 Qwen2.5-7B-Instruct (TP=2, GPU 0-1)", "M1_qwen7b_auto", "M1_qwen7b_fp8"),
        ("M2 Llama-3.1-70B-Instruct (TP=4, GPU 0-3)", "M2_llama70b_auto", "M2_llama70b_fp8"),
        ("M3 DeepSeek-R1 671B (TP=8, GPU 0-7)", "M3_r1_auto", "M3_r1_fp8"),
    ]
    out = ["# L1 KV cache dtype lever — measurement summary", ""]
    for label, base, new in pairs:
        if (RUNS / f"{base}.json").exists() or (RUNS / f"{new}.json").exists():
            out.append(cmp_table(label, base, new))
        else:
            out.append(f"### {label}\n\n_no data_\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
