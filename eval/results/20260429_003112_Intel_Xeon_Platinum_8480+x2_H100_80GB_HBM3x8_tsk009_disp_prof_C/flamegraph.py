#!/usr/bin/env python3
"""TSK_009 dispatcher profile flame graph (SVG, pure Python).

Compares wall-time breakdown across modes for input_heavy
(15360 input × 1024 output × 100 prompts) on prod (Llama-3.3-70B + TP=8).
"""
from __future__ import annotations

import os
from xml.sax.saxutils import escape

# --- measured values (s, wall-time on 100 prompts batch) -------------
totals = {
    "A: cold-tier OFF":               164.1,
    "B: cold-tier ON, IDE_006 OFF":   200.0,
    "C1: IDE_006 ON, fix v1":         219.3,
    "C2: IDE_006 ON, fix v2 (fast)":  215.2,
}

A = totals["A: cold-tier OFF"]
B = totals["B: cold-tier ON, IDE_006 OFF"]
C1 = totals["C1: IDE_006 ON, fix v1"]
C2 = totals["C2: IDE_006 ON, fix v2 (fast)"]

components = {
    "A: cold-tier OFF": [
        ("baseline GPU forward + decode", A, "#4ECDC4"),
    ],
    "B: cold-tier ON, IDE_006 OFF": [
        ("baseline GPU forward + decode", A, "#4ECDC4"),
        ("vLLM cold-tier store/metadata", B - A, "#FFA500"),
    ],
    "C1: IDE_006 ON, fix v1": [
        ("baseline GPU forward + decode", A, "#4ECDC4"),
        ("vLLM cold-tier store/metadata", B - A, "#FFA500"),
        ("hot_cold_attention pre-check (8.4 ms × 5200 calls/wkr)", C1 - B, "#FF6B6B"),
    ],
    "C2: IDE_006 ON, fix v2 (fast-path bypass)": [
        ("baseline GPU forward + decode", A, "#4ECDC4"),
        ("vLLM cold-tier store/metadata", B - A, "#FFA500"),
        ("dispatcher fast-path (80.6 μs × 13800 calls/wkr)", C2 - B, "#9B59B6"),
    ],
}

# --- SVG layout ------------------------------------------------------
W = 1200          # canvas width
H_BAR = 60        # bar height
Y_GAP = 25
LABEL_W = 320     # left label width
RIGHT_PAD = 130   # right side total label
PLOT_W = W - LABEL_W - RIGHT_PAD
TOP_PAD = 80
BOT_PAD = 60
N = len(components)
H = TOP_PAD + N * (H_BAR + Y_GAP) + BOT_PAD

max_total = max(t for t in totals.values()) + 5
SCALE = PLOT_W / max_total


def text(x, y, s, *, size=12, weight="normal", anchor="start", fill="#000"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'font-family="DejaVu Sans, sans-serif" '
            f'font-weight="{weight}" text-anchor="{anchor}" '
            f'fill="{fill}">{escape(s)}</text>')


lines = []
lines.append(f'<?xml version="1.0" encoding="UTF-8"?>')
lines.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}">'
)
# background
lines.append(f'<rect width="{W}" height="{H}" fill="white"/>')
# title
lines.append(text(W / 2, 28,
    "TSK_009 dispatcher profile — overhead breakdown",
    size=18, weight="bold", anchor="middle"))
lines.append(text(W / 2, 50,
    "input_heavy: 100 prompts × 15360 input × 1024 output  ·  Llama-3.3-70B + TP=8",
    size=12, anchor="middle", fill="#444"))

# bars
for i, (mode, segs) in enumerate(components.items()):
    y = TOP_PAD + i * (H_BAR + Y_GAP)
    cy = y + H_BAR / 2
    # left label
    lines.append(text(LABEL_W - 10, cy + 4, mode, size=12, weight="bold", anchor="end"))
    # segments
    x = LABEL_W
    total = 0.0
    for name, dt, color in segs:
        w = dt * SCALE
        total += dt
        lines.append(
            f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{H_BAR}" '
            f'fill="{color}" stroke="black" stroke-width="0.7"/>'
        )
        # segment label
        if w > 90:
            lines.append(text(x + w / 2, cy - 2, name,
                              size=11, anchor="middle", weight="bold"))
            lines.append(text(x + w / 2, cy + 14, f"{dt:.1f} s",
                              size=10.5, anchor="middle"))
        elif w > 30:
            lines.append(text(x + w / 2, cy + 4, f"{dt:.1f}s",
                              size=10, anchor="middle"))
        x += w
    # right total
    lines.append(text(x + 8, cy + 4, f"total {total:.1f} s",
                      size=12, weight="bold"))

# x-axis
ax_y = TOP_PAD + N * (H_BAR + Y_GAP) + 5
lines.append(
    f'<line x1="{LABEL_W}" y1="{ax_y}" x2="{LABEL_W + PLOT_W}" '
    f'y2="{ax_y}" stroke="#888" stroke-width="0.8"/>'
)
for s in (0, 50, 100, 150, 200, 250):
    if s > max_total:
        break
    tx = LABEL_W + s * SCALE
    lines.append(
        f'<line x1="{tx}" y1="{ax_y}" x2="{tx}" y2="{ax_y + 5}" '
        f'stroke="#888" stroke-width="0.8"/>'
    )
    lines.append(text(tx, ax_y + 18, f"{s}s", size=10, anchor="middle", fill="#444"))
lines.append(text(LABEL_W + PLOT_W / 2, ax_y + 38,
                  "wall-time (s)", size=11, anchor="middle", fill="#444"))

# fix v2 → v1 delta annotation
delta = C1 - C2
lines.append(text(LABEL_W + 10, H - 15,
    f"fix v2 vs v1: {delta:.1f} s 단축 (dispatcher path 8.4 ms → 80.6 μs/call)",
    size=11, fill="#444"))

lines.append("</svg>")

out = os.path.join(os.path.dirname(__file__), "flamegraph.svg")
with open(out, "w") as f:
    f.write("\n".join(lines))
print(f"saved: {out}")
