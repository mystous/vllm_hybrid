#!/usr/bin/env python3
"""IDE_023 / TSK_043 백서 그림 생성.

구성도는 SVG 로 작성해 `rsvg-convert` 로 PNG 변환, 결과 차트는 matplotlib.
모든 수치는 `eval/results/20260827_*` 원본에서 읽거나 보고서 표에서 인용한다.
"""

from __future__ import annotations

import csv
import os
import statistics as st
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RES = os.path.join(ROOT, "eval", "results")

FONT_DIR = os.path.expanduser("~/.local/share/fonts")
for f in ("NotoSansKR-Regular.otf", "NotoSansKR-Bold.otf"):
    p = os.path.join(FONT_DIR, f)
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams["font.family"] = "Noto Sans KR"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

# 색: GPU 계열 = 녹색, CPU 계열 = 청색, 불가/경고 = 적색
C_GPU, C_CPU, C_BAD, C_NEU, C_ACC = "#2e7d32", "#1565c0", "#c62828", "#78909c", "#ef6c00"
KO = "Noto Sans KR"


# ------------------------------------------------------------------ SVG
def svg(name: str, body: str, w: int, h: int, scale: int = 2) -> str:
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{KO}">'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>'
            '<defs>'
            '<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M0,0 L10,5 L0,10 z" fill="#37474f"/></marker>'
            '<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{C_BAD}"/></marker>'
            '</defs>')
    sp = os.path.join(FIG, name + ".svg")
    pp = os.path.join(FIG, name + ".png")
    with open(sp, "w") as f:
        f.write(head + body + "</svg>")
    subprocess.run(["rsvg-convert", "-z", str(scale), "-o", pp, sp], check=True)
    return pp


def box(x, y, w, h, fill, label, sub="", stroke="#37474f", fs=15, sfs=12,
        tc="#ffffff", rx=8, sw=1.4, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
         f'stroke="{stroke}" stroke-width="{sw}"{d}/>')
    cy = y + h / 2 + (0 if not sub else -7)
    s += (f'<text x="{x+w/2}" y="{cy}" font-size="{fs}" font-weight="600" '
          f'fill="{tc}" text-anchor="middle" dominant-baseline="middle">{label}</text>')
    if sub:
        s += (f'<text x="{x+w/2}" y="{cy+18}" font-size="{sfs}" fill="{tc}" '
              f'opacity="0.9" text-anchor="middle" '
              f'dominant-baseline="middle">{sub}</text>')
    return s


def txt(x, y, s, fs=13, fill="#263238", anchor="start", weight="400",
        style="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{fs}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'font-style="{style}">{s}</text>')


def arrow(x1, y1, x2, y2, color="#37474f", w=1.8, marker="a", dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{color}" stroke-width="{w}" '
            f'fill="none" marker-end="url(#{marker})"{d}/>')


def curve(x1, y1, x2, y2, bend=40, color="#37474f", w=1.8, marker="a", dash=""):
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - bend
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="M{x1},{y1} Q{mx},{my} {x2},{y2}" stroke="{color}" '
            f'stroke-width="{w}" fill="none" marker-end="url(#{marker})"{d}/>')


# ---------------------------------------------- D1. MoE 층과 offload 경계
def d1():
    b = txt(430, 30, "그림 1. MoE 층의 연산 구조와 오프로딩 경계", 19, "#111", "middle", "700")
    b += txt(430, 52, "GPU 는 attention·router·combine 을, CPU 는 expert FFN 을 맡는다", 13,
             "#546e7a", "middle")
    # GPU 영역
    b += (f'<rect x="30" y="80" width="380" height="410" rx="12" fill="#e8f5e9" '
          f'stroke="{C_GPU}" stroke-width="2"/>')
    b += txt(50, 106, "GPU (H100 ×8, TP=8)", 15, C_GPU, "start", "700")
    b += txt(50, 126, "HBM 상주: attention 가중치 + KV cache", 12, "#558b2f")
    b += box(70, 142, 300, 52, C_GPU, "입력 hidden state", "토큰 × d_model")
    b += arrow(220, 194, 220, 214)
    b += box(70, 214, 300, 58, C_GPU, "MLA Attention",
             "Multi-head Latent Attention · KV cache 접근")
    b += arrow(220, 272, 220, 292)
    b += box(70, 292, 300, 58, C_GPU, "Router (gate)",
             "top-k expert 선택 · 토큰별 8/257")
    b += box(70, 400, 300, 58, C_GPU, "Combine + 잔차",
             "expert 출력 가중합 → 다음 층")
    # CPU 영역
    b += (f'<rect x="470" y="80" width="380" height="410" rx="12" fill="#e3f2fd" '
          f'stroke="{C_CPU}" stroke-width="2"/>')
    b += txt(490, 106, "CPU (Xeon 8480+ ×2, AMX)", 15, C_CPU, "start", "700")
    b += txt(490, 126, "DRAM 상주: expert 가중치 전량 (AMXINT4)", 12, "#1976d2")
    for i, (yy, lab) in enumerate([(150, "Expert 1"), (206, "Expert 2"),
                                   (262, "…"), (318, "Expert 256"),
                                   (374, "Shared expert (257)")]):
        f_ = C_CPU if i != 4 else "#0d47a1"
        b += box(500, yy, 320, 44, f_, lab,
                 "" if lab == "…" else "W1·W3 → SiLU → W2  (AMX INT4 GEMM)",
                 fs=13, sfs=10)
    # 교차 화살표
    b += arrow(370, 321, 500, 250, C_ACC, 2.2)
    b += txt(435, 236, "선택된 expert", 12, C_ACC, "middle", "600")
    b += txt(435, 252, "hidden 전송", 11, C_ACC, "middle")
    b += arrow(500, 396, 370, 429, C_ACC, 2.2)
    b += txt(435, 445, "expert 출력 반환", 12, C_ACC, "middle", "600")
    b += txt(30, 520, "오프로딩 경계의 근거: expert FFN 은 토큰마다 top-k 개만 활성되는 "
             "희소 연산이라 층당 실제 계산량이 작고,", 13)
    b += txt(30, 540, "가중치는 전체 파라미터의 대부분을 차지한다. 즉 '용량은 크고 연산은 "
             "드문' 부분이므로 대용량 DRAM 쪽에 두는 것이 유리하다.", 13)
    b += txt(30, 560, "반대로 attention 은 KV cache 를 매 토큰 접근하는 대역폭 민감 연산이라 "
             "HBM 에 남긴다.", 13)
    return svg("d1_moe_boundary", b, 880, 580)


# ---------------------------------------------- D2. 메모리 용량 벽
def d2():
    b = txt(430, 30, "그림 2. 메모리 용량 벽 — GPU-only 가 불가능한 이유", 19, "#111",
            "middle", "700")
    b += txt(430, 52, "DeepSeek-R1-0528 native FP8 = 642 GB", 13, "#546e7a", "middle")
    # 좌: GPU-only — 용량과 모델을 나란히 둔다 (겹치면 라벨이 가려진다)
    b += txt(215, 92, "(a) GPU-only — r0", 16, C_BAD, "middle", "700")
    b += (f'<rect x="60" y="110" width="310" height="300" rx="10" fill="#fafafa" '
          f'stroke="#b0bec5" stroke-width="1.5"/>')
    base = 390                      # 막대 바닥
    scale = 240 / 642               # 642 GB = 240 px
    h_hbm = 608 * scale
    h_mod = 642 * scale
    # 용량 막대
    b += (f'<rect x="105" y="{base-h_hbm}" width="82" height="{h_hbm}" rx="5" '
          f'fill="#c8e6c9" stroke="{C_GPU}" stroke-width="1.6"/>')
    b += txt(146, base + 18, "usable HBM", 12, "#1b5e20", "middle", "700")
    b += txt(146, base + 34, "608 GB", 12, "#33691e", "middle")
    # 모델 막대
    b += (f'<rect x="235" y="{base-h_mod}" width="82" height="{h_mod}" rx="5" '
          f'fill="{C_BAD}" stroke="{C_BAD}" stroke-width="1.6"/>')
    b += txt(276, base + 18, "모델 FP8", 12, C_BAD, "middle", "700")
    b += txt(276, base + 34, "642 GB", 12, C_BAD, "middle")
    # 용량선과 초과분
    b += (f'<path d="M95,{base-h_hbm} L330,{base-h_hbm}" stroke="{C_GPU}" '
          f'stroke-width="2" stroke-dasharray="7 4"/>')
    b += (f'<rect x="235" y="{base-h_mod}" width="82" height="{h_mod-h_hbm}" '
          f'fill="#000" fill-opacity="0.22"/>')
    b += txt(325, base - h_mod - 8, "34 GB 초과", 12, C_BAD, "end", "700")
    b += txt(120, 131, "H100 80 GB ×8 = 640 GB", 11, "#33691e", "start")
    b += txt(120, 146, "gmu 0.95 적용 → 608 GB", 11, "#33691e", "start")
    b += txt(215, 440, "torch.OutOfMemoryError", 14, C_BAD, "middle", "700")
    b += txt(215, 458, "worker 로드 중 90초 만에 실패", 12, "#546e7a", "middle")
    # 우: hybrid
    b += txt(645, 92, "(b) KT Hybrid — r1", 16, C_GPU, "middle", "700")
    b += (f'<rect x="490" y="110" width="310" height="300" rx="10" fill="#fafafa" '
          f'stroke="#b0bec5" stroke-width="1.5"/>')
    b += (f'<rect x="515" y="140" width="120" height="250" rx="6" fill="#c8e6c9" '
          f'stroke="{C_GPU}" stroke-width="1.5"/>')
    b += txt(575, 165, "HBM", 14, "#1b5e20", "middle", "700")
    b += txt(575, 183, "608 GB", 12, "#33691e", "middle")
    b += (f'<rect x="530" y="250" width="90" height="130" rx="5" fill="{C_GPU}"/>')
    b += txt(575, 300, "attention", 12, "#fff", "middle", "700")
    b += txt(575, 318, "+ KV", 12, "#fff", "middle")
    b += txt(575, 340, "≈74 GB", 11, "#fff", "middle")
    b += txt(575, 356, "/GPU 실측", 10, "#fff", "middle")
    b += (f'<rect x="660" y="140" width="120" height="250" rx="6" fill="#bbdefb" '
          f'stroke="{C_CPU}" stroke-width="1.5"/>')
    b += txt(720, 165, "DRAM", 14, "#0d47a1", "middle", "700")
    b += txt(720, 183, "2 TB", 12, "#1565c0", "middle")
    b += (f'<rect x="672" y="210" width="96" height="170" rx="5" fill="{C_CPU}"/>')
    b += txt(720, 275, "expert", 12, "#fff", "middle", "700")
    b += txt(720, 293, "AMXINT4", 12, "#fff", "middle")
    b += txt(720, 315, "328 GB", 11, "#fff", "middle")
    b += txt(645, 440, "서빙 성립 · 19.67 out tok/s", 14, C_GPU, "middle", "700")
    b += txt(645, 458, "기동 160 초 · 32/32 요청 완료", 12, "#546e7a", "middle")
    b += txt(60, 500, "원본: eval/results/20260827_133055_tsk043_main_r1/r0_gpu_only_oom/"
             "oom_evidence.log · 20260827_140008_tsk043_main_r1/RESULTS.md", 11,
             "#78909c")
    return svg("d2_memory_wall", b, 880, 525)


# ---------------------------------------------- D3. step 타임라인
def d3():
    b = txt(450, 30, "그림 3. 하이브리드 디코드 스텝의 시간 구성", 19, "#111", "middle", "700")
    b += txt(450, 52, "실측 TPOT p50 = 305.5 ms · GPU util 평균 75.6% · CPU busy 평균 42.7%",
             13, "#546e7a", "middle")
    y1, y2, hh = 110, 190, 46
    b += txt(40, y1 + 24, "GPU", 15, C_GPU, "start", "700")
    b += txt(40, y2 + 24, "CPU", 15, C_CPU, "start", "700")
    b += f'<line x1="95" y1="{y1-14}" x2="860" y2="{y1-14}" stroke="#cfd8dc"/>'
    segs = [(100, 120, C_GPU, "MLA attn"), (230, 96, C_GPU, "router"),
            (470, 110, C_GPU, "combine"), (590, 120, C_GPU, "MLA attn"),
            (720, 96, C_GPU, "router")]
    for x, wd, c, lab in segs:
        b += box(x, y1, wd, hh, c, lab, fs=12, rx=5)
    cseg = [(330, 135, "expert FFN"), (790, 70, "expert")]
    for x, wd, lab in cseg:
        b += box(x, y2, wd, hh, C_CPU, lab, fs=12, rx=5)
    # 전송 화살표
    b += arrow(326, y1 + hh, 340, y2, C_ACC, 1.6)
    b += arrow(465, y2, 478, y1 + hh, C_ACC, 1.6)
    b += txt(300, y2 + 76, "hidden 전송", 11, C_ACC, "middle")
    b += txt(495, y2 + 76, "출력 반환", 11, C_ACC, "middle")
    # 대기 구간
    b += (f'<rect x="326" y="{y1}" width="144" height="{hh}" rx="5" fill="#eceff1" '
          f'stroke="#b0bec5" stroke-width="1.2" stroke-dasharray="5 4"/>')
    b += txt(398, y1 + 26, "GPU 대기", 12, "#607d8b", "middle", "600")
    b += f'<line x1="100" y1="290" x2="590" y2="290" stroke="#37474f" stroke-width="1.4"/>'
    b += f'<line x1="100" y1="284" x2="100" y2="296" stroke="#37474f" stroke-width="1.4"/>'
    b += f'<line x1="590" y1="284" x2="590" y2="296" stroke="#37474f" stroke-width="1.4"/>'
    b += txt(345, 312, "1 토큰 = TPOT 305.5 ms (실측 p50)", 13, "#37474f", "middle", "600")
    b += txt(40, 352, "· CPU expert 구간이 이 스텝의 지배 항이다. cpuinfer 96 스레드가 "
             "포화하며 CPU busy 42.7% (112C/224T 기준) 를 만든다.", 12)
    b += txt(40, 372, "· 이 경로는 CPU 동기 실행이라 CUDA graph replay 와 호환되지 않아 "
             "--disable-cuda-graph 가 필수였다 (우회 4건 중 결정타).", 12)
    b += txt(40, 392, "· 그림의 구간 폭은 개념적 비율이다. 구간별 개별 계측은 이 캠페인에서 "
             "수행하지 않았다 (총 TPOT 만 실측).", 12, "#c62828")
    return svg("d3_timeline", b, 900, 410)


# ---------------------------------------------- D4. 변환 파이프라인
def d4():
    b = txt(460, 30, "그림 4. 가중치 변환 파이프라인과 실패 경로", 19, "#111", "middle", "700")
    b += box(40, 80, 190, 74, C_NEU, "HF 원본 snapshot",
             "DeepSeek-R1-0528 · FP8 · 642 GB", fs=14, sfs=10)
    b += arrow(230, 117, 290, 117)
    b += box(290, 62, 200, 52, C_BAD, "FP8 직독 시도", "--kt-method FP8", fs=13, sfs=10)
    b += txt(390, 132, "native FP8 per-expert", 11, C_BAD, "middle", "600")
    b += txt(390, 147, "TP source is incomplete", 11, C_BAD, "middle", "600")
    b += txt(390, 162, "→ 기각 (KT 전용 샤딩 요구)", 11, C_BAD, "middle")
    b += arrow(230, 130, 290, 200)
    b += box(290, 180, 200, 74, C_CPU, "kt quant -m int4 -i fp8",
             "96 threads · 65 분 소요", fs=13, sfs=10)
    b += arrow(490, 217, 550, 217)
    b += box(550, 180, 200, 74, C_CPU, "AMXINT4 변환본",
             "328 GB · 63 files", fs=14, sfs=10)
    b += arrow(750, 217, 810, 217)
    b += box(810, 180, 50, 74, C_GPU, "서빙", "", fs=13)
    b += txt(40, 300, "블록별 세부", 15, "#111", "start", "700")
    b += box(40, 318, 250, 52, "#eceff1", "block-wise weight_scale_inv",
             "128×128 스케일 dequant", tc="#263238", fs=13, sfs=10, stroke="#90a4ae")
    b += arrow(290, 344, 340, 344)
    b += box(340, 318, 220, 52, "#eceff1", "INT4 재양자화",
             "AMX 타일 레이아웃", tc="#263238", fs=13, sfs=10, stroke="#90a4ae")
    b += arrow(560, 344, 610, 344)
    b += box(610, 318, 250, 52, "#ffebee", "shared expert (257번째) 폴딩",
             "품질 결함 후보 2 — SUB_167", tc="#b71c1c", fs=13, sfs=10,
             stroke=C_BAD, dash="5 4")
    b += (f'<rect x="40" y="395" width="820" height="72" rx="8" fill="#fff8e1" '
          f'stroke="{C_ACC}" stroke-width="1.4"/>')
    b += txt(56, 420, "판별 실험으로 좁힌 결과", 14, "#e65100", "start", "700")
    b += txt(56, 442, "Qwen3-30B 을 같은 변환기로 INT8·INT4 변환 → 둘 다 출력 정상. "
             "변환 스크립트 main = v0.7.0.post1 태그 IDENTICAL.", 12)
    b += txt(56, 460, "→ INT4 경로 일반 결함·버전 스큐 모두 기각. DeepSeek 계열 특이 처리 "
             "(위 점선 상자 2곳) 로 수렴.", 12)
    return svg("d4_convert", b, 900, 487)


# ---------------------------------------------- D5. 캠페인 순서
def d5():
    b = txt(460, 30, "그림 5. 캠페인 실행 순서 (2026-08-27)", 19, "#111", "middle", "700")
    b += txt(460, 52, "GPU 는 직렬 점유, CPU-only 작업은 병렬", 13, "#546e7a", "middle")
    b += (f'<rect x="30" y="78" width="360" height="120" rx="10" fill="#e3f2fd" '
          f'stroke="{C_CPU}" stroke-width="1.6" stroke-dasharray="6 4"/>')
    b += txt(48, 102, "백그라운드 (병렬, long-pole)", 14, "#0d47a1", "start", "700")
    b += box(48, 112, 160, 34, C_CPU, "R1-0528 다운로드", "642 GB · 약 2 시간",
             fs=12, sfs=9, rx=5)
    b += box(218, 112, 154, 34, C_CPU, "이미지 pull ×2", "vllm · sglang",
             fs=12, sfs=9, rx=5)
    b += box(48, 156, 324, 34, C_CPU, "kt quant int4 변환", "65 분 · 642 GB → 328 GB",
             fs=12, sfs=9, rx=5)
    b += (f'<rect x="30" y="220" width="840" height="120" rx="10" fill="#e8f5e9" '
          f'stroke="{C_GPU}" stroke-width="1.6"/>')
    b += txt(48, 244, "GPU 측정 큐 (직렬 — 8장 공유, 셀 간 서버 재기동)", 14, "#1b5e20",
             "start", "700")
    xs = [(48, "TSK_046", "baseline"), (218, "TSK_045", "KV tier"),
          (388, "TSK_043", "smoke 30B"), (558, "TSK_044", "co-location"),
          (728, "TSK_043", "본판 R1")]
    for i, (x, a, s_) in enumerate(xs):
        fill = C_ACC if a == "TSK_043" else C_GPU
        b += box(x, 258, 142, 56, fill, a, s_, fs=13, sfs=11, rx=6)
        if i < len(xs) - 1:
            b += arrow(x + 142, 286, x + 170, 286)
    b += curve(210, 190, 640, 258, 50, C_CPU, 1.6, "a", "5 4")
    b += txt(430, 205, "변환본 도착", 11, C_CPU, "middle", "600")
    b += arrow(450, 340, 450, 368)
    b += box(280, 368, 340, 50, "#546e7a", "품질 판별 실험 → SUB_167 발급",
             "4 가설 중 3 기각", fs=14, sfs=11)
    b += txt(30, 452, "이 백서가 다루는 범위 = TSK_043 (주황 상자 2개와 그 뒤 판별 실험). "
             "TSK_044·045·046 은 같은 캠페인의 다른 트랙이다.", 12, "#37474f")
    return svg("d5_campaign", b, 900, 472)


# ---------------------------------------------- C1. Qwen3-30B smoke
def c1():
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    a = ax[0]
    v = [1461.77, 96.09]
    bars = a.bar(["GPU-only\n(t1)", "KT hybrid\n(t2)"], v, color=[C_GPU, C_CPU],
                 width=.55)
    a.set_ylabel("출력 처리량 (tok/s)")
    a.set_title("Qwen3-30B-A3B · 처리량", fontsize=12, fontweight="bold")
    for r, x in zip(bars, v):
        a.text(r.get_x() + r.get_width() / 2, x * 1.02, f"{x:,.1f}", ha="center",
               fontsize=11, fontweight="bold")
    a.set_ylim(0, 1700)
    a.text(.5, .55, "GPU 가 15.2배 빠름\n(30B 는 HBM 에 들어가는 모델)",
           transform=a.transAxes, ha="center", fontsize=10, color="#c62828")
    a.grid(axis="y", alpha=.3)
    b = ax[1]
    x = np.arange(2)
    wd = .35
    b.bar(x - wd / 2, [1.98, 43.06], wd, label="CPU busy %", color=C_CPU)
    b.bar(x + wd / 2, [48.8, 4.8], wd, label="GPU0 util %", color=C_GPU)
    b.set_xticks(x)
    b.set_xticklabels(["GPU-only", "KT hybrid"])
    b.set_ylabel("활용률 (%)")
    b.set_title("자원 활용률 (전 구간 재집계)", fontsize=12, fontweight="bold")
    for i, (c_, g_) in enumerate([(1.98, 48.8), (43.06, 4.8)]):
        b.text(i - wd / 2, c_ + 1.5, f"{c_:.1f}", ha="center", fontsize=10)
        b.text(i + wd / 2, g_ + 1.5, f"{g_:.1f}", ha="center", fontsize=10)
    b.legend(fontsize=9)
    b.grid(axis="y", alpha=.3)
    b.set_ylim(0, 60)
    p = os.path.join(FIG, "c1_smoke.png")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------- C2. R1 성립
def c2():
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    a = ax[0]
    bars = a.bar(["GPU-only (r0)", "KT hybrid (r1)"], [0, 19.67],
                 color=[C_BAD, C_CPU], width=.5)
    a.set_ylabel("출력 처리량 (tok/s)")
    a.set_title("DeepSeek-R1-0528 (642 GB)", fontsize=12, fontweight="bold")
    a.text(0, .9, "0\nOutOfMemoryError", ha="center", fontsize=11,
           color=C_BAD, fontweight="bold")
    a.text(1, 20.4, "19.67", ha="center", fontsize=12, fontweight="bold")
    a.set_ylim(0, 25)
    a.grid(axis="y", alpha=.3)
    b = ax[1]
    lab = ["TTFT p50", "TTFT p95", "TPOT p50", "TPOT p95"]
    v = [11199.23, 18098.75, 305.50, 326.34]
    bars = b.barh(lab[::-1], v[::-1], color=[C_ACC, C_ACC, C_NEU, C_NEU][::-1],
                  height=.55)
    b.set_xscale("log")
    b.set_xlabel("지연 (ms, 로그 축)")
    b.set_title("r1 지연 분포", fontsize=12, fontweight="bold")
    for r, x in zip(bars, v[::-1]):
        b.text(x * 1.12, r.get_y() + r.get_height() / 2, f"{x:,.1f}", va="center",
               fontsize=10)
    b.set_xlim(100, 60000)
    b.grid(axis="x", alpha=.3)
    p = os.path.join(FIG, "c2_r1.png")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------- C3. CPU 시계열
def c3():
    R = os.path.join(RES, "20260827_140008_tsk043_main_r1", "r1_kt_hybrid")
    ts, v = [], []
    for line in open(os.path.join(R, "cpu_util.txt")):
        p = line.split("busy=")
        if len(p) == 2:
            ts.append(int(p[0].strip()))
            v.append(float(p[1]))
    t0 = ts[0]
    x = [(t - t0) for t in ts]
    fig, a = plt.subplots(figsize=(9.2, 3.1))
    a.plot(x, v, color=C_CPU, lw=1.6)
    a.fill_between(x, v, color=C_CPU, alpha=.18)
    m = sum(v) / len(v)
    a.axhline(m, color=C_ACC, ls="--", lw=1.3,
              label=f"평균 {m:.2f}%")
    a.axhline(max(v), color=C_BAD, ls=":", lw=1.3, label=f"최대 {max(v):.2f}%")
    a.set_xlabel("경과 시간 (초)")
    a.set_ylabel("CPU busy (%)")
    a.set_title(f"그림 6. r1 실행 중 CPU 사용률 — {len(v)} 표본 / {x[-1]} 초 "
                f"(112C/224T, turbo OFF 2.0 GHz)", fontsize=12, fontweight="bold")
    a.legend(fontsize=9, loc="lower right")
    a.grid(alpha=.3)
    a.set_ylim(0, 60)
    p = os.path.join(FIG, "c3_cpu_series.png")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p, len(v), x[-1], m, max(v)


# ---------------------------------------------- C4. per-GPU
def c4():
    R = os.path.join(RES, "20260827_140008_tsk043_main_r1", "r1_kt_hybrid")
    byg = {}
    for r in csv.reader(open(os.path.join(R, "gpu_util.csv"))):
        if len(r) < 4:
            continue
        try:
            g = int(r[0])
            u = float(r[1].strip().rstrip(" %"))
            mm = float(r[2].strip().split()[0])
            wv = float(r[3].strip().split()[0])
        except Exception:
            continue
        byg.setdefault(g, []).append((u, mm, wv))
    gs = sorted(byg)
    util = [sum(x[0] for x in byg[g]) / len(byg[g]) for g in gs]
    mem = [sum(x[1] for x in byg[g]) / len(byg[g]) / 1024 for g in gs]
    pw = [sum(x[2] for x in byg[g]) / len(byg[g]) for g in gs]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3))
    a = ax[0]
    cols = [C_BAD if u < 40 else C_GPU for u in util]
    bars = a.bar([f"GPU{g}" for g in gs], util, color=cols, width=.62)
    a.axhline(sum(util) / len(util), color=C_ACC, ls="--", lw=1.3,
              label=f"평균 {sum(util)/len(util):.1f}%")
    a.set_ylabel("GPU util (%)")
    a.set_title("GPU별 활용률 (48 표본/장)", fontsize=12, fontweight="bold")
    for r, u in zip(bars, util):
        a.text(r.get_x() + r.get_width() / 2, u + 1.5, f"{u:.1f}", ha="center",
               fontsize=9)
    a.legend(fontsize=9)
    a.grid(axis="y", alpha=.3)
    a.set_ylim(0, 110)
    b = ax[1]
    bars = b.bar([f"GPU{g}" for g in gs], mem, color=C_NEU, width=.62)
    b.set_ylabel("HBM 사용 (GiB)")
    b.set_title("GPU별 HBM 점유 · 평균 전력", fontsize=12, fontweight="bold")
    for r, mv, pv in zip(bars, mem, pw):
        b.text(r.get_x() + r.get_width() / 2, mv + 1, f"{mv:.1f}", ha="center",
               fontsize=9)
        b.text(r.get_x() + r.get_width() / 2, mv / 2, f"{pv:.0f}W", ha="center",
               fontsize=8, color="#fff", rotation=90)
    b.grid(axis="y", alpha=.3)
    b.set_ylim(0, 90)
    p = os.path.join(FIG, "c4_per_gpu.png")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p, util, mem, pw


# ---------------------------------------------- C5. 튜닝 여지
def c5():
    fig, a = plt.subplots(figsize=(9.2, 2.9))
    lab = ["본 측정 r1\n(turbo OFF · GPU expert 0\n· deferral 없음)",
           "KT 공식 참조치\n(8×L20 + Xeon, R1)"]
    v = [19.67, 227.0]
    bars = a.barh(lab, v, color=[C_CPU, C_NEU], height=.5)
    for r, x in zip(bars, v):
        a.text(x + 4, r.get_y() + r.get_height() / 2, f"{x:,.2f} tok/s", va="center",
               fontsize=11, fontweight="bold")
    a.set_xlim(0, 270)
    a.set_xlabel("출력 처리량 (tok/s)")
    a.set_title("그림 9. 측정치와 외부 참조치 — 배수 11.5×",
                fontsize=12, fontweight="bold")
    a.grid(axis="x", alpha=.3)
    a.text(120, -.52, "참조치는 하드웨어·구성이 다르므로 동일 조건 비교가 아니다. "
           "튜닝 여지의 규모를 가리키는 값으로만 인용한다.",
           fontsize=9, color="#c62828", ha="center")
    p = os.path.join(FIG, "c5_headroom.png")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


if __name__ == "__main__":
    outs = [d1(), d2(), d3(), d4(), d5(), c1(), c2()]
    p3 = c3()
    p4 = c4()
    outs += [p3[0], p4[0], c5()]
    for o in outs:
        print(f"  {os.path.basename(o)}  {os.path.getsize(o):,} bytes")
    print(f"\nCPU 시계열: 표본 {p3[1]} / {p3[2]}초 / 평균 {p3[3]:.2f}% / 최대 {p3[4]:.2f}%")
    print("GPU util:", [round(x, 1) for x in p4[1]])
    print("GPU mem GiB:", [round(x, 1) for x in p4[2]])
    print("GPU power W:", [round(x) for x in p4[3]])
