#!/usr/bin/env python3
"""IDE_023 / TSK_043 — CPU MoE Expert Offloading 백서 (docx) 생성.

본문 수치는 `eval/results/20260827_*` 원본과 캠페인 보고서에서 가져온다. 원본에서
직접 재집계한 값은 표 각주에 "재집계" 로 표시한다.
"""

from __future__ import annotations

import os

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "IDE_023_TSK_043_CPU_MoE_Offloading_백서.docx")

BODY = "맑은 고딕"        # 워드 기본 한글 폰트. 없으면 뷰어가 대체한다.
MONO = "Consolas"
ACCENT = RGBColor(0x1A, 0x4F, 0x8B)
MUTED = RGBColor(0x60, 0x6A, 0x72)
BAD = RGBColor(0xB0, 0x2A, 0x2A)

doc = Document()


# ---------------------------------------------------------------- 스타일
def _font(run, name=BODY, size=10.5, bold=False, color=None, mono=False):
    f = run.font
    f.name = name if not mono else MONO
    f.size = Pt(size)
    f.bold = bold
    if color is not None:
        f.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), name if not mono else MONO)


def emit(par, text, size=10.5, bold=False, color=None):
    """인라인 마커를 run 으로 풀어 넣는다.

    문자열 앞의 `**` = 전체 굵게, `!!` = 경고색. 문장 중간의 ``…`` 구간은 고정폭.
    (``로 열고 다음 ``나 문자열 끝에서 닫는다 — 경로·플래그 표기에 쓴다.)
    """
    s = str(text)
    changed = True
    while changed:
        changed = False
        if s.startswith("**"):
            s, bold, changed = s[2:], True, True
        elif s.startswith("!!"):
            s, color, changed = s[2:], BAD, True
    if s.endswith("**"):
        s, bold = s[:-2], True

    # `` 는 여는/닫는 쌍이 아니라 **접두 마커**다. 마커 뒤의 첫 토큰(공백 전까지)만
    # 고정폭으로 찍고 나머지는 일반 문장으로 돌린다. 명령·경로·플래그 표기에 맞춘 규칙.
    parts = s.split("``")
    if parts[0]:
        r = par.add_run(parts[0])
        _font(r, size=size, bold=bold, color=color)
    for seg in parts[1:]:
        if not seg:
            continue
        if " " in seg:
            head, tail = seg.split(" ", 1)
            tail = " " + tail
        else:
            head, tail = seg, ""
        r = par.add_run(head)
        _font(r, size=size * 0.95, bold=bold, color=color, mono=True)
        if tail:
            r = par.add_run(tail)
            _font(r, size=size, bold=bold, color=color)


def setup():
    s = doc.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.left_margin = s.right_margin = Cm(2.2)
    s.top_margin = Cm(2.2)
    s.bottom_margin = Cm(2.0)
    st = doc.styles["Normal"]
    st.font.name = BODY
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), BODY)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.32


def p(text="", size=10.5, bold=False, color=None, align=None, space_after=6,
      indent=0.0, mono=False, style=None):
    par = doc.add_paragraph(style=style)
    if align:
        par.alignment = align
    par.paragraph_format.space_after = Pt(space_after)
    if indent:
        par.paragraph_format.left_indent = Cm(indent)
    if text:
        if mono:
            r = par.add_run(text)
            _font(r, size=size, bold=bold, color=color, mono=True)
        else:
            emit(par, text, size=size, bold=bold, color=color)
    return par


def rich(parts, size=10.5, align=None, space_after=6, indent=0.0):
    """parts = [(text, {bold, color, mono, size}), ...]"""
    par = doc.add_paragraph()
    if align:
        par.alignment = align
    par.paragraph_format.space_after = Pt(space_after)
    if indent:
        par.paragraph_format.left_indent = Cm(indent)
    for t, o in parts:
        r = par.add_run(t)
        _font(r, size=o.get("size", size), bold=o.get("bold", False),
              color=o.get("color"), mono=o.get("mono", False))
    return par


def h1(text, num=None):
    doc.add_page_break()
    par = doc.add_paragraph()
    par.paragraph_format.space_before = Pt(0)
    par.paragraph_format.space_after = Pt(10)
    r = par.add_run(f"{num}. {text}" if num else text)
    _font(r, size=18, bold=True, color=ACCENT)
    _hr(par)


def h2(text):
    par = doc.add_paragraph()
    par.paragraph_format.space_before = Pt(14)
    par.paragraph_format.space_after = Pt(5)
    r = par.add_run(text)
    _font(r, size=13, bold=True, color=RGBColor(0x1F, 0x3A, 0x5F))


def h3(text):
    par = doc.add_paragraph()
    par.paragraph_format.space_before = Pt(10)
    par.paragraph_format.space_after = Pt(4)
    r = par.add_run(text)
    _font(r, size=11.5, bold=True)


def _hr(par, color="1A4F8B", sz=8):
    pr = par._p.get_or_add_pPr()
    b = OxmlElement("w:pBdr")
    bo = OxmlElement("w:bottom")
    bo.set(qn("w:val"), "single")
    bo.set(qn("w:sz"), str(sz))
    bo.set(qn("w:space"), "3")
    bo.set(qn("w:color"), color)
    b.append(bo)
    pr.append(b)


def bullet(text, size=10.5, level=0, color=None, bold_prefix=None):
    par = doc.add_paragraph()
    par.paragraph_format.left_indent = Cm(0.6 + level * 0.6)
    par.paragraph_format.first_line_indent = Cm(-0.35)
    par.paragraph_format.space_after = Pt(3)
    r = par.add_run("· " if level == 0 else "– ")
    _font(r, size=size, color=color)
    if bold_prefix:
        r = par.add_run(bold_prefix)
        _font(r, size=size, bold=True, color=color)
    emit(par, text, size=size, color=color)
    return par


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:fill"), hexcolor)
    tcPr.append(sh)


def table(headers, rows, widths=None, size=9.5, head_bg="1A4F8B",
          zebra="F2F6FA", align_right=(), note=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, htxt in enumerate(headers):
        hdr[i].text = ""
        par = hdr[i].paragraphs[0]
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_after = Pt(2)
        r = par.add_run(htxt)
        _font(r, size=size, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
        shade(hdr[i], head_bg)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci, val in enumerate(row):
            cells[ci].text = ""
            par = cells[ci].paragraphs[0]
            par.paragraph_format.space_after = Pt(2)
            if ci in align_right:
                par.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            for li, line in enumerate(str(val).split("\n")):
                tgt = par if li == 0 else cells[ci].add_paragraph()
                if li:
                    tgt.paragraph_format.space_after = Pt(2)
                    if ci in align_right:
                        tgt.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                emit(tgt, line, size=size)
        if zebra and ri % 2 == 1:
            for c in cells:
                shade(c, zebra)
    if widths:
        # autofit 이 켜져 있으면 Word 가 지정 폭을 무시한다. 고정 레이아웃으로 바꾼다.
        t.autofit = False
        tblPr = t._tbl.tblPr
        layout = OxmlElement("w:tblLayout")
        layout.set(qn("w:type"), "fixed")
        tblPr.append(layout)
        for row in t.rows:
            for i, wd in enumerate(widths):
                row.cells[i].width = Cm(wd)
    if note:
        p(note, size=8.5, color=MUTED, space_after=10)
    else:
        p("", size=4, space_after=4)
    return t


def figure(name, caption, width=16.0):
    path = os.path.join(FIG, name)
    if not os.path.exists(path):
        p(f"[그림 누락: {name}]", color=BAD)
        return
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(8)
    par.paragraph_format.space_after = Pt(3)
    par.add_run().add_picture(path, width=Cm(width))
    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp.paragraph_format.space_after = Pt(12)
    r = cp.add_run(caption)
    _font(r, size=9, color=MUTED)


def code(lines, size=9):
    par = doc.add_paragraph()
    par.paragraph_format.left_indent = Cm(0.5)
    par.paragraph_format.space_before = Pt(4)
    par.paragraph_format.space_after = Pt(8)
    par.paragraph_format.line_spacing = 1.15
    pr = par._p.get_or_add_pPr()
    b = OxmlElement("w:pBdr")
    for side in ("top", "left", "bottom", "right"):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:space"), "4")
        e.set(qn("w:color"), "C9D4DF")
        b.append(e)
    pr.append(b)
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:fill"), "F7F9FB")
    pr.append(sh)
    for i, ln in enumerate(lines):
        if i:
            par.add_run().add_break()
        r = par.add_run(ln)
        _font(r, size=size, mono=True)


def callout(title, lines, bg="FFF6E5", border="E08A2E"):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = t.rows[0].cells[0]
    c.text = ""
    shade(c, bg)
    par = c.paragraphs[0]
    par.paragraph_format.space_after = Pt(3)
    r = par.add_run(title)
    _font(r, size=10.5, bold=True, color=RGBColor(0x8A, 0x4B, 0x00))
    for ln in lines:
        pp_ = c.add_paragraph()
        pp_.paragraph_format.space_after = Pt(2)
        emit(pp_, ln, size=9.5)
    tbl = t._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right"):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "8")
        e.set(qn("w:color"), border)
        borders.append(e)
    tblPr.append(borders)
    p("", size=4, space_after=8)


# ================================================================ 표지
def cover():
    setup()
    p("", space_after=60)
    p("기술 백서", size=12, color=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER,
      space_after=6)
    par = p("CPU MoE Expert Offloading", size=30, bold=True, color=ACCENT,
            align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    p("GPU 메모리에 담기지 않는 MoE 모델을 CPU AMX 로 서빙하기", size=14,
      color=RGBColor(0x33, 0x44, 0x55), align=WD_ALIGN_PARAGRAPH.CENTER,
      space_after=28)
    p("IDE_023 / TSK_043 — PLN_003 Hybrid Regime Sweep", size=12,
      align=WD_ALIGN_PARAGRAPH.CENTER, space_after=40)

    table(
        ["항목", "값"],
        [["대상 트랙", "IDE_023 — MoE Expert Offload (TSK_043)"],
         ["상위 캠페인", "PLN_003 — Hybrid Regime Sweep (IDE_023 · IDE_024 · IDE_025)"],
         ["측정 노드", "violet-h100-016 — Xeon Platinum 8480+ ×2 (AMX) / DDR5 2 TB / H100 80 GB ×8"],
         ["측정일", "2026-08-27"],
         ["대상 모델", "DeepSeek-R1-0528 (native FP8, 642 GB) · Qwen3-30B-A3B (smoke)"],
         ["커밋", "``437c69ae5  (branch feat/hybrid-regime-sweep)"],
         ["판정", "**부분 통과 — 서빙 성립 확인, 출력 품질 게이트 미통과 (SUB_167)"],
         ["원시 데이터", "``eval/results/20260827_124509_tsk043_smoke_qwen30b/\n"
                       "``eval/results/20260827_133055_tsk043_main_r1/\n"
                       "``eval/results/20260827_140008_tsk043_main_r1/"]],
        widths=[3.6, 12.8], size=10)

    p("", space_after=20)
    callout("이 문서의 범위", [
        "PLN_003 캠페인은 4개 트랙(MoE offload · DRAM KV tier · CPU co-location · baseline)으로 "
        "구성된다. 이 백서는 그중 MoE expert offloading 트랙(IDE_023 / TSK_043)만 다룬다.",
        "다른 트랙의 수치는 비교 맥락이 필요한 곳에서만 인용하고, 출처를 함께 적는다.",
        "모든 성능 수치는 CPU turbo OFF (2.0 GHz 고정) 상태에서 측정된 하한이다.",
    ])


# ================================================================ 요약
def s_summary():
    h1("요약", 1)

    h2("1.1 한 문장")
    rich([("8장의 H100(총 640 GB)에 담기지 않는 642 GB 모델을, ", {}),
          ("expert 가중치 전량을 CPU DRAM 으로 내리고 AMX 로 계산", {"bold": True}),
          ("함으로써 ", {}),
          ("서빙 불가(OOM) 상태에서 19.67 tok/s 로 전환", {"bold": True}),
          ("했다.", {})], size=11)

    h2("1.2 핵심 수치")
    table(
        ["구분", "GPU-only", "CPU MoE 오프로딩", "변화"],
        [["서빙 가능 여부", "!!불가 (OutOfMemoryError)", "**가능", "**불가능 → 가능"],
         ["출력 처리량", "0 tok/s", "**19.67 tok/s", "—"],
         ["CPU 사용률", "≈ 0 % (서빙 자체 없음)", "**평균 42.7 % / 최대 49.5 %", "+42.7 %p"],
         ["GPU 사용률", "—", "평균 75.6 %", "—"],
         ["TTFT p50", "—", "11,199 ms", "—"],
         ["TPOT p50", "—", "305.5 ms", "—"],
         ["모델 메모리 배치", "HBM 642 GB 필요 (34 GB 부족)",
          "HBM ≈74 GB/장 + DRAM 328 GB", "—"],
         ["출력 품질", "—", "!!비문 (게이트 미통과)", "!!SUB_167 로 분리"]],
        widths=[3.4, 4.4, 4.6, 3.6], size=9.5, align_right=(),
        note="원본: eval/results/20260827_140008_tsk043_main_r1/RESULTS.md, "
             "20260827_133055_tsk043_main_r1/r0_gpu_only_oom/oom_evidence.log")

    h2("1.3 무엇이 증명되었고 무엇이 증명되지 않았는가")
    table(
        ["", "내용", "근거"],
        [["**증명됨",
          "642 GB 모델은 이 노드의 GPU-only 로 적재 불가하다",
          "r0 — torch.OutOfMemoryError (90 초)"],
         ["**증명됨",
          "expert 전량을 CPU 로 내리면 같은 노드에서 서빙이 성립한다",
          "r1 — 32/32 요청 완료, 19.67 tok/s"],
         ["**증명됨",
          "CPU 가 유휴가 아니라 실제 연산 주체로 소비된다",
          "CPU busy 42.7 %, cpuinfer 96 스레드 포화"],
         ["**증명됨",
          "KT 스택(변환→AMX 적재→연산→정확성)이 이 노드에서 동작한다",
          "Qwen3-30B smoke — 출력 정상"],
         ["!!미증명",
          "이 경로의 출력이 정확하다",
          "R1 greedy 출력 비문 — SUB_167"],
         ["!!미증명",
          "19.67 tok/s 가 이 구성의 달성 가능 성능이다",
          "turbo OFF · GPU expert 0 · deferral 미사용의 하한"],
         ["!!미증명",
          "GPU 에 들어가는 모델에도 이득이 있다",
          "Qwen3-30B 은 GPU-only 가 15.2배 빠름"]],
        widths=[2.0, 7.4, 7.0], size=9.5)

    figure("d2_memory_wall.png",
           "그림 2. 메모리 용량 벽 — 같은 모델을 (a) GPU-only 로 적재하면 34 GB 가 모자라 "
           "실패하고, (b) expert 를 DRAM 으로 내리면 성립한다.")


# ================================================================ 배경
def s_background():
    h1("배경과 문제 정의", 2)

    h2("2.1 왜 오프로딩인가 — 용량과 연산의 비대칭")
    p("대규모 언어 모델 서빙에서 GPU 가 제공하는 자원은 두 가지다. 하나는 연산 처리량이고 "
      "다른 하나는 HBM 용량이다. 이 둘은 함께 늘어나지 않는다. H100 한 장은 약 1,000 "
      "TFLOPS 급 연산을 제공하지만 HBM 은 80 GB 에 묶여 있다. 모델 크기가 커질 때 먼저 "
      "닿는 벽은 연산이 아니라 용량이다.")
    p("Mixture-of-Experts(MoE) 구조는 이 비대칭을 극단으로 밀어붙인다. MoE 는 파라미터 "
      "수를 크게 늘리면서도 토큰당 활성 파라미터는 그대로 두는 설계다. 결과적으로 "
      "가중치의 대부분을 차지하는 expert 는 "
      "'용량은 막대하지만 토큰마다 극히 일부만 쓰이는' 자원이 된다.")

    h2("2.2 이 노드에서의 구체적 벽")
    table(
        ["항목", "값", "비고"],
        [["DeepSeek-R1-0528 native FP8", "642 GB", "HF snapshot 실측 용량"],
         ["H100 80 GB × 8 물리 HBM", "640 GB", "—"],
         ["usable HBM", "≈ 608 GB", "gpu-memory-utilization 0.95 적용"],
         ["**부족분", "**34 GB", "가중치만으로 초과. KV cache 는 계산 이전"]],
        widths=[6.4, 3.4, 6.6], size=9.5, align_right=(1,))
    p("부족분 34 GB 는 가중치만 따진 값이다. 실제 서빙에는 KV cache·활성화·통신 버퍼가 "
      "추가로 필요하므로 실질 부족분은 이보다 크다. 즉 이 모델은 이 노드에서 "
      "'조금 빠듯한' 것이 아니라 '적재 자체가 불가능한' 영역에 있다.")

    h2("2.3 선행 연구·선행 시도가 정한 경계")
    p("이 저장소는 CPU 를 활용해 시스템 처리량을 올리려는 시도를 여러 세대에 걸쳐 진행했고, "
      "그 과정에서 명확히 기각된 경로가 있다.")
    table(
        ["선행 ID", "시도", "결과"],
        [["SUB_036 / SUB_040 / SUB_041 / SUB_042",
          "dense 모델의 decode attention 을 CPU 로 오프로드",
          "!!본 노드 클래스에서 재실증 기각"],
         ["IDE_006", "KV 의존성이 있는 하이브리드 경로 (Q-dependency dilemma)",
          "구조적 제약 확인"]],
        widths=[5.6, 6.4, 4.4], size=9.5,
        note="출처: shadow_assists/features/IDE_023/CLAUDE.md, PLN_003.md §5")
    rich([("IDE_023 이 다루는 범위는 위 결론과 충돌하지 않는 ", {}),
          ("유일한 영역", {"bold": True}),
          (" — 즉 ", {}),
          ("GPU-only 로는 애초에 실행이 불가능한 모델", {"bold": True}),
          (" 로 한정된다. GPU 가 할 수 있는 일을 CPU 가 대신하는 것이 아니라, "
           "GPU 가 할 수 없는 일을 CPU 가 가능하게 만드는 구도다.", {})])

    callout("판정 원칙 (캠페인 전체에 적용)", [
        "· \"CPU 가 바빠졌다\"는 성공이 아니다. binding 지표는 언제나 처리량과 지연이며 "
        "CPU 사용률은 보조 지표다.",
        "· 카운터 없는 이득 주장은 금지한다. 선행 캠페인(IDE_006)의 \"merged 0 %\" 교훈이다.",
        "· 게이트는 측정 전에 정의한다 (TST_020~023).",
    ])


# ================================================================ 원리
def s_principle():
    h1("동작 원리", 3)

    h2("3.1 MoE 층의 구조와 오프로딩 경계")
    p("Transformer 의 한 층은 attention 블록과 feed-forward 블록으로 나뉜다. MoE 는 이 "
      "feed-forward 블록을 여러 개의 독립적인 expert 로 복제하고, 토큰마다 router 가 "
      "그중 일부만 선택해 통과시킨다.")
    p("DeepSeek-R1 계열은 층마다 256 개의 routed expert 와 1 개의 shared expert (총 257 개) "
      "를 갖고, 토큰당 routed expert 중 8 개가 활성된다. 즉 층당 실제 계산에 참여하는 "
      "expert 는 전체의 약 3 % 다. 나머지 97 % 의 가중치는 그 토큰에 대해서는 "
      "읽히지 않는다.")
    p("오프로딩 설계는 이 비대칭을 그대로 이용한다. 가중치 용량의 대부분을 차지하면서 "
      "토큰당 계산량은 작은 expert 를 용량이 큰 DRAM 에 두고, KV cache 를 매 토큰 "
      "접근해야 하는 attention 은 대역폭이 큰 HBM 에 남긴다.")

    figure("d1_moe_boundary.png",
           "그림 1. MoE 층의 연산 구조와 오프로딩 경계. Router 까지는 GPU, expert FFN 은 "
           "CPU, 결과 합산은 다시 GPU 에서 이뤄진다.")

    callout("모델 구조 수치의 출처", [
        "층당 expert 257 개는 이 캠페인의 서버 로그에서 확인된 값이다 "
        "(RESULTS.md: \"257 experts/layer 전량 CPU\").",
        "토큰당 top-8 활성은 DeepSeek-R1 모델 사양이며 이 캠페인에서 직접 계측하지 않았다. "
        "본문에서 구조 설명 목적으로만 인용한다.",
    ])

    h2("3.2 왜 expert 가 CPU 에 적합한가")
    table(
        ["성질", "attention (GPU 유지)", "expert FFN (CPU 이전)"],
        [["가중치 용량", "상대적으로 작다", "**모델 대부분을 차지"],
         ["토큰당 접근량", "KV cache 전체를 매 토큰 읽는다", "선택된 소수 expert 만 읽는다"],
         ["병목 자원", "메모리 대역폭", "가중치 적재 용량"],
         ["동일 연산 재사용", "요청·토큰마다 KV 가 달라 재사용 없음",
          "배치 내 같은 expert 로 모인 토큰끼리 GEMM 병합 가능"],
         ["적합한 메모리", "**HBM (고대역폭·소용량)", "**DRAM (저대역폭·대용량)"]],
        widths=[3.4, 6.2, 6.4], size=9.5)
    p("expert 연산은 동일 expert 로 라우팅된 토큰들을 묶어 하나의 행렬 곱으로 만들 수 있다. "
      "배치가 클수록 이 묶음이 커지고 연산 대비 가중치 적재 비용이 상대적으로 낮아진다. "
      "따라서 CPU 오프로딩은 배치가 큰 서빙 구간에서 상대적으로 유리하다.")

    h2("3.3 AMX 와 INT4 양자화")
    p("Intel AMX(Advanced Matrix Extensions)는 Xeon Sapphire Rapids 세대부터 제공되는 "
      "행렬 곱 전용 명령 집합이다. 레지스터가 아닌 타일(tile) 단위로 데이터를 잡고 한 "
      "명령으로 타일 단위 누적 곱을 수행하므로, 같은 코어에서 AVX-512 대비 정수 행렬 곱 "
      "처리량이 크게 올라간다. 이 트랙이 CPU 로 expert 를 돌릴 수 있는 근거가 이 명령 "
      "집합이다.")
    p("AMX 는 정수와 bfloat16 경로를 제공하므로 FP8 원본을 그대로 쓸 수 없다. 이 트랙은 "
      "kt-kernel 의 변환기로 FP8 원본을 AMXINT4 로 재양자화했다. 부수 효과로 용량도 "
      "642 GB → 328 GB 로 줄어 DRAM 배치가 쉬워진다.")
    table(
        ["항목", "값", "비고"],
        [["원본", "DeepSeek-R1-0528 native FP8 · 642 GB", "HF snapshot"],
         ["변환 명령", "``kt quant -m int4 -i fp8", "96 threads"],
         ["변환 시간", "**65 분", "예상 3 시간 대비 단축"],
         ["출력", "**AMXINT4 · 328 GB · 63 files", "``~/.cache/huggingface/kt/r1-0528-int4"],
         ["압축률", "0.51× (642 → 328 GB)", "—"]],
        widths=[3.4, 6.6, 6.0], size=9.5,
        note="원본: eval/results/20260827_140008_tsk043_main_r1/RESULTS.md, "
             "PROGRESS_20260827.md (16:3x, 17:4x 항목)")

    h2("3.4 하이브리드 실행 경로")
    p("서빙 시 한 층의 처리는 다음 순서로 진행된다.")
    bullet("GPU 가 MLA(Multi-head Latent Attention)를 계산한다. KV cache 는 HBM 에 있다.",
           bold_prefix="① ")
    bullet("GPU 의 router 가 토큰별로 활성 expert 를 고른다.", bold_prefix="② ")
    bullet("선택 정보와 hidden state 가 CPU 로 전달된다.", bold_prefix="③ ")
    bullet("CPU 가 해당 expert 의 FFN(W1·W3 → SiLU → W2)을 AMX INT4 GEMM 으로 계산한다. "
           "cpuinfer 스레드 96 개가 이 구간을 담당한다.", bold_prefix="④ ")
    bullet("결과가 GPU 로 돌아가 가중합·잔차 연결을 거쳐 다음 층으로 넘어간다.",
           bold_prefix="⑤ ")
    p("이 경로는 CPU 구간이 동기적이다. 즉 GPU 는 ④ 가 끝날 때까지 해당 층을 진행할 수 "
      "없다. 이것이 다음 두 가지 결과로 직접 이어진다 — (가) TPOT 이 305 ms 수준으로 "
      "커지고, (나) CUDA graph replay 와 호환되지 않아 graph 를 끌 수밖에 없다.")

    figure("d3_timeline.png",
           "그림 3. 하이브리드 디코드 스텝의 시간 구성. CPU expert 구간이 지배 항이며 "
           "그 동안 GPU 는 대기한다. 구간 폭은 개념적 비율이다.")

    h2("3.5 언제 이득이고 언제 손해인가")
    p("오프로딩의 손익은 단순한 비교로 결정된다.")
    table(
        ["상황", "GPU-only", "CPU 오프로딩", "선택"],
        [["모델이 HBM 에 들어간다",
          "HBM 대역폭으로 전속 실행",
          "CPU 연산·전송이 추가되어 느려진다",
          "**GPU-only"],
         ["모델이 HBM 에 들어가지 않는다",
          "!!실행 불가 (0)",
          "느리지만 실행된다",
          "**오프로딩"]],
        widths=[4.2, 4.2, 4.8, 2.8], size=9.5)
    rich([("이 캠페인은 두 상황을 각각 측정해 이 경계를 실증했다. ", {}),
          ("Qwen3-30B-A3B", {"bold": True}),
          (" 은 HBM 에 들어가는 모델이고 GPU-only 가 15.2 배 빨랐다. ", {}),
          ("DeepSeek-R1-0528", {"bold": True}),
          (" 은 들어가지 않는 모델이고 GPU-only 는 0 이었다.", {})])


# ================================================================ 관련 자료
def s_related():
    h1("관련 자료", 4)

    h2("4.1 외부 스택")
    table(
        ["구성 요소", "버전", "역할"],
        [["SGLang", "0.5.18 (``lmsysorg/sglang:latest, digest 9e148f5ac788)",
          "서빙 엔진. GPU attention·router·스케줄러"],
         ["kt-kernel", "0.7.0.post2 (``pip install --no-deps)",
          "CPU expert 커널(AMX)과 가중치 변환기"],
         ["KTransformers", "``kvcache-ai/ktransformers (main)",
          "kt-kernel 상위 프로젝트. 변환 스크립트 출처"],
         ["vLLM", "0.28.0 (``vllm/vllm-openai:latest)",
          "GPU-only 대조군(r0) 및 다른 트랙"]],
        widths=[3.0, 6.6, 6.4], size=9.5)
    p("kt-kernel 을 기본 설치하면 이미지의 torch 2.13.0+cu130 이 2.9.1 로 내려가 "
      "환경이 깨진다. 따라서 의존성 해석을 차단한 설치가 필수였다.")
    code(["pip install --no-deps kt-kernel==0.7.0.post2",
          "",
          "# 변환 스크립트는 wheel 에 포함되지 않아 별도 수급",
          "mkdir -p /usr/local/lib/python3.12/dist-packages/scripts",
          "curl -sfL https://raw.githubusercontent.com/kvcache-ai/ktransformers/main/\\",
          "  kt-kernel/scripts/convert_cpu_weights.py \\",
          "  -o /usr/local/lib/python3.12/dist-packages/scripts/convert_cpu_weights.py"])

    h2("4.2 외부 참조 성능치")
    p("KTransformers 프로젝트가 공개한 DeepSeek-R1 하이브리드 서빙 참조치는 8×L20 + Xeon "
      "구성에서 227 tok/s 다. 하드웨어와 구성이 본 측정과 다르므로 동일 조건 비교가 "
      "아니지만, 이 경로의 튜닝 여지 규모를 가늠하는 값으로 인용한다.")

    h2("4.3 저장소 내부 선행 ID")
    table(
        ["ID", "내용", "이 트랙과의 관계"],
        [["SUB_036 / 040 / 041 / 042", "dense 모델 CPU offload",
          "!!기각 — 본 트랙이 그 결론을 피해 설계된 근거"],
         ["IDE_006", "하이브리드 KV 경로의 Q-dependency dilemma",
          "본 트랙은 KV 를 옮기지 않아 무관함이 확인됨"],
         ["IDE_024 / TSK_044", "CPU co-location 간섭 측정", "같은 캠페인 병행 트랙"],
         ["IDE_025 / TSK_045", "DRAM KV/prefix tier (+51.8 %)", "같은 캠페인 병행 트랙"],
         ["TSK_046", "노드 baseline re-anchor (70B-FP8 = 3,039.0 tok/s)",
          "같은 캠페인 기준선"],
         ["SUB_167", "R1 INT4 출력 품질 결함", "**본 트랙에서 파생된 미결 항목"],
         ["IDE_030 / IDE_031", "hot/cold expert 배치·예측 모델 (Qwen3-480B)",
          "이후 9월 작업. 본 트랙의 후속 계열"]],
        widths=[4.4, 6.2, 5.8], size=9.5,
        note="출처: shadow_assists/id_registry.md")


# ================================================================ 실험 구성
def s_setup():
    h1("실험 구성", 5)

    h2("5.1 하드웨어")
    table(
        ["구성", "사양", "비고"],
        [["CPU", "Intel Xeon Platinum 8480+ ×2 — 112 코어 / 224 스레드",
          "AMX 네이티브 지원"],
         ["CPU 클럭", "**2.0 GHz 고정 (turbo OFF, ``no_turbo=1)",
          "!!모든 CPU 수치는 이 상태의 하한"],
         ["메모리", "DDR5 2 TB", "expert 328 GB 적재 여유 충분"],
         ["GPU", "NVIDIA H100 80 GB ×8, NVLink", "측정 시 전 GPU 유휴 확인"],
         ["노드 성격", "Kubernetes 워커 (containerd)",
          "Docker Engine 없음 — ``sudo nerdctl 셔임 사용, 전 컨테이너 ``--net host"],
         ["네트워크", "약 105 MB/s", "70B 68 GB 다운로드 11 분 실측"]],
        widths=[2.6, 7.4, 6.4], size=9.5,
        note="출처: PLN_003.md §2, IDE_023/CLAUDE.md")
    callout("turbo OFF 가 결과에 주는 의미", [
        "이 노드는 k8s 워커이므로 클럭 정책 변경이 사용자 결정 사항이었고, 캠페인 시점에는 "
        "2.0 GHz 로 고정된 상태였다.",
        "expert 연산이 CPU 에서 이뤄지는 이 트랙에서 클럭은 처리량에 직접 영향한다. "
        "따라서 19.67 tok/s 는 클럭 상한이 풀린 상태의 값이 아니다.",
    ])

    h2("5.2 소프트웨어 스택")
    table(
        ["레이어", "구성"],
        [["GPU 서빙", "SGLang 0.5.18 · TP=8 · attention backend = triton"],
         ["CPU expert", "kt-kernel 0.7.0.post2 · AMXINT4 · cpuinfer 96 threads · "
                        "threadpool 2"],
         ["대조군(r0)", "vLLM 0.28.0 · TP=8 · gpu-memory-utilization 0.95"],
         ["벤치", "``vllm bench serve --backend openai (SGLang 미지원으로 openai 경로 사용)"],
         ["CPU 계측", "``/proc/stat 자작 샘플러 (노드에 mpstat 부재)"],
         ["GPU 계측", "``nvidia-smi 주기 질의 — util / memory / power"]],
        widths=[3.2, 13.2], size=9.5)

    h2("5.3 필요했던 호환성 패치와 우회")
    p("이 트랙은 저장소의 vLLM fork 소스를 한 줄도 바꾸지 않았다. 필요한 수정은 전부 "
      "컨테이너 내부 패치와 실행 플래그였다.")
    table(
        ["#", "증상", "원인", "조치"],
        [["1", "SGLang FA3 crash", "이미지 내부 cutlass / nvvm 버전 불일치",
          "``--attention-backend triton"],
         ["2", "``KTMoEWrapper TypeError", "SGLang 0.5.18 ↔ kt-kernel 0.7.0 API 불일치",
          "래퍼에 ``gpu_experts_mask=None 인자 추가"],
         ["3", "``moe_align_block_size 인자 누락",
          "이미지 내 sgl_kernel 파이썬 래퍼 ↔ 컴파일된 op 불일치 (GPU-only 도 crash)",
          "래퍼에 ``ignore_invalid_expert 인자 추가"],
         ["4", "**KT 경로 CUDA graph crash (``invalid argument)",
          "**CPU 동기 expert 경로가 graph replay 와 비호환",
          "**``--disable-cuda-graph"],
         ["5", "``LocalEntryNotFoundError",
          "``~/hetero-exp/models/hub 이 컨테이너 안에서 깨지는 심볼릭 링크",
          "실제 캐시 ``~/.cache/huggingface 직접 마운트"],
         ["6", "70B 서버 로딩 정지",
          "캐시가 실제로는 불완전 (``.incomplete blob) + 비인증 rate-limit 재다운로드",
          "호스트 ``hf download 로 완성 후 ``HF_HUB_OFFLINE=1"]],
        widths=[0.8, 4.0, 6.4, 5.2], size=9,
        note="출처: COMPREHENSIVE_REPORT_20260827.md §2.3, §3.2 · PROGRESS_20260827.md (15:0x). "
             "#4 가 서빙 성립의 결정타였다.")
    p("패치 2·3 의 실제 diff 는 다음과 같다.")
    code(["# /sgl-workspace/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py  (~L224)",
          "         self.wrapper = KTMoEWrapper(",
          "             ...",
          "             moe_intermediate_size=intermediate_size_full,",
          "+            gpu_experts_mask=None,   # kt-kernel 0.7.0 필수 인자. None = 전 expert CPU",
          "             num_gpu_experts=self.num_gpu_experts,",
          "",
          "# /usr/local/lib/python3.12/dist-packages/sgl_kernel/moe.py",
          " def moe_align_block_size(..., pad_sorted_token_ids=False,",
          "+    ignore_invalid_expert=False,",
          " ):",
          "     torch.ops.sgl_kernel.moe_align_block_size.default(..., pad_sorted_token_ids,",
          "+        ignore_invalid_expert,",
          "     )"])

    h2("5.4 가중치 변환 파이프라인")
    p("FP8 원본을 그대로 읽히려는 시도가 먼저 있었고 실패했다. kt-kernel 의 native FP8 "
      "로더는 KT 전용 샤딩 레이아웃을 요구하므로 HF 원본 직독이 불가능했다.")
    code(["# 실패한 경로",
          "--kt-method FP8  +  원본 HF snapshot",
          "  → native FP8 per-expert TP source is incomplete",
          "  (FP8_MOE_TP pool 생성 backend=AMX 까지는 정상 진행)",
          "",
          "# 채택한 경로",
          "kt quant -m int4 -i fp8   # 96 threads, 65 분, 642 GB → 328 GB"])
    figure("d4_convert.png",
           "그림 4. 가중치 변환 파이프라인과 실패 경로. 점선 상자 두 곳이 이후 품질 결함의 "
           "원인 후보로 남았다.")

    h2("5.5 서빙 구성")
    p("Qwen3-30B smoke 와 R1 본판에 공통으로 사용한 기동 명령의 골격이다.")
    code(["python3 -m sglang.launch_server \\",
          "  --model-path <HF snapshot> --tp <1|8> \\",
          "  --attention-backend triton \\",
          "  --disable-cuda-graph \\",
          "  --trust-remote-code \\",
          "  --kt-weight-path <kt quant 출력 dir> \\",
          "  --kt-method <AMXINT8|AMXINT4> \\",
          "  --kt-cpuinfer 96 \\",
          "  --kt-threadpool-count 2 \\",
          "  --kt-num-gpu-experts 0          # 0 = expert 전량 CPU"])
    table(
        ["플래그", "값", "의미"],
        [["``--kt-cpuinfer", "96", "expert 연산 스레드 수. 112 코어 중 96 사용"],
         ["``--kt-threadpool-count", "2", "소켓당 1 개 (2 소켓 구성)"],
         ["``--kt-num-gpu-experts", "**0", "**hot expert 의 GPU 배치를 쓰지 않음 — 하한 조건"],
         ["``--kt-method", "AMXINT4", "CPU 커널 정밀도"],
         ["``--disable-cuda-graph", "—", "CPU 동기 경로와 graph replay 비호환"],
         ["``--attention-backend", "triton", "이미지 내 FA3 빌드 불일치 우회"]],
        widths=[4.6, 2.2, 9.6], size=9.5)

    h2("5.6 워크로드와 계측")
    table(
        ["단계", "모델", "TP", "요청", "입력/출력 토큰", "동시성"],
        [["smoke t1 (GPU-only)", "Qwen3-30B-A3B-FP8", "1", "128", "1024 / 256", "16"],
         ["smoke t2 (KT hybrid)", "Qwen3-30B-A3B → AMXINT8", "1", "128", "1024 / 256", "16"],
         ["r0 (GPU-only)", "DeepSeek-R1-0528 FP8", "8", "—", "—", "—"],
         ["**r1 (KT hybrid)", "**DeepSeek-R1-0528 → AMXINT4", "**8", "**32",
          "**1024 / 128", "**8"]],
        widths=[3.8, 4.6, 1.0, 1.4, 3.0, 2.2], size=9,
        note="입력 토큰은 dataset 실측 합계 기준 — smoke 129,773 tok / r1 32,510 tok. "
             "r0 는 서버 기동 단계에서 실패하여 벤치를 실행하지 못했다.")

    h2("5.7 게이트 정의 (TST_021, 측정 전 확정)")
    table(
        ["#", "게이트", "충족 여부"],
        [["1", "GPU-only 로 적재 불가한 regime 임을 실증한다",
          "**충족 — r0 OutOfMemoryError"],
         ["2", "하이브리드 경로로 서빙이 성립한다 (전 요청 완료)",
          "**충족 — r1 32/32 완료"],
         ["3", "CPU 가 연산 주체로 소비된다",
          "**충족 — CPU busy 42.7 %, cpuinfer 96 포화"],
         ["4", "출력 품질이 정상이다",
          "!!미충족 — greedy 출력 비문 (SUB_167)"]],
        widths=[0.8, 9.2, 6.4], size=9.5,
        note="출처: PLN_003.md §4, eval/results/20260827_140008_tsk043_main_r1/RESULTS.md")

    h2("5.8 캠페인 내 실행 순서")
    p("GPU 8 장은 셀 간 서버 재기동이 필요해 직렬로 점유했고, 다운로드·변환·문서화 같은 "
      "CPU-only 작업은 병렬로 진행했다. R1 다운로드(642 GB)와 INT4 변환(65 분)이 "
      "전체 일정의 long-pole 이었다.")
    figure("d5_campaign.png",
           "그림 5. 캠페인 실행 순서. 주황 상자가 이 백서의 범위인 TSK_043 이다.")


# ================================================================ 결과
def s_results():
    h1("실험 결과", 6)

    h2("6.1 단계별 개요")
    table(
        ["단계", "목적", "결과", "판정"],
        [["smoke t1", "GPU-only 기준선 확보 (GPU-fit 모델)", "1,461.77 tok/s", "**기준선"],
         ["smoke t2", "KT 스택 전 구간 검증", "96.09 tok/s · 출력 정상", "**통과"],
         ["r0", "GPU-only 불가 regime 실증", "``torch.OutOfMemoryError (90 초)",
          "**통과"],
         ["r1 (1차)", "FP8 원본 직독 하이브리드",
          "!!native FP8 per-expert TP source is incomplete", "!!실패"],
         ["변환", "AMXINT4 재양자화", "328 GB · 65 분", "**완료"],
         ["**r1 (최종)", "**AMXINT4 하이브리드 서빙",
          "**19.67 tok/s · 32/32 완료 · CPU 42.7 %", "**부분 통과"],
         ["판별 실험", "출력 품질 결함 원인 분해", "4 가설 중 3 기각", "!!SUB_167 이관"]],
        widths=[2.4, 5.0, 6.4, 2.6], size=9.5)

    h2("6.2 Qwen3-30B-A3B smoke — 스택 검증")
    p("본판 전에 GPU 에 들어가는 MoE 모델로 스택 전 구간(변환 → AMX 적재 → 연산 → 정확성 "
      "→ CPU 포화)을 검증했다. 이 단계의 목적은 성능 비교가 아니다.")
    table(
        ["지표", "t1 GPU-only", "t2 KT hybrid", "비"],
        [["출력 처리량", "**1,461.77 tok/s", "96.09 tok/s", "0.066×"],
         ["벤치 소요", "22.42 s", "341.03 s", "15.2×"],
         ["TTFT p50", "179.28 ms", "2,381.02 ms", "13.3×"],
         ["TPOT p50", "9.66 ms", "157.69 ms", "16.3×"],
         ["완료 요청", "128 / 128", "128 / 128", "—"],
         ["입력 / 출력 토큰", "129,773 / 32,768", "129,773 / 32,768", "동일"],
         ["CPU busy 평균", "1.98 %", "**43.06 %", "21.7×"],
         ["GPU0 util 평균", "48.8 %", "4.8 %", "0.10×"],
         ["출력 정확성", "정상", "**정상", "—"]],
        widths=[3.2, 4.2, 4.2, 2.2], size=9.5, align_right=(1, 2, 3),
        note="CPU / GPU 활용률은 원본 cpu_util.txt · gpu_util.csv 를 전 구간 재집계한 값이다 "
             "(t1 CPU 19 표본 / GPU 8 표본, t2 CPU 179 표본 / GPU 73 표본). "
             "캠페인 보고서는 벤치 구간 한정 GPU 0.6 % 를 기록했다 — 표본에 타임스탬프가 "
             "없어 본 문서에서는 창을 좁히지 못했다.")
    figure("c1_smoke.png",
           "그림 6. Qwen3-30B-A3B — GPU-only 와 하이브리드의 처리량·자원 활용률. "
           "HBM 에 들어가는 모델에서는 GPU-only 가 15.2 배 빠르다.", width=15.5)
    rich([("이 결과는 오프로딩의 실패가 아니다. ", {}),
          ("HBM 에 들어가는 모델에서 GPU 가 이기는 것은 설계상 당연한 결과", {"bold": True}),
          ("이며, 이 단계에서 확인하려던 것은 '스택이 동작하는가' 였다. 출력이 정상이고 "
           "CPU 가 포화했으므로 목적은 달성되었다.", {})])

    h2("6.3 r0 — GPU-only 적재 불가 실증")
    p("vLLM 0.28, TP=8, gpu-memory-utilization 0.95 로 DeepSeek-R1-0528 을 적재하면 90 초 "
      "만에 worker 로드 단계에서 실패한다.")
    code(["torch.OutOfMemoryError: CUDA out of memory",
          "",
          "(EngineCore) ERROR  multiproc_executor.py:805 in wait_for_ready",
          "(EngineCore) ERROR  Exception: WorkerProc initialization failed due to an",
          "                    exception in a background process.",
          "(APIServer)  RuntimeError: Engine core initialization failed."])
    p("원본: ``eval/results/20260827_133055_tsk043_main_r1/r0_gpu_only_oom/oom_evidence.log",
      size=8.5, color=MUTED)
    p("이 실패는 설정 오류가 아니라 용량 산술의 결과다. 642 GB 가중치 > usable HBM 608 GB "
      "이므로 KV cache 를 0 으로 잡아도 적재되지 않는다.")

    h2("6.4 r1 — 하이브리드 서빙 성립")
    table(
        ["지표", "값", "원본 필드"],
        [["서버 기동 (HEALTH OK)", "**160 s", "RUN.log"],
         ["완료 요청", "**32 / 32 (실패 0)", "Successful requests"],
         ["벤치 소요", "208.26 s", "Benchmark duration"],
         ["출력 처리량", "**19.67 tok/s", "Output token throughput"],
         ["최대 순간 출력 처리량", "32.00 tok/s", "Peak output token throughput"],
         ["전체 토큰 처리량", "175.77 tok/s", "Total token throughput"],
         ["요청 처리량", "0.15 req/s", "Request throughput"],
         ["입력 / 생성 토큰", "32,510 / 4,096", "Total input / generated tokens"],
         ["TTFT 평균 / p50 / p95", "12,380.89 / 11,199.23 / 18,098.75 ms",
          "Time to First Token"],
         ["TPOT 평균 / p50 / p95", "312.46 / 305.50 / 326.34 ms",
          "Time per Output Token"],
         ["ITL 평균 / p50 / p95", "312.46 / 304.82 / 318.48 ms",
          "Inter-token Latency"],
         ["최대 동시 요청", "8 (관측 peak 16)", "Maximum / Peak concurrent requests"]],
        widths=[4.6, 6.4, 5.4], size=9.5,
        note="원본: eval/results/20260827_140008_tsk043_main_r1/r1_kt_hybrid/bench.log")
    figure("c2_r1.png",
           "그림 7. DeepSeek-R1-0528 — GPU-only 는 0 (적재 실패), 하이브리드는 19.67 tok/s. "
           "오른쪽은 r1 의 지연 분포(로그 축).", width=15.5)
    p("기동 160 초는 642 GB 원본이 아니라 328 GB 변환본을 DRAM 으로 올리는 시간이며, "
      "직전 변환 작업으로 page cache 가 더워진 상태의 값이다.")

    h2("6.5 자원 활용")
    h3("CPU")
    table(
        ["지표", "값"],
        [["표본 수 / 관측 구간", "118 표본 / 235 초 (2 초 간격)"],
         ["CPU busy 평균", "**42.75 %"],
         ["CPU busy 중앙값", "46.91 %"],
         ["CPU busy 최대", "**49.49 %"],
         ["스레드 구성", "cpuinfer 96 / 224 논리 스레드 (112 물리 코어)"],
         ["클럭", "2.0 GHz 고정 (turbo OFF)"]],
        widths=[6.0, 10.4], size=9.5,
        note="원본 cpu_util.txt 재집계. 96/224 = 42.9 % 이므로 관측된 42.75 % 는 "
             "cpuinfer 스레드가 사실상 포화 상태였음을 뜻한다.")
    figure("c3_cpu_series.png",
           "그림 8. r1 실행 중 CPU 사용률 시계열. 초기 상승 후 약 47 % 수준에서 "
           "평탄해진다.", width=15.5)
    h3("GPU")
    table(
        ["GPU", "util 평균 %", "HBM 점유 GiB", "평균 전력 W"],
        [["GPU0", "**15.5", "73.0", "154"],
         ["GPU1", "81.1", "72.3", "175"],
         ["GPU2", "67.2", "72.3", "166"],
         ["GPU3", "90.9", "72.3", "191"],
         ["GPU4", "91.0", "72.3", "190"],
         ["GPU5", "84.3", "72.3", "176"],
         ["GPU6", "87.1", "72.3", "184"],
         ["GPU7", "87.6", "71.9", "187"],
         ["**전체 평균", "**75.6", "**72.4", "**178"]],
        widths=[2.6, 4.6, 4.6, 4.6], size=9.5, align_right=(1, 2, 3),
        note="원본 gpu_util.csv 재집계 — GPU 장당 48 표본. HBM 점유 합계는 약 579 GiB 로, "
             "expert 를 DRAM 으로 내린 뒤 남은 attention 가중치와 KV pool 이다.")
    figure("c4_per_gpu.png",
           "그림 9. GPU별 활용률·HBM 점유·전력. GPU0 만 15.5 % 로 낮다.", width=15.5)
    callout("GPU0 의 낮은 활용률", [
        "GPU0 은 다른 7 장(67~91 %)과 달리 15.5 % 에 머물렀다. HBM 점유는 73.0 GiB 로 "
        "오히려 가장 크다.",
        "이 캠페인은 원인을 규명하지 않았다. 일반적으로 rank 0 이 스케줄러·토크나이저·"
        "샘플링 등 직렬 작업을 겸하면 나타나는 패턴이지만, 이 측정만으로는 확정할 수 없다.",
        "후속 튜닝에서 확인 대상이다.",
    ])

    h2("6.6 출력 품질 게이트 미통과")
    p("r1 의 greedy smoke 출력이 비문이었다. 따옴표·기호가 반복되는 형태이며, 정답 토큰이 "
      "부분적으로 등장하기는 하나 문장이 성립하지 않았다. 게이트 4 를 통과하지 못했다.")
    p("원인을 네 가설로 분해해 판별 실험을 수행했다.")
    table(
        ["#", "가설", "실험", "결과"],
        [["1", "chat template 처리 문제", "chat endpoint 경유 재측정",
          "!!기각 — 동일하게 깨짐"],
         ["2", "INT4 경로 일반 결함",
          "Qwen3-30B-A3B-FP8 → AMXINT8 변환 후 smoke", "!!기각 — 출력 정상"],
         ["2′", "INT4 경로 일반 결함 (같은 변환기)",
          "Qwen3-30B-A3B-FP8 → AMXINT4 변환 후 smoke", "!!기각 — 출력 정상"],
         ["3", "변환 스크립트 버전 스큐",
          "``main 과 ``v0.7.0.post1 태그 비교", "!!기각 — IDENTICAL"],
         ["4", "kt-kernel 버전 문제", "kt-kernel 0.7.0 으로 스왑",
          "!!판정 불가 — 무관한 import 오류로 부팅 실패"],
         ["**5", "**DeepSeek 계열 특이 처리",
          "**위 기각의 잔여 후보", "**수렴 — SUB_167 로 이관"]],
        widths=[0.8, 4.6, 5.6, 5.4], size=9,
        note="원본: eval/results/20260827_140008_tsk043_main_r1/RESULTS.md, "
             "PROGRESS_20260827.md")
    p("남은 후보는 두 가지로 좁혀졌다.")
    bullet("128×128 블록 단위 ``weight_scale_inv 의 dequant 처리 — DeepSeek FP8 체크포인트 "
           "특유의 스케일 저장 방식이다.", bold_prefix="후보 A  ")
    bullet("shared expert(257 번째)의 폴딩 방식 — routed expert 와 다른 취급이 필요하다.",
           bold_prefix="후보 B  ")
    rich([("이 결함이 미해결인 동안 ", {}),
          ("R1 의 성능 튜닝은 의미가 없다", {"bold": True}),
          (". 잘못된 출력을 더 빠르게 만드는 작업이 되기 때문이다. 캠페인은 이 원칙에 따라 "
           "튜닝을 중단하고 SUB_167 을 선행 게이트로 지정했다.", {})])


# ================================================================ 분석
def s_analysis():
    h1("분석", 7)

    h2("7.1 19.67 tok/s 를 어떻게 읽어야 하는가")
    p("이 수치는 성능 주장이 아니라 성립 증명이다. 비교 대상이 '다른 구성의 같은 모델' 이 "
      "아니라 '서빙 자체가 불가능한 상태' 이기 때문이다. 개선율을 백분율로 쓸 수 없다 — "
      "분모가 0 이다.")
    p("동시에 이 수치는 이 구성이 낼 수 있는 값의 하한이다. 성능을 올릴 수 있는 수단 세 "
      "가지가 모두 사용되지 않은 상태에서 측정되었다.")
    table(
        ["미사용 수단", "측정 시 상태", "예상 방향"],
        [["CPU 클럭", "**2.0 GHz 고정 (turbo OFF)",
          "expert 연산이 지배 항이므로 클럭에 거의 비례해 개선"],
         ["hot expert 의 GPU 배치", "**``--kt-num-gpu-experts 0 (전량 CPU)",
          "자주 쓰이는 expert 를 HBM 여유에 올려 CPU 구간 축소"],
         ["expert deferral", "**미사용",
          "CPU expert 계산을 다음 층 attention 과 겹쳐 대기 제거"],
         ["cpuinfer / NUMA 튜닝", "96 threads · threadpool 2 · 기본 배치",
          "소켓 지역성 조정 여지"]],
        widths=[4.0, 5.8, 6.6], size=9.5)
    figure("c5_headroom.png",
           "그림 10. 본 측정치와 KTransformers 공개 참조치. 하드웨어·구성이 다르므로 "
           "동일 조건 비교가 아니며, 여지의 규모를 가리키는 값으로만 인용한다.", width=15.5)

    h2("7.2 스텝 시간의 구성")
    p("TPOT p50 이 305.5 ms 이고 ITL p50 이 304.82 ms 로 거의 같다. 토큰 간 간격이 "
      "토큰당 처리 시간과 일치한다는 뜻이며, 파이프라인이 겹쳐지지 않고 직렬로 "
      "진행된다는 신호다. 그림 3 의 'GPU 대기' 구간이 실제로 존재한다는 근거다.")
    p("한편 GPU util 평균은 75.6 % 로 낮지 않다. 두 수치는 모순이 아니다 — TP=8 의 "
      "MLA attention 과 all-reduce 통신이 GPU 를 점유하는 구간이 있고, expert 대기 "
      "구간에서도 nvidia-smi 의 표본이 busy 로 잡힐 수 있다. 이 측정은 스텝 내부를 "
      "구간별로 분해하지 않았으므로 두 구간의 정확한 비율은 알 수 없다.")
    callout("이 측정이 답하지 못하는 것", [
        "· 한 스텝에서 GPU attention · 전송 · CPU expert · combine 이 각각 몇 ms 인지 — "
        "구간별 계측을 하지 않았다. 총 TPOT 만 있다.",
        "· CPU expert 구간에서 메모리 대역폭이 포화했는지 — DRAM 대역폭 카운터를 "
        "수집하지 않았다.",
        "· GPU0 의 낮은 활용률의 원인.",
    ])

    h2("7.3 두 모델의 대비가 말하는 것")
    table(
        ["", "Qwen3-30B-A3B", "DeepSeek-R1-0528"],
        [["가중치 (서빙 정밀도)", "HBM 에 적재 가능", "**642 GB — 적재 불가"],
         ["GPU-only", "**1,461.77 tok/s", "!!0 (OutOfMemoryError)"],
         ["CPU 하이브리드", "96.09 tok/s", "**19.67 tok/s"],
         ["하이브리드 선택 이유", "없음 (GPU 가 15.2× 빠름)", "**유일한 실행 경로"],
         ["CPU busy", "43.06 %", "42.75 %"],
         ["출력 정확성", "정상", "!!비문"]],
        widths=[4.6, 5.8, 6.0], size=9.5)
    p("두 행의 CPU busy 가 거의 같다(43.06 % vs 42.75 %)는 점이 이 경로의 성격을 보여준다. "
      "CPU 측 부하는 모델 크기와 무관하게 cpuinfer 스레드 수(96)가 결정한다. 즉 CPU 는 "
      "두 경우 모두 동일하게 포화했고, 달라진 것은 그 포화가 만들어내는 가치다 — "
      "30B 에서는 GPU 가 더 잘 하는 일을 대신한 것이고, R1 에서는 아무도 할 수 없던 일을 "
      "한 것이다.")


# ================================================================ 한계
def s_limits():
    h1("한계와 미결 항목", 8)

    h2("8.1 이 백서가 주장하지 않는 것")
    bullet("CPU MoE 오프로딩이 일반적으로 빠르다 — 주장하지 않는다. GPU 에 들어가는 모델에서는 "
           "15.2 배 느렸다.")
    bullet("19.67 tok/s 가 이 구성의 성능이다 — 주장하지 않는다. 튜닝 수단 3 가지가 "
           "미사용인 하한이다.")
    bullet("이 경로의 출력이 정확하다 — 주장하지 않는다. R1 에서 품질 게이트를 통과하지 "
           "못했다.")
    bullet("dense 모델에도 적용된다 — 주장하지 않는다. 선행 SUB_036/040/041/042 에서 "
           "기각된 영역이다.")

    h2("8.2 미결 항목")
    table(
        ["우선", "항목", "내용", "선행 조건"],
        [["**1", "**SUB_167",
          "R1 INT4 출력 품질 결함. 후보 = block-wise ``weight_scale_inv dequant 또는 "
          "shared expert 폴딩. upstream 이슈 조회·제보 대상",
          "—"],
         ["2", "turbo unlock",
          "``no_turbo=1 해제. 본 백서의 모든 CPU 수치가 하한인 원인",
          "k8s 워커 정책 — 사용자 결정"],
         ["3", "R1 성능 튜닝",
          "hot expert GPU 배치 (``--kt-num-gpu-experts), expert deferral, "
          "cpuinfer / NUMA 튜닝",
          "**SUB_167 통과"],
         ["4", "스텝 내부 계측",
          "GPU attention · 전송 · CPU expert 구간 분해",
          "—"],
         ["5", "GPU0 활용률 규명", "rank 0 직렬 작업 가설 확인", "—"],
         ["6", "Phase 2 — fork 내재화",
          "IDE_023 경로를 vLLM fork 에 통합",
          "**SUB_167 + 튜닝 수치 확보"]],
        widths=[1.2, 3.0, 8.0, 4.2], size=9,
        note="출처: COMPREHENSIVE_REPORT_20260827.md §4")

    h2("8.3 재현 시 주의")
    bullet("컨테이너-로컬 패치는 휘발성이다. 컨테이너를 다시 만들면 §5.3 의 패치 4 건을 "
           "재적용해야 한다.")
    bullet("kt-kernel 은 반드시 ``--no-deps 로 설치한다. 기본 설치는 이미지의 torch 를 "
           "교체해 환경을 깨뜨린다.")
    bullet("``--disable-cuda-graph 없이는 첫 prefill 에서 ``invalid argument 로 죽는다.")
    bullet("FP8 원본 직독은 되지 않는다. ``kt quant 변환이 필수이며 R1 기준 65 분이 걸린다.")
    bullet("이 노드는 Docker Engine 이 없다. ``~/bin/docker 는 ``sudo nerdctl 셔임이며 "
           "모든 컨테이너를 ``--net host 로 띄운다.")


# ================================================================ 부록
def s_appendix():
    h1("부록", 9)

    h2("A. 원시 데이터 목록")
    table(
        ["경로", "내용"],
        [["``eval/results/20260827_124509_tsk043_smoke_qwen30b/t1_gpu_only/",
          "``bench.log · ``server.log · ``cpu_util.txt · ``gpu_util.csv · ``smoke.json"],
         ["``eval/results/20260827_124509_tsk043_smoke_qwen30b/t2_kt_hybrid/",
          "위와 동일 + ``server_fail.log (CUDA graph crash 기록)"],
         ["``eval/results/20260827_133055_tsk043_main_r1/r0_gpu_only_oom/",
          "``oom_evidence.log · ``server.log"],
         ["``eval/results/20260827_133055_tsk043_main_r1/r1_kt_hybrid/",
          "``server_fail.log (FP8 직독 실패 기록)"],
         ["``eval/results/20260827_140008_tsk043_main_r1/",
          "``RESULTS.md · ``RUN.log"],
         ["``eval/results/20260827_140008_tsk043_main_r1/r1_kt_hybrid/",
          "``bench.log · ``server.log · ``cpu_util.txt (118 표본) · "
          "``gpu_util.csv (384 행) · ``smoke.json"]],
        widths=[8.4, 8.0], size=9)

    h2("B. 가중치 산출물 (디스크, 미커밋)")
    table(
        ["경로", "크기", "용도"],
        [["``~/.cache/huggingface/.../DeepSeek-R1-0528", "642 GB", "원본 FP8"],
         ["``~/.cache/huggingface/kt/r1-0528-int4", "**328 GB",
          "AMXINT4 변환본 — r1 본판"],
         ["``~/.cache/huggingface/kt/qwen3-30b-a3b-int8", "31 GB", "smoke t2"],
         ["``~/.cache/huggingface/kt/qwen3-30b-a3b-int4", "15 GB", "품질 판별 실험"]],
        widths=[8.0, 2.6, 5.8], size=9.5)

    h2("C. ID 대응표")
    table(
        ["ID", "종류", "내용", "상태"],
        [["``PLN_003", "계획", "Hybrid Regime Sweep 캠페인", "활성 (2026-08-27)"],
         ["``IDE_023", "아이디어", "MoE Expert Offload", "**본 백서의 주제"],
         ["``TSK_043", "작업", "SGLang + KT 로 R1-0528 하이브리드 서빙",
          "**완료 (품질 follow-up 분리)"],
         ["``TST_021", "테스트", "TSK_043 게이트 4 항", "3/4 충족"],
         ["``SUB_167", "하위 문제", "R1 INT4 출력 품질 결함", "!!대기"],
         ["``IDE_024 / TSK_044", "병행", "CPU co-location", "완료"],
         ["``IDE_025 / TSK_045", "병행", "DRAM KV/prefix tier", "완료 (+51.8 %)"],
         ["``TSK_046", "병행", "baseline re-anchor", "완료 (scope 축소)"]],
        widths=[3.6, 2.0, 6.4, 4.4], size=9.5,
        note="출처: shadow_assists/id_registry.md")

    h2("D. 같은 캠페인의 다른 트랙 (비교 맥락)")
    p("이 백서의 범위는 아니지만, 같은 날 같은 노드에서 측정된 다른 트랙의 결과를 "
      "맥락으로 남긴다.")
    table(
        ["트랙", "결과", "CPU 의 역할"],
        [["IDE_025 / TSK_045 — DRAM KV tier",
          "**압박 구성에서 418.0 → 634.4 tok/s (+51.8 %), TTFT p50 −64.9 %",
          "CPU 는 연산하지 않는다. 2 TB DRAM 이 저장 tier 로 쓰인다"],
         ["IDE_024 / TSK_044 — co-location",
          "BG 56 proc 시 GPU −0.50 %, BG 112 proc 시 −3.69 %",
          "CPU 가 별도 작업을 수행 (간섭 비용 곡선)"],
         ["**IDE_023 / TSK_043 — MoE offload",
          "**서빙 불가(0) → 19.67 tok/s",
          "**CPU 가 expert 연산의 주체"]],
        widths=[4.6, 6.4, 5.4], size=9,
        note="출처: COMPREHENSIVE_REPORT_20260827.md §1.1. 세 트랙 모두 저장소 "
             "fork 소스 변경 0 줄로 수행되었다.")

    h2("E. 문서·그림 생성")
    table(
        ["파일", "내용"],
        [["``shadow_assists/features/IDE_023/whitepaper/make_figures.py",
          "그림 10 장 생성 (구성도 5 = SVG → PNG, 차트 5 = matplotlib)"],
         ["``shadow_assists/features/IDE_023/whitepaper/make_docx.py",
          "이 문서 생성"],
         ["``shadow_assists/features/IDE_023/whitepaper/figures/",
          "PNG + SVG 원본"]],
        widths=[9.0, 7.4], size=9)
    p("그림의 수치는 모두 §A 의 원시 데이터에서 읽거나 캠페인 보고서 표에서 인용했다. "
      "재집계 값은 해당 표 각주에 명시했다.", size=9, color=MUTED)


if __name__ == "__main__":
    cover()
    s_summary()
    s_background()
    s_principle()
    s_related()
    s_setup()
    s_results()
    s_analysis()
    s_limits()
    s_appendix()
    doc.save(OUT)
    print(f"→ {OUT}")
    print(f"   {os.path.getsize(OUT):,} bytes")
