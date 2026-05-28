# METRONOME — IEEE/ACM 국제 학술대회 논문

> **제목**: METRONOME: 투기적 디코딩 환경의 박자 정렬 교차 도메인 CPU 자원 오케스트레이션 알고리즘 — 24-기법 실험 기반 설계와 검증
>
> **저자**: Kyunam Cho (1저자), Heon-Chang Yoo (교신저자)
>
> **target**: IEEE/ACM 국제 학술대회 (IEEEtran conference 클래스, 10 pages × 2-column)
>
> **언어**: 한국어 본문 (XeLaTeX + kotex)

---

## 디렉토리 구조

```
paper/
├── main.tex                            ← 진입점 (IEEEtran preamble + \input)
├── references.bib                      ← BibTeX (3-agent verification 후 작성)
├── sections/
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_background.tex
│   ├── 04_problem_formulation.tex
│   ├── 05_metronome_algorithm.tex      ← ★ 핵심 contribution
│   ├── 06_methodology.tex
│   ├── 07_results.tex
│   ├── 08_mechanism_analysis.tex
│   ├── 09_discussion.tex
│   └── 10_conclusion.tex
├── figures/                            ← TikZ 그림 (Fig.1 ~ Fig.6)
├── tables/                             ← booktabs 표 (Table I ~ V)
└── README.md                           ← (본 문서)
```

---

## 빌드 환경

본 프로젝트는 **XeLaTeX + kotex + fontspec + Nanum 폰트** 조합 사용.

### Overleaf (권장)

1. 프로젝트를 Overleaf 에 업로드 (zip 또는 git import)
2. Menu → Settings → **Compiler: XeLaTeX** 선택
3. Recompile

Overleaf 는 NanumMyeongjo / NanumGothic 폰트 사전 설치되어 있어 별도 작업 불필요.
`main.tex` 첫 줄의 `% !TEX program = xelatex` 매직 코멘트는 일부 에디터 영역만
인식하므로 Overleaf 의 Settings 에서 직접 지정해야 합니다.

### 로컬 (Ubuntu/Debian)

```bash
sudo apt install texlive-xetex texlive-lang-korean texlive-bibtex-extra \
                 texlive-pictures texlive-fonts-extra chktex \
                 fonts-nanum
fc-cache -fv
```

### 빌드 명령

```bash
# 1. lint (스타일 + 기본 문법)
chktex main.tex -q

# 2. 정식 빌드 (PDF 생성)
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex

# 3. 문법 검사만 (PDF 생성 X — 본 환경 권장)
xelatex -draftmode -interaction=nonstopmode main.tex
bibtex main
xelatex -draftmode -interaction=nonstopmode main.tex
xelatex -draftmode -interaction=nonstopmode main.tex
grep -E "Warning|Error|undefined" main.log | head -50
```

### latexmk 통합 (선택)

```bash
latexmk -xelatex -bibtex main.tex     # PDF
latexmk -xelatex -bibtex -pretex='\AtBeginDocument{\let\paperpdfmode\nullfont}' main.tex
```

---

## 본 논문의 contribution

1. **METRONOME 알고리즘** — 5-stage 음악적 명명 (TEMPO → CHORD → METER → ACCENT → RITARDANDO). 24 단일 후보 최적화 기법이 모두 미달한 유의 임계점 (5%) 을 단일 통합 알고리즘으로 도달.
2. **Cross-Domain Superposition Theorem** — Theorem 1. LLM 추론 서빙에서 CPU 최적화 기법의 stacking 시 도메인 간섭 비용을 처음으로 정량적으로 정식화.
3. **Step-Boundary Alignment + Branchy Pattern (counter-intuitive)** — Theorem 2 + 분기 패턴의 prefetcher inhibit 메커니즘. high-rate + regular 보다 5.4× 효과.

---

## reference 검증 protocol (Phase 3)

3 개 독립 Agent 가 병렬로 30+ reference link alive 검증:
- Agent A: arXiv ID HEAD check
- Agent B: DBLP / OpenReview entry 검증
- Agent C: third-party cross-search (Google Scholar / Semantic Scholar)

3 agent 모두 alive 통과 + 제목/저자 일치 시에만 `references.bib` 등록.

---

## AI 사용 disclosure

본 논문 §10 Acknowledgments 에 IEEE/ACM 2024+ policy 준수 표현으로 명시.
