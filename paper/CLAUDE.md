# paper/ — CLAUDE.md

논문 편집 시 이 파일을 먼저 읽을 것.
단일 진실 공급원: `paper/main.tex` (구조), `paper/sections/` (본문), `paper/figures/`, `paper/tables/`.

---

## 빌드

```bash
# Overleaf: Menu → Settings → Compiler: XeLaTeX
# 로컬:
xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex
```

- **컴파일러: XeLaTeX 필수** (kotex + fontspec 사용). pdflatex 빌드 불가.
- 폰트: NanumMyeongjo (본문), NanumGothic (sans). 로컬 빌드 시 `fc-cache -fv` 후 확인.

---

## 컬럼 레이아웃 규칙 (절대 규칙)

| 환경 | 너비 지정 | 용도 |
|------|-----------|------|
| 1컬럼 figure | `\resizebox{\columnwidth}{!}{...}` | 단일 컬럼 fit |
| 2컬럼 figure | `\begin{figure*}` + `\resizebox{\textwidth}{!}{...}` | 전체 페이지 너비 |
| 1컬럼 table | `\begin{table}` (resizebox 불필요 시) | 단일 컬럼 fit |
| 2컬럼 table | `\begin{table*}` + `\resizebox{\textwidth}{!}{...}` | 전체 페이지 너비 |

### 금지 사항
- `\columnwidth`의 배수(예: `1.5\columnwidth`, `2\columnwidth`) 사용 금지
- `\textwidth`를 1-column 환경(`\begin{figure}`)에서 사용 금지 → 컬럼 경계 침범
- resizebox 없이 절대 크기(cm/pt)로 폭을 지정하는 경우, 반드시 `\columnwidth` 또는 `\textwidth`보다 작음을 수식으로 확인할 것
- `\resizebox{0.92\columnwidth}{!}` 같은 90% 이하 축소는 허용하나 1.0 초과는 금지

### TikZ 그림 폭 안전 체크
```latex
% 1컬럼 그림: 아래 두 가지 중 하나만 사용
\resizebox{\columnwidth}{!}{%      % 정확히 1컬럼 폭으로 스케일
\resizebox{0.95\columnwidth}{!}{% % 여백 확보 (권장)

% 2컬럼 그림: figure* 환경과 함께
\resizebox{\textwidth}{!}{%
```

---

## 그림 객체 간섭 최소화 규칙

그림 내 객체(노드, 화살표, 레이블, 박스) 사이의 **시각적 겹침(overlap)** 을 원천 차단한다.

### 1. 노드 간 최소 간격

```latex
% node distance는 최소 0.3cm 이상
node distance=0.35cm   % 수직 방향 최소값
node distance=0.5cm    % 권장 기본값
```

- 서로 다른 역할의 노드(stage 박스, 화살표 레이블, 주석 텍스트)는 반드시 다른 y좌표에 배치
- `anchor` 속성으로 위치를 명시할 것. 기본 center anchor는 겹침 위험이 있음

### 2. 레이블/주석 텍스트 겹침 방지

```latex
% ✓ 올바른 예: anchor + 오프셋으로 명시
\node[anchor=south, yshift=2pt] at (x, y) {text};

% ✗ 금지: 위치 미지정으로 다른 객체와 겹칠 수 있음
\node at (x, y) {text};  % center가 기존 요소 위에 얹힐 수 있음
```

- 화살표 위 레이블: `node[midway, above=2pt]` 또는 `node[midway, below=2pt]`로 명시
- 박스 레이블이 박스 크기를 초과하면 레이블을 줄이거나 `inner sep`을 늘릴 것

### 3. 화살표와 노드 경계 겹침

- 화살표 시작/끝은 노드 경계에서 자동으로 처리되지만, curved arrow는 중간 경로가 다른 노드를 통과할 수 있음
- 곡선 화살표(`.. controls ..`)를 사용할 때 컨트롤 포인트가 다른 노드와 겹치는지 수동 확인
- 가능하면 `to[bend left=xx]`, `to[bend right=xx]`로 대체 (겹침 가시성이 높음)

### 4. 시퀀스 다이어그램 / 타이밍 다이어그램

- 레인(actor)별로 y좌표를 충분히 분리: `\foreach` 루프로 y를 균등 배분
- 좌우 방향 화살표는 레인 경계를 넘지 않도록 x 범위를 `xmin`, `xmax`로 clamp
- 겹침 위험 구간에는 `fill=white, draw=none` 투명 박스를 레이블 뒤에 배치 (배경 지우기)

```latex
% 레이블 배경 지우기 패턴
\node[fill=white, inner sep=1pt, font=\tiny] at (x, y) {text};
```

### 5. `\resizebox`로 축소 시 텍스트 최소 크기

- 최종 출력에서 텍스트가 5pt 미만이 되지 않도록 소스에서 `\scriptsize` 또는 `\tiny` 최대 사용
- `\resizebox{0.9\columnwidth}{!}`로 10% 축소 시 소스의 `\small` = 출력 약 8pt (허용)
- 지나친 축소(`0.5\columnwidth` 이하)는 텍스트 가독성을 해치므로 그림을 재설계

---

## 그림/표 배치 규칙

```latex
% 그림: [t] (페이지 상단) 우선, [b] 허용, [h] 사용 금지
\begin{figure}[t]   % 1컬럼
\begin{figure*}[t]  % 2컬럼

% 표: 동일
\begin{table}[t]
\begin{table*}[t]
```

- `[h]` (here) 배치는 IEEE 2-column 레이아웃에서 컬럼 균형을 깨뜨리므로 사용 금지
- 2컬럼(`figure*`, `table*`)은 반드시 단락의 **시작** 근처에 배치하여 LaTeX 배치 알고리즘이 페이지 상단에 띄울 수 있도록

---

## 도메인 커맨드 / 표기

```latex
\algname          % → \textsc{Metronome}
\tgpu             % → T_{\mathrm{gpu}}
\tcpu             % → T_{\mathrm{cpu}}
\dom              % → \mathrm{dom}
```

- 직접 `T_{gpu}` 입력 금지. 커맨드 사용 강제.
- 알고리즘 참조: `Algorithm~\ref{alg:metronome-init}`, `Algorithm~\ref{alg:metronome-serve}`
- 정리 참조: `Theorem~\ref{thm:superposition}`, `Theorem~\ref{thm:alignment}`

---

## 파일 구조

```
paper/
├── main.tex               # 문서 루트 (XeLaTeX, input 목록)
├── sections/
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_background.tex
│   ├── 04_problem_formulation.tex
│   ├── 05_metronome_algorithm.tex
│   ├── 06_methodology.tex
│   ├── 07_results.tex
│   ├── 08_mechanism_analysis.tex
│   ├── 09_discussion.tex
│   └── 10_conclusion.tex
├── figures/
│   ├── fig1_metronome_pipeline.tex   (1컬럼, TikZ)
│   ├── fig2_step_alignment.tex       (1컬럼, TikZ 타이밍)
│   ├── fig3_pattern_cycle_grid.tex   (1컬럼, TikZ 히트맵)
│   ├── fig4_prefetcher_contention.tex(1컬럼, TikZ 시퀀스)
│   ├── fig5_period_drift.tex         (1컬럼, TikZ 시계열)
│   └── fig6_class_taxonomy.tex       (1컬럼, TikZ 덴드로그램)
├── tables/
│   ├── tbl_vs_baseline.tex           (2컬럼, table*)
│   ├── tbl_class_representatives.tex (2컬럼, table*)
│   ├── tbl_superposition.tex
│   ├── tbl_multirun.tex
│   └── tbl_metronome_hyperparam.tex
├── references.bib
└── CLAUDE.md                         # 이 파일
```

---

## 새 그림 추가 체크리스트

1. `figures/figN_name.tex` 파일 생성
2. `\begin{figure}[t]` 또는 `\begin{figure*}[t]` 선택
3. `\resizebox{0.95\columnwidth}{!}` (1컬럼) 또는 `\resizebox{\textwidth}{!}` (2컬럼) 적용
4. 모든 노드에 `anchor` 명시 또는 `node distance`로 명확한 상대 위치 지정
5. 화살표 레이블에 `above=2pt`/`below=2pt` 명시
6. `\caption{...}` 과 `\label{fig:name}` 추가
7. 해당 section 파일에 `\input{figures/figN_name}` 삽입

## 새 표 추가 체크리스트

1. `tables/tbl_name.tex` 파일 생성
2. 너비에 따라 `table` / `table*` 선택
3. 2컬럼이면 `\resizebox{\textwidth}{!}{...}` 적용
4. `\toprule`, `\midrule`, `\bottomrule` (booktabs) 사용, `\hline` 사용 금지
5. `\caption{...}` 과 `\label{tbl:name}` 추가
6. 해당 section 파일에 `\input{tables/tbl_name}` 삽입
