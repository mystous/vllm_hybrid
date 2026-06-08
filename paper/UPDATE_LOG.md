# paper/ UPDATE_LOG — 100+ lever 측정 결과 통합 (2026-06-08)

## 작업 범위

`shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/poc/`
하 6 회 cumulative sweep ($\ge75$ host-path lever + 6 워크로드 $\times$ 4 메서드 sweep
+ CPU AMX BF16 draft) 의 정직한 측정 결과를 paper 에 통합.

## 변경 sections (in-place edit)

| 파일 | 변경 내용 |
|---|---|
| `sections/00_abstract.tex` | `\todo{suffix 대비 +XX% ...}` 제거 → host reclamation 75+ lever 0/$\ge75$ net-positive 정직 보고 추가. hw gap 4500$\times$/9.2$\times$/213$\times$ 명시. |
| `sections/01_introduction.tex` | contribution (iv) `\todo{회수 처리량 이득 ...}` 제거 → host reclamation 부정 결론 + fp8 KV 직교 lever + 적용 영역 재정의로 갱신. |
| `sections/06_ceres_algorithm.tex` | 4 개 `\todo{}` (게이트, $T_{\mathrm{host}}$ 분해, 식~\ref{eq:hide-condition} 좌변 등) 모두 측정 결과로 대체. $T_{\mathrm{host}}\!\le\!\epsilon$ 부정 결론 명시. |
| `sections/08_results.tex` | `\todo{}` 3 개 모두 채움. 신규 4 § subsec 추가: `res-eagle3` (6/6 negative), `res-fp8` (production candidate), `res-amx-draft` (hw gap 9.2$\times$), `res-100lever` (누적 verdict). |
| `sections/10_discussion.tex` | "host reclamation 미측정" → "75+ lever 측정 모두 net-negative" 갱신. 신규 §subsec `hw-gap-analysis` (4500$\times$ compute / 9.2$\times$ latency / 213$\times$ BW) 추가. |
| `sections/11_conclusion.tex` | `\todo{최종 성능 이득 ...}` 제거 → measured-negative 정직 결론 + 적용 영역 재정의. |

## 신규 tables (4)

| 파일 | 내용 | 출처 |
|---|---|---|
| `tables/tbl_six_workload.tex` (신규) | 6 workload $\times$ 4 method, Llama-3.1-8B B200 TP=8, suffix 5/6 +4.92$\sim$+27.19%, eagle3 6/6 -26$\sim$-32%, fp8 KV 5/6 +6.14$\sim$+7.32%. 5-sweep mean$\pm$std (CV $\le$2.43%). | `poc/six_workload_sweep/runs/precision/` |
| `tables/tbl_lever_summary.tex` (신규) | 100+ lever 카테고리별 요약 (env-flag/NUMA/AVX-512/scheduler/cudagraph/async/CPU sampling/n-gram/Eagle3/suffix/AMX/KV tiering/fp8 KV) $\times$ count $\times$ $\Delta$\% range $\times$ verdict. 0/$\ge75$ host-path net-positive. | `poc/{ide023_levers, hw_custom_round_{1,2,3}, cpu_heavy_*, cpu_continuous, amx_cpu_draft, eagle3_suffix_final}` |
| `tables/tbl_cpu_amx_draft.tex` (신규) | Llama-3.2-1B CPU AMX BF16 draft + Llama-3.1-8B GPU verify, 5 워크로드 W1-W5 모두 -98 ~ -99.8%. accept rate $\alpha\!=\!0.65$ (K=5) 인데도 hw gap 9.2$\times$ 로 collapse. | `poc/amx_cpu_draft/MEASUREMENTS.md` |
| `tables/tbl_reclamation.tex` (갱신) | 기존 placeholder (TBD) 표 → 75+ lever 의 measured-negative 표로 완전 재작성. 대표 14 lever 발췌 + fp8 KV 비교군. | `poc/{hw_custom_round_*, cpu_heavy_*, cpu_continuous, ide023_levers, amx_cpu_draft}` |

## 정량 데이터 (5-sweep precision, seeds 42-46)

Llama-3.1-8B-Instruct, TP=8, B200$\times$8, 500p $\times$ conc=64 $\times$ max\_tok=2048:

| workload | vanilla mean$\pm$std | suffix $\Delta$% | fp8_kv $\Delta$% |
|---|---:|---:|---:|
| sonnet | 21,012 ± 88 (CV 0.42%) | **+18.77%** | **+6.28%** |
| code | 20,576 ± 21 (CV 0.10%) | **+27.19%** | **+7.13%** |
| balanced | 19,018 ± 89 (CV 0.47%) | **+13.05%** | **+6.14%** |
| sonnet-heavy | 19,680 ± 105 (CV 0.53%) | **+8.54%** | **+6.71%** |
| code-heavy | 19,431 ± 64 (CV 0.33%) | **+4.92%** | **+7.32%** |
| chat (1-sweep) | 16,406 | -24.57% | -4.39% |

suffix mean CV $\le$ 2.43%, fp8_kv $\le$ 0.84%, vanilla $\le$ 0.53% — 통계적 유의.

## 검증 결과

| 항목 | 결과 |
|---|---|
| `\todo{}` 잔류 (sections + tables) | **0** (이전 8 개 → 0) |
| 신규 label / ref 매칭 | 8/8 (tbl:six-workload, tbl:lever-summary, tbl:cpu-amx-draft, subsec:res-eagle3, subsec:res-fp8, subsec:res-amx-draft, subsec:res-100lever, subsec:hw-gap-analysis) |
| 신규 `\input{tables/...}` 4 개 in 08_results.tex | 4/4 (line 116, 161, 202, 224) |
| brace balance (10 변경 파일) | 모두 diff=0 |
| 2-column 컴플라이언스 | `tbl_six_workload` = `table*` + `\resizebox{\textwidth}{!}` (와이드), 나머지 3 표 = `table` + `\resizebox{\columnwidth}{!}` (1-column) |
| booktabs (`\toprule`/`\midrule`/`\bottomrule`) | 4/4 |
| 도메인 커맨드 (`\algname`, `\tgpu`, `\tcpu`) 보존 | 유지 |
| xelatex 빌드 | **컨테이너 미설치** — 정적 sanity check 만 통과 |

## 정직성 게이트

- 모든 수치는 raw measurement (`poc/*/MEASUREMENTS.md`, `precision/*.json`) 에 직접 기인.
- 양수 lever 가 1 개 (fp8 KV) 임을 abstract / introduction / results / discussion / conclusion 5 곳에서 동일 정직 보고.
- "host reclamation 75+ lever 모두 net-negative" 와 "구조적 원인 = hw gap 4500$\times$/9.2$\times$/213$\times$" 가 paper 전체에 일관 명시.
- 적용 영역 재정의 (host-bound regime / hw-gap-smaller arch / long-context multi-tier KV) 가 discussion + conclusion 양쪽에서 명시.

## 보존된 paper 무결성

- 기존 sections (02_background, 03_related_work, 04_motivation, 05_problem_formulation, 07_methodology, 09_mechanism_analysis) 미변경.
- 기존 tables (tbl_b200_alpha, tbl_b200_xl, tbl_b200_resource, tbl_b200_oracle, tbl_b200_latency, tbl_gate, tbl_roadmap, tbl_multirun, tbl_class_representatives, ...) 미변경.
- main.tex 의 input 순서 유지 (신규 section 없음, 기존 subsec 만 확장).
- IEEEtran 2-column 레이아웃 / kotex 한국어 본문 / 도메인 커맨드 모두 보존.

## 후속 작업 (paper 외)

- xelatex 빌드 검증 (호스트 또는 Overleaf 에서). 명령:
  `xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex`
- Overleaf sync 는 사용자 동의 후 진행 (`paper/CLAUDE.md` 의 절차 따름).
- git commit / push 보류 — 사용자 허락 후.
