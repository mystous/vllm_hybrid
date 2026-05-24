# SUB_072 — idea backlog manager (TSK_020 의 follow-up 관리)

> **parent**: TSK_020
> **status**: 활성 (2026-05-24 신설)
> **effort**: ongoing — idea 생성/진행에 따라 incremental
> **위치**: [`features/IDE_006/TSK_020/idea/`](../idea/)

## 1. 역할

TSK_020 진행 중 발견된 follow-up idea / framing 정정 / 측정 candidate / upstream 기여 candidate 를 backlog 으로 모은다. 각 idea 는 "어떤 doc 을 확인하고, 어떻게 업데이트해야 하는지" 의 procedure 와 영향 doc list 를 self-contained 명세.

idea 가 실제 진행 단계에 들어가면 (a) 단순 doc 정정 → 본 SUB 안에서 처리, (b) 측정/코드 변경 필요 → 새 SUB 신설 (SUB_073~) 후 그 SUB 가 owner, 본 SUB 는 backlog tracker.

## 2. 현 backlog (6 idea)

[`../idea/README.md`](../idea/README.md) 의 표 그대로:

| ID | 제목 | priority | status | 관련 SUB (진행 시) |
|---|---|---|---|---|
| I001 | vanilla 대비 +134% framing 의 fork-patch / vLLM built-in 분리 | ★★★ | 대기 | (doc only) |
| I002 | SuffixDecoding 측정 — code workload 회귀 mitigation candidate | ★★ | 대기 | SUB_073 candidate |
| I003 | acceptance rate 직접 측정 → R/K 분리 | ★★ | 대기 | SUB_074 candidate |
| I004 | workload-aware predictive gating PoC | ★ | 대기 | SUB_075 candidate |
| I005 | SUB_047 patch 의 vLLM upstream PR 제출 | ★ | 대기 | (외부 PR) |
| I006 | vLLM Issue #16258 reproduction | ◐ | 대기 | SUB_076 candidate |

## 3. idea 관리 procedure

### 3.1 새 idea 등록

1. `idea/` 디렉토리에 `I###_<slug>.md` 신규 생성 (다음 빈 번호).
2. idea md 작성 — template:
   - §1 fact (발견 turn / 출처 / 외부 reference)
   - §2 가설 / 측정 계획
   - §3 effort 추정
   - §4 진행 시 신설 SUB (candidate 번호)
   - §5 확인 / 업데이트 필요 doc (영향 매트릭스)
   - §6 risk / caveat
   - §7 결과 (after 진행)
3. [`idea/README.md`](../idea/README.md) 의 표에 한 줄 추가 (priority + status).
4. 본 SUB plan 의 §2 표 도 동기화.

### 3.2 idea 진행

- **단순 doc 정정 (I001 같은 case)**: idea md 의 §3 procedure 그대로 따라 진행, doc 변경 후 status → `완료`. SUB 신설 없음.
- **측정 / 코드 변경 필요 (I002~I006 같은 case)**:
  1. 새 SUB_### 신설 (id_registry 갱신, 다음 번호 +1).
  2. `planning/SUB_###_<topic>.md` 신규 plan.
  3. idea md 의 §4 "진행 시 신설 SUB" 에 실제 SUB 번호 + 링크 추가.
  4. SUB 진행 → 결과 → idea md 의 §7 결과 작성.
  5. idea status → `완료`.

### 3.3 idea 폐기

- 외부 변화 (예: vLLM 영역 upstream 영역 이미 fix) 로 obsolete 면 idea md status → `기각`.
- README 표 동기화.

## 4. 우선순위 권장

[`idea/README.md`](../idea/README.md) §진행 우선순위 권장 참조. 요약:

1. **I001** (즉시) — 후속 작업의 framing 정확성 prerequisite.
2. **I002 SuffixDecoding** (~2 시간) — code 회귀 mitigation 의 직접 검증.
3. **I003 acceptance 직접 측정** (~30 분) — R/K framework 정확화.
4. **I004 workload-aware gating PoC** — I002/I003 결과 종합 후.
5. **I005 upstream PR** — I001 후.
6. **I006 외부 reproduction** — 짬내서.

## 5. doc 영향 매트릭스 (inverse index)

어떤 doc 이 어떤 idea 와 연관되는지 — idea 진행 시 *어디를 확인해야 하는지* 의 inverse view:

| doc | I001 | I002 | I003 | I004 | I005 | I006 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| `analysis/workload_acceptance_analysis_20260524.md` | §1·§11.4 | §10.4·§11 | §3·§4 | §6·§10.3 | §10 R28·§11.2.1 | §11.3 |
| `Best_SpecDecode_10778tps.md` | §1·§3·§5·§6 | §7 | — | §7 | §4.4 | — |
| `INDEX.md` | §0·§1 | §1·§4 | §0 | §1·§4 | — | — |
| `measurements/sub044_*/RESULTS.md` | §3 vs vanilla | — | — | — | — | — |
| `measurements/sub047_*/RESULTS.md` | §3 vs vanilla | — | — | — | — | — |
| `measurements/sub071_*/RESULTS.md` | §3 비교 표 | (참조) | — | — | — | — |
| `id_registry.md` | SUB_044·SUB_047·TSK_020 | SUB_073 entry | SUB_074 entry | SUB_075 entry | SUB_047 갱신 | SUB_076 entry |

→ 본 매트릭스는 idea 진행 시 *어디를 update 해야 하는지* 의 single-source-of-truth. idea md 의 §5 와 본 표가 항상 동기화되어야 함.

## 6. 본 SUB 의 생명주기

- 활성 — TSK_020 가 활성인 동안 본 SUB 도 활성. 새 idea 발견 시 본 backlog 에 등록.
- 종료 조건: TSK_020 종결 시 (SUB_071 + 모든 idea 진행/폐기 후) 본 SUB 도 종결.

## 7. raw 자료

| 항목 | 위치 |
|---|---|
| idea backlog README | [`../idea/README.md`](../idea/README.md) |
| 각 idea md | `../idea/I001_*.md` ~ `../idea/I006_*.md` |
| 본 SUB plan | (본 doc) |
