# IDE_013 — SUB_047 patch 의 vLLM upstream PR 제출

> **parent backlog**: [`README.md`](README.md) (TSK_020 / SUB_072)
> **자식 SUB**: [`SUB_077`](../planning/SUB_077_upstream_pr_draft.md)
> **발견**: 2026-05-24, analysis doc §10 R28 (PR #24986) 의 TODO 해소 가능성
> **priority**: ★ (upstream 기여)
> **status**: ◐ **draft 완료, human submit 대기** (2026-05-24)

## 1. fact

vLLM upstream PR #24986 (ekagra-ranjan + Nick Hill, 2025-09-25) 의 origin code:

```python
# vllm/v1/spec_decode/ngram_proposer.py:36-48 (PR #24986 merged 상태):
# TODO(ekagra-ranjan): bump up the cap from 1 to 8
# when TP parallelization for ngram is implemented.
self.num_numba_thread_available = min(1, (cpu_count // 2))
self.num_numba_thread_available //= tp_size
self.num_tokens_threshold = 8192
```

본 fork 의 SUB_047 patch:

```python
cap = int(os.environ.get("VLLM_NGRAM_NUM_THREADS_CAP", "1"))
divide_by_tp = int(os.environ.get("VLLM_NGRAM_DIVIDE_BY_TP", "1"))
self.num_numba_thread_available = max(1, min(cap, (cpu_count // 2)))
if divide_by_tp:
    self.num_numba_thread_available //= tp_size
self.num_numba_thread_available = max(1, self.num_numba_thread_available)
```

→ env-tunable, default 는 upstream 동일 (cap=1, div_tp=1 → no behavior change). 사용자가 명시적으로 `VLLM_NGRAM_NUM_THREADS_CAP=8 + VLLM_NGRAM_DIVIDE_BY_TP=0` set 할 때만 threading enable.

본 patch 는 **upstream PR #24986 의 TODO 를 backward-compatible 하게 해소**. PR review (benchislett: "threading plateaus after 4-8") 와 정합.

## 2. PR 제출 plan

### 2.1 AGENTS.md (vllm-project/vllm 영역 contribution policy) 준수

1. **Duplicate check** (필수):
   ```bash
   gh issue view 24986 --repo vllm-project/vllm --comments
   gh pr list --repo vllm-project/vllm --state open --search "ngram thread cap"
   gh pr list --repo vllm-project/vllm --state open --search "VLLM_NGRAM_NUM_THREADS_CAP"
   gh pr list --repo vllm-project/vllm --state open --search "ngram TP parallelization"
   ```
   유사 PR 가 open 이면 본 PR 제출 보류, 기존 PR 에 comment 추가.

2. **Accountability**:
   - PR description 에 명시: "AI assistance was used (Claude)"
   - human submitter (mystous@gmail.com) 가 모든 변경 line review 후 제출.
   - 테스트 실행 + 결과 PR description 에 첨부.

3. **fail-closed**: duplicate 발견 시 PR 제출 안 함.

### 2.2 PR scope

- 변경 file: `vllm/v1/spec_decode/ngram_proposer.py` (단일 file, ~10 줄)
- 변경 안 함:
  - default 동작 (backward compat): cap=1, div_tp=1 → upstream 와 동일 결과
  - 다른 spec decoding method (eagle, medusa, suffix 등): 영향 없음
  - vLLM v0: 본 file 은 v1 only

### 2.3 PR description (draft)

```markdown
## Summary

Make the ngram thread cap and tp-divisor in `NgramProposer` env-tunable, resolving the TODO from #24986:

> `# TODO(ekagra-ranjan): bump up the cap from 1 to 8 when TP parallelization for ngram is implemented.`

Two new env vars:
- `VLLM_NGRAM_NUM_THREADS_CAP` (default 1 — upstream behavior unchanged)
- `VLLM_NGRAM_DIVIDE_BY_TP` (default 1 — upstream behavior unchanged)

When users set `VLLM_NGRAM_NUM_THREADS_CAP=8 + VLLM_NGRAM_DIVIDE_BY_TP=0`, the disabled batch-parallel threading from #24986 is enabled (8 thread/rank).

## Why this is not duplicating an existing PR

(grep result of `gh pr list ... | head`)

## Measurement (Llama-3.3-70B + H100×8 + TP=8, sonnet 500p × 8192in × 8192max)

| config | tps | wall (s) | vs default |
|---|---:|---:|---:|
| default (cap=1, div_tp=1) | 10,778.6 | 372.9 | — |
| cap=8, div_tp=0 (this PR enabled) | 10,956.5 (3-run avg) | 366.83 | +1.65% |

Aligned with #24986 review thread by @benchislett:
> "threading performance plateaus after 4 threads, no improvement beyond 8 threads"

## Test plan

- [ ] `pytest tests/v1/spec_decode/test_ngram_proposer.py` (default cap=1)
- [ ] `VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0 pytest tests/v1/spec_decode/test_ngram_proposer.py` (cap=8 enabled)
- [ ] Backward compat: default env 미설정 시 upstream 동일 동작 확인.

AI assistance was used (Claude Code).

Signed-off-by: Kyunam Cho <mystous@gmail.com>
```

### 2.4 effort

- duplicate check: 30 분
- PR draft + description: 1 시간
- 테스트 추가 / 실행: 2-3 시간
- PR submit + review 대응: 수 일 (review cycle)
- **총 effort (PR submit 까지): 0.5-1 일**

## 3. 진행 조건

1. **I001 (contribution framing 정정) 완료 후** — PR description 의 measurement 표가 정확한 framing 으로 작성되어야 over-claim 없음.
2. **AGENTS.md 의 duplicate check 통과** — open PR 가 같은 영역에 없을 것.
3. **human submitter (mystous@gmail.com) 의 사전 동의** — AI 기여 명시 정책 준수.

## 4. 확인 / 업데이트 필요 doc

| 파일 | 갱신 위치 |
|---|---|
| `Best_SpecDecode_10778tps.md` | §4.4 본 lever 의 출처 — "upstream PR #PR_NUMBER submitted" 추가 |
| `analysis/workload_acceptance_analysis_20260524.md` | §10.1 R28 + §11.2.1 SUB_047 patch 영역 — upstream PR 링크 추가 |
| `id_registry.md` | SUB_047 entry 의 "fork patch" → "upstream-submitted patch" 갱신 |
| 본 idea md | upstream PR URL + merge status |

## 5. risk

- vLLM maintainer 가 본 env-tunable approach 보다 PR #29184 의 NGram GPU 또는 다른 방향을 선호할 가능성 → PR rejection.
- backward compat 가 maintainer 의 default behavior 정책과 다를 가능성 (예: maintainer 가 "default 도 cap=8 로 올려야" 선호).
- benchmark reproduction 요구 — maintainer 가 본 측정의 reproducibility 검증 요구 시 측정 환경 (Llama-3.3-70B + H100×8 + 본 prompt set) 가 일반화 가능한지 의문 제기 가능.

## 6. 결과 (SUB_077, 2026-05-24)

본 idea 영역 측정 SUB 아님 (PR draft 작업). sonnet/chat/code workload 영역 N/A.

### 6.1 duplicate check 결과 (WebFetch on vllm-project/vllm)

| 검색 | 결과 |
|---|---|
| `is:pr ngram thread cap` | **0 open PR** (3 closed: #24986, #12388, #5649 — scope 무관) |
| `is:issue ngram threading OR VLLM_NGRAM` | **0 매칭** |

→ AGENTS.md fail-closed condition 미발현. **PR 제출 가능**.

### 6.2 PR draft 산출물

- `measurements/sub077_pr_draft_20260524/PR_DRAFT.md` (영어):
  - 제목: `[V1][Spec Decode] Make ngram numba thread cap and tp-divisor env-tunable to enable batch-parallel threading from #24986`
  - 본문: motivation (PR #24986 TODO 해소) + isolated patch + measurement 표 (IDE_009 정정된 contribution framing 사용) + test plan + backward compat 명시 + AI assistance disclosure
- isolated patch (~8 줄, `vllm/v1/spec_decode/ngram_proposer.py` 의 cap = env-tunable 부분만, SUB_065/066/067/057 lever 제외)

### 6.3 다음 step (human, mystous@gmail.com)

1. PR_DRAFT.md 모든 line review
2. vllm-project/vllm fork branch 생성 (`feat/ngram-thread-cap-env`)
3. isolated patch apply + AGENTS.md 의 uv venv / pre-commit 환경 setup
4. `pytest tests/v1/spec_decode/test_ngram_proposer.py` (default + cap=8 env)
5. `gh pr create` (gh CLI 별도 설치 필요)
6. PR URL 회수 → 본 idea §6.3 갱신 + id_registry SUB_077 status 갱신

### 6.4 본 idea 의 contribution

본 SUB 영역 측정 결과 아닌 doc 산출물. 단 IDE_009 (vanilla framing 정정) 후 본 PR description 의 contribution framing 정확화 가능 (over-claim 영역 제거됨, "+1.65% vs upstream default" 영역 honest representation).
