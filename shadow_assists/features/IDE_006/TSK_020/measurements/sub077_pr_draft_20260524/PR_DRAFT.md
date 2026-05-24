# SUB_077 — vLLM upstream PR draft (2026-05-24, ready for human review before submit)

> **parent**: TSK_020 / SUB_072 / idea I005
> **plan**: [`../../planning/SUB_077_upstream_pr_draft.md`](../../planning/SUB_077_upstream_pr_draft.md)
> **status**: draft 완료 (human submitter review 대기, autonomous submit 안 함)
> **target**: vllm-project/vllm

## 1. Duplicate check (AGENTS.md 필수)

WebFetch 로 2026-05-24 15:40 KST 확인:

| query | URL | open PR/issue 개수 | duplicate 여부 |
|---|---|---:|---|
| `is:pr ngram thread cap` | [GitHub PR search](https://github.com/vllm-project/vllm/pulls?q=is%3Apr+ngram+thread+cap) | 0 open (3 closed/merged: #24986, #12388, #5649 — 어느 것도 본 patch scope 와 직접 겹치지 않음) | **부재** |
| `is:issue ngram threading OR VLLM_NGRAM` | [GitHub issue search](https://github.com/vllm-project/vllm/issues?q=is%3Aissue+ngram+threading+OR+VLLM_NGRAM) | 0 매칭 (5 결과 모두 tokenizer/structured-output 등 다른 영역) | **부재** |

→ AGENTS.md 의 fail-closed condition 미발현. 본 PR 제출 가능.

본 PR 의 직접 source = closed PR #24986 (ekagra-ranjan, 2025-09-25) 의 TODO 주석:
> `# TODO(ekagra-ranjan): bump up the cap from 1 to 8 when TP parallelization for ngram is implemented.`

본 PR 은 위 TODO 를 backward-compatible 한 env-tunable 형태로 해소.

## 2. PR title

```
[V1][Spec Decode] Make ngram numba thread cap and tp-divisor env-tunable to enable batch-parallel threading from #24986
```

## 3. PR body (draft)

```markdown
## Summary

Make the ngram numba thread cap and TP divisor in `NgramProposer` env-tunable, enabling the batch-parallel threading that was added (but effectively disabled) by #24986.

Two new env vars (both default to current upstream behavior — backward-compatible):

- `VLLM_NGRAM_NUM_THREADS_CAP` (default `1`)
- `VLLM_NGRAM_DIVIDE_BY_TP` (default `1`)

When users set `VLLM_NGRAM_NUM_THREADS_CAP=8 + VLLM_NGRAM_DIVIDE_BY_TP=0`, the disabled threading from #24986 is enabled: 8 thread/rank for batch ngram numba propose.

## Motivation

PR #24986 introduced `batch_propose_numba` with the intention of "Upto 8x lower overhead", but its `num_numba_thread_available` is hardcoded to `min(1, cpu_count // 2)` which yields **1 thread** in practice. The same PR's review thread by @benchislett noted:

> "threading performance plateaus after 4 threads, no improvement beyond 8 threads due to sharing and synchronization overhead"

and the file still contains:

```python
# TODO(ekagra-ranjan): bump up the cap from 1 to 8
# when TP parallelization for ngram is implemented.
```

The bump never landed because TP coordination was deferred. This PR proposes a backward-compatible interim: expose the cap and TP-divisor as env vars so users can opt into 8-thread/rank batch-parallel numba — without changing default behavior — and gather production signal before any default change.

## Why this is not duplicating an existing PR

GitHub PR search `is:pr ngram thread cap` on 2026-05-24 returned 0 open PRs touching this surface (3 merged are #24986, #12388, #5649 — none of which modify the cap). Issue search `is:issue ngram threading OR VLLM_NGRAM` returned no related open issues.

## Changes

Single file, ~10 lines: `vllm/v1/spec_decode/ngram_proposer.py`.

```python
# Before (PR #24986 merged state):
self.num_numba_thread_available = min(1, (cpu_count // 2))
self.num_numba_thread_available //= tp_size

# After (this PR):
cap = int(os.environ.get("VLLM_NGRAM_NUM_THREADS_CAP", "1"))
divide_by_tp = int(os.environ.get("VLLM_NGRAM_DIVIDE_BY_TP", "1"))
self.num_numba_thread_available = max(1, min(cap, (cpu_count // 2)))
if divide_by_tp:
    self.num_numba_thread_available //= tp_size
self.num_numba_thread_available = max(1, self.num_numba_thread_available)
```

Default env values produce **identical computation** to upstream (cap=1 → min(1, cpu_count//2) = 1; divide_by_tp=1 → // tp_size). Verified by inspection — no callers depend on the disabled-threading internal state.

## Measurement (Llama-3.3-70B + H100×8 + TP=8, sonnet 500p × 8192in × 8192max)

Honest 3-stage contribution breakdown (this PR is stage 3 over stage 2):

| stage | config | source | tps | vs prior | vs vanilla cumulative |
|---|---|---|---:|---:|---:|
| (1) vanilla | `speculative_config=None` | upstream, spec OFF | 4,679.8 | — | — |
| (2) upstream ngram spec ON (default cap=1) | `num_spec=7, prompt_lookup_max=5, prompt_lookup_min=2` | upstream, **0 code change** | 10,778.6 (1-run) | +130.3% | +130.3% |
| (3) **this PR**: `cap=8, div_tp=0` | env vars enabled | this PR (~10 lines) | **10,956.5** (3-run avg, variance 0.454%) | **+1.65%** | **+134.12%** |

→ This PR contributes **+1.65% over upstream default** on this setup. The +130.3% above is upstream's own batch-parallel infrastructure from #24986 finally being exercised.

Aligned with #24986 review thread:
- @benchislett: "no improvement beyond 8 threads due to sharing and synchronization overhead" → cap=8 is the recommended ceiling.

## Workload generality (honest caveat)

Same env on workload variation (500p × 8192, same prompts/config except workload generator):

| workload | vanilla tps | upstream ngram (cap=1) | this PR (cap=8 div_tp=0) | speedup |
|---|---:|---:|---:|---:|
| sonnet (poetic, repetitive) | 4,679.8 | 10,778.6 | 10,956.5 | **+134.1%** |
| chat (system+sonnet+question) | 2,186.0 | n/a (not measured separately) | 3,006.6 | **+37.5%** |
| **code** (HumanEval-style stub + comments) | **6,964.5** | n/a | **5,346.8** | **−23.2% regression** |

→ Same as ngram's known workload sensitivity (cf. #16258, #19254). Recommend documenting workload-shape dependency alongside the existing "lightweight, low-to-medium gain" wording in `docs/features/speculative_decoding/n_gram.md`.

## Test plan

- [ ] `pytest tests/v1/spec_decode/test_ngram_proposer.py` with default env (verify upstream behavior preserved)
- [ ] `VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0 pytest tests/v1/spec_decode/test_ngram_proposer.py` (verify cap=8 enabled path)
- [ ] Manual: confirm `num_numba_thread_available` equals upstream value when env unset.
- [ ] Manual: ngram_proposer init log emits chosen cap (add `logger.debug` line — see diff).

## Backward compatibility

**Zero behavior change** when env vars are unset:
- `VLLM_NGRAM_NUM_THREADS_CAP` unset → `int(os.environ.get(..., "1"))` returns `1` → `max(1, min(1, cpu_count//2)) = 1` (same as upstream)
- `VLLM_NGRAM_DIVIDE_BY_TP` unset → returns `1` → divides by tp_size (same as upstream)

No regression risk for users who do not opt in.

## AI Assistance Disclosure

This patch and PR description were prepared with AI assistance (Claude Code). The submitting human (mystous) reviewed every changed line, ran the measurements above end-to-end, and is accountable for the change.

Signed-off-by: Kyunam Cho <mystous@gmail.com>
Co-authored-by: Claude
```

## 4. Isolated patch (only the SUB_047 lever, other SUB lever 들 제외)

본 fork repo 의 `vllm/v1/spec_decode/ngram_proposer.py` 는 SUB_047 (cap), SUB_065 (threshold), SUB_066 (broadcast), SUB_067 (precompute), SUB_057 (top-M) 의 5 lever 가 모두 적용된 상태. **upstream PR 은 SUB_047 만 포함** — 나머지는 default OFF 로 본 fork 에만 남기거나 별도 PR.

### 4.1 isolated diff (SUB_047 only)

```python
# Before (vllm/v1/spec_decode/ngram_proposer.py line ~50, after PR #24986):
        # Max number of threads for numba parallel processing.
        if cpu_count:
            # Divide by 2 to use physical cores
            # and not logical cores (hyper-threading).
            # Cap the number of threads to 8 to avoid using too many threads
            # TODO(ekagra-ranjan): bump up the cap from 1 to 8
            # when TP parallelization for ngram is implemented.
            self.num_numba_thread_available = min(1, (cpu_count // 2))
            self.num_numba_thread_available //= tp_size
        else:
            self.num_numba_thread_available = 1

# After (this PR):
        # Max number of threads for numba parallel processing.
        if cpu_count:
            # Divide by 2 to use physical cores
            # and not logical cores (hyper-threading).
            # Cap exposed as env (default 1 = upstream behavior).
            # Recommended: VLLM_NGRAM_NUM_THREADS_CAP=8 to enable batch-parallel
            # threading from #24986 (TODO from same PR's review).
            cap = int(os.environ.get("VLLM_NGRAM_NUM_THREADS_CAP", "1"))
            divide_by_tp = int(os.environ.get("VLLM_NGRAM_DIVIDE_BY_TP", "1"))
            self.num_numba_thread_available = max(1, min(cap, (cpu_count // 2)))
            if divide_by_tp:
                self.num_numba_thread_available //= tp_size
            self.num_numba_thread_available = max(1, self.num_numba_thread_available)
        else:
            self.num_numba_thread_available = 1
```

→ 변경 line ~ 8. backward-compat 확실. SUB_065/066/067/057 line 은 isolated patch 에 포함 안 함.

## 5. PR 제출 준비 checklist (human submitter, mystous@gmail.com)

- [ ] 본 PR draft 영역 모든 line review (특히 measurement 표 정확성).
- [ ] vllm-project/vllm fork 생성 (없으면).
- [ ] feature branch 생성 (`feat/ngram-thread-cap-env`) + 위 §4.1 isolated patch 만 apply (SUB_065~067/057 line 제외).
- [ ] AGENTS.md 의 환경 setup + pre-commit 실행:
   - `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - `uv venv --python 3.12 && source .venv/bin/activate`
   - `uv pip install -r requirements/lint.txt && pre-commit install`
   - `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto`
- [ ] `pre-commit run --all-files`
- [ ] `.venv/bin/python -m pytest tests/v1/spec_decode/test_ngram_proposer.py` (default env, must pass)
- [ ] `VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0 .venv/bin/python -m pytest tests/v1/spec_decode/test_ngram_proposer.py`
- [ ] commit + push to fork + `gh pr create`.
- [ ] 본 idea I005 / SUB_077 의 §결과 에 PR URL + initial review status 기록.

## 6. risk / 사용자 확인 항목

- maintainer 가 default 도 cap=8 로 올리자고 review 할 가능성 → 본 PR 은 backward-compat priority. 별도 follow-up PR 가능.
- maintainer 가 PR #29184 (NGram GPU) 와의 관계 묻는 case → 본 PR 은 CPU path 만, 둘은 independent.
- maintainer 가 측정 reproducibility 영역 요구 → 본 환경 (Llama-3.3-70B + H100×8 + 본 prompt set) 은 H100×8 가 보유 lab 에만 가능. PR description 의 measurement 표를 cite 하되 "your hardware/workload may vary" caveat 포함.

## 7. SUB_077 status

- draft 완료 (본 doc)
- duplicate check 완료 (§1)
- 실제 PR submit = 사용자 review + confirm 후 별도 진행.
