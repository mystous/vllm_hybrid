# SUB_077 — I005 vLLM upstream PR (duplicate check + draft)

> **parent**: TSK_020 / idea I005
> **status**: 활성 (2026-05-24 신설)
> **effort**: 1-2 시간 (duplicate check + PR description draft. 실제 submit 은 사용자 confirm 후)
> **idea**: [`../idea/IDE_013_vllm_upstream_pr.md`](../idea/IDE_013_vllm_upstream_pr.md)

## 1. scope

본 SUB 는 idea I005 의 PR 제출 전 준비 단계:
1. AGENTS.md 정책의 duplicate check 진행 (gh CLI 미설치이므로 WebFetch 로 vllm-project/vllm PR list 검색).
2. PR description draft 작성 (영어).
3. 본 fork 의 patch 를 isolated diff 로 추출 (clean PR 용).
4. 측정 표 첨부 — I001 (SUB_073) 정정 후 framing 으로.

**실제 PR submit 은 본 SUB 의 산출물 사용자 review 후 별도 진행** (autonomous mode 라도 외부 PR submit 는 사용자 confirm 필수).

## 2. 진행 절차

### 2.1 duplicate check

WebFetch 로:
- `https://github.com/vllm-project/vllm/pulls?q=is%3Apr+ngram+thread+cap`
- `https://github.com/vllm-project/vllm/pulls?q=is%3Apr+VLLM_NGRAM_NUM_THREADS_CAP`
- `https://github.com/vllm-project/vllm/pulls?q=is%3Apr+ngram+TP+parallelization`
- `https://github.com/vllm-project/vllm/pulls?q=is%3Apr+24986+follow-up`

duplicate 부재 확인 후 진행.

### 2.2 PR description draft

idea I005 §2.3 의 draft 그대로 + I001 정정된 framing 사용.

### 2.3 patch isolation

본 fork 의 `vllm/v1/spec_decode/ngram_proposer.py` 변경분 git diff 추출 → patch file 또는 PR body 의 코드 블록 첨부.

## 3. risk / 사용자 confirm 항목

- AI 기여 명시 (AGENTS.md 정책)
- benchmark reproducibility — 본 환경 (Llama-3.3-70B + H100×8 + sonnet 500p) 외부 reproduce 어려움. PR description 에 명시.
- 사용자 (mystous@gmail.com) 가 실제 submit 시점에 PR description / patch 최종 review 필수.

## 4. 산출물

- `measurements/sub077_pr_draft_<TS>/` (duplicate check 결과 + PR description draft + isolated patch file)
- idea I005 §2 갱신 (실제 search 결과 + draft 첨부)
- 사용자 confirm 대기 → submit
