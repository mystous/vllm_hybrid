# SUB_213 — test.md

## U. CPU 단위 테스트 (GPU 불필요)

### U1. pad 로직 — 길이 보장

`tests/unit/test_pad_uniform.py` (본 디렉토리):

- `VLLM_SUFFIX_PAD_UNIFORM=1` 세팅 후 `SuffixDecodingProposer.propose()` 호출
  (SuffixDecodingCache 는 짧은 합성 시퀀스로 구성).
- **기대**: 모든 비-경계 요청의 draft 길이 == `num_speculative_tokens` 정확히.
- 경계 케이스: `num_tokens + K + 1 > max_model_len` 인 요청은
  `max_model_len - num_tokens - 1` 로 truncate.

### U2. OFF = upstream no-op

- env/flag 둘 다 부재 시 `propose()` 출력이 lever 추가 이전과 bit-동일.
- flag file fallback: env 없이 `/tmp/vllm_l3_VLLM_SUFFIX_PAD_UNIFORM.flag` 만으로 ON.

### U3. 정확도 등가 논거 (코드 리뷰 수준)

- pad 토큰은 suffix tree 의 match 가 아니므로 verify 에서 target 토큰과 불일치
  → rejection sampler 가 **항상 기각** → 최종 출력 분포 불변 (greedy/sampling 모두).
- 단, bonus token 경로와 무관함을 확인: 기각 위치 이후 토큰은 모두 폐기됨.

## E. 검증 실험 (GPU 필요 — H100×8 가용 시)

| 셀 | 스크립트 | 사전 예측 | 판정 기준 |
|---|---|---:|---|
| E1 van+PW mix | `verify_fap.sh` | ~8,850 | ±10% 내 → H-FaP 확정, host DSA 효과=0 |
| E2 suf K32+PW mix | `verify_fap.sh` | ~27,851 | ±10% 내 → suffix "−12%" 도 FaP 부작용 재귀속 |

- E1 이 12,089 근처로 나오면 H-FaP 기각 → host DSA 가설 재부활 (추가 격리 필요).

## P. Lever 측정 (GPU 필요)

| 셀 | 사전 예측 | GO / kill |
|---|---:|---|
| P1 K8+pad+FaP | 30.6k~34.8k (+10~25% vs 27,851) | P1 > 27,851 → GO |
| P2 K15+pad+FaP | P1 대비 ± (verify 비용 vs α tail) | 정보용 |
| P3 K8+nopad+FaP | ≈ P4 (FULL 미적중 → FaP 무관) | P3 ≫ P4 면 모델 오류 신호 |
| P4 K8+nopad+PW | K 축소 단독 기준점 | — |

- **uniform 적중 확인 방법**: boot log 의 cudagraph dispatch 통계 또는
  P1 vs P3 의 유의미한 차이 (FULL 적중이 P1 에만 존재).
- **정확도 게이트**: P1 출력 vs P4 출력의 corpus-level 분포 유사성
  (CLAUDE.md Constraint 운영 해석 — token-level 일치는 informational).

## 실행 절차 (GPU 가용 시)

```bash
nohup bash -c 'bash shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_212_optimal_dsa_6point/verify_fap.sh; \
  bash shadow_assists/features/IDE_023_cpu_slack_harvest/SUB_213_fap_suffix_uniform/sub213_sweep.sh' \
  > /tmp/sub213_chain.out 2>&1 &
```

예상 wall time: E1/E2 ~25분 + P1~P4 ~50분 ≈ **1h15m** (Llama-8B mix 1셀 ~12분 기준).
