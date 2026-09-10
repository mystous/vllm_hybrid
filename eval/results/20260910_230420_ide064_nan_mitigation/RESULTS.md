# IDE_064 — NaN 사망의 완화 수단 검증 (`SGLANG_SANITIZE_NAN_LOGITS`)

`IDE_063` 에서 원인 특정에 실패했다 (KV dtype·지연 cold expert·누적 이력 모두 기각, 확률적 실패).
대신 SGLang 소스에 방어 수단이 있고 **기본값이 꺼져 있음**을 확인했다.

```
srt/environ.py:1071   SGLANG_SANITIZE_NAN_LOGITS = EnvBool(False)
srt/environ.py:456    SGLANG_ENABLE_ASYNC_ASSERT  = EnvBool(False)

utils/async_probe.py: sanitize_nan_logits(logits, msg)
  -> maybe_warn_nan(...); torch.nan_to_num_(logits, nan=-1e30, posinf=1e30, neginf=-1e30)
```

켜면 NaN·±Inf logit 을 ±1e30 으로 치환하고 경고를 남긴 뒤 샘플링을 계속한다.
꺼져 있으면 PyTorch `multinomial` 내부 검사에서 device-side assert 가 나고 스케줄러 전 랭크가 죽는다.

검증: EPOCH-OPS 구성 (fp8_e5m2 KV + mixed forward + 지연 expert τ=0.25, graph 224) 에
`SGLANG_SANITIZE_NAN_LOGITS=1` 만 추가하고 도착률 5.0 × 3,000요청 (약 10분) 을 4회 반복.
2026-09-10 23:06~2026-09-11 00:0x. 하네스 `eval/harness/20260910_e12/run_ide062.sh` 말미.

## 결과

| 회차 | 성공 | 출력 tok/s | TPOT 평균/p99 | 서버 | NaN 발생 |
|---|---|---|---|---|---|
| run1 | 2,999/3,000 | 621.24 | 214.84 / 236.19 | 생존 | 0 |
| run2 | 2,998/3,000 | 622.87 | 218.54 / 241.25 | 생존 | **1회 (4랭크 동시)** |
| run3 | 3,000/3,000 | 621.79 | 218.17 / 248.48 | 생존 | 0 |
| run4 | 3,000/3,000 | 621.68 | 222.48 / 256.04 | 생존 | 0 |

run2 의 서버 로그 (2026-09-10 23:18:12, TP0~TP3 전부):

```
NaN detected in sampler: next_token_logits; values were sanitized before sampling.
This usually indicates numerical overflow (e.g. fp16 activations) or an upstream bug producing NaN.
```

같은 시각에 `probability tensor contains ...` assert 는 **0건**이고 서버는 정상 완료했다.

품질: GSM8K 100문항 **95/100**. 완화 없이 측정한 96/100 과 한 문항 차이로 n=100 의 표본 오차 범위다.

## 판정 — 채택

| 항목 | 완화 끔 | 완화 켬 |
|---|---|---|
| 도착률 5.0 · mixed forward 켬 조건에서 사망 | **6회 중 2회 (33%)** | **4회 중 0회** |
| NaN 발생 시 결과 | 스케줄러 4랭크 종료, 진행 중 요청 전량 실패 (E12: 6,000건) | 해당 토큰만 치환, 요청 정상 완료 |
| 처리량 대가 | — | 없음 (621~623 tok/s, 변동 0.3%) |
| 품질 | GSM100 96/100 | GSM100 95/100 |

사망률 33% 하에서 4회 연속 생존이 우연일 확률은 0.67⁴ ≈ 20% 다. 단독으로는 결정적이지 않으나,
**run2 에서 NaN 이 실제로 발생했는데도 생존했다는 직접 관측**이 있으므로 기전 수준에서 확정된다.

따라서 EPOCH-OPS 구성을 권고할 때 `SGLANG_SANITIZE_NAN_LOGITS=1` 을 필수 설정으로 포함한다.

## 남는 문제

1. **NaN 의 근본 원인은 여전히 미상이다.** SGLang 자체 진단은 "수치 오버플로 또는 상류 버그" 를 지목한다.
   본 경로에서 의심 지점은 CPU expert 의 INT4 역양자화 결과와 MoE 합산이다.
   발생률이 3,000요청당 약 1회 (토큰 기준 약 40만 토큰당 1회) 로 낮아 추적이 어렵다.
2. **완화는 정확성을 손상시킨다** (해당 토큰이 임의 값이 된다). 다만 NaN 이 난 시점의 출력은
   이미 무의미하므로, 서버 전체를 잃는 것보다 엄격히 낫다.
3. 기본값이 꺼져 있다는 점은 SGLang 상류에 보고할 가치가 있다. `AGENTS.md` 의 중복 확인 절차가
   선행돼야 하므로 여기서는 기록만 남긴다.
