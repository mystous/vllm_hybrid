# IDE_068 test

## TST_024 (실험 B)

| # | 검사 | 통과 조건 |
|---|---|---|
| 1 | b0 기준선 성립 | HEALTH OK + 64/64 완료 + greedy 4문항 정상 |
| 2 | 하이브리드 각 셀 성립 | 동일 |
| 3 | 최소 장수 | 성립한 최소 TP 를 답으로 삼는다. 부팅만 되고 요청이 실패하면 성립 아님 |
| 4 | 처리량 보고 | 기준선 대비 비율과 절대값을 함께. 손실을 숨기지 않는다 |

## TST_025 (실험 A)

| # | 검사 | 통과 조건 |
|---|---|---|
| 1 | 적재 | 변환본 DRAM 적재 + HEALTH OK. `free -g` 로 사용량 기록 |
| 2 | 완료 | 전 요청 완료 |
| 3 | 품질 | greedy 4문항 정상. 미통과 시 "적재·서빙 성립, 품질 미통과" 로 기록하고 SUB_167 연계 |
| 4 | 용량 | DRAM/HBM 실측 사용량과 2 TB 대비 여유를 표로 |

## greedy 4문항 (선행 캠페인과 동일)

1. `The capital of France is` → Paris 포함, 문장 성립
2. `def fibonacci(n):` → 올바른 재귀/반복 구현
3. `1+2+3+...+100 =` → 5050 또는 n(n+1)/2
4. chat: "Write a Python function that reverses a string. Just the code." → `s[::-1]` 류
