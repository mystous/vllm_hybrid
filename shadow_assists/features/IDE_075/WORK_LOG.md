# WORK_LOG — IDE_075

## 2026-09-17
- 15:27 하네스 v2 의 `import harness as H74` 가 자기 자신(eval/ide075/harness.py)을 가리켜 부팅 뒤 runtime_proofs 호출에서 AttributeError → OFF_OPEN 부팅(OFF_OPEN_152426, 108 s, smoke·warmup 완료) 이 세션 없이 소진됨 (원장에 boot_end 기록, 세션 0). 체인의 CORE 부팅은 진행 중이었으므로 하네스만 중단하고 서버는 살려 ATTACH 모드로 재사용 (새 부팅 아님, boot_end 는 attached 로 기록). 이후 OFF 두 부팅(OFF_A, OFF_B)은 CORE 뒤에 실행 → 지시서의 OFF_OPEN(앞)/OFF_CLOSE(뒤) 시간 배치와 다름을 한계로 기록. 예산: 부팅 소진 예상 4/5 (낭비 1 포함), 세션 6/7.
- 15:29 두 번째 하네스 결함: main() 이 ATTACH 분기보다 먼저 PLANS[plan] 을 조회해 KeyError → 체인이 다음 plan(OFF_A) 으로 넘어가며 stop_server 로 로딩 중이던 CORE 서버를 종료. CORE_152703 부팅 시도는 세션 0 으로 소진 (원장 boot_end=ABORTED_HARNESS_BUG). 이후 예산 계수를 boot_start(시도) 기준으로 변경. 남은 부팅: OFF_A(진행 중), OFF_B, CORE → 잔여 0, 예비 S7 없음. 실행 순서는 OFF_A → OFF_B → CORE 가 되어 지시서의 OFF 앞/뒤 배치와 다름 (시간순 한계 기록).
- 15:45~16:05 분석: S3/S4 두 CORR 세션에서 enqueue→start 의 99.9 % 가 직전 층 deferred 실행 점유(pred_deferred p50 606/592 µs), 미분리 0.75 µs → FIFO/dispatch 후보를 기각. 서비스 span 이 unique cold expert 수와 r=0.925 로 비례, 단계는 up_gate 55 %/down 34 %, 전 expert AVX vec_mul (n_amx 0). tail 10/15.6k 은 한 서브풀의 down 단계(5 ms) → NUMA 원인 미분리. lifecycle: fwd 미기록 5/2/1 건 (SPSC 대조 실패 추정, 건수 공개, ≤0.1 % 기준).
- RESOURCE 세션 처리량 781.6 (OFF 평균 대비 1.11) 은 계측 ON 이 더 빠른 단발값 → DIAGNOSTIC_ONLY, 성능 향상으로 해석하지 않음.
- READINESS: 후보 A(Cold expert 서비스 시간)·B(관측 비용 기반 Hot 예산 재배치) READY_WITH_LIMITED_SCOPE. FIFO·복사·Hot 커널은 근거 없음/선택 안 함. NUMA tail 은 CONDITIONAL. GLM 은 별도 트랙.
