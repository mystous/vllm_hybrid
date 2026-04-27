# 20260427_091915_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_cold_verify

- timestamp: 20260427_091915
- hw_tag:    Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8
- branch:    feat/ide006-cold-kv-cpu-partial-attention
- commit:    abe0564ade78b0293277a735fd98ccdd8b262c8f
- python:    /workspace/vllm_dev_prj/bin/python
- vllm:      0.1.dev15917+g0a6396b45
- NUM_PROMPTS: 50
- OUTPUT_LEN:  16
- MONITOR_INTERVAL: 1s

## components
- eval/run_e2e_accuracy.py --split-on-only — IDE_006 cold path 실 발화
- eval/monitor.py — CPU/GPU 사용률 시계열 캡처 (interval=1s)

## 통과 기준
- e2e RC = 0 (engine 안 죽음)
- e2e_artifacts/split_on.json 의 num_completed = 50
- e2e.stderr.log 에 cold path 발화 흔적 (IDE_006 dispatcher / AMX trace)
- monitor_cpu.csv 평균 CPU 사용률이 baseline 대비 의미있는 수준

## exit codes
- e2e:                       0

## 발화 신호
- IDE_006 dispatcher 라인:    40 회
- AMX trace / partial profile: 0
0 회
- bench 완료 라인:
      [split_on] batched generate complete: 50 prompts in 161.1s (avg 3.22s/prompt)
    [split_on] done in 841.3s

## 사용률 (시계열 평균)
- CPU 사용률 평균: 12.4 %
- GPU 사용률 평균: 9.4 %

## 다음 작업
- 발화 신호 = 0 이면 cold path 안 탔음 → NUM_PROMPTS 더 키워야 함
- 발화 신호 > 0 + e2e RC = 0 이면 검증 통과
- CPU 평균이 baseline 대비 거의 같으면 overlap 미작동 → §4.6 작업 필요
