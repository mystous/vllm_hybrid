# cpu_heavy_C4 — fp8 + CPU sampling stack + max-num-seqs 변형 (FAILED)

## Attempted

- C4a: fp8 + `VLLM_CPU_SAMPLING=1`
- C4b: fp8 + `--max-num-seqs 512`
- C4c: fp8 + `--max-num-seqs 512` + `VLLM_CPU_SAMPLING=1`

## Result

| case | result |
|---|---|
| C4a | **Killed at s1** (547.8 tps, 6/500 succ). engine OOM / segfault after ~5s. |
| C4b | **wait_ready fatal** (boot failed). |
| C4c | boot 시도 중 cancel (선행 case 실패 + 시간 cost). |

근거: L11 MEASUREMENTS.md 의 기록 "CPU sampling mode 에서는 async_scheduling=True 와 우리
patch 의 D2H/H2D 가 stream 간에 race". B200 + Llama-8B + fp8 환경에서도 동일 race condition
재발. fp8 와 cpu_sampling 의 결합은 stream 동기화 충돌이 발생.

또한 `--max-num-seqs 512` (default 256 → 2배) 와 fp8 결합은 KV cache 메모리 추정에 변화를
주어 vllm engine 초기화에 실패 (KV slot allocation 단계).

## Verdict

- C4a/b/c **stack failed** by engine instability, 측정 불가.
- fp8 + cpu_sampling 결합은 production-safe 하지 않은 patch 상태. 별도 stream isolation
  patch 가 필요.

## 후속 작업 권장

- 본 round 에서는 stack 시도 종료. 추가 lever 가 single config 에서 양수 winning 을
  보이지 않으므로 stacking 으로 +10% 달성 가능성이 낮음.
- 만약 stack 을 재시도하려면 (1) fp8 + max-num-seqs 256 (default) 결합, (2) CPU sampling
  의 stream isolation patch 가 선행되어야 함.
