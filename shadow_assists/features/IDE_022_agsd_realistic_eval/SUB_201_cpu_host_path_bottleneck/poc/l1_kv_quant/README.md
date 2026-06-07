# SUB_201 후속 lever L1 — CPU AMX KV quantization (vLLM `--kv-cache-dtype`)

## 목적

SUB_201 b1/b3 finding 에서 도출된 후속 lever 후보 중 **L1 (KV cache dtype 양자화)** 단독 효과 검증.

- **Baseline**: `--kv-cache-dtype auto` (= 모델 dtype, BF16/FP16)
- **Lever**: `--kv-cache-dtype fp8` (B200 native = e4m3 FP8)
- **가설**: KV cache 메모리 1/2 → effective concurrency capacity ↑ → memory-bound 회복

vLLM 의 KV dtype lever 만 단독 측정. CPU AMX 자체로 KV cache 를 양자화하는 경로는 본 lever 가 미구현 (vLLM 의 KV 양자화는 GPU side 이고, "CPU AMX KV 양자화" 는 future work). 본 POC 는 **GPU side fp8 KV** 의 net 효과만 측정하여, 같은 효과를 CPU AMX 로 확장했을 때 의미가 있는지 정량 평가.

## 측정 plan

| 모델 | TP | GPU | KV baseline | KV lever | max-model-len |
|---|---:|---|---|---|---:|
| Qwen2.5-7B-Instruct | 2 | 0,1 | auto (bf16) | fp8 | 8192 |
| Llama-3.1-70B-Instruct | 4 | 0-3 | auto (bf16) | fp8 | 8192 |
| DeepSeek-R1 671B | 8 | 0-7 | auto | fp8 | 8192 |

- corpus: `sharegpt 100p × conc=16 × max-tokens=512` (capacity-focused short-output)
- runner: `vllm_config_perf/gating/realistic_eval/throughput_runner.py` (b3 sweep 과 동일)
- 동일 corpus 파일 재사용: `../b3_8gpu_full/sharegpt200.parquet` (`--limit 100`)

## 산출물

- `sweep.sh` — phase 별 (`PHASE=m1|m2|m3|all`) 실행
- `summarize.py` — runs/*.json + gpu csv 표 집계
- `runs/M{1,2,3}_*` — bench json/raw/log + gpu csv
- `MEASUREMENTS.md` — 최종 보고

## 실행

```bash
PHASE=m1 bash sweep.sh   # Qwen-7B
PHASE=m2 bash sweep.sh   # Llama-70B
PHASE=m3 bash sweep.sh   # R1-671B (boot 5-7min)
/workspace/vllm_dev_prj/bin/python summarize.py
```

LD_LIBRARY_PATH 는 sweep.sh 내부에서 export. 모든 vllm 명령에 `torch/lib + nvidia/nccl/lib` prefix 적용.
