# IDE_025 — DRAM KV/Prefix-Cache Tier (연산이 아닌 저장)

> parent: `PLN_003` / `TSK_045` / `TST_023` / 진행 로그: [`../IDE_023/PROGRESS_20260827.md`](../IDE_023/PROGRESS_20260827.md)

## 1. 배경 — IDE_006 과의 결정적 차이

IDE_006 (기각 계열) 은 CPU 가 cold KV 위에서 **attention 을 계산**하려 했고 Q-dependency dilemma 로 구조 기각되었다. 본 IDE 는 CPU 가 **계산하지 않는다** — 2TB DRAM 은 GPU pool 에서 밀려난 KV/prefix 의 **저장 tier** 이며, 이득의 원천은 prefix hit 시 **prefill 재계산 회피** (GPU 연산 절약) 다. 이는 legacy 운영 규칙의 "resource-separable work + reduced GPU reload/transfer" 조건에 정확히 부합하고, 업계 표준 (vLLM OffloadingConnector, LMCache, llm-d) 으로 net-win 이 검증된 구조다.

- 용량: 70B 기준 KV ≈ 320KB/token → 2TB ≈ **~6M tokens** (GPU pool 의 ~4.6배)
- 본 fork 는 upstream 의 `vllm/v1/kv_offload/` + `offloading_connector.py` 를 이미 inherit — **운영 결정 + 측정**이 작업의 본체

## 2. 측정 설계 (TSK_045)

압박이 없으면 이득도 없다 (IDE_006 1차 기각의 교훈: 128/128 에서 cold KV 미발생). 따라서:

1. **공유 prefix workload**: 8K prefix × 여러 그룹 × 총 N req (bench serve `--random-prefix-len` 계열)
2. **GPU pool 압박**: `--gpu-memory-utilization` 축소 또는 `--num-gpu-blocks-override` 로 prefix 가 GPU pool 에서 evict 되는 구성을 명시적으로 만든다 (정직한 emulation — 실제 long-context 대체)
3. 셀: (a) vanilla (prefix cache off) / (b) GPU prefix cache only (evict 후 재계산) / (c) + CPU offloading connector (DRAM 에서 reload)
4. 비압박 대조군: 동일 config 에서 압박 제거 시 (c) 의 회귀 ≤1% 확인
