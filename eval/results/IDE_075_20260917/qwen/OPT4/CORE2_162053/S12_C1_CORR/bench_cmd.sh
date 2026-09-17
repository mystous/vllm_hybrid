#!/usr/bin/env bash
# container vllm-h100
python3 /tmp/probe_client.py /home/mystous/.cache/huggingface/kt/ide073/inputs/qwen/perf_LOW_CONCURRENCY.custom.jsonl 16 1 128 qwen480 /home/mystous/.cache/huggingface/kt/ide075/CORE2_162053/ctl_S12_C1_CORR /home/mystous/.cache/huggingface/kt/ide075/CORE2_162053/probe_S12_C1_CORR.json
