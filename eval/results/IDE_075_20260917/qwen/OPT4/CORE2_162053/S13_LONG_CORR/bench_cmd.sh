#!/usr/bin/env bash
# container vllm-h100
python3 /tmp/probe_client.py /home/mystous/.cache/huggingface/kt/ide073/inputs/qwen/perf_LONGER_PREFILL.custom.jsonl 32 8 128 qwen480 /home/mystous/.cache/huggingface/kt/ide075/CORE2_162053/ctl_S13_LONG_CORR /home/mystous/.cache/huggingface/kt/ide075/CORE2_162053/probe_S13_LONG_CORR.json
