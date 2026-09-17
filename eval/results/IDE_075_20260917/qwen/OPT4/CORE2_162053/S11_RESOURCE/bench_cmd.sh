#!/usr/bin/env bash
# container vllm-h100
python3 /tmp/probe_client.py /home/mystous/.cache/huggingface/kt/ide073/inputs/qwen/perf_MAIN_SHORT.custom.jsonl 128 64 128 qwen480 /home/mystous/.cache/huggingface/kt/ide075/CORE2_162053/ctl_S11_RESOURCE /home/mystous/.cache/huggingface/kt/ide075/CORE2_162053/probe_S11_RESOURCE.json
