#!/usr/bin/env bash
# container vllm-h100
python3 /tmp/probe_client.py /home/mystous/.cache/huggingface/kt/ide073/inputs/qwen/perf_MAIN_SHORT.custom.jsonl 128 64 128 qwen480 /models/kt/ide075/OFF_OPEN2_161158/ctl_S8b_OFF_OPEN2_probe /models/kt/ide075/OFF_OPEN2_161158/probe_S8b_OFF_OPEN2_probe.json
