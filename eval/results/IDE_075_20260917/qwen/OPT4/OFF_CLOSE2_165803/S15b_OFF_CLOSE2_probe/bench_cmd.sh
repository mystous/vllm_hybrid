#!/usr/bin/env bash
# container vllm-h100
python3 /tmp/probe_client.py /home/mystous/.cache/huggingface/kt/ide073/inputs/qwen/perf_MAIN_SHORT.custom.jsonl 128 64 128 qwen480 /home/mystous/.cache/huggingface/kt/ide075/OFF_CLOSE2_165803/ctl_S15b_OFF_CLOSE2_probe /home/mystous/.cache/huggingface/kt/ide075/OFF_CLOSE2_165803/probe_S15b_OFF_CLOSE2_probe.json
