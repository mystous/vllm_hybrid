#!/usr/bin/env bash
# container vllm-h100
taskset -c 48-55,104-111,160-167,216-223 python3 /tmp/probe_client.py /home/mystous/.cache/huggingface/kt/ide073/inputs/qwen/perf_MAIN_SHORT.custom.jsonl 128 64 128 qwen480 /home/mystous/.cache/huggingface/kt/ide075/OFF_X_171100/ctl_S16b_OFF_X_probe_pinned /home/mystous/.cache/huggingface/kt/ide075/OFF_X_171100/probe_S16b_OFF_X_probe_pinned.json
