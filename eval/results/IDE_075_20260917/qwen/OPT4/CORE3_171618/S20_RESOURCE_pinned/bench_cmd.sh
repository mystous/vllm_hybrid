#!/usr/bin/env bash
# container vllm-h100
taskset -c 48-55,104-111,160-167,216-223 python3 /tmp/probe_client.py /home/mystous/.cache/huggingface/kt/ide073/inputs/qwen/perf_MAIN_SHORT.custom.jsonl 128 64 128 qwen480 /home/mystous/.cache/huggingface/kt/ide075/CORE3_171618/ctl_S20_RESOURCE_pinned /home/mystous/.cache/huggingface/kt/ide075/CORE3_171618/probe_S20_RESOURCE_pinned.json
