# 20260428_065709_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_partial_profile_64k
- VLLM_PARTIAL_ATTN_PROFILE=1
- VLLM_COLD_KV_FALLBACK_DEADLINE_MS=0 (fallback 비활성, partition only)
- max-prompts=5, input=65536, output=4
