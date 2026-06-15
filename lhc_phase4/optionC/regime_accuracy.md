# Option C — Regime classification accuracy

Per-run distribution of regime classifications inferred from
boot-log scheduler `Engine 000:` lines (KV%, gen tps, waiting).

| run | samples | GPU_SAT | KV_HEAVY | BALANCED | mean KV% | mean gen tps |
|---|---|---|---|---|---|---|
| bl_balanced_lhc_adaptive_s1 | 2 | 100.0% | 0.0% | 0.0% | 0.70% | 18511 |
| bl_balanced_lhc_adaptive_s2 | 2 | 100.0% | 0.0% | 0.0% | 0.65% | 18914 |
| bl_balanced_vanilla_s1 | 2 | 100.0% | 0.0% | 0.0% | 0.60% | 17756 |
| bl_balanced_vanilla_s2 | 3 | 66.7% | 0.0% | 33.3% | 0.37% | 12516 |
| bl_chat_lhc_adaptive_s1 | 1 | 100.0% | 0.0% | 0.0% | 0.00% | 11968 |
| bl_chat_lhc_adaptive_s2 | 1 | 100.0% | 0.0% | 0.0% | 0.20% | 6486 |
| bl_chat_vanilla_s1 | 1 | 100.0% | 0.0% | 0.0% | 0.00% | 12034 |
| bl_chat_vanilla_s2 | 1 | 100.0% | 0.0% | 0.0% | 0.00% | 11989 |
| bl_code-heavy_lhc_adaptive_s1 | 2 | 100.0% | 0.0% | 0.0% | 1.45% | 10147 |
| bl_code-heavy_lhc_adaptive_s2 | 2 | 100.0% | 0.0% | 0.0% | 1.40% | 10201 |
| bl_code-heavy_vanilla_s1 | 1 | 100.0% | 0.0% | 0.0% | 2.30% | 13354 |
| bl_code-heavy_vanilla_s2 | 2 | 100.0% | 0.0% | 0.0% | 0.35% | 10083 |
| bl_code_lhc_adaptive_s1 | 2 | 100.0% | 0.0% | 0.0% | 0.35% | 12798 |
| bl_code_lhc_adaptive_s2 | 2 | 100.0% | 0.0% | 0.0% | 0.40% | 12799 |
| bl_code_vanilla_s1 | 1 | 100.0% | 0.0% | 0.0% | 0.70% | 17820 |
| bl_code_vanilla_s2 | 2 | 100.0% | 0.0% | 0.0% | 0.40% | 12799 |
| bl_sonnet-heavy_lhc_adaptive_s1 | 3 | 100.0% | 0.0% | 0.0% | 1.10% | 13501 |
| bl_sonnet-heavy_lhc_adaptive_s2 | 3 | 100.0% | 0.0% | 0.0% | 1.17% | 13445 |
| bl_sonnet-heavy_vanilla_s1 | 3 | 100.0% | 0.0% | 0.0% | 1.30% | 13269 |
| bl_sonnet-heavy_vanilla_s2 | 2 | 100.0% | 0.0% | 0.0% | 1.80% | 18154 |
| bl_sonnet_lhc_adaptive_s1 | 3 | 66.7% | 0.0% | 33.3% | 0.27% | 8193 |
| bl_sonnet_lhc_adaptive_s2 | 2 | 50.0% | 0.0% | 50.0% | 0.40% | 8977 |
| bl_sonnet_vanilla_s1 | 3 | 66.7% | 33.3% | 0.0% | 0.13% | 8335 |
| bl_sonnet_vanilla_s2 | 2 | 100.0% | 0.0% | 0.0% | 0.20% | 12443 |
| wd1_lhc | 26 | 3.8% | 88.5% | 7.7% | 1.72% | 1988 |
| wd1_lhc_adaptive_s1 | 8 | 62.5% | 0.0% | 37.5% | 4.65% | 3258 |
| wd1_lhc_adaptive_s2 | 7 | 71.4% | 0.0% | 28.6% | 5.09% | 3577 |
| wd1_lhc_adaptive_sfx_s1 | 9 | 55.6% | 0.0% | 44.4% | 4.79% | 2882 |
| wd1_lhc_adaptive_sfx_s2 | 8 | 62.5% | 0.0% | 37.5% | 4.78% | 3198 |
| wd1_lhc_always_off_s1 | 7 | 71.4% | 0.0% | 28.6% | 5.14% | 3613 |
| wd1_lhc_always_off_s2 | 8 | 62.5% | 0.0% | 37.5% | 4.69% | 3212 |
| wd1_lhc_always_on_s1 | 8 | 62.5% | 0.0% | 37.5% | 4.56% | 3178 |
| wd1_lhc_always_on_s2 | 8 | 75.0% | 0.0% | 25.0% | 4.58% | 3222 |
| wd1_vanilla | 13 | 92.3% | 0.0% | 7.7% | 1.74% | 1973 |
| wd1_vanilla_s1 | 8 | 62.5% | 0.0% | 37.5% | 4.69% | 3184 |
| wd1_vanilla_s2 | 8 | 75.0% | 0.0% | 25.0% | 4.66% | 3256 |