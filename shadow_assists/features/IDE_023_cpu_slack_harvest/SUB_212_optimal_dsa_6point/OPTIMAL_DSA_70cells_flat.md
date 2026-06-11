# Optimal+DSA — Flat 70-Cell Matrix (10 models × 7 corpus × 6 points)

> Single-flat representation: 70 rows, one per (model, corpus). 6 throughput columns by
> measurement point. Winner column = best of 6 points per cell.

## Column definitions

| Column | Host DSA WQ | spec decode | vllm DSA env | Source |
|---|:---:|:---:|:---:|---|
| **van OFF** | disabled (2026-06-02) | — | — | TSK_042 baseline |
| **suf OFF** | disabled (2026-06-02) | suffix K=32 | — | TSK_042 baseline |
| **van ON** | enabled (2026-06-08+) | — | — | fresh sweep |
| **suf ON** | enabled (2026-06-08+) | suffix K=32 | — | fresh sweep |
| **DSA ON** | enabled | — | `VLLM_LHC_DSA=1 VLLM_LEVER_N9=1` | fresh sweep |
| **suf+DSA ON** | enabled | suffix K=32 | vllm DSA on | fresh sweep |

## Hardware / Software

- **HW**: DGX B200 × 8 (sm_100, 183 GiB HBM3e), Xeon Platinum 8570 (Emerald Rapids) + AMX + DSA 8 SWQ
- **vLLM**: `1.7.dev16107+gffe20fb09.d20260601` (sm_100 build)
- **Setup**: corpus 500p × conc=32 × max_tok=8192, streaming, `cudagraph_mode=FULL_AND_PIECEWISE` (FaP), gpu_mem_util=0.85, max_model_len=16384
- **TP**: head%8==0 → TP=8 (GPU 0-7) / 7B Qwen models → TP=4 (GPU 0-3)

## Notable

- **Llama-3.1-405B-FP8** ⑤ ⑥ (suf ON / suf+DSA ON): engine init failure → 14 cells 영구 미측정
- 호스트 DSA WQ enable mtime: **2026-06-08 00:40** — TSK_042 측정 (2026-06-02) 시에는 disabled

---

## Full Matrix — output_tps (70 cells)

| model | corpus | van OFF | suf OFF | van ON | suf ON | DSA ON | suf+DSA ON | winner |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| **Qwen-7B** | sharegpt | 4,189 | 6,167 | 5,600 | 6,058 | 5,640 | 6,164 | **suf OFF** |
|  | swebench | 4,120 | 5,416 | 5,871 | 5,322 | 5,973 | 5,551 | **DSA ON** |
|  | humaneval | 3,754 | 5,213 | 5,331 | 4,863 | 5,336 | 4,989 | **DSA ON** |
|  | mbpp | 3,814 | 5,506 | 5,931 | 5,346 | 5,965 | 5,390 | **DSA ON** |
|  | wildchat | 4,184 | 6,285 | 5,694 | 5,974 | 5,644 | 6,293 | **suf+DSA ON** |
|  | lmsys | 4,090 | 5,956 | 5,409 | 5,906 | 5,427 | 6,038 | **suf+DSA ON** |
|  | mix | 4,169 | 7,803 | 5,564 | 7,478 | 5,572 | 7,457 | **suf OFF** |
| **DS-Qwen-7B** | sharegpt | 8,724 | 11,961 | 12,232 | 11,234 | 12,170 | 11,240 | **van ON** |
|  | swebench | 8,835 | 15,422 | 11,891 | 14,671 | 11,888 | 14,682 | **suf OFF** |
|  | humaneval | 8,159 | 11,459 | 11,273 | 11,035 | 11,240 | 10,519 | **suf OFF** |
|  | mbpp | 8,440 | 12,398 | 11,694 | 11,481 | 11,676 | 12,260 | **suf OFF** |
|  | wildchat | 8,925 | 11,717 | 12,319 | 10,795 | 12,210 | 11,263 | **van ON** |
|  | lmsys | 8,811 | 11,360 | 12,055 | 11,052 | 12,057 | 11,390 | **DSA ON** |
|  | mix | 9,058 | 24,458 | 12,277 | 22,467 | 12,301 | 22,193 | **suf OFF** |
| **Llama-8B** | sharegpt | 8,868 | 19,054 | 12,091 | 18,073 | 12,088 | 19,328 | **suf+DSA ON** |
|  | swebench | 8,348 | 21,353 | 11,970 | 20,735 | 11,518 | 20,518 | **suf OFF** |
|  | humaneval | 9,048 | 15,126 | 10,967 | 14,794 | 11,061 | 15,601 | **suf+DSA ON** |
|  | mbpp | 8,730 | 17,825 | 12,190 | 17,976 | 12,066 | 17,360 | **suf ON** |
|  | wildchat | 9,002 | 19,856 | 12,210 | 19,602 | 12,197 | 19,451 | **suf OFF** |
|  | lmsys | 9,074 | 19,862 | 12,528 | 19,361 | 11,993 | 18,905 | **suf OFF** |
|  | mix | 8,850 | 27,851 | 12,089 | 24,407 | 12,058 | 26,615 | **suf OFF** |
| **Qwen-32B** | sharegpt | 3,079 | 4,662 | 4,591 | 4,499 | 4,607 | 4,474 | **suf OFF** |
|  | swebench | 2,892 | 5,002 | 4,148 | 4,348 | 4,244 | 4,566 | **suf OFF** |
|  | humaneval | 2,571 | 4,859 | 3,602 | 4,325 | 3,527 | 4,269 | **suf OFF** |
|  | mbpp | 2,915 | 5,138 | 4,295 | 4,826 | 4,425 | 4,817 | **suf OFF** |
|  | wildchat | 3,128 | 4,884 | 4,804 | 4,651 | 4,738 | 4,504 | **suf OFF** |
|  | lmsys | 3,053 | 4,478 | 4,686 | 4,578 | 4,628 | 4,249 | **van ON** |
|  | mix | 3,056 | 6,597 | 4,694 | 5,979 | 4,698 | 6,256 | **suf OFF** |
| **DS-Qwen-32B** | sharegpt | 4,803 | 4,996 | 4,931 | 4,682 | 4,902 | 4,613 | **suf OFF** |
|  | swebench | 4,409 | 5,241 | 4,524 | 5,589 | 4,561 | 5,444 | **suf ON** |
|  | humaneval | 3,462 | 3,771 | 3,729 | 3,935 | 4,208 | 3,435 | **DSA ON** |
|  | mbpp | 4,690 | 5,690 | 4,905 | 5,097 | 4,806 | 5,221 | **suf OFF** |
|  | wildchat | 4,891 | 5,729 | 5,066 | 5,363 | 5,102 | 5,539 | **suf OFF** |
|  | lmsys | 4,898 | 5,356 | 4,993 | 4,980 | 5,011 | 5,116 | **suf OFF** |
|  | mix | 4,938 | 9,056 | 5,134 | 8,378 | 5,060 | 9,240 | **suf+DSA ON** |
| **Qwen-72B** | sharegpt | 2,688 | 3,219 | 2,830 | 3,095 | 2,906 | 3,006 | **suf OFF** |
|  | swebench | 2,361 | 2,647 | 2,474 | 2,743 | 2,444 | 2,635 | **suf ON** |
|  | humaneval | 806 | 2,489 | 1,989 | 2,358 | 2,542 | 2,022 | **DSA ON** |
|  | mbpp | 3,395 | 3,234 | 3,417 | 2,976 | 3,441 | 2,910 | **DSA ON** |
|  | wildchat | 2,803 | 2,621 | 2,929 | 2,434 | 2,909 | 2,591 | **van ON** |
|  | lmsys | 2,807 | 3,429 | 3,169 | 2,978 | 3,083 | 3,153 | **suf OFF** |
|  | mix | 2,735 | 5,268 | 2,967 | 5,643 | 2,902 | 5,266 | **suf ON** |
| **Llama-70B** | sharegpt | 3,091 | 4,864 | 3,177 | 4,542 | 3,139 | 4,634 | **suf OFF** |
|  | swebench | 2,878 | 6,026 | 2,809 | 5,949 | 2,968 | 5,455 | **suf OFF** |
|  | humaneval | 3,391 | 4,728 | 3,456 | 4,598 | 2,899 | 4,549 | **suf OFF** |
|  | mbpp | 1,773 | 3,266 | 1,699 | 3,243 | 1,778 | 2,273 | **suf OFF** |
|  | wildchat | 3,172 | 5,261 | 3,213 | 5,142 | 3,268 | 4,966 | **suf OFF** |
|  | lmsys | 3,040 | 3,958 | 3,145 | 3,677 | 3,123 | 3,818 | **suf OFF** |
|  | mix | 3,129 | 10,400 | 3,206 | 10,247 | 3,192 | 8,829 | **suf OFF** |
| **DS-Llama-70B** | sharegpt | 3,033 | 2,660 | 3,018 | 2,579 | 3,142 | 2,503 | **DSA ON** |
|  | swebench | 3,236 | 2,739 | 3,142 | 2,739 | 3,182 | 2,642 | **van OFF** |
|  | humaneval | 2,852 | 2,788 | 2,828 | 2,809 | 2,812 | 2,718 | **van OFF** |
|  | mbpp | 2,777 | 2,426 | 2,989 | 2,328 | 2,954 | 2,265 | **van ON** |
|  | wildchat | 3,127 | 2,658 | 3,208 | 2,544 | 3,166 | 2,661 | **van ON** |
|  | lmsys | 2,992 | 2,848 | 3,046 | 2,756 | 3,045 | 2,844 | **van ON** |
|  | mix | 3,164 | 6,127 | 3,244 | 6,175 | 3,198 | 5,818 | **suf ON** |
| **Llama-405B-FP8** | sharegpt | 1,217 | 2,061 | 1,239 | — | 1,229 | — | **suf OFF** |
|  | swebench | 1,204 | 2,639 | 1,211 | — | 1,239 | — | **suf OFF** |
|  | humaneval | 1,253 | 2,112 | 1,237 | — | 1,192 | — | **suf OFF** |
|  | mbpp | 916 | 1,725 | 883 | — | 815 | — | **suf OFF** |
|  | wildchat | 1,280 | 2,290 | 1,267 | — | 1,263 | — | **suf OFF** |
|  | lmsys | 1,220 | 2,243 | 1,247 | — | 1,221 | — | **suf OFF** |
|  | mix | 1,252 | 2,829 | 1,252 | — | 1,271 | — | **suf OFF** |
| **R1-671B** | sharegpt | 1,475 | 797 | 1,565 | 730 | 1,559 | 794 | **van ON** |
|  | swebench | 1,474 | 538 | 1,536 | 496 | 1,518 | 542 | **van ON** |
|  | humaneval | 1,004 | 606 | 961 | 1,219 | 858 | 670 | **suf ON** |
|  | mbpp | 1,437 | 677 | 1,482 | 661 | 1,490 | 669 | **DSA ON** |
|  | wildchat | 1,556 | 858 | 1,614 | 880 | 1,614 | 824 | **van ON** |
|  | lmsys | 1,533 | 811 | 1,587 | 808 | 1,592 | 773 | **DSA ON** |
|  | mix | 1,538 | 781 | 1,599 | 754 | 1,601 | 727 | **DSA ON** |

---

## Winner distribution (70 cells)

| Winner | count | % |
|---|---:|---:|
| **suf OFF** | 36 | 51.4% |
| **DSA ON** | 11 | 15.7% |
| **van ON** | 10 | 14.3% |
| **suf+DSA ON** | 5 | 7.1% |
| **suf ON** | 6 | 8.6% |
| **van OFF** | 2 | 2.9% |

## Group analysis — DSA OFF vs DSA ON regime

- **DSA OFF regime winner (① ②)**: 38/70 (54.3%)
- **DSA ON regime winner (③④⑤⑥)**: 32/70 (45.7%)

---

## Source

- TSK_042 baseline: `vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/summ_<TAG>_<METHOD>_<CORPUS>.json`
- Fresh sweep: `lhc_phase4/optimal_dsa/runs/summ_<TAG>_<METHOD>_<CORPUS>.json`
- Total cells (70 × 6 — 14 fail = 406): see commit `2d2142254` / SUB_212
