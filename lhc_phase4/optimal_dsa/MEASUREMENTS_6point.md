# Optimal+DSA 6-Point Coverage — Multi-Model Real-Corpus Validation

**HW**: DGX B200 × 8 (sm_100), Xeon Platinum 8570 + AMX + DSA 8 SWQ
**Harness**: `vllm_config_perf/gating/realistic_eval/throughput_runner.py`
**Setup**: corpus 500p × conc=32 × max_tok=8192, streaming, `cudagraph_mode=FULL_AND_PIECEWISE` (FaP)

## 6 measurement points per (model, corpus)

| ID | label | host DSA WQ | vllm spec decode | vllm DSA env | source |
|---|---|:---:|:---:|:---:|---|
| ① | van(OFF) | disabled | none | none | TSK_042 (2026-06-02) |
| ② | van(ON) | **enabled** | none | none | fresh (2026-06-10+) |
| ③ | DSA(ON) | enabled | none | **on** | fresh |
| ④ | suf(OFF) | disabled | suffix K=32 | none | TSK_042 |
| ⑤ | suf(ON) | **enabled** | suffix K=32 | none | fresh |
| ⑥ | suf+dsa(ON) | enabled | suffix K=32 | **on** | fresh |

---

## Coverage

| model | ① | ② | ③ | ④ | ⑤ | ⑥ | 셀 합 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `Qwen2.5-7B-Instruct` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **42/42** |
| `DeepSeek-R1-Distill-Qwen-7B` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **42/42** |
| `Llama-3.1-8B-Instruct` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **42/42** |
| `Qwen2.5-32B-Instruct` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **42/42** |
| `DeepSeek-R1-Distill-Qwen-32B` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **42/42** |
| `Qwen2.5-72B-Instruct` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **42/42** |
| `Llama-3.1-70B-Instruct` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **42/42** |
| `DeepSeek-R1-Distill-Llama-70B` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **42/42** |
| `Llama-3.1-405B-Instruct-FP8` | ✅ | ✅ | ✅ | ✅ | 0/7 | 0/7 | **28/42** |
| `DeepSeek-R1` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **42/42** |

**전체: 406/420 = 96.7%**

---

## Headline — mix corpus (10 모델 × 6 points)

| model | ① van(OFF) | ② van(ON) | ③ DSA(ON) | ④ suf(OFF) | ⑤ suf(ON) | ⑥ suf+dsa(ON) | **best** |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `Qwen2.5-7B-Instruct` | 4,169 | 5,564 | 5,572 | 7,803 | 7,478 | 7,457 | **suf(OFF) 7,803** |
| `DeepSeek-R1-Distill-Qwen-7B` | 9,058 | 12,277 | 12,301 | 24,458 | 22,467 | 22,193 | **suf(OFF) 24,458** |
| `Llama-3.1-8B-Instruct` | 8,850 | 12,089 | 12,058 | 27,851 | 24,407 | 26,615 | **suf(OFF) 27,851** |
| `Qwen2.5-32B-Instruct` | 3,056 | 4,694 | 4,698 | 6,597 | 5,979 | 6,256 | **suf(OFF) 6,597** |
| `DeepSeek-R1-Distill-Qwen-32B` | 4,938 | 5,134 | 5,060 | 9,056 | 8,378 | 9,240 | **suf+dsa(ON) 9,240** |
| `Qwen2.5-72B-Instruct` | 2,735 | 2,967 | 2,902 | 5,268 | 5,643 | 5,266 | **suf(ON) 5,643** |
| `Llama-3.1-70B-Instruct` | 3,129 | 3,206 | 3,192 | 10,400 | 10,247 | 8,829 | **suf(OFF) 10,400** |
| `DeepSeek-R1-Distill-Llama-70B` | 3,164 | 3,244 | 3,198 | 6,127 | 6,175 | 5,818 | **suf(ON) 6,175** |
| `Llama-3.1-405B-Instruct-FP8` | 1,252 | 1,252 | 1,271 | 2,829 | — | — | **suf(OFF) 2,829** |
| `DeepSeek-R1` | 1,538 | 1,599 | 1,601 | 781 | 754 | 727 | **DSA(ON) 1,601** |

---

## `Qwen2.5-7B-Instruct` — 6 points × 7 corpus

| corpus | ① van(OFF) | ② van(ON) | ③ DSA(ON) | ④ suf(OFF) | ⑤ suf(ON) | ⑥ suf+dsa(ON) |
|---|---:|---:|---:|---:|---:|---:|
| sharegpt | 4,189 | 5,600 | 5,640 | 6,167 | 6,058 | 6,164 |
| swebench | 4,120 | 5,871 | 5,973 | 5,416 | 5,322 | 5,551 |
| humaneval | 3,754 | 5,331 | 5,336 | 5,213 | 4,863 | 4,989 |
| mbpp | 3,814 | 5,931 | 5,965 | 5,506 | 5,346 | 5,390 |
| wildchat | 4,184 | 5,694 | 5,644 | 6,285 | 5,974 | 6,293 |
| lmsys | 4,090 | 5,409 | 5,427 | 5,956 | 5,906 | 6,038 |
| mix | 4,169 | 5,564 | 5,572 | 7,803 | 7,478 | 7,457 |

---

## `DeepSeek-R1-Distill-Qwen-7B` — 6 points × 7 corpus

| corpus | ① van(OFF) | ② van(ON) | ③ DSA(ON) | ④ suf(OFF) | ⑤ suf(ON) | ⑥ suf+dsa(ON) |
|---|---:|---:|---:|---:|---:|---:|
| sharegpt | 8,724 | 12,232 | 12,170 | 11,961 | 11,234 | 11,240 |
| swebench | 8,835 | 11,891 | 11,888 | 15,422 | 14,671 | 14,682 |
| humaneval | 8,159 | 11,273 | 11,240 | 11,459 | 11,035 | 10,519 |
| mbpp | 8,440 | 11,694 | 11,676 | 12,398 | 11,481 | 12,260 |
| wildchat | 8,925 | 12,319 | 12,210 | 11,717 | 10,795 | 11,263 |
| lmsys | 8,811 | 12,055 | 12,057 | 11,360 | 11,052 | 11,390 |
| mix | 9,058 | 12,277 | 12,301 | 24,458 | 22,467 | 22,193 |

---

## `Llama-3.1-8B-Instruct` — 6 points × 7 corpus

| corpus | ① van(OFF) | ② van(ON) | ③ DSA(ON) | ④ suf(OFF) | ⑤ suf(ON) | ⑥ suf+dsa(ON) |
|---|---:|---:|---:|---:|---:|---:|
| sharegpt | 8,868 | 12,091 | 12,088 | 19,054 | 18,073 | 19,328 |
| swebench | 8,348 | 11,970 | 11,518 | 21,353 | 20,735 | 20,518 |
| humaneval | 9,048 | 10,967 | 11,061 | 15,126 | 14,794 | 15,601 |
| mbpp | 8,730 | 12,190 | 12,066 | 17,825 | 17,976 | 17,360 |
| wildchat | 9,002 | 12,210 | 12,197 | 19,856 | 19,602 | 19,451 |
| lmsys | 9,074 | 12,528 | 11,993 | 19,862 | 19,361 | 18,905 |
| mix | 8,850 | 12,089 | 12,058 | 27,851 | 24,407 | 26,615 |

---

## `Qwen2.5-32B-Instruct` — 6 points × 7 corpus

| corpus | ① van(OFF) | ② van(ON) | ③ DSA(ON) | ④ suf(OFF) | ⑤ suf(ON) | ⑥ suf+dsa(ON) |
|---|---:|---:|---:|---:|---:|---:|
| sharegpt | 3,079 | 4,591 | 4,607 | 4,662 | 4,499 | 4,474 |
| swebench | 2,892 | 4,148 | 4,244 | 5,002 | 4,348 | 4,566 |
| humaneval | 2,571 | 3,602 | 3,527 | 4,859 | 4,325 | 4,269 |
| mbpp | 2,915 | 4,295 | 4,425 | 5,138 | 4,826 | 4,817 |
| wildchat | 3,128 | 4,804 | 4,738 | 4,884 | 4,651 | 4,504 |
| lmsys | 3,053 | 4,686 | 4,628 | 4,478 | 4,578 | 4,249 |
| mix | 3,056 | 4,694 | 4,698 | 6,597 | 5,979 | 6,256 |

---

## `DeepSeek-R1-Distill-Qwen-32B` — 6 points × 7 corpus

| corpus | ① van(OFF) | ② van(ON) | ③ DSA(ON) | ④ suf(OFF) | ⑤ suf(ON) | ⑥ suf+dsa(ON) |
|---|---:|---:|---:|---:|---:|---:|
| sharegpt | 4,803 | 4,931 | 4,902 | 4,996 | 4,682 | 4,613 |
| swebench | 4,409 | 4,524 | 4,561 | 5,241 | 5,589 | 5,444 |
| humaneval | 3,462 | 3,729 | 4,208 | 3,771 | 3,935 | 3,435 |
| mbpp | 4,690 | 4,905 | 4,806 | 5,690 | 5,097 | 5,221 |
| wildchat | 4,891 | 5,066 | 5,102 | 5,729 | 5,363 | 5,539 |
| lmsys | 4,898 | 4,993 | 5,011 | 5,356 | 4,980 | 5,116 |
| mix | 4,938 | 5,134 | 5,060 | 9,056 | 8,378 | 9,240 |

---

## `Qwen2.5-72B-Instruct` — 6 points × 7 corpus

| corpus | ① van(OFF) | ② van(ON) | ③ DSA(ON) | ④ suf(OFF) | ⑤ suf(ON) | ⑥ suf+dsa(ON) |
|---|---:|---:|---:|---:|---:|---:|
| sharegpt | 2,688 | 2,830 | 2,906 | 3,219 | 3,095 | 3,006 |
| swebench | 2,361 | 2,474 | 2,444 | 2,647 | 2,743 | 2,635 |
| humaneval | 806 | 1,989 | 2,542 | 2,489 | 2,358 | 2,022 |
| mbpp | 3,395 | 3,417 | 3,441 | 3,234 | 2,976 | 2,910 |
| wildchat | 2,803 | 2,929 | 2,909 | 2,621 | 2,434 | 2,591 |
| lmsys | 2,807 | 3,169 | 3,083 | 3,429 | 2,978 | 3,153 |
| mix | 2,735 | 2,967 | 2,902 | 5,268 | 5,643 | 5,266 |

---

## `Llama-3.1-70B-Instruct` — 6 points × 7 corpus

| corpus | ① van(OFF) | ② van(ON) | ③ DSA(ON) | ④ suf(OFF) | ⑤ suf(ON) | ⑥ suf+dsa(ON) |
|---|---:|---:|---:|---:|---:|---:|
| sharegpt | 3,091 | 3,177 | 3,139 | 4,864 | 4,542 | 4,634 |
| swebench | 2,878 | 2,809 | 2,968 | 6,026 | 5,949 | 5,455 |
| humaneval | 3,391 | 3,456 | 2,899 | 4,728 | 4,598 | 4,549 |
| mbpp | 1,773 | 1,699 | 1,778 | 3,266 | 3,243 | 2,273 |
| wildchat | 3,172 | 3,213 | 3,268 | 5,261 | 5,142 | 4,966 |
| lmsys | 3,040 | 3,145 | 3,123 | 3,958 | 3,677 | 3,818 |
| mix | 3,129 | 3,206 | 3,192 | 10,400 | 10,247 | 8,829 |

---

## `DeepSeek-R1-Distill-Llama-70B` — 6 points × 7 corpus

| corpus | ① van(OFF) | ② van(ON) | ③ DSA(ON) | ④ suf(OFF) | ⑤ suf(ON) | ⑥ suf+dsa(ON) |
|---|---:|---:|---:|---:|---:|---:|
| sharegpt | 3,033 | 3,018 | 3,142 | 2,660 | 2,579 | 2,503 |
| swebench | 3,236 | 3,142 | 3,182 | 2,739 | 2,739 | 2,642 |
| humaneval | 2,852 | 2,828 | 2,812 | 2,788 | 2,809 | 2,718 |
| mbpp | 2,777 | 2,989 | 2,954 | 2,426 | 2,328 | 2,265 |
| wildchat | 3,127 | 3,208 | 3,166 | 2,658 | 2,544 | 2,661 |
| lmsys | 2,992 | 3,046 | 3,045 | 2,848 | 2,756 | 2,844 |
| mix | 3,164 | 3,244 | 3,198 | 6,127 | 6,175 | 5,818 |

---

## `Llama-3.1-405B-Instruct-FP8` — 6 points × 7 corpus

| corpus | ① van(OFF) | ② van(ON) | ③ DSA(ON) | ④ suf(OFF) | ⑤ suf(ON) | ⑥ suf+dsa(ON) |
|---|---:|---:|---:|---:|---:|---:|
| sharegpt | 1,217 | 1,239 | 1,229 | 2,061 | — | — |
| swebench | 1,204 | 1,211 | 1,239 | 2,639 | — | — |
| humaneval | 1,253 | 1,237 | 1,192 | 2,112 | — | — |
| mbpp | 916 | 883 | 815 | 1,725 | — | — |
| wildchat | 1,280 | 1,267 | 1,263 | 2,290 | — | — |
| lmsys | 1,220 | 1,247 | 1,221 | 2,243 | — | — |
| mix | 1,252 | 1,252 | 1,271 | 2,829 | — | — |

---

## `DeepSeek-R1` — 6 points × 7 corpus

| corpus | ① van(OFF) | ② van(ON) | ③ DSA(ON) | ④ suf(OFF) | ⑤ suf(ON) | ⑥ suf+dsa(ON) |
|---|---:|---:|---:|---:|---:|---:|
| sharegpt | 1,475 | 1,565 | 1,559 | 797 | 730 | 794 |
| swebench | 1,474 | 1,536 | 1,518 | 538 | 496 | 542 |
| humaneval | 1,004 | 961 | 858 | 606 | 1,219 | 670 |
| mbpp | 1,437 | 1,482 | 1,490 | 677 | 661 | 669 |
| wildchat | 1,556 | 1,614 | 1,614 | 858 | 880 | 824 |
| lmsys | 1,533 | 1,587 | 1,592 | 811 | 808 | 773 |
| mix | 1,538 | 1,599 | 1,601 | 781 | 754 | 727 |

---

## Effect decomposition — mix corpus (Δ% 분해)

- **host DSA effect on vanilla**: ② vs ①
- **host DSA effect on suffix**: ⑤ vs ④
- **vllm env effect on vanilla (host ON)**: ③ vs ②
- **vllm env effect on suffix (host ON)**: ⑥ vs ⑤
- **suffix effect (same-state host OFF)**: ④ vs ①
- **suffix effect (same-state host ON)**: ⑤ vs ②

| model | DSA on van | DSA on suf | vllm env on van | vllm env on suf | suf-gain (OFF) | suf-gain (ON) |
|---|---:|---:|---:|---:|---:|---:|
| `Qwen2.5-7B-Instruct` | +33.5% | -4.2% | +0.2% | -0.3% | +87.2% | +34.4% |
| `DeepSeek-R1-Distill-Qwen-7B` | +35.5% | -8.1% | +0.2% | -1.2% | +170.0% | +83.0% |
| `Llama-3.1-8B-Instruct` | +36.6% | -12.4% | -0.3% | +9.0% | +214.7% | +101.9% |
| `Qwen2.5-32B-Instruct` | +53.6% | -9.4% | +0.1% | +4.6% | +115.9% | +27.4% |
| `DeepSeek-R1-Distill-Qwen-32B` | +4.0% | -7.5% | -1.4% | +10.3% | +83.4% | +63.2% |
| `Qwen2.5-72B-Instruct` | +8.5% | +7.1% | -2.2% | -6.7% | +92.6% | +90.2% |
| `Llama-3.1-70B-Instruct` | +2.4% | -1.5% | -0.4% | -13.8% | +232.4% | +219.6% |
| `DeepSeek-R1-Distill-Llama-70B` | +2.5% | +0.8% | -1.4% | -5.8% | +93.7% | +90.3% |
| `Llama-3.1-405B-Instruct-FP8` | -0.0% | — | +1.5% | — | +125.9% | — |
| `DeepSeek-R1` | +4.0% | -3.5% | +0.1% | -3.6% | -49.2% | -52.8% |

---

## Winner distribution — best point per (model, corpus) 70 셀

| Winner point | count | % |
|---|---:|---:|
| van(OFF) | **2** | 2.9% |
| van(ON) | **10** | 14.3% |
| DSA(ON) | **11** | 15.7% |
| suf(OFF) | **36** | 51.4% |
| suf(ON) | **6** | 8.6% |
| suf+dsa(ON) | **5** | 7.1% |
