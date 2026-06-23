# SUB_248 — 70B 서빙 레버 10종 sweep (2026-06-15)

> **판정: 신규 GPU-side 레버는 전부 중립. 서빙 향상은 spec-decode(+82~91%)·TP스케일(+34%)
> = 기존(축적) 레버로만 수렴.** 깨끗한 측정(고유프롬프트 APC무효, GPU util 100%=GPU-bound).

baseline = 1436.9 tps (Llama-3.1-70B TP4, gmu0.85, mml4096).

| 레버 | best tps | vs base | 신규? | 판정 |
|---|---:|---:|---|---|
| baseline | 1436.9 | — | — | 기준 |
| spec_ngram_k8 | 2750.6 | **+91.4%** | 기존(spec) | ⭐ 최고 |
| fp8kv+spec | 2644.0 | +84.0% | 조합 | ⭐(spec 지배) |
| spec_ngram_k5 | 2618.8 | +82.3% | 기존(spec) | ⭐ |
| tp8 | 1927.2 | +34.1% | 기존(TP) | ⭐ |
| fp8_kv | 1453.9 | +1.2% | **신규** | 중립 |
| cudagraph_full | 1437.9 | +0.1% | 신규 | 중립 |
| batched_tokens_8192 | 1437.8 | +0.1% | 신규 | 중립 |
| max_num_seqs_512 | 1437.1 | +0.0% | 신규 | 중립 |
| attn_flashinfer | BOOT_FAIL | — | 신규 | 스크립트버그(env 확장)·미확정 |
| attn_flash_attn | BOOT_FAIL | — | 신규 | 동일 |

**결론**:
- 향상 레버 = spec-decode(K↑일수록↑: K5 +82%→K8 +91%)·TP스케일 — **둘 다 이미 알려진 축적 레버**.
- 신규 GPU-side 레버(FP8 KV·max_seqs·batched·cudagraph_full)는 **전부 ~0% 중립** — 워크로드가
  GPU-compute-bound(util 100%)라 메모리/스케줄/그래프 레버가 병목을 못 바꿈(세션 발견과 정합).
- attn 백엔드 2종은 sweep 스크립트의 `$env` 확장 버그(`VLLM_ATTENTION_BACKEND=...: command
  not found`)로 미부팅 — vLLM 비호환 아님. 재시도 시 `env VAR=val` 형식 필요.
- **10회 탐색 종합: 축적 레버(spec/TP)와 구별되는 신규 서빙-win 레버는 발견되지 않음.**

산출물: `sweep10.sh`, `bench_unique.py`, `runs/sweep_results.csv`.

## 라운드2 — 신규 바이트단위/연산축소 레버 10종 (2026-06-15) ⭐ 돌파

> **FP8 가중치 양자화 = 바이트단위 HW 착취로 GPU-compute-bound 돌파 첫 성공. (정확도 게이트 별도 필요 — lossy)**

baseline=1436.9 (bf16). 깨끗한 고유프롬프트, GPU util.

| 레버 | best tps | vs base |
|---|---:|---:|
| **fp8_weight + spec** | 3259.6 | **+126.9%** ⭐⭐ |
| **fp8_weight + tp8** | 2238.2 | +55.8% |
| **fp8_weight (단독)** | 1817.3 | **+26.5%** ⭐ |
| tp2_spec | 1617.9 | +12.6% |
| attn flashinfer/flash_attn/triton | ~1436 | +0%(중립) |
| tp2 | 914.5 | −36% |
| enforce_eager | 760.8 | −47% |
| kv_fp8_e5m2 | BOOT_FAIL | — |

**판정**: 비트폭(2B→1B)이 B200 FP8 텐서코어를 가속 = compute-bound를 깨는 유일 신규 레버.
커널 백엔드 교체는 중립(연산-bound). FP8×spec 곱셈 스택 +127%. **단 FP8은 lossy →
정확도 게이트(분포 동등) 통과해야 유효** = 라운드3.

## 라운드3 — FP8 정확도 게이트 PASS + FP4 부팅 확인 (2026-06-15) ⭐⭐ 검증완료

- **FP8 가중치 정확도 게이트 PASS** (bf16 vs fp8, 8 프롬프트 greedy logprob):
  max_abs_logprob_diff=0.135(≤0.5), ppl_rel mean0.018/max0.042(≤0.1). token_match 69%
  =informational(greedy cascading). → **+26.5%(단독)/+127%(spec조합) win이 분포동등으로 유효.**
- **FP4(mxfp4) online 부팅 OK** — 0.5B 양자화가 B200/이 빌드서 online 동작. tps/정확도 미측정(다음).
- 산출물: `round3.sh`, `collect_logprobs.py`, `runs/lp_{bf16,fp8}.json`.

**결론**: 멀티라운드 탐색의 결실 = **FP8 가중치 양자화(바이트단위 2B→1B, B200 FP8 TC)가
검증된 신규 서빙 win.** 세션 내내 "신규 win 없음"이었으나, GPU-bound를 깨는 정답은
"비트폭 축소"였고 정확도 게이트까지 통과. FP4(0.5B)가 다음 프론티어.

## 라운드4 — 코드(compute-graph) 구조 fusion 10종 (2026-06-15) — 전부 중립~음(-)

> **판정: 그래프-구조 변경(커널 fusion 패스)은 성능 향상 없음.** inductor가 이미 최적 fusion
> 적용 → 수동 패스는 중립~소폭 손해. FP8 베이스(1812.8) 위, GPU util 100% 동일.

| fusion 패스 | tps | vs fp8 |
|---|---:|---:|
| fuse_gemm_comms | 1813.1 | +0.0% |
| fuse_rope_kvcache | 1812.3 | −0.0% |
| enable_sp | 1809.4 | −0.2% |
| fuse_act_quant | 1807.1 | −0.3% |
| fuse_attn_quant | 1793.5 | −1.1% |
| all_fuse_gemmcomm | 1762.5 | −2.8% |
| all_fuse | 1758.8 | −3.0% |
| fuse_norm_quant | 1767.2 | −2.5% |
| qk_norm_rope_fusion | 1751.7 | −3.4% |

**결론**: 컴파일 그래프는 이미 최적화돼 있어 추가 fusion 패스가 안 닿음(일부는 오버헤드로
−). 코드/그래프 구조 변경 ≠ 서빙 향상. **GPU-compute-bound를 깬 유일 레버는 비트폭(FP8,
+26%)** 로 재확인. (1차 BOOT_FAIL 전부는 ready-체크 `assert` 패턴이 정상로그 `size_asserts`
에 오매칭된 false-fail, 수정 후 재측정한 본 표가 유효.)

## CPU-AMX 검증 + GPU 무검증 아이디어 — 무성립 확정 (2026-06-15)

> **판정: 70B 타깃에서 무성립.** 검증=타깃 forward라 CPU로 옮기면 (a)70B를 CPU서 돌려 느리거나
> (b)작은 모델로 대체돼 출력이 그 작은 모델 분포가 됨 → 출력 동등 위반.

프록시 측정(vLLM 무수정): "CPU 작은모델(Qwen2.5-0.5B, AMX draft 모델) 검증 + GPU 무검증" =
출력이 0.5B 분포. 70B 레퍼(lp_bf16) vs 0.5B(lp_0p5b) 분포 비교:
- token_match = **0.4%**, first-divergence = **0.2 토큰**(첫 토큰부터 갈림), ppl 70B 1.29 / 0.5B 1.86.
- → **분포동등 게이트 FAIL.** CPU 근사검증은 70B 출력을 보존 못 함(rejection sampling 의 p=타깃
  확률이 필수, 작은 모델 p'≠p → 채택 편향 → 출력=작은모델).
- 정합: repo `cpu_amx.py`는 올바른 방향(CPU=draft, GPU=verify). verify(=타깃 forward)는 이동 불가.
산출물: `runs/lp_0p5b.json`, collect/비교.

## 비동기 디코드(--async-scheduling) A/B (2026-06-16) — 중립

| conc | async_off | async_on | Δ |
|---|---:|---:|---:|
| 24 (throughput) | 1596.6 | 1597.0 | +0.0% |
| 4 (저-conc/host-overhead) | 409.6 | 409.8 | +0.0% |

**판정**: vLLM `--async-scheduling`(스케줄러-GPU 겹침)은 throughput·저-conc 둘 다 **중립**.
GPU-bound라 호스트 겹침이 안 닿고, 저-conc에서도 오버헤드가 이미 작음. "낙관적 비동기
디코드"(미검증 토큰 선출력)는 별도 큰 코드변경이나 latency-only·correctness-위험이라 미진행.

## 란초스(Lanczos) 적용성 검토 결론
직접 서빙-속도 레버 아님. 간접 2경로(둘 다 lossy·heavy·off-regime):
① 저랭크 KV 압축(Krylov 절단SVD) — 롱컨텍스트 KV-대역폭-bound 레짐(현 벤치 밖, KV 비병목).
② **FP4 eigen-rotation** — 가중치를 양자화-친화 기저로 회전해 FP4(0.5B) 정확도↑ → byte-level
   프론티어 확장(세션 win=FP8과 가장 연결). 오프라인 캘리브, QuIP/AWQ-rotation과 중복.
→ 권고: 란초스 단독보다 ②로 "FP4 게이트 통과"를 노리는 게 가치.

## FP4(mxfp4) 실측 + 란초스 ② 재평가 (2026-06-16)

- FP4 mxfp4 online: tps 1438.6/1433.5 **≈ bf16(1436.9)**, FP8(1817)보다 느림. 정확도
  token_match **100%**/logprob diff **0.0000** = **bf16 비트동일** → **online mxfp4가 bf16
  체크포인트를 실제 양자화 안 함**(FP4 weights/scale 부재 → bf16 폴백). round3 "부팅 OK"는
  부팅만, 연산은 bf16였음.
- **결론**: FP4 byte-level은 **오프라인 FP4 양자화 체크포인트**(modelopt/llm-compressor +
  캘리브레이션) 필요 = 멀티시간 프로젝트. **란초스 eigen-rotation(②)은 그 오프라인 파이프라인
  의 정확도 도구**(QuIP/AWQ-rotation류), 빠른 레버 아님.
- → **지금 얻을 수 있는 byte-level 천장 = FP8(online, +26.5%/+127%, SR-002)**. FP4+란초스는
  오프라인 양자화 후속 프로젝트로 분류(현 레짐 즉시 이득 없음).
산출물: `fp4_test.sh`, `runs/lp_fp4.json`.

## FP4 오프라인양자화 10-method sweep (2026-06-16) ⭐⭐⭐ FP4가 FP8 돌파

> **W4A4 NVFP4 = FP8 대비 +23%, bf16 대비 +54.8%, 정확도 게이트 PASS = 새 최고 단일-config win.**
> FP4+spec = bf16 대비 +194.5%(최고속) 단 게이트 근소 FAIL(rotation으로 보강 여지).

bf16=1437.2, fp8=1810.1 레퍼. 각 분포동등 게이트(vs bf16, max_logprob_diff≤0.5 & ppl_rel≤0.1).

| 방법 | tps | vs fp8 | vs bf16 | gate | (diff/rel) |
|---|---:|---:|---:|---|---|
| **w4a4 NVFP4** | 2225.2 | **+23.0%** | +54.8% | **PASS ✅** | 0.43/0.068 |
| w4a4 + KV-fp8 | 2258.7 | +24.8% | +57.2% | FAIL | 0.23/0.145 |
| **w4a4 + spec** | 4231.9 | **+133.8%** | **+194.5%** | FAIL(근소) | 0.43/0.128 |
| w4a4 + TP8 | 2449.7 | +35.3% | +70.4% | FAIL | 0.45/0.103 |
| w4a16 | 763.3 | −58% | −47% | PASS | (FP4→FP16 dequant 오버헤드로 느림) |
| w4a16 + spec | 936.8 | −48% | −35% | FAIL | |
| spinquant(회전=란초스②) | BOOT_FAIL | — | — | — | `Online transforms+TP 미지원`(TP=1 필요) |
| fp8 (ref) | 1810.1 | — | +25.9% | PASS | 0.135/0.042 |
| bf16 (ref) | 1437.2 | — | — | PASS | — |

**판정**:
- **W4A4 NVFP4(사전양자화)가 FP8을 +23% 넘고 게이트 통과** → byte-level 천장이 FP8→FP4로 갱신.
  단 정확도 여유는 FP8보다 작음(diff 0.43 vs 0.135, 게이트 경계 근접).
- **W4A4+spec = +194.5%(bf16 대비) 최고속**이나 ppl_rel 0.128>0.1로 근소 FAIL → **rotation
  (SpinQuant/란초스②)로 정확도 살리면 게이트 통과 가능** = 다음 타깃(TP=1로 회전 테스트).
- W4A16(FP4가중치/FP16활성)은 dequant 오버헤드로 **오히려 느림** → 속도엔 W4A4(가중치+활성 FP4).
산출물: `fp4_sweep.sh`, `runs/fp4_results.csv`, `runs/lp_w4a4*.json`.
