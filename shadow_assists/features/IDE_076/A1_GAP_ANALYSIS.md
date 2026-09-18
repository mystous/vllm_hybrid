# A1_GAP_ANALYSIS — 기존 RB 내부의 실제 R=1·2 잔여 비용 (PLAN.md §6.1, A1-0)

작성 2026-09-17 22:58 KST, 23:08 replay 근거 추가

## 1. 대응표 (§6.1)
| 확인 대상 | 현재 소스 증거 |
|---|---|
| RB 진입 조건·symbol | `integer_mat_mul<amx::GemmKernel224Int4, false>` (amx_kernels.hpp:2635) → `K::avx_rb_on()` = env `KT_AVX_RB` 존재 여부(값 무관; OPT4 `=0` → ON) → `avx_kernel_rb` (:1832). compile 분기 `!amx_or_avx && is_same<K, GemmKernel224Int4>`. 설치 .so 에서는 전부 인라인(nm 에 symbol 없음) → 증거는 동일 플래그로 별도 TU 컴파일한 asm (`/tmp/ide076_a1asm.s`, 아래 §3) |
| blocking 축 | row(RB=8/4/2/1) × output(N_STEP 32 = 2×16 lane) × K(K_STEP 64 = 16 k_i × 4 B) ; acc[RB][2][SPLIT] zmm, SPLIT 은 의존 사슬 분리(R=1: 4, R=2: 2) |
| 현재 row batching | **정적**: `avx_kernel_rb` 가 남은 행 수로 8→4→2→1 블록을 선택. R=1 → `avx_rb_rows<1,4>`, R=2 → `<2,2>`, R=3 → `<2,2>`+`<1,4>`. padding 없음(M_STEP 32 안에서 실제 행 수만 순회) |
| W load·unpack·scale | k_i 당 B 로드 2(64 B: lo/hi 32열), 언팩 `and`+`slli`(lo)·`and`(hi) ×2 = 6 op. **행 블록(RB) 안에서만 재사용** → R=1 이면 W 언팩이 행 1개에만 쓰임. 스케일은 K 블록 끝 `apply_scale` (행·열 스케일 곱, 32 lane) |
| gate/up 실행 | 서로 다른 task (task_id2 % 2). 같은 A(입력 BufferA) 를 gate task 와 up task 가 각각 읽음 → 입력 접근 2회 (A 는 행당 K B = 6 KB, L1/L2 상주 → DRAM 재접근은 아님) |
| down 실행 | activation 출력을 `from_mat` 으로 int8 양자화한 BufferA (K=1,280), W down N=6,144 → nth 48 task |
| NUMA shard | 서브풀별 intermediate 1,280 (moe-tp 균등 분할), hidden 6,144. gate/up: K=6,144·N=1,280; down: K=1,280·N=6,144. hidden/intermediate 전체 차원을 tile 로 쓰지 않음 |
| asm 증거 | §3 |

## 2. 실제 shape (IDE_075 expert 표본, bs=63)
서브풀당 unique cold expert p50 17~18, expert 별 rows: ≤2 가 67.7 % (표본), 1~5. 따라서 gate/up·down 의 대부분 task 가 R=1 또는 R=2 블록 1개로 실행된다.

## 3. asm 계수 (g++ 13.3, 서버 빌드와 같은 플래그 `-O3 -ffast-math -march=native …`, 함수 본문 전체)
| 인스턴스 | 총 명령 | vpdpbusd | vpbroadcastd(A) | zmm 메모리 로드 | zmm 저장 | vpand | vpslld | 스택 zmm 참조 | 분기 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `avx_rb_rows<1,4>` (R=1) | 1,026 | 64 | 34 | 42 | 8 | 64 | 32 | 14 | 15 |
| `avx_rb_rows<2,2>` (R=2) | 1,577 | 128 | 66 | 52 | 16 | 64 | 32 | 28 | 15 |
| `avx_rb_rows<4,1>` (R=4) | 2,777 | 256 | 130 | 72 | 32 | 64 | 32 | 56 | 15 |

K_STEP(16 k_i) 당: B 로드 32 + 언팩 96(and 64 + slli 32) 은 R 과 무관하게 고정, dpbusd 64×R, A 브로드캐스트 32×R(+2). 즉 R=1 에서 dot 명령은 전체의 6 %(64/1,026) 이고 **B 언팩(96)·B 로드(32) 가 dot 의 2 배**다. R=2 에서 dot 8 %, R=4 에서 9 %. (`_mm512_dpbssd_epi32` 는 헤더 매크로로 `vpdpbusd` 로 컴파일; A 는 `prepare_a` 로 unsigned 변환, lo 니블은 `<<4` 로 16 배 스케일 후 `apply_scale` 에서 보정.)

## 4. 줄일 수 있는 잔여 비용 후보 (A1 범위 안, §2.3 허용)
| ID | 잔여 비용 | 근거 | 변경 범위 | 반증 조건 |
|---|---|---|---|---|
| A1-a | lo 니블 언팩 2 op(and+slli) → 1 op: GFNI `vgf2p8affineqb` 로 바이트 단위 `<<4` (CPU 지원 확인: cpuinfo gfni) | k_i 당 언팩 6 → 4 op (R=1 총 명령 −6 %) | `avx_rb_rows` 의 B 언팩 4곳 (정밀도·스케일·누산 순서 불변; hi 경로 그대로) | 커널이 DRAM 대역 한계이면 시간 불변 (§5 의 R 의존성 시험으로 판정) |
| A1-b | R=1 에서 SPLIT=4 누산기 4개 합산·초기화(vpaddd 6, zero) 와 acc 스택 spill 14 | 작은 비용; R=1 전용 경로에서 acc 를 zmm 에 고정 | `<1,4>` 대체 함수 | 이득 <1 % 면 기록만 |
| A1-c | gate/up 이 같은 A 를 두 task 에서 브로드캐스트 (k_i 당 2 op ×2) | 입력 접근 공유 subvariant (§6.5): 한 task 가 gate·up 두 W 타일을 처리하며 A 브로드캐스트 1회 → 명령 −2/k_i, task 수 절반(장벽 안 task 20→10/expert) | `forward_prefill` S2 task 구성 + `do_gate_up_gemm` fused 변형 (W·scale 분리 유지, activation 위치 불변) | 병렬성 감소(task 10/expert × 18 = 180 < 워커 48 ×?) 로 tail 악화 시 기각 |
| A1-d | `to_mat` 별도 pass (C tile int32→fp32→bf16, N 범위 재순회) | apply_scale 뒤 별도 루프에서 C 를 다시 읽음 | apply_scale 에 bf16 변환 융합 (수치: fp32 곱 후 bf16 반올림 순서 동일) | 이득 <1 % 면 기록만 |

## 5. 판정 계획 (A1-0 → A1-1)
1. expert replay (`eval/ide076/expert_comparator.py`, 같은 pool 구성 96/2): R=1·2·3 × unique 160(전 expert)·16(서버 유사) 에서 forward 시간. **R=2 시간 ≈ R=1 시간이면 W 스트리밍(DRAM/L3) 한계**, R 에 비례하면 명령 한계 → A1-a/b 의 기대 이득 상한을 정한다.
2. RB OFF(unset) 대조는 기존 RB 의 기여 크기 확인용(정식 대조군 아님).
3. 후보 A1-a 부터 구현 → tile comparator(동일 descriptor 출력 bitwise 비교, 정수 누산이므로 bit 동일 기대) → job replay → CORR/OFF MAIN.

## 6. expert replay 근거 (v4, RB ON, 96 스레드/NUMA 2, 난수 가중치, cache-hot 반복; `replay_rb_on`, `replay_nscale`, 2026-09-17 23:00)
| 시나리오 | unique expert | p50 (µs) | expert 당 (µs) | 서브풀 W 스트리밍 (GB/s/socket) |
|---|---:|---:|---:|---:|
| R=1, n=160 | 160 | 9,890 | 61.8 | 191 |
| R=2, n=160 | 160 | 10,586 | 66.2 | 178 |
| R=3, n=160 | 160 | 12,116 | 75.7 | 156 |
| R=1, n=16 (서버 유사) | 16 | 830 | 51.9 | 228 |
| R=2, n=16 | 16 | 956 | 59.7 | 198 |
| 무작위 top-8, qlen 64 (rows 1~5 혼합) | 159 | 11,918 | 75.0 | 157 |
| n 스케일 R=1: 8 / 16 / 32 / 64 / 128 | | 325 / 771 / 1,925 / 4,068 / 7,839 | 40.6 / 48.2 / 60.2 / 63.6 / 61.2 | 290 / 245 / 196 / 186 / 193 |

- **R=1→R=2 는 +7 % (n=160), +15 % (n=16); R=1→R=3 +22 %.** 행 하나를 더해도 시간이 거의 늘지 않으므로 R=1 경로의 시간은 W 스트리밍(로드·언팩·DRAM/L3 대역)이 지배하고, 행당 계산분은 expert 당 ≈4~8 µs 이다. n=8/16 에서 expert 당 시간이 낮은 것은 반복 실행에서 W(8×11.8 = 94 MB/NUMA) 가 L3 에 부분 상주하기 때문(대역 290 GB/s 는 DRAM 아님).
- n≥32 의 선형 적합 기울기 ≈ 62 µs/expert, 절편 ≈ 0 (1,925 − 32×60.2 ≈ −1) → **job 당 dispatch·장벽 고정비는 격리 replay 에서 50 µs 미만** (A2 §7.1 참고).
- RB OFF 대조: R=1 n=160 10,780 (+9 %), n=16 1,058 (+27 %), R=3 13,603 (+12 %) → 기존 RB 의 기여. 출력은 RB ON/OFF bit 동일 (6 시나리오 전부 `torch.equal`).
- 서버(S29): 서브풀 unique 12~18, service 905~942 µs, up_gate 497 + down 304 → expert 당 ≈52 µs 로 replay n=16 (51.9) 과 일치. 즉 서버의 Cold 서비스도 W 스트리밍 한계(≈230 GB/s/socket) 에 있다.

## 7. 잠정 결론 (A1-0)
- 기존 RB 는 R=1/2 정적 블록을 이미 갖고 있다(ALREADY_IMPLEMENTED: row blocking). 남은 비용 중 명령 수준(언팩·브로드캐스트·spill)의 절감은 **W 스트리밍이 지배하는 구간에서 시간으로 드러나기 어렵다**: R 을 1→2 로 늘려도 +7 % 뿐이므로 R=1 시간의 대부분은 W 바이트 처리량에 묶여 있다. A1-a(GFNI 언팩, 명령 −6 %) 의 기대 이득 상한은 그 계산 몫(≈7~15 %) 의 일부다.
- 그래도 §6.7 대로 A1-a 를 구현·측정한다 (bitwise 동일 요구, replay·CORR·OFF MAIN 순). 이득이 판정 한도(2 %) 미만이면 `NO_MEANINGFUL_GAIN` 또는 `KERNEL_ONLY_GAIN` 으로 기록하고 기본 OFF.
- 언팩 자체를 없애는 W 형식 변경(int8 저장, 2× 바이트) 은 §2.3 금지(형식 변경·DRAM 증가) → 제외.

## 8. A1-a 구현·replay 결과 (v5 941a83c4…, 2026-09-17 23:11)
| 시나리오 | v4 (RB ON) | v5 A1=0 | v5 A1=1 | A1 on/off |
|---|---:|---:|---:|---:|
| R=1 n160 | 9,890 | 9,951 | 9,967 | +0.2 % |
| R=1 n16 | 830 | 838 | 837 | 0 |
| R=2 n160 | 10,586 | 11,190 | 11,021 | −1.5 % |
| R=2 n16 | 956 | 1,051 | 1,030 | −2.0 % |
| R=3 n160 | 12,116 | 12,865 | 12,784 | −0.6 % |
| 무작위 top-8 | 11,918 | 12,331 | 12,128 | −1.6 % |
출력: a1on == a1off == v4 bitwise (6 시나리오). proof 카운터: 112,640 eligible / 112,640 r1 / fallback 0 (hot_r1_e18). 실행 간 변동(v4 vs v5 A1=0, 같은 경로) 이 0.6~6 % 이므로 A1-a 의 −0~2 % 는 판정 한도 아래 → E2E OFF MAIN 으로 최종 판정(예상: NO_MEANINGFUL_GAIN 또는 KERNEL_ONLY_GAIN).

## 9. A1-e: 기존 RB 의 W 타일 프리페치 거리 (KT_AVX_PF, 기본 0) — 잔여 비용 = 노출된 W 로드 지연
| 시나리오 | PF=0 (v4) | PF=1 | PF=2 | PF=4 |
|---|---:|---:|---:|---:|
| R=1 n160 | 9,890 | 9,490 (−4.0 %) | 9,437 (−4.6 %) | 9,550 (−3.4 %) |
| R=1 n16 | 830 | 728 (−12.3 %) | 760 (−8.4 %) | 731 (−11.9 %) |
| R=2 n160 | 10,586 | 9,975 (−5.8 %) | 10,006 (−5.5 %) | 10,059 (−5.0 %) |
| R=2 n16 | 956 | 858 (−10.3 %) | 856 (−10.5 %) | 870 (−9.0 %) |
프리페치는 산술·순서를 바꾸지 않으므로 출력 bit 동일(설계상; E2E 변형에서 smoke 로 확인). 서버 유사 규모(n16) 에서 −10~12 % 로 가장 큰 잔여 비용 절감 후보. A1 subvariant 로 E2E 비교 (A1e = PF 1 단독, A1ae = GFNI + PF 1).
