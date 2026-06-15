# SUB_233 — Suffix tree walk SW prefetch (negative result, 2026-06-14)

> **판정: 기각 (no improvement)**. SW prefetch 3 variant 모두 suffix tree walk 를
> 가속하지 못함 (중립~유해). 기본 알고리즘 유지(설치 `_C.so` 원본 복원), 다음 SUB
> 자동 진행 안 함 — 사용자 지시 대기.

## 1. 가설 / 적용점

프로파일(IDE_026 PROFILE_ANALYSIS): pad 적용 후 **suffix draft = 워커 CPU의 새 1위(~32%)**.
suffix tree walk 는 `node->children.find()` → child deref → `_seqs[ref_seq]` 재lookup 의
**dependent-load 포인터 체이스** 이므로 SW prefetch(`__builtin_prefetch`)로 지연 은닉 시도.

코드: `arctic_inference/csrc/suffix_decoding/{suffix_tree.cc,int32_map.h}` (C++ 네이티브,
nanobind `_C.so`). 빌드 경로 확보: nanobind 2.12 + `nb_combined.cpp` + robin_map,
`g++ -O3 -std=c++20` 로 in-place 재컴파일 (무변경 baseline 재빌드 → import·speculate 동등 검증).

## 2. 변형 (3 attempts)

| variant | prefetch 위치 |
|---|---|
| v1 | `_speculate_path`(grandchild head) + `_match_context`(자식 bucket `Int32Map::prefetch`) |
| v2 | `_speculate_path` 만 (2-deep: child→head_child→head_child) |
| v3 | `_match_context` 만 (자식 bucket prefetch) |

## 3. CPU 마이크로벤치 (격리 측정, GPU 무관, taskset -c 0)

대형 트리(400 seq × 400 tok, vocab 32k, 반복 모티프) + 200k speculate 콜, 4회 평균.
drafted 토큰 동일(1,192,308) = 동일 walk. 단위 ns/call (낮을수록 빠름):

| variant | 평균 ns/call | vs baseline |
|---|---:|---:|
| **baseline (재빌드)** | **8,147** | — |
| v1 | 8,326 | **+2.4% 느림** |
| v2 | 8,201 | +0.7% 느림 |
| v3 | 8,310 | +2.0% 느림 |

→ 어떤 변형도 가속 못 함. v3(match-only) +2% = `_match_context` prefetch 가 주 해악.

## 4. end-to-end 70B (K6 pad, taskset 0-47,56-103, 동일 세션·동일 nanobind 빌드)

| corpus | baseline | prefetch(v1) | 델타 |
|---|---:|---:|---:|
| mix | 6,259.2 | 6,264.5 | +0.08% |
| lmsys | 6,213.6 | 6,367.0 | +2.5% |
| swebench | 7,694.1 | 6,861.4 | −10.8% |

swebench tps 는 측정간 6826/7896/6861/7993 = **±17% 고분산**(단일값 신뢰 낮음).
mix·lmsys 평탄 → **명확한 향상 없음**. (기존 fullmatrix baseline 과의 cross-session
비교는 mix −25% 처럼 무효 — taskset 맞춰도 지속, 측정 세션/mix 구성 차이.)

## 5. 진단 — 왜 prefetch 가 안 통하나

walk 1콜 ≈8µs 의 비용은 **포인터 체이스 메모리 지연이 아님**:
1. `speculate()` 가 `match_len = 1..ctx_len` 마다 `_match_context` 로 **root 부터 재탐색**
   → O(ctx²) 해시 probe (재사용 없음).
2. match_len 마다 `Draft` (token_ids/parents/probs `std::vector`) push_back = **힙 할당**.
활성 시퀀스 노드는 직전 `extend` 로 캐시 hot → prefetch 가 은닉할 miss 가 적고,
추가 prefetch 명령/슬롯이 **순수 오버헤드** 가 됨 (특히 _match_context).

**결론**: prefetch 는 이 코드의 병목(할당·재탐색)에 대한 잘못된 레버.
실효 개선은 **알고리즘 변경**(match_len 증분 walk 재사용, Draft 할당 풀링)이라야 하나,
"기본 알고리즘 유지" 제약 하에서는 불가 → SUB_233 기각.

## 6. 재현

```bash
FD=shadow_assists/features/IDE_026_rdt_guarded_harvest/SUB_233_suffix_prefetch
bash $FD/build.sh $FD/src     $FD/_C_prefetch.so   # v1
bash $FD/build.sh $FD/src_v2  $FD/_C_v2.so
bash $FD/build.sh $FD/src_v3  $FD/_C_v3.so
bash $FD/build.sh <pristine-csrc> $FD/_C_baseline.so
for so in _C_baseline _C_prefetch _C_v2 _C_v3; do
  taskset -c 0 .venv/bin/python $FD/microbench.py $FD/$so.so; done
```
산출물: `src/`(v1)·`src_v2/`·`src_v3/`·`microbench.py`·`driver_ab.sh`·`runs/`.
설치 `_C.so` 는 원본 복원 완료 (md5 일치 확인).
