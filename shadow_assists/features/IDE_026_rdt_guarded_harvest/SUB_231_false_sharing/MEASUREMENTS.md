# SUB_231 — False sharing / cache-line layout (2026-06-14)

> **판정: 기각 (표적 부재 + 구조적 천장)** — GPU 측정 불요.

## 1. 표적 실재성 확인 (코드 감사)
SUB_231 원안: per-core shard 카운터 + `alignas(128)` 로 코어간 RFO ping-pong 제거.
원 표적 = "ngram 큐 인덱스 · tempo(LHC) 카운터".

**그러나 suffix+pad serving 경로엔 표적이 없음:**
- arctic `suffix_tree.cc/.h`, `int32_map.h` 에 `atomic`/`std::thread`/`mutex`/`omp`/
  `alignas` **전무** → suffix tree 빌드·walk 는 **단일스레드**. false-sharing 은
  다중 코어가 인접 캐시라인을 동시 write 할 때만 발생 → 단일스레드엔 불가.
- ngram proposer 는 본 경로(suffix) 에서 미사용. tempo/metronome(LHC) 는 이미 기각.
- `suffix_decoding.py` 의 `_l3_lock` 은 L3 텔레메트리(VLLM_L3_TREE_SPEC, 기본 off)
  전용 — draft 핫패스 아님.

## 2. 구조적 천장 (SUB_225 가 증명)
설령 표적이 있어도, 70B serving 은 GPU-bound 이고 suffix draft CPU 는 GPU 와
완전 오버랩(critical path 밖). SUB_225 에서 jemalloc 이 walk 를 −17% 가속해도
70B tps 가 평탄(±1%) 했던 것과 동일 — **CPU-side 미세최적화는 70B tps 를 못 올림**.

## 3. 결론
- SUB_231 은 suffix+pad 70B 경로에서 **N/A (표적 없음) + 무효(구조적 천장)** → 기각.
- false-sharing 감사가 의미있는 곳: 실제 다중스레드 공유 카운터가 있고 CPU 가
  critical path 인 regime (예: ngram 다중스레드 proposer, 소형 모델 고conc). 70B 아님.
