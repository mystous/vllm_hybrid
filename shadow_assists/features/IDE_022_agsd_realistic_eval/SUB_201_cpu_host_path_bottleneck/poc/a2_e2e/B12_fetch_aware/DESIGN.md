# Phase B12 — fetch-aware eviction

> 작성일 2026-06-05 KST. SUB_201 A2 lever 의 B11 fetch-evict race (-16.2%) 해결을 목표로 함.

## 1. 문제 정의 (B11 finding 재요약)

Phase B11 (multi-turn workload, prefix hit 97.9%) 측정:

| metric | native | tier+async | Δ |
|---|---:|---:|---:|
| output_tps | 11,042.1 | 9,255.3 | **-16.2%** |
| n_fetch | — | 6,739 | (fetch path 발화 ✅) |
| n_evict_wait_resolved | — | **6,739** | n_fetch 와 1:1 |

**핵심 관찰**: 모든 fetch 가 evict-complete wait 을 거쳐 발생. async-evict 가 forward overlap 으로 evict cost 를 hide 했지만, 그 block 을 곧 다시 fetch 하는 시점에 D2H 미완료 → fetch path 가 sync block 으로 전락 → forward 직렬 stall.

근본 원인: **현재 LRU cold-block eviction policy (FreeKVCacheBlockQueue + free_blocks 순서) 는 "곧 fetch 될 block" 을 무시하고 evict 한다**. multi-turn 워크로드에서 같은 conversation 의 prior-turn block 은 다음 turn 에서 다시 hit 될 가능성이 매우 높음에도, BlockPool 은 단순히 cached 인 block 을 모두 evict_block 으로 넘긴다.

## 2. 옵션 비교

| 옵션 | 설명 | minimality | correctness risk | 기대 효과 |
|---|---|---|---|---|
| **A. recent-fetch protection** | 최근 N step 내 `fetch_block` 으로 끌어올린 block_id 를 sliding-window set 으로 유지, evict_block 호출 직전에 set 멤버이면 skip | **최소 patch** (KVDramTier 에 deque + set, BlockPool 의 free_blocks 분기에 1 줄 추가) | 매우 낮음 — eviction 을 미루는 것이며 메모리 압박 시 자연스럽게 set 슬라이드 | high (B11 finding 의 직접 대응) |
| B. prediction-based | 같은 conversation 의 prior-turn block 보존 | 큼 — conversation id ↔ block id 매핑이 필요, vLLM scheduler 까지 침투 | 중간 — 잘못된 매핑 시 stale block 보존 | unclear (heuristic 의 정합성에 의존) |
| C. fetch-hot tracking | `fetch_hit_count[block_id]`, threshold ≥ N 인 block 보존 | 중간 — counter 유지, decay 정책 필요 | 낮음 — counter 가 stale 되어도 보존만 늘어남 | medium (single-shot fetch hot 분리 필요) |

→ **옵션 A 선택**. B11 의 finding (모든 fetch 가 evict_wait_resolved 와 1:1) 이 가장 직접적이고, patch 도 최소이며, default OFF env flag 로 regression 보호 가능.

## 3. 옵션 A 구체 설계

### 3.1 KVDramTier 변경 (vllm/v1/core/kv_dram_tiering.py)

```python
class KVDramTier:
    def __init__(self, ...):
        ...
        # SUB_201 A2 Phase B12 — fetch-aware eviction.
        # Sliding window of recently fetched block_ids. evict_block
        # consults this set BEFORE allocating DRAM. The window slides
        # on each successful fetch_block: append the new id, drop the
        # oldest if the window is over capacity. Default window N=512
        # blocks ≈ short conversation continuation horizon.
        from collections import deque
        self._recent_fetch_window: deque[int] = deque()
        self._recent_fetch_set: set[int] = set()
        self._fetch_window_max = int(
            os.environ.get("VLLM_KV_TIER_FETCH_WINDOW", "512")
        )
        self._n_evict_skipped_fetch_aware = 0
        # Captured once — hot-path read-only flag.
        self._fetch_aware = _fetch_aware_enabled()

    def _record_fetch(self, block_id: int) -> None:
        # called from fetch_block on success
        if not self._fetch_aware:
            return
        if block_id in self._recent_fetch_set:
            # already tracked — bring to MRU by removing + re-appending
            try:
                self._recent_fetch_window.remove(block_id)
            except ValueError:
                pass
        else:
            self._recent_fetch_set.add(block_id)
        self._recent_fetch_window.append(block_id)
        while len(self._recent_fetch_window) > self._fetch_window_max:
            old = self._recent_fetch_window.popleft()
            self._recent_fetch_set.discard(old)

    def is_fetch_aware_protected(self, block_id: int) -> bool:
        # called from BlockPool.free_blocks BEFORE evict_block dispatch
        if not self._fetch_aware:
            return False
        return block_id in self._recent_fetch_set
```

`fetch_block` 의 성공 분기 끝에 `self._record_fetch(block_id)` 한 줄 추가.

stats 에 `n_evict_skipped_fetch_aware`, `recent_fetch_window` 의 길이를 추가.

### 3.2 BlockPool 변경 (vllm/v1/core/block_pool.py)

`free_blocks` 의 evict 분기에 가드 추가:

```python
        if tier is not None and tier.has_pointer_binding():
            wait_flag = not self._async_evict
            for block in newly_free:
                if block.block_hash is None:
                    continue
                # SUB_201 A2 Phase B12 — fetch-aware eviction guard.
                # If the block was recently fetched (within the rolling
                # window) skip the evict; it is likely to be re-fetched
                # again soon and the evict→wait→fetch race is what cost
                # us -16.2% in B11.
                if tier.is_fetch_aware_protected(block.block_id):
                    continue
                tier.evict_block(block.block_id, wait=wait_flag)
```

`is_fetch_aware_protected` 가 False 면 옵션 A 가 비활성 (default), 즉 기존 B11 path 완전 보존.

### 3.3 env flag

```
VLLM_KV_TIER_FETCH_AWARE=1   # default OFF
VLLM_KV_TIER_FETCH_WINDOW=512  # default 512 blocks
```

`_fetch_aware_enabled()` 는 `_async_evict_enabled` 와 동일한 strict allowlist (`1/true/yes/on`).

## 4. 정합성 보장

| 위협 | 보호 |
|---|---|
| skipped block 이 영원히 evict 안 됨 → DRAM 누수 | 윈도우가 슬라이드하며 자동 제거. 또한 evict 가 skip 되어도 GPU side 의 free_block_queue 에는 정상 등록 (LRU 으로 재할당 시 `_maybe_evict_cached_block` 가 cache 만 제거) → DRAM 누적 무 |
| 메모리 압박 시 fetch-aware 가 OOM 유발 | DRAM tier 가 가득 차면 어차피 evict_block 가 `False` 반환 + skipped_full counter ↑. fetch-aware 는 evict 시도를 줄일 뿐 GPU memory 와 무관 |
| stale `_recent_fetch_set` 가 잘못된 block 보존 | 윈도우 작아 (N=512) 빠르게 노후화. 잘못 보존되어도 다음 free 라운드에서 evict 가능 |
| async fetch 중인 block 이 set 진입 | fetch_block 의 telemetry tick 후 record. wait=True (PoC default in touch) 이므로 record 시점에서 GPU 로 push 완료 |
| concurrent access | 모든 record/check 가 `KVDramTier._lock` 안에서 수행 (이미 evict/fetch 가 lock 보호). 한 번의 추가 lock 획득은 없음 |

## 5. 측정 hypothesis

H0 (null): fetch-aware 활성화는 net tps 에 영향 없음 (Δ ≈ 0).
H1: fetch-aware 활성화는 B11 의 -16.2% 를 **부분 또는 완전 회복**.

### 측정 지표 (3 run × 동일 multi-turn corpus)

| run | flags |
|---|---|
| native | `VLLM_KV_TIERING_DRAM=0` |
| tier+async | `VLLM_KV_TIERING_DRAM=1 VLLM_KV_TIER_ASYNC=1 VLLM_KV_TIER_FETCH_AWARE=0` |
| tier+async+fetch-aware | `VLLM_KV_TIERING_DRAM=1 VLLM_KV_TIER_ASYNC=1 VLLM_KV_TIER_FETCH_AWARE=1` |

수집: tps / TTFT p50/p99 / TPOT p50/p99 / n_evict / n_fetch / evict_bytes / fetch_bytes / n_evict_wait_resolved / n_evict_skipped_fetch_aware (신규).

### 판정 룰

- B11 의 -16.2% 를 0±2pp 이내로 회복 → "fully recovered"
- -16.2% 의 50% 이상 회복 → "partial recovered"
- 차이 < 5%pp → "neutral"
- 더 악화 → "negative"

## 6. 구현 순서

1. `vllm/v1/core/kv_dram_tiering.py` 패치 (env flag + record + check + stats)
2. `vllm/v1/core/block_pool.py` 패치 (guard 한 줄)
3. unittest 신규 (3개): flag 파싱, fetch_block 후 record, free 시 skip
4. 22 + 3 = 25 testcase regression PASS 확인
5. e2e 3 run × multi-turn corpus
6. MEASUREMENTS.md §13 추가
7. commit (main, push 금지)
