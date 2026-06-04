# B1 Detok Lever — Incremental Wrapper + Micro-batch Design

> SUB_201 §5 B1 lever 의 next-step (2일 scope). PoC 1차 (batch_detokenize) 위에
> **token-by-token streaming path** 와 **N concurrent sequence micro-batch path**
> 를 추가하여 실제 vLLM detok flow 에 dispatch 가능한 surface 확보.
>
> 본 문서는 단일 출처 (single source of truth) 로 사용한다. 추후 prod (Sapphire
> Rapids + H100×8) 통합 검증 단계에서 본 결정사항을 그대로 따른다.

---

## 1. 문제 정의

### 1.1 1차 PoC 의 한계

기존 `AVX512Detokenizer.batch_detokenize(token_ids: list[list[int]])` 는:

- **finalized 한 token sequence 전체** 를 받아서 한 번에 raw byte 시퀀스로
  detokenize. byte-equal 21/21 PASS (GPT-2 / Llama-3.1 / Qwen-2.5).
- 그러나 vLLM 의 실제 detok flow 는 **step 마다 새 token 1개** 가 도착하고
  `decode_next(token_id) -> str` 가 incremental 하게 호출됨.
  (`vllm/v1/engine/detokenizer.py:350` 의 `FastIncrementalDetokenizer.decode_next`)
- HF `DecodeStream.step()` 는 multi-byte char 가 token boundary 를 넘으면
  **valid UTF-8 prefix 만 emit** 하고 trailing incomplete byte 는 internal
  buffer 에 hold. AVX-512 wrapper 도 동일 시멘틱을 따라야 native 와 byte-equal.

### 1.2 ByteLevel BPE 의 핵심 관찰

GPT-2 / Llama-3.1 / Qwen-2.5 모두 ByteLevel BPE 사용. 한 token = 0~수십 raw
byte 시퀀스. 예시 (Llama-3.1 의 "안녕"):

```
"안"  = 0xEC 0x95 0x88  (UTF-8 3 byte, 단일 codepoint)
"녕"  = 0xEB 0x85 0x95
```

이게 4개 token 으로 split 될 수 있고, token i 의 raw byte = `0xEC 0x95`
(2 byte) + token i+1 의 raw byte = `0x88 0xEB` (2 byte) 같은 식으로 codepoint
중간에서 끊어진다. 따라서:

- **token 별 raw bytes 만 가지고는 UTF-8 decode 불가** — accumulated byte
  stream 위에서 boundary detection 이 필요.
- 누적 byte buffer 끝의 incomplete codepoint 는 다음 token 도착까지 hold.

---

## 2. UTF-8 Byte Boundary State Machine

ASCII (0x00-0x7F): 1 byte. high bit = 0.
Continuation (0x80-0xBF): 10xxxxxx — multi-byte sequence 의 trailing.
Lead-2 (0xC0-0xDF): 110xxxxx — 2 byte codepoint 의 leading.
Lead-3 (0xE0-0xEF): 1110xxxx — 3 byte codepoint 의 leading.
Lead-4 (0xF0-0xF7): 11110xxx — 4 byte codepoint 의 leading.

**State machine** (incremental_append 가 token 의 raw bytes 를 buffer 끝에
append 한 직후 실행):

```
algorithm split_valid_prefix(buf: bytearray) -> (str_to_emit, bytes_to_hold):
    n = len(buf)
    if n == 0: return ("", b"")
    # back-scan from end to find latest codepoint boundary
    i = n - 1
    while i >= 0 and i > n - 4:
        b = buf[i]
        if b < 0x80:                          # ASCII — boundary AFTER i
            split = i + 1; break
        if b >= 0xC0:                         # lead byte
            need = 2 if b < 0xE0 else (3 if b < 0xF0 else 4)
            if (n - i) >= need:               # complete sequence
                split = n; break
            else:                              # incomplete — boundary BEFORE i
                split = i; break
        i -= 1                                 # continuation, keep scanning
    else:
        # 4 continuation bytes in a row → garbage; flush all (HF does same)
        split = n
    emit_bytes = bytes(buf[:split])
    hold_bytes = bytes(buf[split:])
    try:
        text = emit_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = emit_bytes.decode("utf-8", errors="replace")
    return (text, hold_bytes)
```

핵심 보장:
1. 한 codepoint 가 split 된 채로 emit 되지 않는다 (HF DecodeStream 와 동일).
2. invalid byte 가 들어와도 `errors="replace"` 로 U+FFFD emit → HF 와 align.
3. back-scan 은 최대 4 byte 만 확인 → O(1).

### 2.1 wrapper API: incremental_append

```python
def incremental_append(self, token_id: int) -> str:
    """Append one token's raw bytes and return any newly emit-able text.
    Maintains per-instance byte buffer for multi-byte codepoint reassembly.
    """
```

상태:
- `self._inc_buf: bytearray` — 미emit byte 누적.
- `self._inc_emit_total: int` — 누적 emit char 수 (informational).

flow:
1. token_id → raw bytes (vocab table 의 `pieces[offsets[tid]:offsets[tid+1]]`).
2. `self._inc_buf.extend(raw)`
3. `text, hold = split_valid_prefix(self._inc_buf)`
4. `self._inc_buf = bytearray(hold)`
5. return text

---

## 3. Micro-batch Path

같은 step 내 N concurrent request 의 finalized token sequence 들을 한 번에
처리. 1차 PoC 의 `batch_detokenize` 위에 얇은 `decode_batch` 래퍼:

```python
def decode_batch(self, token_id_lists: list[list[int]]) -> list[str]:
    """Cache-locality optimized batch decode (vocab loaded once).

    For each input sequence:
        1. AVX-512 kernel returns raw bytes per sequence.
        2. UTF-8 decode with errors='replace' (handles invalid prefixes
           safely; same as HF .decode() default).
    """
```

핵심 결정:
- vocab pieces/offsets/sizes 는 instance 에 이미 alive → kernel 호출 한 번에
  모든 seq 처리. 1차 PoC `batch_detokenize` 가 이미 이 구조.
- `decode_batch` 는 단순 wrapper + bytes→str 변환만 책임.
- error policy: `errors="replace"` 로 통일 (HF `decode()` default 와 일치).

성능 의도:
- N seq × seq_len T 의 detok 작업이 N 개 individual kernel call 이 아니라
  **1 kernel call + 1 vocab load** 로 처리. L1/L2 hit ratio 상승 + Python ↔ C
  call overhead amortize.

---

## 4. Correctness Gate

3 model × 30+ prompt × 3 path 비교:

```
HF tokenizer.decode(ids).encode("utf-8")            # ground truth
== AVX512Detokenizer.decode_batch([ids])[0].encode("utf-8")  # full decode
== "".join(incremental_append(t) for t in ids).encode("utf-8")  # token-by-token
```

모든 prompt 가 3-way byte-equal → PASS.

prompt set (per model 동일):
- 영문 일반 (Lorem-style)
- 한글 (multi-byte char 가 token boundary 를 넘는 케이스 의도)
- 일본어/중문 (3-byte codepoint dense)
- emoji (4-byte codepoint)
- code (special char + escape)
- math/markdown (mixed ASCII + symbols)
- 매우 짧은 prompt (1-2 token)
- 매우 긴 prompt (512+ token)
- 영+한+코드 mix
- 동일 prompt × 다양한 seed

총 30+ prompt 확보 후 expand 가능.

---

## 5. vLLM Wiring (minimal)

### 5.1 Env flag

| flag | default | 의미 |
|---|---|---|
| `VLLM_USE_AVX512_DETOK` | 0 | (1차 PoC) shadow-mode full-batch decode |
| `VLLM_USE_AVX512_DETOK_INC` | 0 | **신규** — incremental_append shadow-mode |

`VLLM_USE_AVX512_DETOK_INC=0` (default) 이면 dispatch path 가 완전 no-op.
flag=1 이면 `decode_next` 매 호출마다 shadow-mode 로 `incremental_append` 도
실행 (return 값은 사용 안 함, native stream.step 결과가 authoritative).

### 5.2 hook point

`FastIncrementalDetokenizer._protected_step` 에:

```python
if self._avx512_detok_inc is not None:
    try:
        _ = self._avx512_detok_inc.incremental_append(int(next_token_id))
    except Exception:
        pass
```

기존 `_avx512_detok_b1` (batch shadow) 와 동등한 위치/패턴. 충돌 없음.

### 5.3 e2e boot 검증 제외

vllm boot 는 venv link 문제로 본 worktree 에서 불가. syntax 검증
(`python -c "import vllm.v1.engine.detokenizer"`) 까지만. standalone test 가
incremental path 의 byte-equal correctness 를 모두 cover.

---

## 6. 다음 step (out of scope)

1. e2e benchmark (prod Sapphire Rapids + H100×8, Llama-8B, DS-Qwen-7B):
   detok-only step time vs native — speedup 1.1~1.5× 목표.
2. vllm runtime 통합: shadow → authoritative 전환. native stream.step 을
   replace 하려면 (a) state machine 의 special_token / spaces_between 처리
   동등성 확인, (b) `DecodeStream` 의 invalid-prefix recovery 와 동등한
   reset 로직 추가 필요.
3. async tokenizer worker (SUB_190) 와 결합: detok 을 background CPU
   thread 로 fully offload — 본 step 의 incremental_append API 가 그
   worker 의 inner loop body 로 사용 가능.
