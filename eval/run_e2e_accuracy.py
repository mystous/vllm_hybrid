#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TST_003 — e2e accuracy gate orchestration script (IDE_006 / TSK_002).

Runs two vLLM configurations sequentially on the same prompts and compares
the generated outputs:

  baseline:        spec'd by --baseline-env  (typically vllm_original_long_ctx.env)
  split_on:        spec'd by --split-on-env  (typically ide006_cold_kv_split_on
                   _long_ctx.env — Phase 4c dispatcher live)

Two metrics, both must pass for TST_003 to count as 통과:

  D-i  Token-id divergence  greedy decoding, count of mismatched tokens
                            between baseline and split_on per prompt
  D-ii Logprob / PPL diff   per-position max abs logprob diff and
                            relative PPL diff (sequence average)

Why a script and not a pytest test (the TST_003 spec form):
  the spec wires both LLM instances as session fixtures simultaneously,
  which fits the prod box (H100x8 + Llama-3.3-70B) but cannot fit two
  Qwen-7B copies on the dev RTX 3090 (14 GB x 2 = 28 GB > 24 GB). This
  script loads ONE LLM at a time, frees it (gc + cuda.empty_cache),
  then loads the next — works on dev for smoke development AND on prod
  for the real run.

Why batched submission:
  prompts are fed to ``llm.generate`` as a single list so the V1
  scheduler keeps many in flight. Sequential per-prompt submission
  would never put more than one prompt's KV in the GPU pool at a time,
  which on prod (1.2 M-token GPU KV pool) leaves the cold-tier offload
  path dormant — the run would silently revert to a no-op comparison.
  On dev the batched mode also matches what bench.sh does, so dev and
  prod exercise the same scheduler / prefix-caching behaviour.

Usage:

  # dev smoke (single GPU):
  python eval/run_e2e_accuracy.py \\
      --baseline-env eval/envs/vllm_original.env \\
      --split-on-env eval/envs/ide006_cold_kv_split_on.env \\
      --max-tokens 32 --logprobs 10

  # prod (long-context + Llama-3.3-70B + TP=8):
  python eval/run_e2e_accuracy.py \\
      --baseline-env eval/envs/vllm_original_long_ctx.env \\
      --split-on-env eval/envs/ide006_cold_kv_split_on_long_ctx.env \\
      --max-tokens 64 --logprobs 20

Both env files are sourced via bash and the same MODEL / TENSOR_PARALLEL_SIZE
/ GPU_MEMORY_UTIL / MAX_MODEL_LEN / EXTRA_SERVE_ARGS keys are read that
run.sh / serve.sh use — single source of truth. Decoding controls
(max-tokens, logprobs, prompts, tolerances) stay as script CLI args because
they are e2e-accuracy-specific and don't belong in the bench env.

Output layout (under --output-dir):

  baseline.json    per-prompt token_ids + per-position chosen logprobs
  split_on.json    same fields, generated with feature on
  comparison.json  D-i + D-ii per prompt and aggregate verdict
  README.md        run metadata + verdict summary

Exit code 0 iff both D-i and D-ii pass under the configured tolerances —
suitable for CI / run_prod_smoke.sh.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# ---- prompt 생성 -------------------------------------------------------
# 결정적 long prompts. 같은 (input_len, num_prompts, seed) 조합은 항상 동일
# 텍스트 → baseline / split_on 양쪽 실행에서 동일 입력 보장. 별도 파일 없이
# 코드에 self-contained.

# Base passage — 한 단락이 ~200 단어 / ~270 토큰 (BPE 추정). 반복으로
# 임의 길이까지 채움.
_BASE_PASSAGE = (
    "The Roman aqueducts were arguably the most influential engineering "
    "achievement of the ancient world. Stretching across hundreds of "
    "kilometres of varied terrain, they delivered fresh water to public "
    "fountains, baths, and private homes alike. Their construction "
    "demanded mastery of gradients, pressure, masonry, and the use of "
    "inverted siphons and arched bridges, setting design standards that "
    "would persist for over a thousand years. Even after the Western "
    "Empire's decline, fragments continued to function for centuries, "
    "while later civilisations built upon Roman techniques without "
    "fundamentally improving them. The Pont du Gard in southern France, "
    "built in the first century, still stands as a striking example of "
    "their durability and aesthetic ambition. Beyond the engineering "
    "feat, the aqueducts shaped Roman urban life: cities chose their "
    "locations partly based on water availability, public hygiene "
    "improved, and the rise of public baths became a defining cultural "
    "feature. Some historians argue that no other ancient work of "
    "infrastructure had as enduring an influence on the cities that "
    "succeeded Rome. As we examine the surviving plans and physical "
    "remains today, the precision of measurement and the long-term "
    "planning evident in their layout continue to surprise modern "
    "engineers, even as the larger empire that produced them passed "
    "into history.\n\n"
)


def _build_prompt(
    *, prompt_idx: int, target_chars: int, header_template: str
) -> str:
    """Build a deterministic prompt of approximately ``target_chars`` chars.

    Different ``prompt_idx`` values produce slightly different headers so the
    LLM does not see N copies of the same input (which would also confound
    OffloadingConnector's prefix caching). The body is the same base passage
    repeated to length, which keeps token count / vocabulary stable.
    """
    header = header_template.format(idx=prompt_idx)
    body_target = max(target_chars - len(header), 0)
    if body_target == 0:
        return header
    repetitions = (body_target // len(_BASE_PASSAGE)) + 1
    body = (_BASE_PASSAGE * repetitions)[:body_target]
    return header + body


def _generate_prompts(
    *, num_prompts: int, input_len_tokens: int
) -> list[str]:
    """Generate ``num_prompts`` deterministic prompts targeting roughly
    ``input_len_tokens`` tokens each.

    Uses a char-to-token heuristic (~3.7 chars / token for English) so prompts
    are slightly LONGER than target — vLLM tokenizer truncates to the model's
    max if necessary, but that's fine for our use case since baseline and
    split_on apply the same truncation.
    """
    # English BPE typically yields ~3.7 chars / token; round up for safety.
    target_chars = int(input_len_tokens * 4)
    headers = [
        "[Document #{idx}] Summarise the following passage in three concise bullet points, focusing on its long-term influence.\n\n",
        "[Document #{idx}] Identify the three most important engineering principles described and explain why each matters today.\n\n",
        "[Document #{idx}] Compare the cultural impact described to a modern infrastructure system you are familiar with.\n\n",
        "[Document #{idx}] Extract the chronology of key events implied in the text and present it as a timeline.\n\n",
        "[Document #{idx}] Critique one aspect of the engineering or cultural claims and back your critique with reasoning.\n\n",
    ]
    prompts: list[str] = []
    for i in range(num_prompts):
        header_template = headers[i % len(headers)]
        prompts.append(
            _build_prompt(
                prompt_idx=i,
                target_chars=target_chars,
                header_template=header_template,
            )
        )
    return prompts


# ---- 데이터 형 ----------------------------------------------------------


@dataclass
class EnvConfig:
    """eval/envs/*.env 파일에서 추출한 vLLM 서버 + 워크로드 구성."""

    env_path: Path
    model: str
    tensor_parallel_size: int
    gpu_memory_util: float
    max_model_len: int
    # Workload sizing keys — same as run.sh / bench.sh consume. e2e
    # accuracy reuses these so the test workload matches the env's
    # intended scale (e.g. prod long_ctx env → 10 prompts × 8K input ×
    # 128 output, which is enough on Llama-70B + TP=8 to actually
    # trigger cold KV eviction during the split_on run).
    num_prompts: int
    input_len: int
    output_len: int
    # extra_serve_args is the full string after ``EXTRA_SERVE_ARGS=`` (with
    # quoting already stripped by bash). Currently only the
    # ``--kv-transfer-config={...}`` form is parsed; everything else is
    # treated as opaque and not forwarded (LLM(...) does not accept
    # arbitrary serve flags). The parsed kv_transfer_dict is what we use.
    extra_serve_args: str
    kv_transfer_dict: dict[str, Any] | None


@dataclass
class PromptOutputs:
    """단일 prompt 의 generation 결과 — 두 config 간 비교 단위."""

    prompt_index: int
    prompt: str
    token_ids: list[int]
    # per-token chosen logprobs (or None if logprobs were not collected)
    chosen_logprobs: list[float] | None
    generation_seconds: float


@dataclass
class ConfigOutputs:
    config_name: str  # "baseline" or "split_on"
    env: EnvConfig
    max_tokens: int
    logprobs_k: int
    prompts: list[PromptOutputs]
    total_seconds: float


# ---- D-i / D-ii helpers (TST_003 §4.4 spec 시그니처) -------------------


def count_token_divergence(
    token_ids_a: Sequence[int], token_ids_b: Sequence[int]
) -> int:
    """D-i: 두 token_id 시퀀스가 처음으로 갈라지는 위치부터 끝까지의 토큰 수.

    Greedy decoding 의 비결정성이 있을 수 있으므로 "처음 mismatch 이후의
    모든 token 을 divergent 로 계산" 이 spec. 길이가 다르면 짧은 쪽 끝까지
    매치한 후 남은 긴 쪽 길이가 발산.
    """
    n = min(len(token_ids_a), len(token_ids_b))
    diverge_at = n
    for i in range(n):
        if token_ids_a[i] != token_ids_b[i]:
            diverge_at = i
            break
    tail_a = len(token_ids_a) - diverge_at
    tail_b = len(token_ids_b) - diverge_at
    return max(tail_a, tail_b)


def logprob_ppl_diff(
    chosen_logprobs_a: Sequence[float] | None,
    chosen_logprobs_b: Sequence[float] | None,
) -> tuple[float, float]:
    """D-ii: per-position max abs logprob diff + per-sequence PPL relative diff.

    PPL = exp(-mean(logprob)) — 두 시퀀스의 평균 logprob 차이로부터 직접
    relative diff 산출. 길이가 다르면 짧은 쪽까지만 비교.
    """
    if not chosen_logprobs_a or not chosen_logprobs_b:
        return 0.0, 0.0
    n = min(len(chosen_logprobs_a), len(chosen_logprobs_b))
    if n == 0:
        return 0.0, 0.0
    max_abs = 0.0
    for i in range(n):
        diff = abs(chosen_logprobs_a[i] - chosen_logprobs_b[i])
        if diff > max_abs:
            max_abs = diff
    mean_a = sum(chosen_logprobs_a[:n]) / n
    mean_b = sum(chosen_logprobs_b[:n]) / n
    ppl_a = math.exp(-mean_a)
    ppl_b = math.exp(-mean_b)
    ppl_rel = abs(ppl_a - ppl_b) / max(ppl_a, ppl_b, 1e-12)
    return max_abs, ppl_rel


# ---- env 파싱 ----------------------------------------------------------


_KV_TRANSFER_RE = re.compile(r"--kv-transfer-config=(\{.*\})")


def _source_env_dump(env_path: Path) -> dict[str, str]:
    """``env`` 출력에서 본 env 파일이 set 한 변수들만 추출.

    bash subshell 에서 ``set -a; source <env>; env`` 실행 후 결과 parse.
    이미 부모 환경에 있던 변수는 (어차피 동일하게 다시 set 되므로) 그대로
    들어와도 무관 — ``EnvConfig`` 추출은 특정 키만 읽음.
    """
    cmd = f"set -a; source {shlex.quote(str(env_path))}; env -0"
    proc = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        check=True,
        text=False,
    )
    out: dict[str, str] = {}
    for record in proc.stdout.split(b"\x00"):
        if not record:
            continue
        key, sep, value = record.partition(b"=")
        if not sep:
            continue
        out[key.decode("utf-8", errors="replace")] = value.decode(
            "utf-8", errors="replace"
        )
    return out


def _parse_kv_transfer(extra_serve_args: str) -> dict[str, Any] | None:
    """``EXTRA_SERVE_ARGS`` 안의 ``--kv-transfer-config={...}`` JSON 추출."""
    if not extra_serve_args:
        return None
    m = _KV_TRANSFER_RE.search(extra_serve_args)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"failed to parse --kv-transfer-config JSON from EXTRA_SERVE_ARGS:\n"
            f"  raw: {m.group(1)!r}\n"
            f"  error: {exc}"
        ) from exc


def _load_env_config(env_path: Path) -> EnvConfig:
    raw = _source_env_dump(env_path)

    def _need(key: str) -> str:
        if key not in raw or not raw[key]:
            raise ValueError(
                f"{env_path}: missing required key {key} (set in env file)"
            )
        return raw[key]

    extra_serve_args = raw.get("EXTRA_SERVE_ARGS", "") or ""
    kv_transfer_dict = _parse_kv_transfer(extra_serve_args)
    return EnvConfig(
        env_path=env_path,
        model=_need("MODEL"),
        tensor_parallel_size=int(_need("TENSOR_PARALLEL_SIZE")),
        gpu_memory_util=float(_need("GPU_MEMORY_UTIL")),
        max_model_len=int(_need("MAX_MODEL_LEN")),
        num_prompts=int(_need("NUM_PROMPTS")),
        input_len=int(_need("INPUT_LEN")),
        output_len=int(_need("OUTPUT_LEN")),
        extra_serve_args=extra_serve_args,
        kv_transfer_dict=kv_transfer_dict,
    )


def _validate_baseline_split_pair(baseline: EnvConfig, split_on: EnvConfig) -> None:
    """두 config 가 같은 모델/TP/max_model_len/워크로드 인지 — 정확도 비교의 전제."""
    mismatches: list[str] = []
    if baseline.model != split_on.model:
        mismatches.append(f"MODEL: {baseline.model!r} vs {split_on.model!r}")
    if baseline.tensor_parallel_size != split_on.tensor_parallel_size:
        mismatches.append(
            f"TENSOR_PARALLEL_SIZE: {baseline.tensor_parallel_size} vs "
            f"{split_on.tensor_parallel_size}"
        )
    if baseline.max_model_len != split_on.max_model_len:
        mismatches.append(
            f"MAX_MODEL_LEN: {baseline.max_model_len} vs {split_on.max_model_len}"
        )
    if baseline.num_prompts != split_on.num_prompts:
        mismatches.append(
            f"NUM_PROMPTS: {baseline.num_prompts} vs {split_on.num_prompts}"
        )
    if baseline.input_len != split_on.input_len:
        mismatches.append(
            f"INPUT_LEN: {baseline.input_len} vs {split_on.input_len}"
        )
    if baseline.output_len != split_on.output_len:
        mismatches.append(
            f"OUTPUT_LEN: {baseline.output_len} vs {split_on.output_len}"
        )
    if mismatches:
        raise ValueError(
            "baseline / split_on env mismatch — token-divergence comparison "
            "requires identical model / TP / shape / workload:\n  - "
            + "\n  - ".join(mismatches)
        )


def _harmonise_gpu_memory_util(
    baseline: EnvConfig, split_on: EnvConfig
) -> tuple[EnvConfig, float]:
    """두 config 의 GPU_MEMORY_UTIL 을 동일값으로 맞춰 반환.

    baseline env 는 보통 OffloadingConnector 가 없어 더 높은 GPU_MEMORY_UTIL
    (예: 0.9) 을 쓰고, split_on env 는 connector 의 staging 영역을 위해 더
    낮은 값 (예: 0.85) 을 씁니다. 그대로 비교하면 두 config 가 받는 num_gpu_
    blocks 가 달라져 batch / scheduling 패턴 변화로 greedy decoding 결과가
    토큰 단위로 발산할 수 있음 — 알고리즘 변화 (Phase 4c cold path) 와
    독립된 noise. 따라서 e2e 정확도 비교에서는 *둘 중 더 작은 값* 으로
    baseline 을 클램프해 fair 비교를 강제합니다.
    """
    target = min(baseline.gpu_memory_util, split_on.gpu_memory_util)
    if baseline.gpu_memory_util != target or split_on.gpu_memory_util != target:
        print(
            f"[harmonise] aligning GPU_MEMORY_UTIL → {target:.3f} "
            f"(baseline env had {baseline.gpu_memory_util:.3f}, "
            f"split_on env had {split_on.gpu_memory_util:.3f}). The lower "
            "value is used for both so KV block budget is identical and "
            "any divergence is attributable to the cold path itself.",
            flush=True,
        )
    if baseline.gpu_memory_util != target:
        baseline = EnvConfig(
            env_path=baseline.env_path,
            model=baseline.model,
            tensor_parallel_size=baseline.tensor_parallel_size,
            gpu_memory_util=target,
            max_model_len=baseline.max_model_len,
            num_prompts=baseline.num_prompts,
            input_len=baseline.input_len,
            output_len=baseline.output_len,
            extra_serve_args=baseline.extra_serve_args,
            kv_transfer_dict=baseline.kv_transfer_dict,
        )
    return baseline, target


# ---- vLLM 호출 ----------------------------------------------------------


def _kv_transfer_config_from_dict(d: dict[str, Any]):
    """env JSON dict → KVTransferConfig instance (lazy import)."""
    from vllm.config import KVTransferConfig

    return KVTransferConfig(
        kv_connector=d.get("kv_connector"),
        kv_role=d.get("kv_role"),
        enable_cpu_partial_attention=bool(
            d.get("enable_cpu_partial_attention", False)
        ),
        kv_connector_extra_config=dict(d.get("kv_connector_extra_config") or {}),
    )


def _run_one_config(
    *,
    config_name: str,
    env: EnvConfig,
    prompts: list[str],
    max_tokens: int,
    logprobs_k: int,
    seed: int,
) -> ConfigOutputs:
    """한 config 로 LLM 을 띄우고 모든 prompt 를 generate."""
    print(
        f"\n[{config_name}] env={env.env_path.name} model={env.model} "
        f"TP={env.tensor_parallel_size} max_model_len={env.max_model_len}",
        flush=True,
    )
    print(
        f"[{config_name}]   kv_transfer_config="
        + ("None" if env.kv_transfer_dict is None else json.dumps(env.kv_transfer_dict)),
        flush=True,
    )

    t0_total = time.monotonic()

    # Lazy import — script 가 vLLM 없이도 --help 동작 가능.
    from vllm import LLM, SamplingParams

    llm_kwargs: dict[str, Any] = dict(
        model=env.model,
        tensor_parallel_size=env.tensor_parallel_size,
        gpu_memory_utilization=env.gpu_memory_util,
        max_model_len=env.max_model_len,
        seed=seed,
    )
    if env.kv_transfer_dict is not None:
        llm_kwargs["kv_transfer_config"] = _kv_transfer_config_from_dict(
            env.kv_transfer_dict
        )

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        seed=seed,
        logprobs=logprobs_k if logprobs_k > 0 else None,
    )

    # Batched submission — feed *all* prompts to llm.generate at once so the
    # V1 scheduler queues them concurrently. Sequential per-prompt calls keep
    # only one prompt's KV resident at a time, which on prod (H100×8 + Llama-
    # 3.3-70B + TP=8) leaves the 1.2 M-token GPU KV pool 95 %+ empty and the
    # cold-tier offload path never fires — the very thing we are trying to
    # validate. With batched submission and prefix caching, cumulative KV
    # demand crosses the pool threshold and partial-attention actually
    # activates during the split_on run.
    print(
        f"  [{config_name}] batched generate: {len(prompts)} prompts in flight",
        flush=True,
    )
    t0_gen = time.monotonic()
    results = llm.generate(prompts, sampling_params, use_tqdm=True)
    t1_gen = time.monotonic()
    gen_seconds = t1_gen - t0_gen

    # results may not be in input order on all backends — re-sort by request_id
    # is unnecessary for V1 LLM (preserves input order), but defensively map by
    # index using the prompt string. The V1 LLM in fact returns in input order
    # so we can rely on enumerate.
    prompt_outputs: list[PromptOutputs] = []
    avg_per_prompt = gen_seconds / max(len(prompts), 1)
    for idx, request_output in enumerate(results):
        completion = request_output.outputs[0]
        token_ids = list(completion.token_ids)
        chosen_logprobs: list[float] | None = None
        if logprobs_k > 0 and getattr(completion, "logprobs", None):
            chosen_logprobs = []
            for pos, lp_dict in enumerate(completion.logprobs):
                if pos < len(token_ids):
                    tok_id = token_ids[pos]
                    lp_obj = lp_dict.get(tok_id) if lp_dict is not None else None
                    if lp_obj is not None:
                        chosen_logprobs.append(float(lp_obj.logprob))
                    else:
                        chosen_logprobs.append(float("nan"))
        prompt_outputs.append(
            PromptOutputs(
                prompt_index=idx,
                prompt=request_output.prompt,
                token_ids=token_ids,
                chosen_logprobs=chosen_logprobs,
                # Per-prompt timing is not directly observable in batched
                # mode; report the avg as a coarse hint. Total wall is the
                # meaningful number — use ``total_seconds`` on ConfigOutputs.
                generation_seconds=avg_per_prompt,
            )
        )
    print(
        f"  [{config_name}] batched generate complete: "
        f"{len(prompts)} prompts in {gen_seconds:.1f}s "
        f"(avg {avg_per_prompt:.2f}s/prompt)",
        flush=True,
    )

    total = time.monotonic() - t0_total
    print(f"[{config_name}] done in {total:.1f}s", flush=True)

    out = ConfigOutputs(
        config_name=config_name,
        env=env,
        max_tokens=max_tokens,
        logprobs_k=logprobs_k,
        prompts=prompt_outputs,
        total_seconds=total,
    )

    # Free GPU resident model before the next load — critical on dev where
    # two large models would not fit.
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    return out


# ---- comparison + verdict ----------------------------------------------


def _compare_outputs(
    baseline: ConfigOutputs,
    split_on: ConfigOutputs,
    *,
    max_diverging_tokens: int,
    atol_logprob: float,
    rtol_ppl: float,
) -> dict[str, Any]:
    """Compare baseline vs split_on per-prompt outputs and emit verdict.

    A note on the "all-zero diff" guard. If the Cold-KV CPU partial
    attention dispatcher ever silently bypasses (connector mis-wiring,
    KV-pool not under pressure, etc.) the split_on run reduces to the
    same GPU-only arithmetic as baseline and every prompt emerges
    bit-identical (worst_max_abs_logprob == 0.0). This SHOULD trip an
    alarm rather than a green PASS, because BF16 numerical non-
    associativity makes a real partial-attention run incapable of
    being bit-equal to baseline. We catch that case here and degrade
    the verdict to a FAIL with an explicit ``suspicious_no_cold_path``
    flag — this was the false-pass mode we hit in the prod runs of
    20260426 (commits ``9ebf3781``, ``496960a7``, ``2a18e016``,
    ``23cda4b9``) before the connector in-flight ready-signal fix.
    """
    assert len(baseline.prompts) == len(split_on.prompts)
    per_prompt: list[dict[str, Any]] = []
    d_i_pass_all = True
    d_ii_pass_all = True
    worst_div = 0
    worst_max_abs_lp = 0.0
    worst_ppl_rel = 0.0

    for b, s in zip(baseline.prompts, split_on.prompts):
        n_div = count_token_divergence(b.token_ids, s.token_ids)
        max_abs_lp, ppl_rel = logprob_ppl_diff(b.chosen_logprobs, s.chosen_logprobs)
        d_i_pass = n_div <= max_diverging_tokens
        d_ii_pass = (max_abs_lp < atol_logprob) and (ppl_rel < rtol_ppl)
        d_i_pass_all = d_i_pass_all and d_i_pass
        d_ii_pass_all = d_ii_pass_all and d_ii_pass
        worst_div = max(worst_div, n_div)
        worst_max_abs_lp = max(worst_max_abs_lp, max_abs_lp)
        worst_ppl_rel = max(worst_ppl_rel, ppl_rel)
        per_prompt.append(
            dict(
                prompt_index=b.prompt_index,
                len_baseline=len(b.token_ids),
                len_split_on=len(s.token_ids),
                n_diverging_tokens=n_div,
                max_abs_logprob=max_abs_lp,
                ppl_relative_diff=ppl_rel,
                d_i_pass=d_i_pass,
                d_ii_pass=d_ii_pass,
            )
        )

    # Suspicious-PASS guard: bit-identical outputs across all prompts +
    # logprobs == cold path silently did not fire. Logprobs collection
    # may be off (logprobs_k == 0), in which case max_abs_lp / ppl_rel
    # are 0 by construction and the guard cannot tell — only enforce
    # when logprobs were collected.
    logprobs_collected = (
        baseline.logprobs_k > 0
        and split_on.logprobs_k > 0
        and any(p.chosen_logprobs for p in baseline.prompts)
        and any(p.chosen_logprobs for p in split_on.prompts)
    )
    suspicious_no_cold_path = (
        logprobs_collected
        and worst_max_abs_lp == 0.0
        and worst_ppl_rel == 0.0
    )
    if suspicious_no_cold_path:
        d_i_pass_all = False  # force overall FAIL
        d_ii_pass_all = False

    return dict(
        verdict_d_i=d_i_pass_all,
        verdict_d_ii=d_ii_pass_all,
        verdict_overall=d_i_pass_all and d_ii_pass_all,
        worst_diverging_tokens=worst_div,
        worst_max_abs_logprob=worst_max_abs_lp,
        worst_ppl_relative_diff=worst_ppl_rel,
        # IDE_006 / TSK_002 false-pass detector. True iff baseline and
        # split_on produced bit-identical logprobs across every prompt —
        # impossible for a real partial-attention run on BF16, so this
        # flag indicates the cold path was silently bypassed and the
        # verdict was downgraded to FAIL.
        suspicious_no_cold_path=suspicious_no_cold_path,
        tolerances=dict(
            max_diverging_tokens=max_diverging_tokens,
            atol_logprob=atol_logprob,
            rtol_ppl=rtol_ppl,
        ),
        per_prompt=per_prompt,
    )


# ---- I/O ---------------------------------------------------------------


def _serializable(obj: Any) -> Any:
    """asdict 산출물에서 Path 등 비-JSON-serializable 인 항목을 변환."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serializable(v) for v in obj]
    return obj


def _save_config_outputs(path: Path, outputs: ConfigOutputs) -> None:
    payload = _serializable(asdict(outputs))
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _save_comparison(path: Path, comparison: dict[str, Any]) -> None:
    path.write_text(json.dumps(comparison, indent=2))


def _write_readme(
    path: Path,
    *,
    baseline: ConfigOutputs,
    split_on: ConfigOutputs,
    args: argparse.Namespace,
    comparison: dict[str, Any],
) -> None:
    lines = [
        f"# TST_003 e2e accuracy run",
        "",
        f"- timestamp:           {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- baseline env:        {baseline.env.env_path}",
        f"- split_on env:        {split_on.env.env_path}",
        f"- model:               {baseline.env.model}",
        f"- tensor_parallel:     {baseline.env.tensor_parallel_size}",
        f"- max_model_len:       {baseline.env.max_model_len}",
        f"- max_tokens:          {args.max_tokens}",
        f"- logprobs_k:          {args.logprobs}",
        f"- num_prompts:         {len(baseline.prompts)}",
        f"- baseline duration:   {baseline.total_seconds:.1f}s",
        f"- split_on duration:   {split_on.total_seconds:.1f}s",
        "",
        "## Tolerance",
        f"- MAX_DIVERGING_TOKENS: {args.max_diverging_tokens}",
        f"- ATOL_LOGPROB:         {args.atol_logprob}",
        f"- RTOL_PPL:             {args.rtol_ppl}",
        "",
        "## Verdict",
        f"- D-i  (token divergence):  {'PASS' if comparison['verdict_d_i'] else 'FAIL'} "
        f"(worst = {comparison['worst_diverging_tokens']} tokens)",
        f"- D-ii (logprob / PPL):     {'PASS' if comparison['verdict_d_ii'] else 'FAIL'} "
        f"(worst max_abs={comparison['worst_max_abs_logprob']:.4f}, "
        f"worst ppl_rel={comparison['worst_ppl_relative_diff']:.4f})",
        f"- overall:                  {'PASS' if comparison['verdict_overall'] else 'FAIL'}",
        "",
        "## Files",
        "- baseline.json     — per-prompt outputs from baseline (default forward)",
        "- split_on.json     — per-prompt outputs from feature-on (TSK_002 Phase 4c)",
        "- comparison.json   — full per-prompt + aggregate D-i / D-ii verdict",
        "",
    ]
    path.write_text("\n".join(lines))


# ---- main --------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--baseline-env",
        type=Path,
        required=True,
        help="path to env file for the baseline (typically vllm_original*.env)",
    )
    p.add_argument(
        "--split-on-env",
        type=Path,
        required=True,
        help=(
            "path to env file for the split_on case (typically "
            "ide006_cold_kv_split_on*.env). Must enable_cpu_partial_attention=true "
            "in EXTRA_SERVE_ARGS."
        ),
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="override OUTPUT_LEN from the env file (default: use env's OUTPUT_LEN)",
    )
    p.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help=(
            "cap on the number of prompts to run, regardless of NUM_PROMPTS "
            "in the env file. Useful on dev where the env's NUM_PROMPTS is "
            "tuned for benchmarking and would take too long to compare twice. "
            "Default: use env's NUM_PROMPTS."
        ),
    )
    p.add_argument("--logprobs", type=int, default=10, help="0 disables logprob collection")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--split-on-only",
        action="store_true",
        help=(
            "Skip the baseline LLM load + generation. Only the split_on env "
            "is exercised. Useful for quickly verifying that the Phase 4c "
            "cold-path dispatcher fires (via the [IDE_006 diag ...] log "
            "lines) without the cost of a full TST_003 comparison run."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output dir for baseline.json / split_on.json / comparison.json. "
        "Default: eval/results/<TS>_<HW_TAG>_e2e_accuracy/",
    )
    # Tolerance defaults are intentionally generous for the first run; PLN_001
    # §4.1 will tighten these once we have measurements.
    p.add_argument("--max-diverging-tokens", type=int, default=10)
    p.add_argument("--atol-logprob", type=float, default=0.5)
    p.add_argument("--rtol-ppl", type=float, default=0.10)
    return p.parse_args()


def _resolve_workload(
    args: argparse.Namespace, env: EnvConfig
) -> tuple[int, int, int]:
    """Apply --max-prompts / --max-tokens overrides on top of the env."""
    num_prompts = env.num_prompts
    if args.max_prompts is not None and args.max_prompts < num_prompts:
        num_prompts = max(args.max_prompts, 1)
    max_tokens = args.max_tokens if args.max_tokens is not None else env.output_len
    return num_prompts, env.input_len, max_tokens


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        out = args.output_dir
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        hw_tag = os.environ.get("HW_TAG", "unknown_hw")
        eval_dir = Path(__file__).resolve().parent
        out = eval_dir / "results" / f"{ts}_{hw_tag}_e2e_accuracy"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> int:
    args = _parse_args()
    out_dir = _resolve_output_dir(args)
    print(f"output dir: {out_dir}", flush=True)

    baseline_env = _load_env_config(args.baseline_env)
    split_on_env = _load_env_config(args.split_on_env)
    if not args.split_on_only:
        _validate_baseline_split_pair(baseline_env, split_on_env)
        baseline_env, _harmonised = _harmonise_gpu_memory_util(
            baseline_env, split_on_env
        )
    num_prompts, input_len, max_tokens = _resolve_workload(args, split_on_env)

    if (
        split_on_env.kv_transfer_dict is None
        or not split_on_env.kv_transfer_dict.get("enable_cpu_partial_attention", False)
    ):
        # Defence in depth — TST_003 only makes sense when split is actually
        # on for the split_on env. Misconfigured env should fail loudly.
        raise ValueError(
            f"--split-on-env {args.split_on_env} does not have "
            "enable_cpu_partial_attention=true in its EXTRA_SERVE_ARGS "
            "kv-transfer-config — the e2e accuracy gate would compare "
            "two equivalent configurations and trivially pass. Use the "
            "split_on env (e.g. ide006_cold_kv_split_on_long_ctx.env)."
        )

    prompts = _generate_prompts(num_prompts=num_prompts, input_len_tokens=input_len)
    print(
        f"workload: {num_prompts} prompts × ~{input_len} input tokens × "
        f"{max_tokens} max generated tokens (env NUM_PROMPTS={split_on_env.num_prompts}, "
        f"INPUT_LEN={split_on_env.input_len}, OUTPUT_LEN={split_on_env.output_len})",
        flush=True,
    )

    # 1) baseline (skipped when --split-on-only — execution-path 검증 모드)
    if args.split_on_only:
        print(
            "[split-on-only] skipping baseline LLM load + generation. "
            "Only the split_on env runs — useful for verifying cold-path "
            "dispatch via the [IDE_006 diag ...] log lines without the "
            "cost of a full TST_003 comparison.",
            flush=True,
        )
        baseline = None
    else:
        baseline = _run_one_config(
            config_name="baseline",
            env=baseline_env,
            prompts=prompts,
            max_tokens=max_tokens,
            logprobs_k=args.logprobs,
            seed=args.seed,
        )
        _save_config_outputs(out_dir / "baseline.json", baseline)

    # 2) split_on (TSK_002 Phase 4c feature on)
    split_on = _run_one_config(
        config_name="split_on",
        env=split_on_env,
        prompts=prompts,
        max_tokens=max_tokens,
        logprobs_k=args.logprobs,
        seed=args.seed,
    )
    _save_config_outputs(out_dir / "split_on.json", split_on)

    # split-on-only 모드: comparison 단계는 baseline 이 없어 의미 없음.
    # split_on.json + 진단 로그만 보고 path 검증 종료.
    if args.split_on_only:
        print()
        print("=" * 60)
        print("split-on-only run complete — comparison skipped.")
        print(
            "Check the run log for '[IDE_006 diag ...]' lines to confirm "
            "whether the cold-path dispatcher actually fired."
        )
        print(f"results -> {out_dir}")
        print("=" * 60)
        return 0

    # 3) comparison + verdict
    comparison = _compare_outputs(
        baseline,
        split_on,
        max_diverging_tokens=args.max_diverging_tokens,
        atol_logprob=args.atol_logprob,
        rtol_ppl=args.rtol_ppl,
    )
    _save_comparison(out_dir / "comparison.json", comparison)
    # max_tokens used (after override) is reflected in baseline.max_tokens already
    _write_readme(
        out_dir / "README.md",
        baseline=baseline,
        split_on=split_on,
        args=args,
        comparison=comparison,
    )

    print()
    print("=" * 60)
    print(f"D-i  (token divergence):  {'PASS' if comparison['verdict_d_i'] else 'FAIL'}")
    print(f"D-ii (logprob / PPL):     {'PASS' if comparison['verdict_d_ii'] else 'FAIL'}")
    if comparison.get("suspicious_no_cold_path"):
        print(
            "FAIL reason: baseline and split_on produced bit-identical "
            "logprobs across every prompt — impossible for a real "
            "partial-attention run on BF16. The Cold-KV cold path was "
            "silently bypassed (connector mis-wiring or KV-pool under-"
            "pressured). Verdict force-failed."
        )
    print(f"overall:                  {'PASS' if comparison['verdict_overall'] else 'FAIL'}")
    print(f"results -> {out_dir}")
    print("=" * 60)

    return 0 if comparison["verdict_overall"] else 1


if __name__ == "__main__":
    sys.exit(main())
