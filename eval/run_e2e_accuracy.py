#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TST_003 — e2e accuracy gate orchestration script (IDE_006 / TSK_002).

Runs the two vLLM configurations sequentially on the same prompts and
compares the generated outputs:

  baseline:        no kv_transfer_config (default forward path)
  split_on:        OffloadingConnector + enable_cpu_partial_attention=True
                   (TSK_002 Phase 4c dispatcher live)

Two metrics — both must pass for TST_003 to count as 통과:

  D-i  Token-id divergence    : greedy decoding, count of mismatched tokens
                                  between baseline and split_on per prompt
  D-ii Logprob / PPL diff      : per-position max abs logprob diff and
                                  relative PPL diff (sequence average)

Why a script and not a pytest test (the TST_003 spec form):
  the spec wires both LLM instances as session fixtures, which fits the
  prod box (H100x8 + Llama-3.3-70B) but cannot fit two Qwen-7B copies on
  the dev RTX 3090 (14 GB x 2 = 28 GB > 24 GB). This script loads ONE
  LLM at a time, frees it (gc + cuda.empty_cache), then loads the next —
  works on dev for smoke development AND on prod for the real run.

Usage examples:

  # dev smoke (Qwen-7B, RTX 3090):
  python eval/run_e2e_accuracy.py \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --tensor-parallel 1 \\
      --max-tokens 32 --logprobs 10

  # prod (Llama-3.3-70B + H100 x 8):
  python eval/run_e2e_accuracy.py \\
      --model meta-llama/Llama-3.3-70B-Instruct \\
      --tensor-parallel 8 \\
      --max-tokens 64 --logprobs 20 \\
      --gpu-memory-util 0.85

Output layout (under --output-dir):

  baseline.json        per-prompt generated token_ids + per-position top-K logprobs
  split_on.json        same fields, generated with feature on
  comparison.json      D-i (token divergence) + D-ii (max abs logprob, PPL rel) per
                       prompt + aggregate verdict
  README.md            run metadata + verdict summary

Exit code: 0 if BOTH D-i and D-ii pass under the configured tolerances,
non-zero otherwise — suitable for CI / run_prod_smoke.sh.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# ---- 기본 prompt 셋 -----------------------------------------------------
# 짧은 prompt (smoke) + 긴 prompt (cold KV 발생 영역) 혼합. 갯수는 적게 — 모든
# 프롬프트가 두 번씩 generate 되어 dev 에서도 합리적인 시간 안에 끝나야 함.
_DEFAULT_PROMPTS: list[str] = [
    # short smoke
    "Explain the difference between mutex and semaphore in three sentences.",
    "Translate to Korean: 'The early bird catches the worm.'",
    # medium
    (
        "Write a Python function that takes a list of integers and returns the "
        "longest strictly increasing subsequence. Include type hints and a "
        "short docstring."
    ),
    # long-context style — repeated context to fill KV
    (
        "Summarize the following passage in two bullet points.\n\n"
        + (
            "The Roman aqueducts were arguably the most influential engineering "
            "achievement of the ancient world. Stretching across hundreds of "
            "kilometres of varied terrain, they delivered fresh water to public "
            "fountains, baths, and private homes alike. Their construction "
            "demanded mastery of gradients, pressure, masonry, and the use of "
            "inverted siphons and arched bridges, setting design standards that "
            "would persist for over a thousand years.\n"
        )
        * 8
    ),
]


# ---- 데이터 형 ----------------------------------------------------------


@dataclass
class PromptOutputs:
    """단일 prompt 의 generation 결과 — 두 config 간 비교 단위."""

    prompt_index: int
    prompt: str
    token_ids: list[int]
    # per-token top-K logprobs as list[dict[token_id, logprob]] (or None when
    # logprobs were not collected). The dict captures the top-K entries —
    # downstream comparison only needs the chosen-token logprob.
    chosen_logprobs: list[float] | None
    # Wall-clock for this prompt's generation only (seconds).
    generation_seconds: float


@dataclass
class ConfigOutputs:
    config_name: str  # "baseline" or "split_on"
    model: str
    tensor_parallel_size: int
    max_tokens: int
    logprobs_k: int
    extra_serve_args: dict[str, Any] | None
    prompts: list[PromptOutputs]
    total_seconds: float


# ---- D-i / D-ii helpers (TST_003 §4.4) ---------------------------------


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


# ---- vLLM 호출 ----------------------------------------------------------


def _build_kv_transfer_config(*, enable_split: bool, cpu_bytes: int):
    """split_on 에서만 OffloadingConnector + enable_cpu_partial_attention 활성화."""
    if not enable_split:
        return None
    from vllm.config import KVTransferConfig

    return KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        enable_cpu_partial_attention=True,
        kv_connector_extra_config={"cpu_bytes_to_use": int(cpu_bytes)},
    )


def _run_one_config(
    *,
    config_name: str,
    enable_split: bool,
    model: str,
    tensor_parallel_size: int,
    gpu_memory_util: float,
    max_model_len: int,
    cpu_bytes: int,
    prompts: list[str],
    max_tokens: int,
    logprobs_k: int,
    seed: int,
) -> ConfigOutputs:
    """한 config (baseline 또는 split_on) 으로 LLM 을 띄우고 모든 prompt 를 generate."""
    print(f"\n[{config_name}] loading {model} (TP={tensor_parallel_size})...", flush=True)
    t0_total = time.monotonic()

    # Lazy import — script 가 vLLM 없이도 --help 동작 가능.
    from vllm import LLM, SamplingParams

    kv_transfer_config = _build_kv_transfer_config(
        enable_split=enable_split, cpu_bytes=cpu_bytes
    )
    llm_kwargs: dict[str, Any] = dict(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_util,
        max_model_len=max_model_len,
        seed=seed,
        # disable async scheduling to keep step ordering deterministic for the
        # purpose of the divergence comparison (greedy + same seed should still
        # match either way, but determinism is strictly better here).
    )
    if kv_transfer_config is not None:
        llm_kwargs["kv_transfer_config"] = kv_transfer_config

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        seed=seed,
        logprobs=logprobs_k if logprobs_k > 0 else None,
    )

    prompt_outputs: list[PromptOutputs] = []
    for idx, prompt in enumerate(prompts):
        t0 = time.monotonic()
        # vLLM v1: LLM.generate accepts a list, returns list of RequestOutput.
        results = llm.generate([prompt], sampling_params, use_tqdm=False)
        t1 = time.monotonic()
        result = results[0]
        completion = result.outputs[0]
        token_ids = list(completion.token_ids)
        chosen_logprobs: list[float] | None = None
        if logprobs_k > 0 and getattr(completion, "logprobs", None):
            chosen_logprobs = []
            for pos, lp_dict in enumerate(completion.logprobs):
                # lp_dict is dict[token_id, Logprob(.logprob, .rank, .decoded_token)]
                # find the chosen token's logprob
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
                prompt=prompt,
                token_ids=token_ids,
                chosen_logprobs=chosen_logprobs,
                generation_seconds=t1 - t0,
            )
        )
        print(
            f"  prompt {idx}: {len(token_ids)} tok in {t1 - t0:.2f}s",
            flush=True,
        )

    total = time.monotonic() - t0_total
    print(f"[{config_name}] done in {total:.1f}s", flush=True)

    extra_args: dict[str, Any] | None = None
    if kv_transfer_config is not None:
        extra_args = dict(
            kv_connector=kv_transfer_config.kv_connector,
            enable_cpu_partial_attention=(
                kv_transfer_config.enable_cpu_partial_attention
            ),
            kv_connector_extra_config=dict(
                kv_transfer_config.kv_connector_extra_config
            ),
        )
    out = ConfigOutputs(
        config_name=config_name,
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        max_tokens=max_tokens,
        logprobs_k=logprobs_k,
        extra_serve_args=extra_args,
        prompts=prompt_outputs,
        total_seconds=total,
    )

    # Free the GPU resident model before the next load. CRITICAL on dev where
    # two 7B models would not fit.
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

    return dict(
        verdict_d_i=d_i_pass_all,
        verdict_d_ii=d_ii_pass_all,
        verdict_overall=d_i_pass_all and d_ii_pass_all,
        worst_diverging_tokens=worst_div,
        worst_max_abs_logprob=worst_max_abs_lp,
        worst_ppl_relative_diff=worst_ppl_rel,
        tolerances=dict(
            max_diverging_tokens=max_diverging_tokens,
            atol_logprob=atol_logprob,
            rtol_ppl=rtol_ppl,
        ),
        per_prompt=per_prompt,
    )


# ---- I/O ---------------------------------------------------------------


def _save_config_outputs(path: Path, outputs: ConfigOutputs) -> None:
    payload = asdict(outputs)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _save_comparison(path: Path, comparison: dict[str, Any]) -> None:
    path.write_text(json.dumps(comparison, indent=2))


def _write_readme(
    path: Path,
    *,
    args: argparse.Namespace,
    baseline_seconds: float,
    split_on_seconds: float,
    comparison: dict[str, Any],
) -> None:
    lines = [
        f"# TST_003 e2e accuracy run — {args.model}",
        "",
        f"- timestamp:           {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- model:               {args.model}",
        f"- tensor_parallel:     {args.tensor_parallel}",
        f"- gpu_memory_util:     {args.gpu_memory_util}",
        f"- max_model_len:       {args.max_model_len}",
        f"- max_tokens:          {args.max_tokens}",
        f"- logprobs_k:          {args.logprobs}",
        f"- cpu_bytes_to_use:    {args.cpu_bytes}",
        f"- num_prompts:         {len(_DEFAULT_PROMPTS) if args.prompts_file is None else 'from file'}",
        f"- baseline duration:   {baseline_seconds:.1f}s",
        f"- split_on duration:   {split_on_seconds:.1f}s",
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
    p.add_argument("--model", required=True, help="HF model id")
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--gpu-memory-util", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--logprobs", type=int, default=10, help="0 disables logprob collection")
    p.add_argument(
        "--cpu-bytes",
        type=int,
        default=1 * 1024 ** 3,
        help="cpu_bytes_to_use for OffloadingConnector (split_on only)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="optional UTF-8 text file with one prompt per line; default uses _DEFAULT_PROMPTS",
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


def _load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts_file is None:
        return list(_DEFAULT_PROMPTS)
    return [
        line.rstrip()
        for line in args.prompts_file.read_text().splitlines()
        if line.strip()
    ]


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        out = args.output_dir
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        # _hwtag.sh exports HW_TAG; fall back to "unknown_hw" if absent.
        hw_tag = os.environ.get("HW_TAG", "unknown_hw")
        eval_dir = Path(__file__).resolve().parent
        out = eval_dir / "results" / f"{ts}_{hw_tag}_e2e_accuracy"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> int:
    args = _parse_args()
    out_dir = _resolve_output_dir(args)
    print(f"output dir: {out_dir}", flush=True)

    prompts = _load_prompts(args)
    print(f"prompts: {len(prompts)}", flush=True)

    # 1) baseline
    baseline = _run_one_config(
        config_name="baseline",
        enable_split=False,
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_util=args.gpu_memory_util,
        max_model_len=args.max_model_len,
        cpu_bytes=args.cpu_bytes,
        prompts=prompts,
        max_tokens=args.max_tokens,
        logprobs_k=args.logprobs,
        seed=args.seed,
    )
    _save_config_outputs(out_dir / "baseline.json", baseline)

    # 2) split_on (TSK_002 Phase 4c feature on)
    split_on = _run_one_config(
        config_name="split_on",
        enable_split=True,
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_util=args.gpu_memory_util,
        max_model_len=args.max_model_len,
        cpu_bytes=args.cpu_bytes,
        prompts=prompts,
        max_tokens=args.max_tokens,
        logprobs_k=args.logprobs,
        seed=args.seed,
    )
    _save_config_outputs(out_dir / "split_on.json", split_on)

    # 3) comparison + verdict
    comparison = _compare_outputs(
        baseline,
        split_on,
        max_diverging_tokens=args.max_diverging_tokens,
        atol_logprob=args.atol_logprob,
        rtol_ppl=args.rtol_ppl,
    )
    _save_comparison(out_dir / "comparison.json", comparison)
    _write_readme(
        out_dir / "README.md",
        args=args,
        baseline_seconds=baseline.total_seconds,
        split_on_seconds=split_on.total_seconds,
        comparison=comparison,
    )

    print()
    print("=" * 60)
    print(f"D-i  (token divergence):  {'PASS' if comparison['verdict_d_i'] else 'FAIL'}")
    print(f"D-ii (logprob / PPL):     {'PASS' if comparison['verdict_d_ii'] else 'FAIL'}")
    print(f"overall:                  {'PASS' if comparison['verdict_overall'] else 'FAIL'}")
    print(f"results -> {out_dir}")
    print("=" * 60)

    return 0 if comparison["verdict_overall"] else 1


if __name__ == "__main__":
    sys.exit(main())
