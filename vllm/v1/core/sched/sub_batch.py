# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NEO-style SubBatch + BatchPerfData abstractions.

Adapted from NEO (https://github.com/NEO-MLSys25/NEO, MLSys 2025,
Apache 2.0). Algorithms only — no code copied.

A ``SubBatch`` groups four kinds of requests:

* ``gprf_reqs`` — GPU prefilling
* ``cprf_reqs`` — CPU prefilling (rare)
* ``gdec_reqs`` — GPU decoding
* ``cdec_reqs`` — CPU decoding

The ``BatchPerfData`` accessor exposes ``linr_T``, ``pref_T``,
``gdec_T``, ``cdec_T``, ``gpu_time``, and ``cpu_time`` which the
scheduler's ``_decide_mode_and_gen_batch`` uses to balance the two
sub-batches in the asymmetric pipeline.

See ``shadow_assists/features/IDE_006/NEO_code_deepdive.md`` §2.
"""

from __future__ import annotations

from typing import Iterable, Protocol

from vllm.v1.core.sched.perfpredictor import (
    PerfPredictor,
    ZeroPerfPredictor,
)


class _ReqLike(Protocol):
    """Minimum protocol the predictor needs from a request."""
    request_id: int
    @property
    def prompt_len(self) -> int: ...
    @property
    def num_tokens(self) -> int: ...   # current sequence length


class BatchPerfData:
    """Accumulates the four NEO time predictions for a SubBatch.

    Counters
    --------
    x   — total request count
    s   — iteration width (total tokens flowing through linear layers)
    n_g — total GPU-decode KV tokens (sum of seq_len for gdec_reqs)
    x_c — CPU-decode request count
    n_c — total CPU-decode KV tokens

    Time predictions
    ----------------
    pref_T  — accumulated GPU prefill time (sum of per-prefill predictions)
    gdec_T  — GPU decode time (single prediction over n_g)
    lnch_T  — kernel launch overhead (constant per batch)
    linr_T  — derived from ``s`` via ``predictor.get_linr_T``
    cdec_T  — derived from (x_c, n_c) via ``predictor.get_cdec_T``
    """

    def __init__(self, predictor: PerfPredictor | None = None) -> None:
        self.predictor: PerfPredictor = predictor or ZeroPerfPredictor()
        self.x = 0
        self.s = 0
        self.n_g = 0
        self.x_c = 0
        self.n_c = 0
        self.pref_T = 0.0
        self.gdec_T = 0.0
        self.lnch_T = self.predictor.get_lnch_T()

    # ── add / pop ────────────────────────────────────────────────
    def add_pref(self, prompt_len: int) -> None:
        self.x += 1
        self.s += prompt_len
        self.pref_T += self.predictor.get_pref_T(prompt_len)

    def pop_pref(self, prompt_len: int) -> None:
        self.x -= 1
        self.s -= prompt_len
        self.pref_T -= self.predictor.get_pref_T(prompt_len)

    def add_gdec(self, seq_len: int) -> None:
        self.x += 1
        self.s += 1
        self.n_g += seq_len
        self.gdec_T = self.predictor.get_gdec_T(self.n_g)

    def pop_gdec(self, seq_len: int) -> None:
        self.x -= 1
        self.s -= 1
        self.n_g -= seq_len
        self.gdec_T = self.predictor.get_gdec_T(self.n_g)

    def add_cdec(self, seq_len: int) -> None:
        self.x += 1
        self.s += 1
        self.x_c += 1
        self.n_c += seq_len

    def pop_cdec(self, seq_len: int) -> None:
        self.x -= 1
        self.s -= 1
        self.x_c -= 1
        self.n_c -= seq_len

    # ── derived properties ────────────────────────────────────────
    @property
    def linr_T(self) -> float:
        return self.predictor.get_linr_T(self.s)

    @property
    def cdec_T(self) -> float:
        return self.predictor.get_cdec_T(self.x_c, self.n_c)

    @property
    def gpu_time(self) -> float:
        return self.linr_T + self.pref_T + self.gdec_T

    @property
    def cpu_time(self) -> float:
        return self.cdec_T + self.lnch_T


class SubBatch:
    """A NEO-style sub-batch.

    Holds four sub-lists of requests, indexed by where they will run
    (GPU prefill / CPU prefill / GPU decode / CPU decode), and a
    ``BatchPerfData`` that tracks predicted timings as members are
    added or removed.
    """

    def __init__(self, predictor: PerfPredictor | None = None) -> None:
        self.predictor = predictor or ZeroPerfPredictor()
        self.gprf_reqs: list[_ReqLike] = []
        self.cprf_reqs: list[_ReqLike] = []
        self.gdec_reqs: list[_ReqLike] = []
        self.cdec_reqs: list[_ReqLike] = []
        self.perfdata = BatchPerfData(self.predictor)

    def __len__(self) -> int:
        return self.perfdata.x

    # ── add / pop ────────────────────────────────────────────────
    def add_pref(self, req: _ReqLike, *, is_gpu: bool) -> None:
        if is_gpu:
            self.gprf_reqs.append(req)
        else:
            self.cprf_reqs.append(req)
        self.perfdata.add_pref(req.prompt_len)

    def pop_pref(self) -> tuple[_ReqLike, bool]:
        is_gpu = not self.cprf_reqs
        req = self.gprf_reqs.pop() if is_gpu else self.cprf_reqs.pop()
        self.perfdata.pop_pref(req.prompt_len)
        return req, is_gpu

    def add_gdec(self, req: _ReqLike) -> None:
        self.gdec_reqs.append(req)
        self.perfdata.add_gdec(req.num_tokens)

    def pop_gdec(self) -> _ReqLike:
        req = self.gdec_reqs.pop()
        self.perfdata.pop_gdec(req.num_tokens)
        return req

    def add_cdec(self, req: _ReqLike) -> None:
        self.cdec_reqs.append(req)
        self.perfdata.add_cdec(req.num_tokens)

    def pop_cdec(self) -> _ReqLike:
        req = self.cdec_reqs.pop()
        self.perfdata.pop_cdec(req.num_tokens)
        return req

    # ── views ────────────────────────────────────────────────────
    def get_num_prefs(self) -> int:
        return len(self.gprf_reqs) + len(self.cprf_reqs)

    @property
    def num_cprfs(self) -> int:
        return len(self.cprf_reqs)

    @property
    def num_gprfs(self) -> int:
        return len(self.gprf_reqs)

    @property
    def num_gdecs(self) -> int:
        return len(self.gdec_reqs)

    @property
    def num_cdecs(self) -> int:
        return len(self.cdec_reqs)

    @property
    def num_prgds(self) -> int:
        return self.num_gprfs + self.num_cprfs + self.num_gdecs

    @property
    def all_reqs(self) -> list[_ReqLike]:
        # Order matters — the model worker uses this order as the
        # kernel layout (CPU prefill → GPU prefill → GPU decode → CPU decode)
        return [*self.cprf_reqs, *self.gprf_reqs,
                *self.gdec_reqs, *self.cdec_reqs]

    @property
    def gpu_time(self) -> float:
        return self.perfdata.gpu_time

    @property
    def cpu_time(self) -> float:
        return self.perfdata.cpu_time
