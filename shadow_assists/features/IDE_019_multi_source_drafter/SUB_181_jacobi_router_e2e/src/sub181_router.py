"""SUB_181 — AGSD 4-method router (vanilla / ngram / suffix / cpu_jacobi).

Extends `/tmp/sub094_router.py` with cpu_jacobi backend (CPU AVX-512 Jacobi
draft + GPU verify). Backend selection:
- code   -> trident (ngram K=7)
- sonnet -> trident (suffix)
- chat   -> cpu_jacobi (NEW) [ENV gated]
- fallback -> vanilla

cpu_jacobi backend impl: in-process (no separate :8003 endpoint) — Jacobi
kernel produces K candidate ids on CPU, then HTTP-forwards to vanilla GPU
backend for the actual generation (verify-stage equivalent of "forward target
model for K+1 positions"). Honest: the Jacobi candidates are not actually
verified against vllm spec_decode here (since cpu_jacobi proposer is not yet
integrated into vllm core); this router measures router-level orchestration
overhead + end-to-end latency including Jacobi draft cost.

Set AGSD_USE_JACOBI=1 to activate chat -> cpu_jacobi routing.
Set AGSD_USE_JACOBI=0 to fall back to 3-method (chat -> vanilla, baseline).
"""
import asyncio
import ctypes
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, "/tmp")
from sub094_classifier import classify  # reused

# ===== Backend URLs =====
BACKENDS = {
    "vanilla": os.environ.get("AGSD_VANILLA_URL", "http://127.0.0.1:8001/v1/completions"),
    "trident": os.environ.get("AGSD_TRIDENT_URL", "http://127.0.0.1:8002/v1/completions"),
}
MODEL_NAME = os.environ.get("AGSD_MODEL", "Qwen/Qwen2.5-32B-Instruct")
USE_JACOBI = os.environ.get("AGSD_USE_JACOBI", "0") == "1"
JACOBI_LIB = os.environ.get(
    "AGSD_JACOBI_LIB",
    "/workspace/vllm_hybrid/shadow_assists/features/IDE_019_multi_source_drafter/SUB_180_jacobi_canonical/build/libjacobi_avx512.so",
)
JACOBI_HIDDEN = int(os.environ.get("AGSD_JACOBI_HIDDEN", 5120))
JACOBI_VOCAB = int(os.environ.get("AGSD_JACOBI_VOCAB", 152064))
JACOBI_K = int(os.environ.get("AGSD_JACOBI_K", 5))
JACOBI_THREADS = int(os.environ.get("AGSD_JACOBI_THREADS", 64))
JACOBI_BACKEND_URL = os.environ.get(
    "AGSD_JACOBI_VERIFY_URL", BACKENDS["vanilla"]
)  # Jacobi candidates pass to vanilla GPU for verify (proxy)
N_CLASSIFIER_WORKERS = int(os.environ.get(
    "AGSD_CLASSIFIER_WORKERS", min(16, os.cpu_count() or 8)
))

# ===== 4-method decision rule =====
def decide_4method(workload: str, prefix_len: int, jacobi_healthy: bool) -> str:
    """Workload-aware 4-method router decision."""
    if prefix_len < 16:
        return "vanilla"   # fast-path
    if workload == "code":
        return "trident"
    if workload == "sonnet":
        return "trident"
    if workload == "chat":
        if USE_JACOBI and jacobi_healthy:
            return "cpu_jacobi"
        return "vanilla"
    return "vanilla"


# ===== Jacobi backend state =====
class JacobiState:
    def __init__(self):
        self.lib = None
        self.H_buf = None
        self.W_buf = None
        self.argmax_out = None
        self.maxlogit_out = None
        self.healthy = False

    def init(self):
        try:
            self.lib = ctypes.CDLL(JACOBI_LIB)
            self.lib.jacobi_lm_head_argmax_bf16.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
            ]
            self.lib.jacobi_lm_head_argmax_bf16.restype = None
            # Preallocate buffers for K positions × B=1 (single-request fast path)
            rng = np.random.default_rng(0)
            BK_max = JACOBI_K
            # W is the LM-head; in this orchestration we use random BF16 of correct
            # shape as a cost proxy (the actual W would come from the target model
            # via memory-mapped weight file in a full integration).
            self.H_buf = self._mk_bf16(rng.standard_normal((BK_max, JACOBI_HIDDEN),
                                                          dtype=np.float32) * 0.1)
            self.W_buf = self._mk_bf16(rng.standard_normal((JACOBI_HIDDEN, JACOBI_VOCAB),
                                                          dtype=np.float32) * 0.02)
            self.argmax_out = np.zeros(BK_max, dtype=np.int32)
            self.maxlogit_out = np.zeros(BK_max, dtype=np.float32)
            # Single warmup call to verify kernel works
            self.lib.jacobi_lm_head_argmax_bf16(
                self.H_buf.ctypes.data, self.W_buf.ctypes.data,
                self.argmax_out.ctypes.data, self.maxlogit_out.ctypes.data,
                BK_max, JACOBI_HIDDEN, JACOBI_VOCAB, JACOBI_THREADS,
            )
            self.healthy = True
            print(f"[jacobi] init OK K={JACOBI_K} hidden={JACOBI_HIDDEN}"
                  f" vocab={JACOBI_VOCAB} threads={JACOBI_THREADS}", flush=True)
        except Exception as e:
            self.healthy = False
            print(f"[jacobi] init FAILED: {e}", flush=True)

    @staticmethod
    def _mk_bf16(arr_f32):
        a = arr_f32.astype(np.float32).view(np.uint32)
        bias = 0x7FFF + ((a >> 16) & 1)
        a = (a + bias) >> 16
        return a.astype(np.uint16)

    def draft_step(self) -> float:
        """Run one Jacobi LM-head argmax (single iteration). Returns ms."""
        BK = JACOBI_K
        t0 = time.perf_counter()
        self.lib.jacobi_lm_head_argmax_bf16(
            self.H_buf.ctypes.data, self.W_buf.ctypes.data,
            self.argmax_out.ctypes.data, self.maxlogit_out.ctypes.data,
            BK, JACOBI_HIDDEN, JACOBI_VOCAB, JACOBI_THREADS,
        )
        return (time.perf_counter() - t0) * 1000.0


jacobi = JacobiState()
executor: ProcessPoolExecutor | None = None
http_client: httpx.AsyncClient | None = None
# Concurrency cap for jacobi backend (CPU kernel: thread-bound)
JACOBI_SEM_LIMIT = int(os.environ.get("AGSD_JACOBI_CONCURRENCY", 1))
jacobi_sem: asyncio.Semaphore | None = None

stats: dict = {
    "total": 0,
    "by_backend": {"vanilla": 0, "trident": 0, "cpu_jacobi": 0},
    "by_workload": {"sonnet": 0, "chat": 0, "code": 0},
    "classify_time_total_s": 0.0,
    "forward_time_total_s": 0.0,
    "jacobi_draft_time_total_ms": 0.0,
    "jacobi_calls": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor, http_client, jacobi_sem
    executor = ProcessPoolExecutor(max_workers=N_CLASSIFIER_WORKERS)
    limits = httpx.Limits(max_keepalive_connections=512, max_connections=1024)
    http_client = httpx.AsyncClient(timeout=600.0, limits=limits)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, classify, "warmup")
    if USE_JACOBI:
        jacobi.init()
    jacobi_sem = asyncio.Semaphore(JACOBI_SEM_LIMIT)
    print(f"[router] ready USE_JACOBI={USE_JACOBI} backends={BACKENDS}"
          f" jacobi_healthy={jacobi.healthy}", flush=True)
    yield
    await http_client.aclose()
    executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)


class GenerateReq(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0


@app.post("/generate")
async def generate(req: GenerateReq):
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    wl = await loop.run_in_executor(executor, classify, req.prompt)
    prefix_len = len(req.prompt.split())
    backend = decide_4method(wl, prefix_len, jacobi.healthy)
    t_classify = time.perf_counter() - t0
    stats["total"] += 1
    stats["by_workload"][wl] += 1
    stats["classify_time_total_s"] += t_classify

    # cpu_jacobi: pre-step CPU Jacobi draft (cost proxy), then forward to GPU
    jacobi_ms = 0.0
    if backend == "cpu_jacobi":
        # acquire semaphore (CPU thread-bound). Honest cost: each chat req pays
        # the Jacobi draft cost serially.
        async with jacobi_sem:
            # Run kernel in worker thread to avoid blocking event loop
            jacobi_ms = await loop.run_in_executor(None, jacobi.draft_step)
            stats["jacobi_calls"] += 1
            stats["jacobi_draft_time_total_ms"] += jacobi_ms
        # Forward to vanilla GPU endpoint (verify-equivalent forward)
        url = JACOBI_BACKEND_URL
        stats["by_backend"]["cpu_jacobi"] += 1
    elif backend == "trident":
        url = BACKENDS["trident"]
        stats["by_backend"]["trident"] += 1
    else:
        url = BACKENDS["vanilla"]
        stats["by_backend"]["vanilla"] += 1

    body = {
        "model": MODEL_NAME,
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "seed": req.seed,
    }
    t1 = time.perf_counter()
    try:
        resp = await http_client.post(url, json=body)
        data = resp.json()
        out_text = data.get("choices", [{}])[0].get("text", "")
        out_tokens = data.get("usage", {}).get("completion_tokens", 0)
    except Exception as e:
        return {"error": str(e), "backend": backend, "workload": wl}
    t_forward = time.perf_counter() - t1
    stats["forward_time_total_s"] += t_forward
    return {
        "workload": wl,
        "backend": backend,
        "text": out_text,
        "tokens": out_tokens,
        "classify_ms": t_classify * 1000,
        "jacobi_ms": jacobi_ms,
        "forward_ms": t_forward * 1000,
    }


@app.get("/stats")
async def get_stats():
    n = max(stats["total"], 1)
    nj = max(stats["jacobi_calls"], 1)
    return {
        **stats,
        "classify_ms_avg": stats["classify_time_total_s"] / n * 1000,
        "forward_ms_avg": stats["forward_time_total_s"] / n * 1000,
        "jacobi_ms_avg": stats["jacobi_draft_time_total_ms"] / nj,
        "use_jacobi": USE_JACOBI,
        "jacobi_healthy": jacobi.healthy,
    }


@app.post("/reset")
async def reset_stats():
    stats["total"] = 0
    stats["classify_time_total_s"] = 0.0
    stats["forward_time_total_s"] = 0.0
    stats["jacobi_draft_time_total_ms"] = 0.0
    stats["jacobi_calls"] = 0
    for k in ("by_backend", "by_workload"):
        for sub_k in stats[k]:
            stats[k][sub_k] = 0
    return {"ok": True}


@app.get("/health")
async def health():
    return {"ok": True, "use_jacobi": USE_JACOBI, "jacobi_healthy": jacobi.healthy}


if __name__ == "__main__":
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    port = int(os.environ.get("AGSD_ROUTER_PORT", 8000))
    uvicorn.run(
        "sub181_router:app",
        host="0.0.0.0",
        port=port,
        log_level="warning",
        workers=1,
        loop="uvloop" if "uvloop" in sys.modules else "asyncio",
    )
