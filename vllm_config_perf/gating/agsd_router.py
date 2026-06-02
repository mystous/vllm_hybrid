"""AGSD end-to-end router — FastAPI + uvloop + ProcessPool + httpx async.

SUB_094 (2026-05-25) 의 production-ready 영역 재구성.

아키텍처:
    client (mixed traffic, concurrency N)
                  │
                  ▼
        ┌─────────────────────────┐
        │  CPU router (본 스크립트)│
        │  - FastAPI + uvloop     │
        │  - ProcessPool × 16     │ ← parallel regex classify
        │  - httpx (conn pool 1024)│ ← async forwarder
        └──────────┬──────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
   vLLM vanilla         vLLM Trident (suffix+PIECEWISE)
   port 8001            port 8002

routing 정책 (workload → backend):
    chat   → vanilla     (large model 환경에서도 chat 은 Trident 도 좋지만,
                          parallel GPU 활용을 위해 vanilla 로 분배)
    sonnet → trident
    code   → trident

본 정책은 SUB_094 측정에서 검증됨 (3 mix scenario × 200 prompt 모두 net positive).
gating decision 정확도 100%.

실행:
    uvicorn agsd_router:app --host 0.0.0.0 --port 8000 \\
        --workers 1 --loop uvloop

환경변수:
    AGSD_VANILLA_URL    (default http://127.0.0.1:8001/v1)
    AGSD_TRIDENT_URL    (default http://127.0.0.1:8002/v1)
    AGSD_CLASSIFY_POOL  (default 16)
    AGSD_HTTP_POOL_SIZE (default 1024)
    AGSD_CHAT_BACKEND   (default vanilla)   ← chat workload routing target
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm_config_perf.gating.workload_classifier import classify  # noqa: E402

LOGGER = logging.getLogger("agsd_router")

# ---- config ----
VANILLA_URL = os.environ.get("AGSD_VANILLA_URL", "http://127.0.0.1:8001/v1")
TRIDENT_URL = os.environ.get("AGSD_TRIDENT_URL", "http://127.0.0.1:8002/v1")
CLASSIFY_POOL_SIZE = int(os.environ.get("AGSD_CLASSIFY_POOL", "16"))
HTTP_POOL_SIZE = int(os.environ.get("AGSD_HTTP_POOL_SIZE", "1024"))
CHAT_BACKEND = os.environ.get("AGSD_CHAT_BACKEND", "vanilla")  # "vanilla" or "trident"

# workload → backend mapping (SUB_094 검증)
BACKEND_FOR_WORKLOAD: dict[str, str] = {
    "chat": CHAT_BACKEND,
    "sonnet": "trident",
    "code": "trident",
}

BACKEND_URL: dict[str, str] = {
    "vanilla": VANILLA_URL,
    "trident": TRIDENT_URL,
}


# ---- lifecycle ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    """startup/shutdown — ProcessPool + httpx client."""
    app.state.executor = ProcessPoolExecutor(max_workers=CLASSIFY_POOL_SIZE)
    limits = httpx.Limits(
        max_connections=HTTP_POOL_SIZE,
        max_keepalive_connections=HTTP_POOL_SIZE,
    )
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(600.0, connect=10.0),
        limits=limits,
    )
    app.state.stats = {
        "total_requests": 0,
        "classify_total_ms": 0.0,
        "by_backend": {"vanilla": 0, "trident": 0},
        "by_workload": {"chat": 0, "sonnet": 0, "code": 0},
    }
    LOGGER.info(
        "agsd_router started: classify_pool=%d  http_pool=%d  chat_backend=%s",
        CLASSIFY_POOL_SIZE,
        HTTP_POOL_SIZE,
        CHAT_BACKEND,
    )
    try:
        yield
    finally:
        await app.state.client.aclose()
        app.state.executor.shutdown(wait=False)


app = FastAPI(title="AGSD Router", lifespan=lifespan)


# ---- helpers ----
def _extract_prompt(payload: dict[str, Any]) -> str:
    """OpenAI completion / chat completion payload 에서 prompt 텍스트 추출."""
    if "prompt" in payload:
        prompt = payload["prompt"]
        if isinstance(prompt, list):
            return "\n".join(str(p) for p in prompt)
        return str(prompt)
    if "messages" in payload:
        # chat completion format
        parts = []
        for msg in payload["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts) + "\n<|assistant|>\n"
    return ""


async def _classify_async(prompt: str, executor: ProcessPoolExecutor) -> str:
    """ProcessPool 에서 regex classify (CPU intensive 영역)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, classify, prompt)


# ---- endpoints ----
@app.post("/route")
async def route_single(req: Request) -> JSONResponse:
    """단일 prompt routing — backend 결정만 반환 (forward 안 함)."""
    payload = await req.json()
    prompt = _extract_prompt(payload)
    t0 = time.perf_counter()
    workload = await _classify_async(prompt, req.app.state.executor)
    classify_ms = (time.perf_counter() - t0) * 1000.0

    backend = BACKEND_FOR_WORKLOAD[workload]
    req.app.state.stats["total_requests"] += 1
    req.app.state.stats["classify_total_ms"] += classify_ms
    req.app.state.stats["by_backend"][backend] += 1
    req.app.state.stats["by_workload"][workload] += 1

    return JSONResponse(
        {
            "workload": workload,
            "backend": backend,
            "backend_url": BACKEND_URL[backend],
            "classify_ms": classify_ms,
        }
    )


@app.post("/v1/completions")
@app.post("/v1/chat/completions")
async def forward(req: Request):
    """OpenAI-compat forward — classify + forward to selected backend."""
    payload = await req.json()
    prompt = _extract_prompt(payload)

    t_classify = time.perf_counter()
    workload = await _classify_async(prompt, req.app.state.executor)
    classify_ms = (time.perf_counter() - t_classify) * 1000.0

    backend = BACKEND_FOR_WORKLOAD[workload]
    url = BACKEND_URL[backend] + req.url.path[3:]  # strip /v1 prefix duplicate

    req.app.state.stats["total_requests"] += 1
    req.app.state.stats["classify_total_ms"] += classify_ms
    req.app.state.stats["by_backend"][backend] += 1
    req.app.state.stats["by_workload"][workload] += 1

    is_stream = bool(payload.get("stream", False))

    if is_stream:
        async def _stream():
            async with req.app.state.client.stream(
                "POST", url, json=payload, timeout=600.0
            ) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk

        return StreamingResponse(_stream(), media_type="text/event-stream")

    resp = await req.app.state.client.post(url, json=payload, timeout=600.0)
    return JSONResponse(resp.json(), status_code=resp.status_code)


@app.post("/batch_route")
async def batch_route(req: Request) -> JSONResponse:
    """multi-prompt batch routing decision (forward 안 함)."""
    payload = await req.json()
    prompts: list[str] = payload.get("prompts", [])
    executor: ProcessPoolExecutor = req.app.state.executor

    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()
    workloads = await asyncio.gather(
        *(loop.run_in_executor(executor, classify, p) for p in prompts)
    )
    classify_ms = (time.perf_counter() - t0) * 1000.0

    backends = [BACKEND_FOR_WORKLOAD[w] for w in workloads]
    distribution = {
        "vanilla": backends.count("vanilla"),
        "trident": backends.count("trident"),
    }

    return JSONResponse(
        {
            "n_prompts": len(prompts),
            "workloads": workloads,
            "backends": backends,
            "distribution": distribution,
            "classify_total_ms": classify_ms,
            "classify_avg_ms": classify_ms / max(1, len(prompts)),
        }
    )


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/stats")
async def stats(req: Request) -> JSONResponse:
    s = req.app.state.stats
    total = max(1, s["total_requests"])
    return JSONResponse(
        {
            "total_requests": s["total_requests"],
            "classify_avg_ms": s["classify_total_ms"] / total,
            "by_backend": s["by_backend"],
            "by_workload": s["by_workload"],
        }
    )


@app.get("/recommendations")
async def recommendations() -> JSONResponse:
    """recommendations table 반환 (gating decision matrix 영역 self-doc)."""
    from vllm_config_perf.gating.recommendations import RECOMMENDATIONS

    out = {}
    for (workload, size), rec in RECOMMENDATIONS.items():
        key = f"{workload}_{size}"
        out[key] = dict(rec)  # TypedDict → plain dict
    return JSONResponse(out)


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "agsd_router:app",
        host=os.environ.get("AGSD_HOST", "0.0.0.0"),
        port=int(os.environ.get("AGSD_PORT", "8000")),
        loop="uvloop",
        workers=1,  # single worker — ProcessPool 이 classifier 병렬화
        access_log=False,
    )
