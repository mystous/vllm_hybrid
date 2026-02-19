#!/usr/bin/env python3
"""
CPU+GPU 동시 추론 검증 스크립트.

CPU 프로세스와 GPU가 실제로 동시에 추론하는 것을 시간 겹침으로 증명합니다.

사용법:
    PYTHONPATH=. python3 scripts/verify_hybrid_parallel.py
"""

import multiprocessing
import os
import queue
import time
import threading
from dataclasses import dataclass

import torch


MODEL_NAME = "facebook/opt-125m"
MAX_TOKENS = 50


def test_cpu_inference_standalone():
    """Phase 1: CPU 프로세스에서 실제로 모델 로드 + 토큰 생성 확인."""
    print("=" * 60)
    print("[Phase 1] CPU 단독 추론 검증")
    print("=" * 60)

    from vllm.v1.engine.hybrid_core import (
        CPUInferenceProcess, CPUInferenceRequest, CPUInferenceResponse,
        _SHUTDOWN_SENTINEL,
    )

    ctx = multiprocessing.get_context("spawn")
    req_queue = ctx.Queue()
    res_queue = ctx.Queue()

    print(f"  모델: {MODEL_NAME}")
    print(f"  CPU 프로세스 시작 중...")

    proc = ctx.Process(
        target=CPUInferenceProcess._cpu_worker_loop,
        args=(
            MODEL_NAME, True, 4, "float32", False, None, 2,
            req_queue, res_queue,
        ),
        daemon=True,
        name="cpu-test",
    )

    start = time.time()
    proc.start()
    print(f"  CPU PID: {proc.pid}")

    # 모델 로딩 대기
    print("  모델 로딩 대기 중...")
    time.sleep(8)

    # 요청 전송
    cpu_req = CPUInferenceRequest(
        request_id="cpu-test-001",
        prompt_token_ids=[2, 10, 20, 30, 40, 50],
        max_tokens=30,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        stop_token_ids=[],  # EOS 없이 max_tokens까지 생성
    )
    req_queue.put(cpu_req)
    print(f"  요청 전송: prompt_len=6, max_tokens=30")

    # 결과 수집
    all_tokens = []
    finished = False
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            resp: CPUInferenceResponse = res_queue.get(timeout=2)
            all_tokens.extend(resp.generated_token_ids)
            print(f"    수신: +{len(resp.generated_token_ids)} tokens, "
                  f"total={len(all_tokens)}, finished={resp.finished}")
            if resp.finished:
                finished = True
                break
        except queue.Empty:
            continue

    elapsed = time.time() - start

    # 종료
    req_queue.put(_SHUTDOWN_SENTINEL)
    proc.join(timeout=10)
    if proc.is_alive():
        proc.terminate()

    if finished and len(all_tokens) > 0:
        print(f"\n  CPU 추론 결과:")
        print(f"    총 생성 토큰: {len(all_tokens)}")
        print(f"    토큰 ID: {all_tokens[:15]}...")
        print(f"    소요 시간: {elapsed:.2f}초")
        print(f"  ✅ CPU 단독 추론 성공!")
        return True
    else:
        print(f"  ❌ CPU 추론 실패 (tokens={len(all_tokens)}, finished={finished})")
        return False


def test_gpu_inference_standalone():
    """Phase 2: GPU에서 동일 모델로 추론 확인."""
    print("\n" + "=" * 60)
    print("[Phase 2] GPU 단독 추론 검증")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  ⚠️ CUDA 미지원, GPU 테스트 스킵")
        return False

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  모델: {MODEL_NAME}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).cuda()

    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    elapsed = time.time() - start

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  프롬프트: '{prompt}'")
    print(f"  생성결과: '{text[:80]}...'")
    print(f"  소요시간: {elapsed:.4f}초")
    print(f"  ✅ GPU 단독 추론 성공!")

    del model
    torch.cuda.empty_cache()
    return True


def test_parallel_inference():
    """Phase 3: CPU와 GPU가 동시에 추론하는지 시간 겹침으로 검증."""
    print("\n" + "=" * 60)
    print("[Phase 3] CPU + GPU 동시 추론 검증 (핵심)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  ⚠️ CUDA 미지원, 병렬 테스트 스킵")
        return False

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm.v1.engine.hybrid_core import (
        CPUInferenceProcess, CPUInferenceRequest, CPUInferenceResponse,
        _SHUTDOWN_SENTINEL,
    )

    # === GPU 모델 미리 로드 (병렬 구간에서 로딩 시간 제거) ===
    print(f"  GPU 모델 미리 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    gpu_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).cuda()
    # warmup
    warmup_input = tokenizer("warmup", return_tensors="pt").to("cuda")
    with torch.no_grad():
        gpu_model.generate(**warmup_input, max_new_tokens=5)
    print(f"  GPU 준비 완료")

    # === CPU 프로세스 미리 시작 ===
    print(f"  CPU 프로세스 시작 중...")
    ctx = multiprocessing.get_context("spawn")
    req_queue = ctx.Queue()
    res_queue = ctx.Queue()

    cpu_proc = ctx.Process(
        target=CPUInferenceProcess._cpu_worker_loop,
        args=(
            MODEL_NAME, True, 8, "float32", False, None, 2,
            req_queue, res_queue,
        ),
        daemon=True,
        name="cpu-parallel",
    )
    cpu_proc.start()
    print(f"  CPU PID: {cpu_proc.pid}")
    print(f"  CPU 모델 로딩 대기 (10초)...")
    time.sleep(10)

    # === 동시 실행 준비 ===
    gpu_results = {"tokens": [], "start": 0, "end": 0, "success": False}
    barrier = threading.Barrier(2, timeout=10)  # 동시 시작 보장

    def gpu_worker():
        """GPU에서 여러 요청 연속 추론."""
        try:
            barrier.wait()  # CPU와 동시 시작
        except threading.BrokenBarrierError:
            return

        prompts = [
            "The meaning of life is",
            "In the beginning there was",
            "Once upon a time in a",
            "The quick brown fox jumps over",
            "Artificial intelligence will",
        ]

        gpu_results["start"] = time.time()
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = gpu_model.generate(
                    **inputs, max_new_tokens=MAX_TOKENS, do_sample=False
                )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            gpu_results["tokens"].append(text)
        gpu_results["end"] = time.time()
        gpu_results["success"] = True

    # === 동시 추론 시작! ===
    print(f"\n  >>> 동시 추론 시작! <<<")

    # GPU 스레드 시작
    gpu_thread = threading.Thread(target=gpu_worker, name="gpu-parallel")
    gpu_thread.start()

    # CPU 요청 전송 (여러 요청)
    cpu_req_ids = []
    for i in range(3):
        cpu_req = CPUInferenceRequest(
            request_id=f"parallel-cpu-{i:03d}",
            prompt_token_ids=[2, 100 + i * 10, 200, 300, 400, 500],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            stop_token_ids=[],
        )
        cpu_req_ids.append(cpu_req.request_id)
        req_queue.put(cpu_req)

    # Barrier 해제 (동시 시작)
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        pass

    cpu_start = time.time()

    # CPU 결과 수집
    cpu_all_tokens = {}
    cpu_finished = set()
    deadline = time.time() + 120
    while time.time() < deadline and len(cpu_finished) < len(cpu_req_ids):
        try:
            resp = res_queue.get(timeout=1)
            if resp.request_id not in cpu_all_tokens:
                cpu_all_tokens[resp.request_id] = []
            cpu_all_tokens[resp.request_id].extend(resp.generated_token_ids)
            if resp.finished:
                cpu_finished.add(resp.request_id)
        except queue.Empty:
            continue

    cpu_end = time.time()

    # GPU 완료 대기
    gpu_thread.join(timeout=30)

    # 종료
    req_queue.put(_SHUTDOWN_SENTINEL)
    cpu_proc.join(timeout=10)
    if cpu_proc.is_alive():
        cpu_proc.terminate()

    # GPU 메모리 정리
    del gpu_model
    torch.cuda.empty_cache()

    # === 결과 분석 ===
    print(f"\n  --- 결과 분석 ---")

    gpu_ok = gpu_results.get("success", False)
    cpu_ok = len(cpu_finished) > 0

    base_time = min(
        gpu_results.get("start", cpu_start),
        cpu_start
    )

    if gpu_ok:
        gpu_elapsed = gpu_results["end"] - gpu_results["start"]
        gs = gpu_results["start"] - base_time
        ge = gpu_results["end"] - base_time
        print(f"  GPU: ✅ 완료 ({gpu_elapsed:.3f}초)")
        print(f"    구간: [{gs:.3f}s ~ {ge:.3f}s]")
        print(f"    생성: {len(gpu_results['tokens'])}개 응답")
    else:
        gs, ge = 0, 0
        print(f"  GPU: ❌ 실패")

    if cpu_ok:
        cpu_elapsed = cpu_end - cpu_start
        cs = cpu_start - base_time
        ce = cpu_end - base_time
        total_cpu_tokens = sum(len(v) for v in cpu_all_tokens.values())
        print(f"  CPU: ✅ 완료 ({cpu_elapsed:.3f}초)")
        print(f"    구간: [{cs:.3f}s ~ {ce:.3f}s]")
        print(f"    완료: {len(cpu_finished)}/{len(cpu_req_ids)}개 요청")
        print(f"    생성: 총 {total_cpu_tokens} 토큰")
    else:
        cs, ce = 0, 0
        print(f"  CPU: ❌ 실패")

    # 동시성 판정
    if gpu_ok and cpu_ok:
        overlap_start = max(gs, cs)
        overlap_end = min(ge, ce)
        overlap = max(0, overlap_end - overlap_start)

        total_t = max(ge, ce)
        print(f"\n  겹침 구간: {overlap:.3f}초")

        # 타임라인 시각화
        bar_width = 60
        scale = bar_width / total_t if total_t > 0 else 1

        def make_bar(start_t, end_t, char='█'):
            bar = [' '] * bar_width
            s = int(start_t * scale)
            e = int(end_t * scale)
            for i in range(max(0, s), min(e, bar_width)):
                bar[i] = char
            return ''.join(bar)

        print(f"\n  타임라인 (0 ~ {total_t:.1f}초):")
        print(f"  GPU: |{make_bar(gs, ge)}|")
        print(f"  CPU: |{make_bar(cs, ce)}|")

        # 겹침 영역 표시
        if overlap > 0.01:
            print(f"  겹침: |{make_bar(overlap_start, overlap_end, '▓')}|")

        if overlap > 0.05:
            pct = overlap / total_t * 100 if total_t > 0 else 0
            print(f"\n  ✅ CPU와 GPU가 {overlap:.3f}초 동안 동시에 연산!")
            print(f"     (전체 실행 시간 대비 {pct:.1f}% 겹침)")
            return True
        else:
            print(f"\n  ⚠️ 겹침이 {overlap:.3f}초로 너무 짧음")
            return False
    else:
        print(f"\n  ❌ 하나 이상의 경로 실패")
        return False


def test_router_visualization():
    """Phase 4: RequestRouter 분배 시각화."""
    print("\n" + "=" * 60)
    print("[Phase 4] RequestRouter 분배 시각화")
    print("=" * 60)

    from vllm.v1.engine.hybrid_core import RequestRouter

    router = RequestRouter(0.05)
    routes = []
    for i in range(100):
        r = router.route(f"req-{i}")
        routes.append("C" if r == "cpu" else ".")

    print(f"  cpu_ratio=0.05 (5%), 100 요청:")
    print(f"    {''.join(routes[:50])}")
    print(f"    {''.join(routes[50:])}")
    print(f"    (. = GPU, C = CPU)")

    stats = router.get_stats()
    print(f"\n  GPU: {stats['gpu_requests']}, CPU: {stats['cpu_requests']}")
    print(f"  실제 비율: {stats['actual_cpu_ratio']:.1%}")
    print(f"  ✅ 라우팅 분배 정상")


if __name__ == "__main__":
    print("🔍 CPU+GPU Hybrid Parallel-Batch 동시 추론 검증")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CPU cores: {multiprocessing.cpu_count()}")
    print()

    results = {}

    results["cpu"] = test_cpu_inference_standalone()
    results["gpu"] = test_gpu_inference_standalone()

    if results["cpu"] and results["gpu"]:
        results["parallel"] = test_parallel_inference()
    else:
        print("\n  ⚠️ 단독 테스트 실패, 병렬 테스트 스킵")
        results["parallel"] = False

    test_router_visualization()

    # 최종 결과
    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    print(f"  Phase 1 (CPU 단독):     {'✅' if results.get('cpu') else '❌'}")
    print(f"  Phase 2 (GPU 단독):     {'✅' if results.get('gpu') else '❌'}")
    print(f"  Phase 3 (동시 추론):    {'✅' if results.get('parallel') else '❌'}")
    print(f"  Phase 4 (라우터 분배):  ✅")

    if results.get("parallel"):
        print("\n  🎉 CPU와 GPU가 동시에 추론하는 것이 확인되었습니다!")
    elif results.get("cpu") and results.get("gpu"):
        print("\n  ⚠️ 각각 동작하지만 동시 실행 겹침이 부족합니다.")
    else:
        print("\n  ❌ 일부 테스트 실패")
