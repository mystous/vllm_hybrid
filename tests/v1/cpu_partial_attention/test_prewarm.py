# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TSK_002 §4.6 — cpu_partial_attention.prewarm() contract.

worker init 에서 한 번 호출되어 첫 cold-path call 의 ~1 s warmup 비용을
평탄화하는 hook. 본 파일은:

* idempotent — 두 번 이상 호출 시 raise 없이 통과
* select_isa_path() 를 호출하므로 그 cache 가 채워짐
* _NUMA_PIN_DONE flag 를 True 로 세팅 (NUMA 미사용 환경에서도 동일)
* AMX 미가용 (i9-12900KF 등 dev) 에서도 silent — _ensure_amx_permission_once
  가 자동 skip
"""

from __future__ import annotations

import pytest

from vllm.v1.attention.ops import cpu_partial_attention as cpa
from vllm.v1.attention.ops.cpu_partial_attention import ISAPath


def test_prewarm_is_idempotent(monkeypatch):
    monkeypatch.setattr(cpa, "_NUMA_PIN_DONE", False)
    cpa.prewarm()
    # 두 번째 호출 — _NUMA_PIN_DONE 이 이미 True 이므로 _maybe_pin_numa_once
    # 가 fast-return, select_isa_path 도 cache 된 결과만 반환.
    cpa.prewarm()
    cpa.prewarm()  # 세 번도 안전


def test_prewarm_sets_numa_pin_done(monkeypatch):
    monkeypatch.setattr(cpa, "_NUMA_PIN_DONE", False)
    cpa.prewarm()
    assert cpa._NUMA_PIN_DONE is True, (
        "prewarm 후에는 _NUMA_PIN_DONE=True — 다음 forward_partial_with_lse "
        "호출이 _maybe_pin_numa_once 의 lock-free fast path 를 탐"
    )


def test_prewarm_invokes_select_isa_path(monkeypatch):
    """prewarm 은 ISA detection 을 트리거 (첫 호출의 ~1s cpuinfo 비용을 흡수)."""
    calls: list = []

    real_select = cpa.select_isa_path

    def spy_select_isa_path():
        calls.append(1)
        return real_select()

    monkeypatch.setattr(cpa, "select_isa_path", spy_select_isa_path)
    monkeypatch.setattr(cpa, "_NUMA_PIN_DONE", False)
    cpa.prewarm()
    assert len(calls) == 1


def test_prewarm_handles_amx_path_safely(monkeypatch):
    """select_isa_path 가 AMX 를 반환해도 _has_amx_kernel 이 False 면 prctl 시도 안 함.
    (dev 머신은 AMX hw 자체가 없어 _has_amx_kernel=False — 자연 silent)"""
    monkeypatch.setattr(cpa, "select_isa_path", lambda: ISAPath.AMX)
    monkeypatch.setattr(cpa, "_has_amx_kernel", lambda: False)
    monkeypatch.setattr(cpa, "_NUMA_PIN_DONE", False)

    permission_calls: list = []

    def fail_if_called():
        permission_calls.append(1)

    monkeypatch.setattr(cpa, "_ensure_amx_permission_once", fail_if_called)
    cpa.prewarm()
    assert permission_calls == [], (
        "_has_amx_kernel=False 면 prctl 호출 안 함 — AMX hw 없는 머신에서 SIGILL 회피"
    )


def test_prewarm_calls_amx_permission_when_kernel_available(monkeypatch):
    """반대로 AMX kernel 도 load 됐고 ISA path 도 AMX 면 prctl 한 번 호출."""
    monkeypatch.setattr(cpa, "select_isa_path", lambda: ISAPath.AMX)
    monkeypatch.setattr(cpa, "_has_amx_kernel", lambda: True)
    monkeypatch.setattr(cpa, "_NUMA_PIN_DONE", False)

    permission_calls: list = []
    monkeypatch.setattr(
        cpa, "_ensure_amx_permission_once",
        lambda: permission_calls.append(1),
    )
    cpa.prewarm()
    assert permission_calls == [1]


def test_prewarm_skips_numa_pin_when_already_done(monkeypatch):
    """_NUMA_PIN_DONE 이 이미 True 면 _pin_threads_to_local_numa 호출 없음."""
    monkeypatch.setattr(cpa, "_NUMA_PIN_DONE", True)

    pin_calls: list = []
    if cpa._pin_threads_to_local_numa is not None:
        monkeypatch.setattr(
            cpa, "_pin_threads_to_local_numa",
            lambda: pin_calls.append(1),
        )
    cpa.prewarm()
    assert pin_calls == []
