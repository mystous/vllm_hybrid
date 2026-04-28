# 20260428_023936_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_quick_check

- timestamp: 20260428_023936
- hw_tag:    Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8
- branch:    feat/ide006-cold-kv-cpu-partial-attention
- commit:    bf525c24d9ad3fb85a721654245356957de14669
- python:    /workspace/vllm_dev_prj/bin/python
- vllm:      0.1.dev15917+g0a6396b45

## scope
- tests/v1/cpu_partial_attention/  (TSK_001 + TSK_002 회귀)
- tests/v1/kv_offload/             (TSK_004 NUMA + reload sync)
- 제외: tests/v1/kv_offload/test_cpu_offloading.py (vLLM upstream
  inherited 테스트 — IDE_006 본 회귀 아님. Llama-3.2-1B 의존)
- e2e 미포함 — '오래 진행하지 말고 개발된 내용 확인만' (사용자 요구)
- 모드: offline (방화벽 환경, HF_HUB_OFFLINE=1)

## result
- pytest_rc: 1

### tail
```
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
- generated xml file: /workspace/vllm_hybrid/eval/results/20260428_023936_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_quick_check/pytest_junit.xml -
=========================== short test summary info ============================
FAILED tests/v1/cpu_partial_attention/test_portable_cross_check.py::test_default_dispatch_uses_portable_when_available[bf16]
FAILED tests/v1/cpu_partial_attention/test_portable_cross_check.py::test_default_dispatch_uses_portable_when_available[fp16]
FAILED tests/v1/cpu_partial_attention/test_wrapper_dispatch.py::test_force_simd_path_cascades_to_portable[bf16-avx512]
FAILED tests/v1/cpu_partial_attention/test_wrapper_dispatch.py::test_force_simd_path_cascades_to_portable[bf16-amx]
FAILED tests/v1/cpu_partial_attention/test_wrapper_dispatch.py::test_force_simd_path_cascades_to_portable[fp16-avx512]
FAILED tests/v1/cpu_partial_attention/test_wrapper_dispatch.py::test_force_simd_path_cascades_to_portable[fp16-amx]
FAILED tests/v1/cpu_partial_attention/test_wrapper_dispatch.py::test_default_path_matches_forced_portable[bf16]
FAILED tests/v1/cpu_partial_attention/test_wrapper_dispatch.py::test_default_path_matches_forced_portable[fp16]
===== 8 failed, 404 passed, 3 deselected, 16 warnings in 98.15s (0:01:38) ======
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```
