"""AGSD spec config recommendations table.

SUB_080 / SUB_092 의 analytical gating decision matrix.
워크로드 + 모델 크기에 따라 best spec config + 예상 speedup 권장.

본 표는 SUB_093 (Llama-3.3-70B + TP=8 × H100×8) 측정 결과 기반.
prod hardware 동등 환경에서 권장. 다른 hw 에선 직접 재측정 필요.
"""

from __future__ import annotations

from typing import Literal, TypedDict

ModelSize = Literal["small", "medium", "large"]
WorkloadType = Literal["sonnet", "chat", "code"]


class SpecRecommendation(TypedDict):
    method: str | None  # "suffix" / "ngram" / None (= vanilla)
    num_speculative_tokens: int | None
    cudagraph_mode: str  # "PIECEWISE" / "FULL"
    gpu_memory_utilization: float
    expected_speedup_pct: float  # vanilla 대비 %
    source_sub: str  # 측정 출처


def _model_size_from_param_count(num_params_billion: float) -> ModelSize:
    """모델 파라미터 수 (B) → size category."""
    if num_params_billion <= 7:
        return "small"
    elif num_params_billion <= 32:
        return "medium"
    else:
        return "large"


# ---- 결정 매트릭스 (workload, model_size) → SpecRecommendation ----
RECOMMENDATIONS: dict[
    tuple[WorkloadType, ModelSize], SpecRecommendation
] = {
    # ===== large model (≥70B) — Trident core 가 모든 workload best =====
    ("sonnet", "large"): {
        "method": "suffix",
        "num_speculative_tokens": 32,
        "cudagraph_mode": "PIECEWISE",
        "gpu_memory_utilization": 0.80,
        "expected_speedup_pct": 52.1,  # SUB_093 sonnet
        "source_sub": "SUB_093 / SUB_089",
    },
    ("chat", "large"): {
        "method": "suffix",
        "num_speculative_tokens": 32,
        "cudagraph_mode": "PIECEWISE",
        "gpu_memory_utilization": 0.80,
        "expected_speedup_pct": 68.9,  # SUB_093 chat
        "source_sub": "SUB_093 / SUB_085",
    },
    ("code", "large"): {
        "method": "suffix",
        "num_speculative_tokens": 32,
        "cudagraph_mode": "PIECEWISE",
        "gpu_memory_utilization": 0.80,
        "expected_speedup_pct": 18.8,  # SUB_093 code (ngram -20.2% 회귀 mitigation)
        "source_sub": "SUB_093 / SUB_085 v2",
    },
    # ===== medium model (≤32B) — 부분 적용 =====
    ("sonnet", "medium"): {
        "method": "suffix",
        "num_speculative_tokens": 32,
        "cudagraph_mode": "PIECEWISE",
        "gpu_memory_utilization": 0.80,
        "expected_speedup_pct": 43.9,  # Qwen 32B avg
        "source_sub": "SUB_097-A",
    },
    ("chat", "medium"): {
        "method": "suffix",
        "num_speculative_tokens": 32,
        "cudagraph_mode": "PIECEWISE",
        "gpu_memory_utilization": 0.80,
        "expected_speedup_pct": 30.0,  # 추정
        "source_sub": "SUB_097-A",
    },
    ("code", "medium"): {
        "method": "suffix",
        "num_speculative_tokens": 32,
        "cudagraph_mode": "PIECEWISE",
        "gpu_memory_utilization": 0.80,
        "expected_speedup_pct": 15.0,  # 추정
        "source_sub": "SUB_097-A",
    },
    # ===== small model (≤7B) — 모든 spec method universal regression =====
    #   issue #16258 hardware-independent. AGSD gating 시 vanilla 권장.
    ("sonnet", "small"): {
        "method": None,
        "num_speculative_tokens": None,
        "cudagraph_mode": "PIECEWISE",
        "gpu_memory_utilization": 0.85,
        "expected_speedup_pct": 0.0,
        "source_sub": "SUB_088 / SUB_090 / SUB_091",
    },
    ("chat", "small"): {
        "method": None,
        "num_speculative_tokens": None,
        "cudagraph_mode": "PIECEWISE",
        "gpu_memory_utilization": 0.85,
        "expected_speedup_pct": 0.0,
        "source_sub": "SUB_088 / SUB_090 / SUB_091",
    },
    ("code", "small"): {
        "method": None,
        "num_speculative_tokens": None,
        "cudagraph_mode": "PIECEWISE",
        "gpu_memory_utilization": 0.85,
        "expected_speedup_pct": 0.0,
        "source_sub": "SUB_088 / SUB_090 / SUB_091",
    },
}


def recommend(workload: WorkloadType, model_size: ModelSize) -> SpecRecommendation:
    """워크로드 + 모델 크기 → spec config 권장값."""
    return RECOMMENDATIONS[(workload, model_size)]


def recommend_for_model(
    workload: WorkloadType, num_params_billion: float
) -> SpecRecommendation:
    """파라미터 수 기준 자동 size 결정 + 권장."""
    return recommend(workload, _model_size_from_param_count(num_params_billion))


def to_speculative_config(rec: SpecRecommendation) -> dict | None:
    """SpecRecommendation → vLLM LLM constructor 의 speculative_config dict.

    None 반환 = vanilla (spec OFF).
    """
    if rec["method"] is None:
        return None
    return {
        "method": rec["method"],
        "num_speculative_tokens": rec["num_speculative_tokens"],
    }


if __name__ == "__main__":
    import json

    print("=== AGSD recommendations matrix ===")
    for (workload, size), rec in RECOMMENDATIONS.items():
        method = rec["method"] or "vanilla"
        print(
            f"  {workload:>6} × {size:>6} → {method:>10} K={rec['num_speculative_tokens']} "
            f"PIECEWISE  gmu={rec['gpu_memory_utilization']}  "
            f"+{rec['expected_speedup_pct']:.1f}%  ({rec['source_sub']})"
        )
