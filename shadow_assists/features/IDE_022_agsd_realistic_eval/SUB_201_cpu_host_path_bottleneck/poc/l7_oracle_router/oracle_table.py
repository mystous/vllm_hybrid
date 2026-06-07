"""L7 — Oracle table loader for model-type level routing.

TSK_042 routing_combined/metrics_table.csv 의 215 measured cells 를
(model_family, workload_type) → {method: tps, best_method: str, best_tps: float}
형식으로 변환한다.

본 oracle 은 model-level (모델 family 가 결정되면 method 가 결정됨) 이며,
TSK_044 의 per-request workload 분류기와는 다음 점에서 다르다:
  - TSK_044: 각 prompt 마다 C0~C3 regex / ONNX classifier 호출 → host overhead
  - L7    : 라우터 진입 시 model_name 만 보고 instance 선택 → O(1) lookup, host overhead ≈ 0

산출: oracle.lookup(model_family, workload_type) → (best_method, best_tps)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_CSV = Path(
    "/workspace/host_vllm_hybrid/vllm_config_perf/gating/realistic_eval/runs/"
    "routing_combined/metrics_table.csv"
)

# ---------------------------------------------------------------------------
# Model family classification
#   하나의 model_family 가 곧 boot 되는 vLLM instance 의 단위가 된다.
# ---------------------------------------------------------------------------
MODEL_FAMILY: dict[str, str] = {
    "Qwen2.5-7B-Instruct": "Qwen-7B",
    "Qwen2.5-32B-Instruct": "Qwen-32B",
    "Qwen2.5-72B-Instruct": "Qwen-72B",
    "Llama-3.1-8B-Instruct": "Llama-8B",
    "Llama-3.1-70B-Instruct": "Llama-70B",
    "Llama-3.1-405B-Instruct-FP8": "Llama-405B",
    "DeepSeek-R1-Distill-Qwen-7B": "DS-Qwen-7B",
    "DeepSeek-R1-Distill-Qwen-32B": "DS-Qwen-32B",
    "DeepSeek-R1-Distill-Llama-70B": "DS-Llama-70B",
    "DeepSeek-R1": "DS-R1-671B",
}

WORKLOAD_TYPES = ["sharegpt", "lmsys", "wildchat", "humaneval", "mbpp", "swebench", "mix"]
METHODS = ["vanilla", "suffix", "ngram", "llm-d"]


@dataclass(frozen=True)
class OracleRow:
    model_family: str
    workload_type: str
    by_method: dict[str, float]   # method → output_tps
    best_method: str
    best_tps: float
    default_tps: float            # vanilla 기준 (default == no spec, no router)
    uplift_pct: float             # (best - vanilla) / vanilla * 100


class Oracle:
    """O(1) lookup oracle. 라우터는 본 객체 1 회만 build."""

    def __init__(self, rows: list[OracleRow]):
        self._rows: dict[tuple[str, str], OracleRow] = {
            (r.model_family, r.workload_type): r for r in rows
        }

    @classmethod
    def from_csv(cls, csv_path: Path = DEFAULT_CSV) -> "Oracle":
        df = pd.read_csv(csv_path)
        df = df[df["model"].isin(MODEL_FAMILY.keys())].copy()
        df["model_family"] = df["model"].map(MODEL_FAMILY)
        df["workload_type"] = df["condition"]

        rows: list[OracleRow] = []
        for (mf, wt), sub in df.groupby(["model_family", "workload_type"]):
            by_method = dict(zip(sub["method"], sub["output_tps"]))
            # vanilla 를 default 로
            default = by_method.get("vanilla", float("nan"))
            # best 는 max tps
            best_method = max(by_method, key=lambda m: by_method[m])
            best_tps = by_method[best_method]
            uplift = (
                (best_tps - default) / default * 100.0 if default and default == default else 0.0
            )
            rows.append(
                OracleRow(
                    model_family=mf,
                    workload_type=wt,
                    by_method=by_method,
                    best_method=best_method,
                    best_tps=best_tps,
                    default_tps=default,
                    uplift_pct=uplift,
                )
            )
        return cls(rows)

    # --- API -----------------------------------------------------------------
    def lookup(self, model_family: str, workload_type: str) -> OracleRow:
        """라우터가 매 request 마다 부르는 hot-path. dict get → O(1)."""
        return self._rows[(model_family, workload_type)]

    def lookup_best_method(self, model_family: str, workload_type: str) -> str:
        return self._rows[(model_family, workload_type)].best_method

    def iter_rows(self):
        return iter(self._rows.values())

    def to_dataframe(self) -> pd.DataFrame:
        recs = []
        for r in self.iter_rows():
            rec = {
                "model_family": r.model_family,
                "workload_type": r.workload_type,
                "best_method": r.best_method,
                "best_tps": r.best_tps,
                "default_tps": r.default_tps,
                "uplift_pct": r.uplift_pct,
            }
            for m in METHODS:
                rec[f"tps_{m}"] = r.by_method.get(m, float("nan"))
            recs.append(rec)
        return pd.DataFrame.from_records(recs).sort_values(
            ["model_family", "workload_type"]
        )

    def to_json(self) -> str:
        """라우터가 cold-start 시 메모리에 올릴 JSON snapshot."""
        return json.dumps(
            {
                f"{r.model_family}|{r.workload_type}": {
                    "best_method": r.best_method,
                    "best_tps": r.best_tps,
                    "default_tps": r.default_tps,
                    "uplift_pct": r.uplift_pct,
                    "by_method": r.by_method,
                }
                for r in self.iter_rows()
            },
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    o = Oracle.from_csv()
    df = o.to_dataframe()
    print(df.to_string(index=False))
    print()
    print(f"# cells: {len(df)}")
    print(f"# model_family: {df['model_family'].nunique()}")
    print(f"# workload_type: {df['workload_type'].nunique()}")
