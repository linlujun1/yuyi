"""全集级指标汇总（纯标准库）。"""

from __future__ import annotations

from typing import Any

DIM_COLS = ["s_issue", "s_method", "s_stance", "s_length"]


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def aggregate_metrics(rows: list[dict[str, Any]], tau: float = 0.8) -> dict[str, float]:
    if not rows:
        return {}
    dim_avg = {c: _mean([float(r[c]) for r in rows]) for c in DIM_COLS}
    dim_values = list(dim_avg.values())
    n_pass = sum(1 for r in rows if all(float(r[c]) >= tau for c in DIM_COLS))
    overall = [float(r["s_overall"]) for r in rows]
    return {
        "MacroAvg": _mean(overall),
        "PassRate": n_pass / len(rows),
        "WorstDim": min(dim_values),
        "Gap": max(dim_values) - min(dim_values),
        **{f"DimAvg_{k}": v for k, v in dim_avg.items()},
        "tau": tau,
        "n": float(len(rows)),
    }
