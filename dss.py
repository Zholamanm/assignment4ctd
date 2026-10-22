"""Multi-criteria DSS for choosing an internship offer.

Pipeline: load -> validate -> min-max normalize -> WSM -> TOPSIS -> compare -> charts.

Run:
  python dss.py
  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "internships.csv"
OUTPUT_DIR = ROOT / "output"

# key, label, benefit|cost, unit
CRITERIA = [
    ("pay", "Monthly pay", "benefit", "thousand tenge / month"),
    ("commute", "Commute", "cost", "minutes"),
    ("learning", "Learning", "benefit", "score 1-10"),
    ("contract", "Contract length", "benefit", "months"),
    ("overtime", "Overtime", "cost", "hours / week"),
]

DEFAULT_WEIGHTS = {
    "pay": 0.25,
    "commute": 0.15,
    "learning": 0.30,
    "contract": 0.10,
    "overtime": 0.20,
}

# Offers outside these bounds are dropped before ranking.
CONSTRAINTS = {
    "pay": ("min", 80),
    "commute": ("max", 60),
}

WEIGHT_TOLERANCE = 1e-6


def criterion_keys() -> list[str]:
    return [key for key, _, _, _ in CRITERIA]


def criterion_types() -> dict[str, str]:
    return {key: kind for key, _, kind, _ in CRITERIA}


def load_matrix(path: Path = DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "alternative" not in frame.columns:
        raise ValueError("CSV must have an 'alternative' column.")
    frame = frame.set_index("alternative")
    return frame


def apply_constraints(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept, rejected). Rejected rows fail a hard limit."""
    mask = pd.Series(True, index=frame.index)
    reasons: dict[str, str] = {}
    for key, (bound, limit) in CONSTRAINTS.items():
        if key not in frame.columns:
            continue
        if bound == "min":
            failed = frame[key] < limit
            text = f"{key} < {limit}"
        else:
            failed = frame[key] > limit
            text = f"{key} > {limit}"
        for name in frame.index[failed]:
            reasons[name] = text
        mask &= ~failed
    kept = frame.loc[mask].copy()
    rejected = frame.loc[~mask].copy()
    rejected.insert(0, "failed_constraint", [reasons[name] for name in rejected.index])
    return kept, rejected


def validate(frame: pd.DataFrame, weights: dict[str, float]) -> None:
    keys = criterion_keys()
    missing_cols = [key for key in keys if key not in frame.columns]
    if missing_cols:
        raise ValueError(f"Missing criteria columns: {missing_cols}")

    if frame.empty:
        raise ValueError("No alternatives left after constraints.")

    if frame[keys].isna().any().any():
        raise ValueError("Missing values in the decision matrix.")

    numeric = frame[keys].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Every criterion value must be numeric.")

    extra = [key for key in weights if key not in keys]
    absent = [key for key in keys if key not in weights]
    if extra or absent:
        raise ValueError(f"Weights must cover exactly {keys}. Extra={extra} missing={absent}")

    values = [float(weights[key]) for key in keys]
    if any(value < 0 for value in values):
        raise ValueError("Weights must be >= 0.")
    total = float(sum(values))
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(f"Weights must sum to 1. Current sum is {total:.6f}.")


def minmax_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Map every column to 0..1 where 1 is best, for both benefit and cost."""
    types = criterion_types()
    normalized = pd.DataFrame(index=frame.index)
    for key in criterion_keys():
        column = frame[key].astype(float)
        low = float(column.min())
        high = float(column.max())
        if high == low:
            normalized[key] = 1.0
            continue
        if types[key] == "benefit":
            normalized[key] = (column - low) / (high - low)
        elif types[key] == "cost":
            normalized[key] = (high - column) / (high - low)
        else:
            raise ValueError(f"Unknown criterion type for {key}: {types[key]}")
    return normalized


def weighted_sum(normalized: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    keys = criterion_keys()
    score = sum(normalized[key] * float(weights[key]) for key in keys)
    score.name = "wsm_score"
    return score.sort_values(ascending=False)


def topsis(frame: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Vector-normalize, weight, then closeness to the ideal offer."""
    keys = criterion_keys()
    types = criterion_types()
    matrix = frame[keys].astype(float).to_numpy()
    denom = np.sqrt((matrix ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    vector = matrix / denom
    weight_row = np.array([float(weights[key]) for key in keys])
    weighted = vector * weight_row

    ideal = np.zeros(len(keys))
    worst = np.zeros(len(keys))
    for j, key in enumerate(keys):
        column = weighted[:, j]
        if types[key] == "benefit":
            ideal[j] = column.max()
            worst[j] = column.min()
        else:
            ideal[j] = column.min()
            worst[j] = column.max()

    d_best = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted - worst) ** 2).sum(axis=1))
    denom_c = d_best + d_worst
    closeness = np.divide(d_worst, denom_c, out=np.zeros_like(d_worst), where=denom_c != 0)

    result = pd.DataFrame(
        {
            "topsis_score": closeness,
            "distance_to_best": d_best,
            "distance_to_worst": d_worst,
        },
        index=frame.index,
    )
    return result.sort_values("topsis_score", ascending=False)


def ranking_table(wsm_scores: pd.Series, topsis_scores: pd.Series) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "wsm_score": wsm_scores,
            "topsis_score": topsis_scores,
        }
    )
    table["wsm_rank"] = table["wsm_score"].rank(ascending=False, method="min").astype(int)
    table["topsis_rank"] = table["topsis_score"].rank(ascending=False, method="min").astype(int)
    return table.sort_values(["wsm_rank", "topsis_rank"])


def compare_text(table: pd.DataFrame) -> str:
    wsm_order = table.sort_values("wsm_rank").index.tolist()
    topsis_order = table.sort_values("topsis_rank").index.tolist()
    same_winner = wsm_order[0] == topsis_order[0]
    same_order = wsm_order == topsis_order
    lines = [
        f"Same winner: {'yes' if same_winner else 'no'} ({wsm_order[0]} by WSM, {topsis_order[0]} by TOPSIS).",
        f"Same ranking: {'yes' if same_order else 'no'}.",
        "WSM adds weighted normalized scores, so a strong column can offset a weak one.",
        "TOPSIS measures distance to an ideal and an anti-ideal, so one very bad cost can drop an offer even if another column is perfect.",
    ]
    if not same_order:
        jumps = (table["topsis_rank"] - table["wsm_rank"]).abs().sort_values(ascending=False)
        mover = jumps.index[0]
        lines.append(
            f"Largest move: {mover} goes from WSM rank {int(table.loc[mover, 'wsm_rank'])} "
            f"to TOPSIS rank {int(table.loc[mover, 'topsis_rank'])}."
        )
    return " ".join(lines)


def plot_scores(table: pd.DataFrame, path: Path) -> None:
    order = table.sort_values("wsm_score", ascending=True)
    y = np.arange(len(order))
    height = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(y - height / 2, order["wsm_score"], height=height, label="WSM", color="#1F7A6B")
    ax.barh(y + height / 2, order["topsis_score"], height=height, label="TOPSIS", color="#12324A")
    ax.set_yticks(y)
    ax.set_yticklabels(order.index)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score (higher is better)")
    ax.set_title("Internship scores: WSM vs TOPSIS")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_heatmap(normalized: pd.DataFrame, path: Path) -> None:
    labels = [label for _, label, _, _ in CRITERIA]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    image = ax.imshow(normalized.to_numpy(), cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(range(len(normalized.index)))
    ax.set_yticklabels(normalized.index)
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            value = float(normalized.iloc[i, j])
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="#142033", fontsize=9)
    ax.set_title("Normalized matrix (1 = best on that criterion)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(weights: dict[str, float] | None = None, save_charts: bool = True) -> dict:
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    raw = load_matrix()
    kept, rejected = apply_constraints(raw)
    validate(kept, weights)
    normalized = minmax_normalize(kept)
    wsm_scores = weighted_sum(normalized, weights)
    topsis_result = topsis(kept, weights)
    table = ranking_table(wsm_scores, topsis_result["topsis_score"])
    charts = {}
    if save_charts:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        score_path = OUTPUT_DIR / "scores.png"
        heat_path = OUTPUT_DIR / "heatmap.png"
        plot_scores(table, score_path)
        plot_heatmap(normalized, heat_path)
        charts = {"scores": score_path, "heatmap": heat_path}
    return {
        "raw": raw,
        "kept": kept,
        "rejected": rejected,
        "normalized": normalized,
        "wsm": wsm_scores,
        "topsis": topsis_result,
        "table": table,
        "explanation": compare_text(table),
        "charts": charts,
        "weights": weights,
    }


def main() -> None:
    result = run()
    print("DECISION MATRIX")
    print(result["kept"].round(2).to_string())
    print()
    if result["rejected"].empty:
        print("CONSTRAINTS: all offers pass (pay >= 80, commute <= 60).")
    else:
        print("REJECTED")
        print(result["rejected"].to_string())
    print()
    print("WEIGHTS", {k: result["weights"][k] for k in criterion_keys()})
    print()
    print("NORMALIZED (1 = best)")
    print(result["normalized"].round(3).to_string())
    print()
    print("WSM")
    for name, score in result["wsm"].items():
        print(f"  {name}: {score:.3f}")
    print()
    print("TOPSIS")
    for name, row in result["topsis"].iterrows():
        print(
            f"  {name}: C={row['topsis_score']:.3f}  "
            f"d+={row['distance_to_best']:.3f}  d-={row['distance_to_worst']:.3f}"
        )
    print()
    print("COMPARISON")
    show = result["table"][["wsm_score", "wsm_rank", "topsis_score", "topsis_rank"]].copy()
    print(show.round(3).to_string())
    print()
    print(result["explanation"])
    print()
    print("CHARTS")
    for label, path in result["charts"].items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
