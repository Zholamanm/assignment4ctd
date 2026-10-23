"""Streamlit UI for the internship DSS: friendly weights, clear ranking, charts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dss import CRITERIA, DEFAULT_WEIGHTS, WEIGHT_TOLERANCE, run

st.set_page_config(
    page_title="Internship DSS",
    page_icon=":briefcase:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1180px; }
      h1, h2, h3 { letter-spacing: -0.02em; }
      div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #E2DDD4;
        border-radius: 14px;
        padding: 0.75rem 1rem;
      }
      .winner-card {
        background: linear-gradient(135deg, #E7F3EF 0%, #F7F5F1 70%);
        border: 1px solid #1F7A6B;
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        margin: 0.4rem 0 1rem 0;
      }
      .winner-card h3 { margin: 0 0 0.35rem 0; color: #12324A; }
      .winner-card p { margin: 0; color: #5C6B7A; }
      .soft-note {
        background: #F7F5F1;
        border-left: 4px solid #1F7A6B;
        padding: 0.7rem 0.9rem;
        border-radius: 0 10px 10px 0;
        color: #142033;
        margin-bottom: 0.8rem;
      }
      .method-pill {
        display: inline-block;
        background: #12324A;
        color: #F4F7F8;
        border-radius: 999px;
        padding: 0.15rem 0.65rem;
        font-size: 0.78rem;
        margin-right: 0.35rem;
      }
      .benefit { color: #1F7A6B; font-weight: 600; }
      .cost { color: #C47B2B; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

PRESETS = {
    "Balanced (default)": dict(DEFAULT_WEIGHTS),
    "Learning first": {
        "pay": 0.15,
        "commute": 0.10,
        "learning": 0.45,
        "contract": 0.10,
        "overtime": 0.20,
    },
    "Pay first": {
        "pay": 0.45,
        "commute": 0.15,
        "learning": 0.15,
        "contract": 0.10,
        "overtime": 0.15,
    },
    "Low stress": {
        "pay": 0.15,
        "commute": 0.25,
        "learning": 0.20,
        "contract": 0.10,
        "overtime": 0.30,
    },
}


def medal(rank: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")


def friendly_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {key: label for key, label, _, _ in CRITERIA}
    out = frame.rename(columns=rename).copy()
    out.index.name = "Offer"
    return out


def normalize_weights(raw: dict[str, float]) -> dict[str, float]:
    total = float(sum(raw.values()))
    if total <= 0:
        equal = 1.0 / len(raw)
        return {key: equal for key in raw}
    return {key: value / total for key, value in raw.items()}


def slider_key(criterion: str) -> str:
    return f"imp_{criterion}"


def apply_preset(name: str) -> None:
    """Write preset values into the widget session keys Streamlit actually uses."""
    values = PRESETS[name]
    for key, value in values.items():
        st.session_state[slider_key(key)] = float(value)
    st.session_state.active_preset = name


def plot_score_bars(table: pd.DataFrame):
    order = table.sort_values("wsm_score", ascending=True)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    y = range(len(order))
    height = 0.35
    ax.barh(
        [i - height / 2 for i in y],
        order["wsm_score"],
        height=height,
        label="Weighted Sum",
        color="#1F7A6B",
    )
    ax.barh(
        [i + height / 2 for i in y],
        order["topsis_score"],
        height=height,
        label="TOPSIS",
        color="#12324A",
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(order.index)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score (higher is better)")
    ax.set_title("How the two methods score each offer")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_heatmap(normalized: pd.DataFrame):
    labels = [label for _, label, _, _ in CRITERIA]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    image = ax.imshow(normalized.to_numpy(), cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_yticks(range(len(normalized.index)))
    ax.set_yticklabels(normalized.index)
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            ax.text(
                j,
                i,
                f"{float(normalized.iloc[i, j]):.2f}",
                ha="center",
                va="center",
                color="#142033",
                fontsize=9,
            )
    ax.set_title("Normalized values (1 = best on that criterion)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


if "active_preset" not in st.session_state:
    apply_preset("Balanced (default)")

# ----- Sidebar -----
with st.sidebar:
    st.header("What matters to you?")

    weight_mode = st.radio(
        "How should weights work?",
        [
            "Relative importance (auto-fix to 100%)",
            "Exact weights (must total 1.00)",
        ],
        index=0,
    )
    exact_mode = weight_mode.startswith("Exact")

    if exact_mode:
        st.caption(
            "Each slider is a real weight. The five values must add up to exactly 1.00. "
            "If Monthly pay is 1.00, the other four must be 0.00."
        )
    else:
        st.caption(
            "Sliders are relative importance, not final weights. "
            "The app rescales them so the final weights always sum to 100%. "
            "Monthly pay at 1.00 with others still above 0 is fine - it just gets a larger share."
        )

    preset_name = st.selectbox(
        "Quick preset",
        list(PRESETS.keys()),
        index=list(PRESETS.keys()).index(st.session_state.active_preset)
        if st.session_state.active_preset in PRESETS
        else 0,
    )

    if preset_name != st.session_state.active_preset:
        apply_preset(preset_name)
        st.rerun()

    if st.button("Apply this preset now", use_container_width=True):
        apply_preset(preset_name)
        st.rerun()

    st.divider()
    raw_weights: dict[str, float] = {}
    for key, label, kind, unit in CRITERIA:
        kind_html = (
            '<span class="benefit">higher is better</span>'
            if kind == "benefit"
            else '<span class="cost">lower is better</span>'
        )
        st.markdown(f"**{label}**  \n{kind_html} | {unit}", unsafe_allow_html=True)
        raw_weights[key] = st.slider(
            label,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key=slider_key(key),
            label_visibility="collapsed",
        )

    raw_total = float(sum(raw_weights.values()))

    st.divider()
    st.subheader("Final weights used for ranking")

    if raw_total <= 0:
        st.error("All criteria are 0. Raise at least one slider.")
        st.stop()

    if exact_mode:
        if abs(raw_total - 1.0) > WEIGHT_TOLERANCE:
            st.error(
                f"Exact weights must total 1.00. Current total is {raw_total:.2f}. "
                "Lower some sliders or switch to Relative importance."
            )
            st.stop()
        weights = dict(raw_weights)
        st.success(f"Exact total = {raw_total:.2f}")
    else:
        weights = normalize_weights(raw_weights)
        if abs(raw_total - 1.0) > 0.001:
            st.info(
                f"Slider total is {raw_total:.2f}. "
                f"Auto-normalized to 100% for ranking."
            )
        else:
            st.success("Slider total is already 1.00")

    weight_df = pd.DataFrame(
        {
            "Criterion": [label for _, label, _, _ in CRITERIA],
            "Slider": [raw_weights[key] for key, _, _, _ in CRITERIA],
            "Final weight": [weights[key] for key, _, _, _ in CRITERIA],
        }
    )
    st.dataframe(
        weight_df.style.format({"Slider": "{:.2f}", "Final weight": "{:.0%}"}).bar(
            subset=["Final weight"], color="#1F7A6B", vmin=0, vmax=1
        ),
        hide_index=True,
        use_container_width=True,
    )

# ----- Main -----
st.title("Choose an internship offer")
st.markdown(
    '<div class="soft-note">'
    "Set what matters in the sidebar, then compare the ranked offers. "
    '<span class="method-pill">WSM</span>'
    '<span class="method-pill">TOPSIS</span>'
    "are two ranking methods. They often agree on the winner and disagree on the middle."
    "</div>",
    unsafe_allow_html=True,
)

try:
    result = run(weights=weights, save_charts=False)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

table = result["table"]
wsm_winner = table.sort_values("wsm_rank").index[0]
topsis_winner = table.sort_values("topsis_rank").index[0]
same_winner = wsm_winner == topsis_winner

if same_winner:
    st.markdown(
        f"""
        <div class="winner-card">
          <h3>Recommended offer: {wsm_winner}</h3>
          <p>Both Weighted Sum and TOPSIS pick this offer with your current preferences.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="winner-card">
          <h3>Methods disagree on the winner</h3>
          <p>Weighted Sum prefers <b>{wsm_winner}</b>. TOPSIS prefers <b>{topsis_winner}</b>.
          Check the ranks below and decide which trade-off you accept.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("WSM winner", wsm_winner, medal(1))
with m2:
    st.metric("WSM score", f"{float(table.loc[wsm_winner, 'wsm_score']):.2f}")
with m3:
    st.metric("TOPSIS winner", topsis_winner, medal(1))
with m4:
    st.metric("TOPSIS score", f"{float(table.loc[topsis_winner, 'topsis_score']):.2f}")

st.subheader("Ranked offers")
rank_view = (
    table.reset_index()
    .rename(
        columns={
            "alternative": "Offer",
            "wsm_score": "WSM score",
            "wsm_rank": "WSM rank",
            "topsis_score": "TOPSIS score",
            "topsis_rank": "TOPSIS rank",
        }
    )
    .sort_values("WSM rank")
)
rank_view["WSM rank"] = rank_view["WSM rank"].map(medal)
rank_view["TOPSIS rank"] = rank_view["TOPSIS rank"].map(medal)
st.dataframe(
    rank_view.style.format({"WSM score": "{:.3f}", "TOPSIS score": "{:.3f}"}).background_gradient(
        subset=["WSM score", "TOPSIS score"], cmap="YlGn"
    ),
    hide_index=True,
    use_container_width=True,
)

with st.expander("Why do the methods look different?", expanded=True):
    st.write(result["explanation"])
    st.caption(
        "Weighted Sum rewards a strong strength that can balance a weakness. "
        "TOPSIS punishes being far from the ideal offer, so one bad cost can hurt more."
    )

chart_left, chart_right = st.columns(2)
with chart_left:
    st.subheader("Score chart")
    st.pyplot(plot_score_bars(table), clear_figure=True)
with chart_right:
    st.subheader("Normalized matrix")
    st.pyplot(plot_heatmap(result["normalized"]), clear_figure=True)

st.subheader("Offer details")
if not result["rejected"].empty:
    st.warning("Some offers failed a hard limit and were removed before ranking.")
    st.dataframe(
        friendly_matrix(result["rejected"].drop(columns=["failed_constraint"], errors="ignore")),
        use_container_width=True,
    )

detail_tabs = st.tabs(["Raw scores", "Normalized (0-1)", "How to read this"])
with detail_tabs[0]:
    st.caption("Original numbers for each internship.")
    units = " | ".join(f"{label}: {unit}" for _, label, _, unit in CRITERIA)
    st.caption(units)
    st.dataframe(friendly_matrix(result["kept"]), use_container_width=True)
with detail_tabs[1]:
    st.caption("After min-max normalization. 1 always means best on that column, including costs.")
    st.dataframe(friendly_matrix(result["normalized"].round(3)), use_container_width=True)
with detail_tabs[2]:
    st.markdown(
        """
**Pay, learning, contract** - higher values are better.  
**Commute, overtime** - lower values are better.  
**Constraints** - pay must be at least 80 thousand tenge/month, commute at most 60 minutes.  
**Sidebar** - use Relative mode for easy exploration, or Exact mode when you want weights that already sum to 1.00.
        """
    )
