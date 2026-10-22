"""Streamlit UI: change internship weights and rerank with WSM and TOPSIS."""

from __future__ import annotations

import streamlit as st

from dss import CRITERIA, DEFAULT_WEIGHTS, WEIGHT_TOLERANCE, run

st.set_page_config(page_title="Internship DSS", layout="wide")
st.title("Internship offer DSS")
st.caption("Change the weights. They must sum to 1. The ranking is recomputed with WSM and TOPSIS.")

st.subheader("Weights")
cols = st.columns(len(CRITERIA))
weights: dict[str, float] = {}
for column, (key, label, kind, unit) in zip(cols, CRITERIA):
    with column:
        st.markdown(f"**{label}**")
        st.caption(f"{kind} | {unit}")
        weights[key] = st.number_input(
            key,
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_WEIGHTS[key]),
            step=0.01,
            format="%.2f",
            label_visibility="collapsed",
        )

total = float(sum(weights.values()))
if abs(total - 1.0) <= WEIGHT_TOLERANCE:
    st.success(f"Weight sum = {total:.2f}")
else:
    st.error(f"Weight sum = {total:.2f}. It must be 1.00 before the ranking runs.")
    st.stop()

try:
    result = run(weights=weights, save_charts=False)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader("Decision matrix")
    st.dataframe(result["kept"], use_container_width=True)
with right:
    st.subheader("Normalized matrix")
    st.caption("1 means best on that column, including cost criteria.")
    st.dataframe(result["normalized"].round(3), use_container_width=True)

if not result["rejected"].empty:
    st.warning("Some offers failed a constraint and were removed.")
    st.dataframe(result["rejected"], use_container_width=True)

st.subheader("Ranking")
show = result["table"][["wsm_score", "wsm_rank", "topsis_score", "topsis_rank"]].round(3)
st.dataframe(show, use_container_width=True)
st.info(result["explanation"])

winner = result["table"].sort_values("wsm_rank").index[0]
st.subheader(f"WSM recommendation: {winner}")

score_col, heat_col = st.columns(2)
with score_col:
    st.subheader("Scores")
    chart = result["table"][["wsm_score", "topsis_score"]].rename(
        columns={"wsm_score": "WSM", "topsis_score": "TOPSIS"}
    )
    st.bar_chart(chart)
with heat_col:
    st.subheader("Normalized heatmap")
    st.dataframe(
        result["normalized"].style.background_gradient(cmap="YlGn", axis=None).format("{:.2f}"),
        use_container_width=True,
    )
