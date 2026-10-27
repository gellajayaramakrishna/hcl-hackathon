# Streamlit report (reads data/app.db read-only): cost per correct answer, savings, Pareto curve,
# routing mix, cache hit / wrong-hit rate, latency per step, failures and retries.
# Run: streamlit run report/app.py
import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "app.db"
CURRENCY = yaml.safe_load((ROOT / "data" / "prices.yaml").read_text()).get("currency", "USD")
SYMBOL = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£"}.get(CURRENCY, CURRENCY + " ")
BLUE, GRAY = "#2a78d6", "#a3a29c"

st.set_page_config(page_title="Router Report", layout="wide")


def read(sql, params=()):
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def money(v):
    return f"{SYMBOL}0" if v == 0 else f"{SYMBOL}{v:.6f}" if v < 0.01 else f"{SYMBOL}{v:.4f}"


def style(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hoverlabel=dict(font_size=12))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig


st.title("Cost-Aware Model Router: Report")
if not DB.exists():
    st.error("data/app.db not found. Run scripts/setup.py, train.py and eval.py first.")
    st.stop()

evals = read("SELECT * FROM eval_results WHERE run_ts = (SELECT MAX(run_ts) FROM eval_results)")
if evals.empty:
    st.warning("No evaluation yet. Run: MOCK_LLM=1 python scripts/eval.py")
    st.stop()
strategies = evals[evals.strategy != "sweep"].set_index("strategy")
sweep = evals[evals.strategy == "sweep"]
ours, large = strategies.loc["ours"], strategies.loc["always-large"]
st.caption(f"Latest evaluation: {evals.run_ts.iloc[0]} · TEST split · costs in {CURRENCY}")

# --- 1. headline: cost per correct answer ---
st.header("Cost per correct answer")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ours", money(ours.cost_per_correct),
          f"{100 * (ours.cost_per_correct / large.cost_per_correct - 1):.1f}% vs always-large", delta_color="inverse")
c2.metric("Always-large", money(large.cost_per_correct))
c3.metric("Savings (total cost)", f"{ours.savings_pct:.1f}%")
c4.metric("Accuracy: ours / large", f"{100 * ours.accuracy:.1f}% / {100 * large.accuracy:.1f}%")

table = strategies[["accuracy", "total_cost", "cost_per_correct", "savings_pct", "avg_ms",
                    "cache_hit_rate", "wrong_hit_rate"]].copy()
for col in ["accuracy", "cache_hit_rate", "wrong_hit_rate"]:
    table[col] = (100 * table[col]).round(1).astype(str) + "%"
table["savings_pct"] = table["savings_pct"].round(1).astype(str) + "%"
table["total_cost"] = table["total_cost"].map(money)
table["cost_per_correct"] = table["cost_per_correct"].map(money)
table["avg_ms"] = table["avg_ms"].round(0).astype(int)
st.dataframe(table, width="stretch")

# --- 2. Pareto curve ---
st.header("Cost vs accuracy (Pareto curve)")
st.caption("Each grey dot is one (τ_small, τ_medium) setting on the test split (routing + escalation, no cache). "
           "The line is the Pareto frontier: no other setting is both cheaper and more accurate. "
           "Diamonds are baselines; the star is our full pipeline run (router + escalation + cache).")
if not sweep.empty:
    pts = sweep.sort_values(["total_cost", "accuracy"], ascending=[True, False])
    frontier = pts[pts.accuracy > pts.accuracy.cummax().shift(fill_value=-1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep.total_cost, y=100 * sweep.accuracy, mode="markers", name="τ settings",
                             marker=dict(color=GRAY, size=7, opacity=0.5),
                             customdata=sweep[["tau_small", "tau_medium"]],
                             hovertemplate="τ_small=%{customdata[0]}, τ_medium=%{customdata[1]}<br>"
                                           "cost=%{x:.5f}<br>accuracy=%{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=frontier.total_cost, y=100 * frontier.accuracy, mode="lines", name="Pareto frontier",
                             line=dict(color=BLUE, width=2), hoverinfo="skip"))
    for name in ["always-small", "always-medium", "always-large", "large+cache", "ours"]:
        r = strategies.loc[name]
        is_ours = name == "ours"
        label = "ours (full pipeline)" if is_ours else name
        fig.add_trace(go.Scatter(x=[r.total_cost], y=[100 * r.accuracy], mode="markers+text", name=name,
                                 text=[label], textposition="top center" if is_ours else "bottom center",
                                 marker=dict(size=14 if is_ours else 10, color=BLUE if is_ours else "#52514e",
                                             symbol="star" if is_ours else "diamond",
                                             line=dict(width=2, color="white")),
                                 hovertemplate=f"{name}<br>cost=%{{x:.5f}}<br>accuracy=%{{y:.1f}}%<extra></extra>"))
    fig.update_layout(xaxis_title=f"Total cost on test ({CURRENCY})", yaxis_title="Accuracy (%)", showlegend=False)
    st.plotly_chart(style(fig, 420), width="stretch")

# --- traffic source for the remaining sections ---
source = st.radio("Traffic for the sections below", ["eval:ours", "live"], horizontal=True,
                  format_func=lambda s: "Evaluation run (ours, test split)" if s == "eval:ours" else "Live UI traffic")
reqs = read("SELECT * FROM requests WHERE source = ?", (source,))
calls = read("SELECT l.* FROM llm_calls l JOIN requests r USING (trace_id) WHERE r.source = ?", (source,))
if reqs.empty:
    st.info("No requests for this source yet.")
    st.stop()

# --- 3. routing mix ---
left, right = st.columns(2)
with left:
    st.header("Routing mix")
    order = ["small", "medium", "large", "cache"]
    mix = reqs.model_used.value_counts().reindex(order, fill_value=0).reset_index()
    mix.columns = ["model", "requests"]
    fig = px.bar(mix, x="model", y="requests", text="requests", color_discrete_sequence=[BLUE])
    fig.update_traces(marker_cornerradius=4, textposition="outside", cliponaxis=False,
                      hovertemplate="%{x}: %{y} requests<extra></extra>")
    st.plotly_chart(style(fig, 300), width="stretch")
    esc = reqs.escalations.value_counts().sort_index()
    st.caption("Escalations per request: " + ", ".join(f"{k} → {v} requests" for k, v in esc.items()))

# --- 4. cache ---
with right:
    st.header("Cache")
    hits = reqs[reqs.cache_type != "none"]
    a, b, c = st.columns(3)
    a.metric("Hit rate", f"{100 * len(hits) / len(reqs):.1f}%")
    a.caption(f"exact {sum(hits.cache_type == 'exact')} · similar {sum(hits.cache_type == 'similar')}")
    if source == "eval:ours":
        b.metric("Wrong-hit rate", f"{100 * ours.wrong_hit_rate:.1f}%")
        b.caption("cache hits whose answer was wrong")
    else:
        bad = read("""SELECT COUNT(*) n FROM feedback f JOIN requests r USING (trace_id)
                      WHERE r.source = 'live' AND r.cache_type != 'none' AND f.thumbs = -1""").n[0]
        b.metric("👎 on cache hits", int(bad))
        b.caption("live traffic has no reference answers")
    c.metric("Cost of hits", money(0.0))
    c.caption("a hit skips the model call")

# --- 5. latency per step ---
st.header("Latency per step")
steps = pd.DataFrame([json.loads(x) for x in reqs.latency_json.dropna()])
if not steps.empty:
    avg = steps.mean().reset_index()
    avg.columns = ["step", "avg_ms"]
    fig = px.bar(avg, x="avg_ms", y="step", orientation="h", text=avg.avg_ms.round(1),
                 color_discrete_sequence=[BLUE])
    fig.update_traces(marker_cornerradius=4, textposition="outside", cliponaxis=False,
                      hovertemplate="%{y}: %{x:.1f} ms average<extra></extra>")
    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=list(avg.step)[::-1]),
                      xaxis_title="Average ms per request (steps that ran)", yaxis_title=None)
    st.plotly_chart(style(fig, 340), width="stretch")
    st.caption(f"Average total time per request: {reqs.total_ms.mean():.0f} ms. "
               "Cache hits stop after the cache steps, so later steps average over fewer requests.")

# --- 6. failures and retries ---
st.header("Failures and retries")
if calls.empty:
    st.info("No model calls for this source.")
else:
    per_model = calls.groupby("model").agg(calls=("ok", "size"), failed=("ok", lambda s: int((s == 0).sum())),
                                           avg_ms=("ms", "mean"), cost=("cost", "sum")).reindex(["small", "medium", "large"]).dropna()
    per_model["failure_rate"] = (100 * per_model.failed / per_model.calls).round(1).astype(str) + "%"
    per_model["avg_ms"] = per_model.avg_ms.round(0).astype(int)
    per_model["cost"] = per_model.cost.map(money)
    per_model[["calls", "failed"]] = per_model[["calls", "failed"]].astype(int)
    st.dataframe(per_model, width="stretch")
    failed_traces = calls[calls.ok == 0].trace_id.nunique()
    st.caption(f"{int((calls.ok == 0).sum())} failed calls in {failed_traces} requests. A failed call is retried "
               "once, then the next bigger model is tried. Calls include the small model's 2 extra "
               "self-consistency samples.")
