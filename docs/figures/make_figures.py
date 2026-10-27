# Regenerates every PNG in docs/figures from data/router.pkl, model_runs.csv and app.db.
# Needs matplotlib (pip install matplotlib). Run from the repo root: MOCK_LLM=1 python docs/figures/make_figures.py
import sys; sys.path.insert(0, ".")
import backend, pickle, json, sqlite3, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score, brier_score_loss
from backend import config, router
from backend.data import load_queries, load_runs
from scripts.train import build_features

OUT = "docs/figures/"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left", "legend.frameon": False})
def save(fig, name): fig.tight_layout(); fig.savefig(OUT + name, facecolor="white"); plt.close(fig); print("saved", name)
def label_bars(ax, bars, fmt):
    for b in bars:
        w = b.get_width() if b.get_width() > b.get_height() * 0 and ax.name == "rectilinear" and hasattr(ax, "_horiz") else None
        ax.annotate(fmt(b.get_height()), (b.get_x() + b.get_width()/2, b.get_height()), ha="center", va="bottom",
                    fontsize=9, color=INK, xytext=(0, 2), textcoords="offset points")

b = pickle.loads(config.ROUTER_PKL.read_bytes())
q, runs = load_queries(), load_runs(); c = runs["correct"]
splits = ["train", "tune", "test"]
db = sqlite3.connect(config.DB_PATH)

# 1 data split
fig, ax = plt.subplots(figsize=(7, 3.6))
ct = pd.crosstab(q.split, q.task_type).loc[splits]
bot = np.zeros(3)
for tt, col in [("mcq", BLUE), ("math", ORANGE)]:
    bars = ax.bar(splits, ct[tt], bottom=bot, color=col, label=tt.upper(), width=0.55, edgecolor="white", linewidth=2)
    for x, (v, bt) in enumerate(zip(ct[tt], bot)): ax.text(x, bt + v/2, str(v), ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    bot += ct[tt].values
for x, t in enumerate(bot): ax.text(x, t + 4, f"{int(t)}", ha="center", color=INK, fontweight="bold")
ax.set_title("Dataset split: 500 questions, 60/20/20"); ax.set_ylabel("questions"); ax.legend(ncols=2, loc="upper right"); ax.grid(axis="x", visible=False)
save(fig, "1_data_split.png")

# 2 label rates
fig, ax = plt.subplots(figsize=(7.5, 3.8))
w = 0.25
for i, (t, col) in enumerate([("small", BLUE), ("medium", ORANGE), ("large", AQUA)]):
    vals = [c.loc[q[q.split == s].id, t].mean() for s in splits]
    bars = ax.bar(np.arange(3) + (i-1)*w, vals, w*0.92, color=col, label=t)
    for bb, v in zip(bars, vals): ax.text(bb.get_x()+bb.get_width()/2, v+0.01, f"{v:.0%}", ha="center", fontsize=9, color=INK)
ax.set_xticks(range(3), splits); ax.set_ylim(0, 1.08); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_title("Training labels: how often each LLM answered correctly"); ax.legend(ncols=3, loc="upper left"); ax.grid(axis="x", visible=False)
save(fig, "2_label_rates.png")

# features + predictions
probs, ys, X = {}, {}, {}
for s in splits:
    df = q[(q.split == s) & q.id.isin(c.index)]
    X[s] = build_features(df, exclude_group=(s == "train"))
    for t in ["small", "medium"]:
        ys[s, t] = c.loc[df.id, t].astype(int).to_numpy(); probs[s, t] = b["models"][t].predict_proba(X[s])[:, 1]
test = q[q.split == "test"]
ceil = {t: roc_auc_score(ys["test", t], test.difficulty.map(config.MOCK_ACCURACY[t]).to_numpy()) for t in ["small", "medium"]}
metrics = {f"{s}/{t}": {"auc": roc_auc_score(ys[s, t], probs[s, t]), "brier": brier_score_loss(ys[s, t], probs[s, t]),
           "base_brier": ys[s, t].mean()*(1-ys[s, t].mean())} for s in splits for t in ["small", "medium"]}

# 3 AUC
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for i, (t, col) in enumerate([("small", BLUE), ("medium", ORANGE)]):
    vals = [metrics[f"{s}/{t}"]["auc"] for s in splits]
    bars = ax.bar(np.arange(3) + (i-0.5)*0.36, vals, 0.33, color=col, label=f"P_{t} classifier")
    for bb, v in zip(bars, vals): ax.text(bb.get_x()+bb.get_width()/2, v+0.01, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.plot([2+(i-0.5)*0.36-0.17, 2+(i-0.5)*0.36+0.17], [ceil[t]]*2, color=INK, ls="--", lw=1.5)
ax.text(2.3, max(ceil.values())+0.02, "mock ceiling", fontsize=8, color=MUTED, ha="center")
ax.axhline(0.5, color=MUTED, lw=1, ls=":"); ax.text(1.5, 0.51, "random = 0.5", fontsize=8, color=MUTED)
ax.set_xticks(range(3), ["train (in-sample)", "tune", "test"]); ax.set_ylim(0.4, 1.08); ax.set_ylabel("ROC AUC")
ax.set_title("Classifier AUC: overfits train, modest on held-out"); ax.legend(loc="upper right"); ax.grid(axis="x", visible=False)
save(fig, "3_auc_by_split.png")

# 4 reliability on test (quantile bins)
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1, label="perfect calibration")
for t, col in [("small", BLUE), ("medium", ORANGE)]:
    p, y = probs["test", t], ys["test", t]
    bins = pd.qcut(p, 5, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "b": bins}).groupby("b", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    ax.plot(g.p, g.y, "-o", color=col, lw=2, ms=8, mec="white", mew=2, label=f"P_{t} (test)")
ax.axvline(b["tau_small"], color=BLUE, lw=1, ls="--"); ax.text(b["tau_small"]-0.01, 0.12, f"τ_small={b['tau_small']}", color=BLUE, fontsize=8, ha="right")
ax.axvline(b["tau_medium"], color=ORANGE, lw=1, ls="--"); ax.text(b["tau_medium"]-0.01, 0.05, f"τ_medium={b['tau_medium']}", color=ORANGE, fontsize=8, ha="right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("predicted probability"); ax.set_ylabel("actual fraction correct")
ax.set_title("Calibration on test (5 equal-size bins)"); ax.legend(loc="upper left", fontsize=9)
save(fig, "4_calibration_test.png")

# 5 tau sweep heatmap (test, from eval)
sw = pd.read_sql("select tau_small, tau_medium, accuracy, total_cost from eval_results where strategy='sweep'", db)
ev = pd.read_sql("select strategy, accuracy, total_cost, cost_per_correct, savings_pct, avg_ms from eval_results where strategy!='sweep'", db).set_index("strategy")
piv_acc = sw.pivot(index="tau_medium", columns="tau_small", values="accuracy")
fig, ax = plt.subplots(figsize=(7.5, 5.6))
im = ax.imshow(piv_acc.values, origin="lower", cmap="Blues", aspect="auto", vmin=piv_acc.values.min(), vmax=piv_acc.values.max())
ax.set_xticks(range(len(piv_acc.columns))[::2], [f"{v:.2f}" for v in piv_acc.columns][::2])
ax.set_yticks(range(len(piv_acc.index))[::2], [f"{v:.2f}" for v in piv_acc.index][::2])
xi, yi = list(piv_acc.columns).index(b["tau_small"]), list(piv_acc.index).index(b["tau_medium"])
ax.scatter([xi], [yi], s=180, facecolor="none", edgecolor=ORANGE, lw=2.5)
ax.annotate(f"chosen τ ({b['tau_small']}, {b['tau_medium']})", (xi, yi), xytext=(xi+1.5, yi-3), color=INK, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=ORANGE))
ax.set_xlabel("τ_small"); ax.set_ylabel("τ_medium"); ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.set_label("accuracy on test"); cb.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_title("τ sweep: test accuracy for all 225 threshold pairs")
save(fig, "5_tau_sweep_heatmap.png")

# 6 pareto
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.scatter(sw.total_cost, sw.accuracy, s=14, color="#9ca3af", label="225 τ pairs (test)", zorder=2)
pts = {"always-small": BLUE, "always-medium": YELLOW, "always-large": AQUA, "large+cache": VIOLET, "ours": ORANGE}
for k, col in pts.items():
    r = ev.loc[k]; ax.scatter(r.total_cost, r.accuracy, s=110, color=col, edgecolor="white", lw=2, zorder=3)
    ax.annotate(k, (r.total_cost, r.accuracy), xytext=(6, -12 if k == "large+cache" else 5), textcoords="offset points", fontsize=9, color=INK, fontweight="bold" if k == "ours" else None)
ax.set_xlabel(f"total cost on test ({config.CURRENCY})"); ax.set_ylabel("accuracy"); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_title("Cost vs accuracy: router vs baselines (test)"); ax.legend(loc="lower right", fontsize=9)
save(fig, "6_pareto.png")

# 7 eval bars: two separate charts
order = ["always-small", "always-medium", "always-large", "large+cache", "ours"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, col, title, fmt in [(axes[0], "accuracy", "Accuracy (test)", lambda v: f"{v:.1%}"), (axes[1], "cost_per_correct", f"Cost per correct answer ({config.CURRENCY})", lambda v: f"{v:.6f}")]:
    vals = ev.loc[order, col]
    bars = ax.barh(order, vals, color=[ORANGE if o == "ours" else "#94a3b8" for o in order], height=0.6)
    for bb, v in zip(bars, vals): ax.text(v, bb.get_y()+bb.get_height()/2, " "+fmt(v), va="center", fontsize=9, color=INK)
    ax.invert_yaxis(); ax.set_title(title); ax.grid(axis="y", visible=False); ax.set_xlim(0, vals.max()*1.3)
    if col == "accuracy": ax.set_xlim(0, 1.15); ax.set_xticks([0, .25, .5, .75, 1]); ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
save(fig, "7_eval_comparison.png")

# 8 routing mix
first = pd.Series([router.pick(a, m, b["tau_small"], b["tau_medium"])[0] for a, m in zip(probs["test", "small"], probs["test", "medium"])]).value_counts()
final = pd.read_sql("select model_used, count(*) n from requests where source='eval:ours' group by 1", db).set_index("model_used").n
cats = [("cache", VIOLET), ("small", BLUE), ("medium", ORANGE), ("large", AQUA)]
fig, ax = plt.subplots(figsize=(8, 2.8))
for row, (lab, s) in enumerate([("router's first choice\n(no cache)", first), ("final model used\n(after cache + escalation)", final)]):
    left = 0
    for k, col in cats:
        v = int(s.get(k, 0))
        if v: ax.barh(row, v, left=left, color=col, edgecolor="white", lw=2, height=0.6, label=k if row == 1 else None)
        if v >= 10: ax.text(left+v/2, row, f"{k}\n{v}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        elif v: ax.text(left+v/2, row, str(v), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        left += v
ax.set_yticks([0, 1], ["router's first choice\n(no cache)", "final model used\n(after cache + escalation)"]); ax.invert_yaxis()
ax.set_xlabel("test questions (102)"); ax.grid(False); ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.35))
ax.set_title("Routing mix on test")
save(fig, "8_routing_mix.png")

# 9 feature importance by group
imp = {t: np.mean([cc.estimator.booster_.feature_importance("gain") for cc in b["models"][t].calibrated_classifiers_], axis=0) for t in ["small", "medium"]}
groups = {"embedding (384)": slice(0, 384), "text features (10)": slice(384, 394), "kNN features (4)": slice(394, 398)}
fig, ax = plt.subplots(figsize=(7.5, 3))
for row, t in enumerate(["small", "medium"]):
    left = 0
    for (g, sl), col in zip(groups.items(), [BLUE, ORANGE, AQUA]):
        v = imp[t][sl].sum()/imp[t].sum()
        ax.barh(row, v, left=left, color=col, edgecolor="white", lw=2, height=0.6, label=g if row == 0 else None)
        ax.text(left+v/2, row, f"{v:.0%}" if v > 0.06 else "", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        left += v
ax.set_yticks([0, 1], ["P_small", "P_medium"]); ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.grid(False); ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.2)); ax.set_title("Share of split gain by feature group")
save(fig, "9_feature_importance.png")
fi = {t: {g: float(imp[t][sl].sum()/imp[t].sum()) for g, sl in groups.items()} for t in imp}

# 10 two-tier experiment
rep = json.load(open("models/2tier/router_report.json"))
keys = ["always_small", "always_large", "oracle", "router"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, m, title, fmt in [(axes[0], "accuracy", "Accuracy (test)", lambda v: f"{v:.1%}"), (axes[1], "cost_usd", "Total cost on test (USD)", lambda v: f"{v:.4f}")]:
    vals = [rep[k][m] for k in keys]
    bars = ax.barh(keys, vals, color=[ORANGE if k == "router" else "#94a3b8" for k in keys], height=0.6)
    for bb, v in zip(bars, vals): ax.text(v, bb.get_y()+bb.get_height()/2, " "+fmt(v), va="center", fontsize=9, color=INK)
    ax.invert_yaxis(); ax.set_title(title); ax.grid(axis="y", visible=False); ax.set_xlim(0, max(vals)*1.3)
    if m == "accuracy": ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
fig.suptitle("Earlier experiment: 2-tier logistic-regression router, real qwen2.5 0.5b vs 1.5b", x=0.01, ha="left", fontsize=11, color=MUTED)
save(fig, "10_two_tier_experiment.png")

# 11 training flow diagram
fig, ax = plt.subplots(figsize=(12, 2.6)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0.3, 2.7)
steps = [("queries.csv\n500 Qs", "train 301 / tune 97\n/ test 102", "#eef2ff"),
         ("setup.py", "each Q × 3 LLMs\n→ graded 0/1 labels", "#fff4ed"),
         ("features", "384 embedding\n+ 10 text + 4 kNN", "#f5f3ff"),
         ("2 × LightGBM", "200 trees, lr 0.05\n+ isotonic (5-fold)", "#f5f3ff"),
         ("τ sweep", "225 pairs on OOF train\n+ tune, replay escalation", "#ecfdf5"),
         ("router.pkl", "τ_small 0.65\nτ_medium 0.90", "#ecfdf5")]
wbox = 1.7
for i, (h, sub, fc) in enumerate(steps):
    x = 0.15 + i*1.98
    ax.add_patch(FancyBboxPatch((x, 0.9), wbox, 1.7, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec="#94a3b8", lw=1.2))
    ax.text(x+wbox/2, 2.2, h, ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(x+wbox/2, 1.45, sub, ha="center", va="center", fontsize=8.5, color=MUTED)
    if i < len(steps)-1: ax.annotate("", (x+wbox+0.27, 1.75), (x+wbox+0.02, 1.75), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
ax.text(0.15, 0.45, "eval.py uses the test split once, after training. scripts/train.py makes no LLM calls; it reuses graded answers from setup.py.", fontsize=9, color=MUTED)
ax.set_title("Router training pipeline", loc="left")
save(fig, "0_training_pipeline.png")

json.dump({"metrics": metrics, "ceiling": ceil, "feature_share": fi, "first": first.to_dict(), "final": final.to_dict(),
           "tau": [b["tau_small"], b["tau_medium"]], "version": b["version"]}, open("docs/figures/metrics.json", "w"), indent=1, default=float)
