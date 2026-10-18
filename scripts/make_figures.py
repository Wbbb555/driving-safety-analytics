"""生成 README 与文档要用的图。

设计原则：每张图只讲一件事，且必须能被一句 caption 说清。
画不出"这图说明了什么"的图就不画。

跑法：
    python scripts/make_figures.py

输出到 docs/figures/。全部基于 data/synthetic 里的合成数据。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paths import load_bundle, resolve_data_dir  # noqa: E402
from task1.run_task1 import build_features, cv_oof, rank_fuse  # noqa: E402
from task2.run_task2 import fit_card, score_from_card  # noqa: E402

# 中文字体：优先用 Windows 自带字体，找不到就退回英文标注
for _f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


# --------------------------------------------------------------- 图 1


def fig_exposure(bundle, out: Path):
    """暴露量混杂：纯里程特征本身就有很高的 AUC，残差化后归零。"""
    lab = bundle["labels"].set_index("gpsno")
    traj = bundle["traj_daily"]
    agg = traj.groupby("gpsno").agg(
        n_points=("n_points", "sum"), km=("dist_km", "sum")
    ).reindex(lab.index)
    y = lab["label"].values.astype(int)
    ok = agg["km"].to_numpy() > 0
    x_raw = agg["n_points"].to_numpy()[ok]
    km = agg["km"].to_numpy()[ok]
    yy = y[ok]

    # 对数—对数残差化（正确口径）
    lx, lk = np.log1p(x_raw), np.log1p(km)
    coef = np.polyfit(lk, lx, 1)
    resid = lx - np.polyval(coef, lk)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))

    ax = axes[0]
    ax.scatter(km, x_raw, s=9, alpha=0.55, c=np.where(yy == 1, "#c0392b", "#7f8c8d"))
    ax.set_xlabel("里程 km")
    ax.set_ylabel("采样点数")
    ax.set_title(f"采样点数 ≈ 里程  (Spearman "
                 f"{pd.Series(x_raw).corr(pd.Series(km), method='spearman'):+.3f})",
                 fontsize=10)

    for ax, vals, title in (
        (axes[1], x_raw, "原始采样点数"),
        (axes[2], resid, "对数—对数残差化后"),
    ):
        fpr, tpr, _ = roc_curve(yy, vals)
        auc = roc_auc_score(yy, vals)
        ax.plot(fpr, tpr, lw=2,
                color="#c0392b" if auc > 0.65 else "#27ae60")
        ax.plot([0, 1], [0, 1], "--", lw=1, color="#bdc3c7")
        ax.set_xlabel("假阳率")
        ax.set_ylabel("真阳率")
        ax.set_title(f"{title}\nAUC = {auc:.4f}", fontsize=10)
        ax.set_aspect("equal")

    fig.suptitle("陷阱一：这不是行为信号，是「跑得多」", fontsize=12, y=1.02)
    _save(fig, out, "trap1_exposure.png")
    return {"auc_raw": float(roc_auc_score(yy, x_raw)),
            "auc_resid": float(roc_auc_score(yy, resid))}


# --------------------------------------------------------------- 图 2


def fig_selection_bias(out: Path, metrics: dict):
    """特征选择偏差：top_k 扫描曲线在噪声带宽内来回跳。"""
    scan = metrics["T2_feature_selection"]["scan"]
    ks = sorted(int(k) for k in scan)
    aucs = [scan[str(k)] if str(k) in scan else scan[k] for k in ks]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(ks, aucs, "o-", lw=2, color="#2980b9")
    lo, hi = min(aucs), max(aucs)
    ax.axhspan(lo, hi, color="#3498db", alpha=0.12)
    ax.axhline(metrics["T2_feature_selection"]["naive"],
               ls="--", lw=1.6, color="#c0392b", label="朴素（全样本选特征）")
    ax.axhline(metrics["T2_feature_selection"]["nested"],
               ls=":", lw=1.6, color="#27ae60", label="嵌套（折内选特征）")
    ax.set_xlabel("选入特征数 top_k")
    ax.set_ylabel("OOF AUC")
    ax.set_title(f"top_k 扫描：全程仅波动 {hi - lo:.4f}，无单调趋势", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.annotate(f"波动带宽 {lo:.4f} ~ {hi:.4f}", xy=(ks[1], hi),
                xytext=(0, 8), textcoords="offset points", fontsize=9,
                color="#2c3e50")
    _save(fig, out, "trap2_selection.png")


# --------------------------------------------------------------- 图 3


def fig_sign_stability(out: Path, metrics: dict):
    """符号稳定性：哪些指标的系数符号一换种子就翻。

    注意 demo_traps 里 ``stab`` 记的是「系数为**负**的重抽样占比」，
    而负系数在这里意味着反常识方向（率越高反而越安全）。
    所以「符合常识方向」的占比是 ``1 - stab``。第一版直接拿 stab 当
    正向占比画，图和 README 表格正好相反——这种错误不会报错，只会静默说谎。
    """
    stab = metrics["T3_collinearity"]["stab"]
    names = list(stab.keys())
    good = [(1.0 - stab[n]) * 100 for n in names]      # 符合常识方向（系数为正）的占比
    order = np.argsort(good)[::-1]
    names = [names[i] for i in order]
    good = [good[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    # 三档配色与 README 的「判读」列一致
    colors = ["#27ae60" if v >= 95 else "#e67e22" if v >= 50 else "#c0392b"
              for v in good]
    ax.barh(names, good, color=colors)
    ax.axvline(95, ls="--", lw=1.2, color="#7f8c8d")
    ax.axvline(50, ls=":", lw=1.2, color="#7f8c8d")
    ax.set_xlabel("系数方向符合常识（事件越多越危险）的重抽样占比 %")
    # 右边留 18% 余量，否则 100.0% 的标签会被裁掉
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_title("陷阱三：单个指标的系数符号本身就不稳（B=200）", fontsize=11)
    for i, v in enumerate(good):
        ax.text(v + 2, i, f"{v:.1f}%", va="center", fontsize=8.5)
    ax.invert_yaxis()
    # 用真正的图例放在右下角（那里是空白区域），比在图外塞一行文字可靠：
    # 图外文字在 bbox_inches="tight" 下很容易与标题或 xlabel 撞上。
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#27ae60", label="稳定且方向正确"),
            Patch(facecolor="#e67e22", label="符号不稳"),
            Patch(facecolor="#c0392b", label="稳定反号"),
        ],
        loc="lower right", fontsize=8.6, framealpha=0.95,
    )
    _save(fig, out, "trap3_sign_stability.png")


# --------------------------------------------------------------- 图 4


def fig_scorecard_lift(bundle, out: Path):
    """评分卡十分位 lift：分数分档后出险率必须单调下降。"""
    cand = bundle["cand"]
    cols = [c for c in cand.columns if "|" in c]
    names = [c.split("|")[-1] for c in cols]
    y = cand["label"].values.astype(int)
    card = fit_card(cand[cols].values.astype(float), y, names)
    sc = score_from_card(card, cand[cols].values.astype(float))

    dec = pd.qcut(pd.Series(sc).rank(method="first"), 10, labels=False)
    tbl = pd.DataFrame({"dec": dec, "y": y}).groupby("dec")["y"].agg(["size", "mean"])

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.bar(range(1, 11), tbl["mean"] * 100, color="#c0392b", alpha=0.82)
    ax.set_xlabel("十分位（1 = 安全分最低 = 最高风险）")
    ax.set_ylabel("出险率 %")
    ax.set_xticks(range(1, 11))
    ax.set_title("任务二：评分卡十分位出险率（合成数据）", fontsize=11)
    base = y.mean() * 100
    ax.axhline(base, ls="--", lw=1.3, color="#2c3e50",
               label=f"整体出险率 {base:.1f}%")
    ax.legend(fontsize=9)
    for i, v in enumerate(tbl["mean"] * 100):
        ax.text(i + 1, v + 0.8, f"{v:.0f}%", ha="center", fontsize=8)
    _save(fig, out, "task2_decile.png")
    return {"deciles": [round(float(v * 100), 2) for v in tbl["mean"]]}


# --------------------------------------------------------------- 图 5


def fig_task1_roc(bundle, out: Path):
    """任务一：三个模型的 ROC 与秩融合对照。"""
    df = build_features(bundle)
    feat_cols = [c for c in df.columns
                 if c not in ("gpsno", "label", "accident_cnt", "near_miss_cnt")]
    X = df[feat_cols].values.astype(float)
    y = df["label"].values.astype(int)
    groups = df["gpsno"].values

    oof, aucs = cv_oof(X, y, groups, 20260601)
    fused = rank_fuse(oof)

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    palette = {"RF": "#95a5a6", "ET": "#2980b9", "LR-L1": "#8e44ad"}
    for k, v in sorted(aucs.items(), key=lambda t: t[1]):
        fpr, tpr, _ = roc_curve(y, oof[k])
        ax.plot(fpr, tpr, lw=1.7, color=palette.get(k, "#7f8c8d"),
                label=f"{k}  {v:.4f}")
    fpr, tpr, _ = roc_curve(y, fused)
    ax.plot(fpr, tpr, lw=2.6, color="#c0392b",
            label=f"秩融合  {roc_auc_score(y, fused):.4f}")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="#bdc3c7")
    ax.set_xlabel("假阳率")
    ax.set_ylabel("真阳率")
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("任务一：按车分组 5 折 OOF ROC（合成数据）", fontsize=11)
    _save(fig, out, "task1_roc.png")


def main():
    ap = argparse.ArgumentParser(description="生成文档用图")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--metrics", default=None,
                    help="demo_traps.py --json 的产物（缺省用 outputs/metrics.json）")
    args = ap.parse_args()

    data_dir = resolve_data_dir(args.data)
    out = Path(args.out).expanduser().resolve() if args.out else ROOT / "docs" / "figures"
    mpath = Path(args.metrics).expanduser().resolve() if args.metrics \
        else ROOT / "outputs" / "metrics.json"
    if not mpath.exists():
        sys.exit(f"缺少 {mpath}，请先跑：python scripts/demo_traps.py --json outputs/metrics.json")

    import json

    metrics = json.loads(mpath.read_text(encoding="utf-8"))
    bundle = load_bundle(data_dir)

    print("生成图：")
    fig_exposure(bundle, out)
    fig_selection_bias(out, metrics)
    fig_sign_stability(out, metrics)
    fig_scorecard_lift(bundle, out)
    fig_task1_roc(bundle, out)
    print(f"\n输出：{out}")


if __name__ == "__main__":
    main()
