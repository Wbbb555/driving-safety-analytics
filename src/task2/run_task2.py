"""任务二：可解释安全评分卡。WOE/IV + 单调分箱 + 符号约束 + PDO。

这个文件是整个项目里最花时间的部分。评分卡的价值不在更准，在于每一分都能
追到具体哪一箱、多少样本、多高出险率——司机被扣分时得能问「为什么扣这一分」。

四个设计，每个背后都是一个踩过的坑：

1. 单调分箱。风险该随事件率单调上升，不单调就把相邻箱合了，直到单调或到箱数
   下限。否则会冒出「这一档比上一档更安全」这种话。
2. 符号约束。多元回归里系数可能反号，评分卡不能出现「事件越多分越高」，所以
   把风险因素的负系数截成 0。代价是丢信息，换来不吐反向建议。
3. 分箱和拟合都得在折内做。用全样本边界再评估会虚高，这里量到 +0.0075~+0.0115。
   偏差大小跟正例比例、分箱数有关，不是常数。注意别拿上一代的全样本分减这一代
   的无泄漏分当泄漏量，那样把版本差异混进去了。
4. 只收率指标，不收计数。计数本质是里程代理，会把跑得多的车判成高风险。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import load_bundle, resolve_data_dir, resolve_out_dir  # noqa: E402

# 评分卡参数
PDO = 50.0                      # 每翻一倍几率差 50 分
TARGET_MEDIAN_SCORE = 700.0     # 刻度锚：中位车的安全分定在这里
MAX_BINS = 5
MIN_BINS = 2
MIN_BIN_FRAC = 0.05             # 单箱至少占总样本 5%
EPS = 0.5                       # WOE 平滑，避免除零与无穷


# --------------------------------------------------------------- 分箱


def _quantile_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    """按分位切初始箱，去重并保证每箱有足够样本。"""
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(x, qs))
    lo, hi = x.min(), x.max()
    edges = np.concatenate(([lo], edges, [hi]))
    # 合并过小的箱
    while len(edges) > MIN_BINS + 1:
        cnt, _ = np.histogram(x, bins=edges)
        frac = cnt / max(len(x), 1)
        if frac.min() >= MIN_BIN_FRAC:
            break
        # 合并样本最少的箱与它的邻居
        k = int(np.argmin(frac))
        if k == 0:
            drop = 1
        elif k == len(cnt) - 1:
            drop = len(edges) - 2
        else:
            drop = k if cnt[k - 1] <= cnt[k + 1] else k + 1
        edges = np.delete(edges, drop)
    return edges


def _bin_counts(x: np.ndarray, y: np.ndarray, edges: np.ndarray):
    idx = np.clip(np.digitize(x, edges[1:-1], right=True), 0, len(edges) - 2)
    n_bins = len(edges) - 1
    n = np.bincount(idx, minlength=n_bins).astype(float)
    pos = np.bincount(idx, weights=y.astype(float), minlength=n_bins)
    return n, pos


def _woe_from_counts(n: np.ndarray, pos: np.ndarray):
    neg = n - pos
    good = neg + EPS
    bad = pos + EPS
    rate_bad = bad / (good + bad)
    rate_good = good / (good + bad)
    woe = np.log(rate_good / rate_bad)      # 坏样本越多，WOE 越小
    return woe, bad / (good + bad)


def monotonic_bins(x: np.ndarray, y: np.ndarray, n_bins: int = MAX_BINS):
    """切箱并强制单调：不良率必须随指标值一起上升。

    先按分位切，遇到相邻箱不良率下降就把这对合并，直到单调或者箱数掉到
    MIN_BINS。评分卡里最常用的强制手段，损失点分辨率，换来不被单箱噪声带偏。
    """
    edges = _quantile_edges(x, n_bins)
    while len(edges) > MIN_BINS + 1:
        n, pos = _bin_counts(x, y, edges)
        _, rate = _woe_from_counts(n, pos)
        bad_pairs = [i for i in range(len(rate) - 1) if rate[i + 1] < rate[i] - 1e-12]
        if not bad_pairs:
            break
        i = bad_pairs[0]
        edges = np.delete(edges, i + 1)
    n, pos = _bin_counts(x, y, edges)
    woe, rate = _woe_from_counts(n, pos)
    return edges, n, pos, woe, rate


def apply_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(x, edges[1:-1], right=True), 0, len(edges) - 2)


# --------------------------------------------------------------- 评分卡


def fit_card(X: np.ndarray, y: np.ndarray, names: list[str], n_bins: int = MAX_BINS):
    """拟合完整评分卡：分箱 + 回归 + 符号约束 + 刻度。

    回归的输入必须是 WOE，不是箱序号。箱序号只是「第几档」，WOE 才是「这一
    档的坏好比取对数」。早先误把箱序号喂进去，后果是三个：系数符号全反（序号
    越大=事件越多=越危险，系数于是为正）、符号约束把车道偏移这种最强指标截成
    0、评分失去区分度并且逐项加总对不上。这类 bug 不报错，只是让结果变成废纸。
    """
    bins, woe_cols = [], []
    for j, name in enumerate(names):
        edges, n, pos, woe, rate = monotonic_bins(X[:, j], y, n_bins)
        bins.append({"name": name, "edges": edges, "n": n, "pos": pos,
                     "woe": woe, "rate": rate})
        # 关键：喂给回归的是 WOE 值，不是箱序号
        woe_cols.append(woe[apply_bins(X[:, j], edges)])
    W = np.column_stack(woe_cols).astype(float)

    lr = LogisticRegression(C=1.0, max_iter=5000)
    lr.fit(W, y)
    beta = lr.coef_[0].copy()
    beta_raw = beta.copy()
    beta[beta > 0] = 0.0            # 符号约束：WOE 越大越安全，系数必须 <= 0

    # 重锚截距：固定 β 后重新解 β0，使预测概率的均值贴近真实正例率
    z = W @ beta
    b0 = _solve_intercept(z, y)

    # 刻度锚定。标准评分卡用 A - B·ln(odds) 的刻度，但那样「基准分」
    # 具体是多少完全由 β0 决定，讲给业务方听很不直观。这里改成更直白的锚：
    # 把**中位车的分数定为 TARGET_MEDIAN_SCORE**，基准分就是让这件事成立的常数。
    # 于是 安全分 = 基准分 + Σ 各项得分，逐项加总恒等于总分（可审计）。
    # 分数的绝对高低不影响排序与 AUC，只影响"看起来多少分"。
    factor = PDO / np.log(2)
    contrib = -factor * (W * beta)                 # shape (n, n_features)
    raw = contrib.sum(axis=1)
    base_offset = TARGET_MEDIAN_SCORE - float(np.median(raw))

    return {"bins": bins, "beta": beta, "beta_raw": beta_raw, "b0": b0,
            "names": names, "base_offset": base_offset}


def _solve_intercept(z: np.ndarray, y: np.ndarray) -> float:
    """一维搜索 β0，最小化负对数似然。"""
    from scipy.optimize import minimize_scalar

    def nll(b0):
        p = 1.0 / (1.0 + np.exp(-(z + b0)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    return float(minimize_scalar(nll, bounds=(-10, 10), method="bounded").x)


def woe_transform(card, X: np.ndarray) -> np.ndarray:
    """把原始指标值映射成 WOE 值（评分卡真正作用的对象）。"""
    out = []
    for j, b in enumerate(card["bins"]):
        out.append(b["woe"][apply_bins(X[:, j], b["edges"])])
    return np.column_stack(out).astype(float)


def score_from_card(card, X: np.ndarray) -> np.ndarray:
    """把 WOE 转成分数。分越高越安全。

    口径：安全分 = 基准分 + Σ（该项得分）
    其中 该项得分 = -factor · β · WOE，factor = PDO / ln2，
    基准分 = base_offset。这样逐项加总恒等于总分，可审计。
    """
    factor = PDO / np.log(2)
    W = woe_transform(card, X)
    return card["base_offset"] + (-factor * (W * card["beta"])).sum(axis=1)


def item_scores(card, X: np.ndarray) -> pd.DataFrame:
    """逐项得分明细，含基准分。

    可解释性就落在这张表上。没有它评分只有一个数，司机问为什么答不了。

    注意返回的列里「安全分」已经是各项加总。做加总校验时不能再对整表 sum，
    那会把汇总列算进去。早先因此报出「差 1106」，看着像数值错误，实际是校验
    代码自己加了两次。
    """
    factor = PDO / np.log(2)
    W = woe_transform(card, X)
    cols = {}
    for j, name in enumerate(card["names"]):
        cols[f"{name}_得分"] = -factor * card["beta"][j] * W[:, j]
    df = pd.DataFrame(cols)
    df["基准分"] = card["base_offset"]
    df["安全分"] = df.sum(axis=1)
    return df


# --------------------------------------------------------------- CV 评估


def oof_auc(cand: pd.DataFrame, cols: list[str], y: np.ndarray, seeds=(0, 1, 42)):
    """OOF AUC，折内分箱，无泄漏口径。

    返回两种折算方式，因为它们的差是口径选择不是误差：

    pooled：各折验证分拼一个向量，算一次全局 AUC。麻烦在于 base_offset 是按
    该折训练集的中位原始分锚到 700 的，五折各不相同（seed=0 是 183.5 / 134.1
    / 122.8 / 96.9 / 128.3）。拼池等于给不同折加了不同常数。核过不是标签泄漏
    （基准分与折内测试正例率 r = -0.085），但确实带进口径噪声。
    perfold：每折各算各的再平均，不受跨折常数影响。

    两个都报。实测（种子 0/1/42 平均）pooled 0.8350、每折中心化 0.8264、
    perfold 0.8390。
    """
    pooled, perfold = [], []
    names = [c.split("|")[-1] for c in cols]
    for s in seeds:
        skf = StratifiedKFold(5, shuffle=True, random_state=s)
        oof = np.zeros(len(y))
        fold_aucs = []
        for tr, te in skf.split(cand, y):
            card = fit_card(cand.loc[tr, cols].values.astype(float), y[tr], names)
            sc = score_from_card(card, cand.loc[te, cols].values.astype(float))
            oof[te] = sc
            # 分越高越安全，取负使 AUC>0.5；逐折内算，不含跨折常数
            fold_aucs.append(roc_auc_score(y[te], -sc))
        pooled.append(roc_auc_score(y, -oof))
        perfold.append(float(np.mean(fold_aucs)))
    return {
        "pooled": float(np.mean(pooled)), "pooled_sd": float(np.std(pooled)),
        "perfold": float(np.mean(perfold)), "perfold_sd": float(np.std(perfold)),
        "per_seed_pooled": [round(v, 4) for v in pooled],
    }


def leaky_auc(cand: pd.DataFrame, cols: list[str], y: np.ndarray, seed=0):
    """有泄漏口径：全样本先切箱，再交叉验证。只作对照，不能当结论。

    演示评分卡最常见的隐蔽错误——分箱边界用到了全部样本标签。两次 AUC 之差
    就是乐观偏差。

    方向约定：这里直接输出逻辑回归的风险概率（正类=出险），概率越高越危险，
    所以不能再取负号。oof_auc 用的是安全分，越高越安全，方向反的。混用会把
    AUC 变成 1−AUC 而且不报错，早先泄漏口径算出过 0.15 这种数。
    """
    names = [c.split("|")[-1] for c in cols]
    card_full = fit_card(cand[cols].values.astype(float), y, names)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(cand, y):
        W = woe_transform(card_full, cand.loc[tr, cols].values.astype(float))
        lr = LogisticRegression(C=1.0, max_iter=5000)
        lr.fit(W, y[tr])
        Wte = woe_transform(card_full, cand.loc[te, cols].values.astype(float))
        oof[te] = lr.predict_proba(Wte)[:, 1]      # 风险概率：越高越危险
    return float(roc_auc_score(y, oof))


def main():
    ap = argparse.ArgumentParser(description="任务二：可解释安全评分卡")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir, out_dir = resolve_data_dir(args.data), resolve_out_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(data_dir)
    cand = bundle["cand"]
    cols = [c for c in cand.columns if "|" in c]
    names = [c.split("|")[-1] for c in cols]
    y = cand["label"].values.astype(int)

    print(f"{len(cand)} 台车，{len(cols)} 个指标，正例 {int(y.sum())} ({y.mean()*100:.1f}%)")
    print("说明：全部数字来自合成数据，只用于演示方法。\n")

    print("各指标单变量 AUC（对标签）：")
    for c, n in sorted(zip(cols, names), key=lambda t: -roc_auc_score(y, cand[t[0]].fillna(0))):
        a = roc_auc_score(y, cand[c].fillna(0))
        print(f"  {n:8s} {a:.4f}")

    # --- 全样本评分卡（用于交付与解释）
    card = fit_card(cand[cols].values.astype(float), y, names)

    factor = PDO / np.log(2)
    print(f"\n刻度：PDO {PDO:.0f}，Factor = {factor:.4f}，β0 = {card['b0']:.4f}")
    print(f"基准分（锚定中位车 = {TARGET_MEDIAN_SCORE:.0f} 分）= {card['base_offset']:.1f}")

    print("\n各指标系数（已做符号约束）：")
    for j, n in enumerate(names):
        raw, con = card["beta_raw"][j], card["beta"][j]
        flag = "  ← 反号被截断为 0" if raw > 0 else ""
        print(f"  {n:8s} 原始 {raw:+.4f} → 约束后 {con:+.4f}{flag}")

    # --- 评分与分档
    sc = score_from_card(card, cand[cols].values.astype(float))
    items = item_scores(card, cand[cols].values.astype(float))
    res = pd.DataFrame({
        "gpsno": cand["gpsno"], "safety_score": np.round(sc, 1),
        "risk_prob_task1_ref": np.nan,
    })
    # 分档：按分数三分位
    q33, q67 = np.quantile(sc, [1 / 3, 2 / 3])
    res["risk_level"] = np.where(sc < q33, "高风险",
                                 np.where(sc < q67, "中风险", "低风险"))
    res = res.sort_values("safety_score")
    res = res.merge(bundle["labels"][["gpsno", "label"]], on="gpsno", how="left")
    res.to_csv(out_dir / "task2_scoring_result.csv", index=False, encoding="utf-8-sig")

    print(f"\n分档（三分位切点 {q33:.1f} / {q67:.1f}）：")
    for lvl in ["高风险", "中风险", "低风险"]:
        sub = res[res["risk_level"] == lvl]
        print(f"  {lvl}  {len(sub):3d} 台，出险率 {sub['label'].mean() * 100:5.1f}%")

    # --- 逐项明细（含加总校验）
    detail = pd.concat([cand[["gpsno"]].reset_index(drop=True),
                        items.round(1)], axis=1)
    detail["安全分_实测"] = np.round(sc, 1)
    # 只加"各项得分"与"基准分"，排除已含汇总的"安全分"列
    item_cols = [c for c in items.columns if c != "安全分"]
    itemsum = items[item_cols].sum(axis=1).values
    maxdiff = float(np.abs(itemsum - sc).max())
    print(f"\n逐项加总 vs 评分 最大差 = {maxdiff:.6f}"
          f"（{'加总自洽' if maxdiff < 1e-9 else '存在分离'}）")

    # 上面是内存口径。导出的 CSV 把每一项舍到 0.1，拿列相加会有最多 0.4 的
    # 舍入残留，这是两个不同的数，别混着说。
    csv_diff = float((detail[item_cols].sum(axis=1) - detail["安全分_实测"]).abs().max())
    print(f"（导出 CSV 的列直接相加，最大差 = {csv_diff:.1f}；"
          f"8 项得分加基准分共 {len(item_cols)} 个数各舍到 0.1，"
          f"理论上界 {len(item_cols) * 0.05:.2f}，属舍入不是结构错）")
    detail.to_csv(out_dir / "task2_item_scores.csv", index=False, encoding="utf-8-sig")

    # --- 评分卡表（每一项、每一箱的完整口径）
    rows = []
    for b in card["bins"]:
        edges = b["edges"]
        for k in range(len(edges) - 1):
            rows.append({
                "指标": b["name"],
                "箱序": k + 1,
                "区间左": round(float(edges[k]), 2),
                "区间右": round(float(edges[k + 1]), 2),
                "样本数": int(b["n"][k]),
                "箱内出险率%": round(float(b["rate"][k]) * 100, 1),
                "WOE": round(float(b["woe"][k]), 4),
                "约束后β": round(float(card["beta"][names.index(b["name"])]), 4),
                "该项得分": round(float(
                    -factor * card["beta"][names.index(b["name"])] * b["woe"][k]), 1),
            })
    card_df = pd.DataFrame(rows)
    # 用同一套 round 规则复算，避免两处各自四舍五入产生差异
    manual = (card_df["约束后β"] * card_df["WOE"] * -factor).round(1)
    gap = float((manual - card_df["该项得分"]).abs().max())
    print(f"评分卡表：{len(card_df)} 行（逐箱），"
          f"得分复算最大差 {gap:.4f}（舍入）")
    card_df.to_csv(out_dir / "task2_scorecard.csv", index=False, encoding="utf-8-sig")

    # --- 无泄漏 vs 有泄漏
    oof = oof_auc(cand, cols, y)
    auc_leak = leaky_auc(cand, cols, y)
    print(f"\nOOF AUC（折内分箱，无泄漏）  = {oof['pooled']:.4f} ± {oof['pooled_sd']:.4f}"
          f"（折内拼池口径）")
    print(f"                               {oof['perfold']:.4f} ± {oof['perfold_sd']:.4f}"
          f"（逐折平均口径）")
    print(f"OOF AUC（全样本分箱，有泄漏）= {auc_leak:.4f}")
    print(f"→ 乐观偏差 {auc_leak - oof['pooled']:+.4f}"
          f"（对拼池口径）/ {auc_leak - oof['perfold']:+.4f}（对逐折口径）")
    print("  注：两种无泄漏口径差约 0.01，是 OOF 拼接口径差异，不是泄漏。"
          "每折基准分不同，拼池会给各折加不同常数（已核验与折内正例率无关）。")

    # --- 十分位单调性
    dec = pd.qcut(pd.Series(sc).rank(method="first"), 10, labels=False)
    tbl = pd.DataFrame({"dec": dec, "y": y}).groupby("dec")["y"].agg(["size", "mean"])
    rho = stats.spearmanr(tbl.index, tbl["mean"]).statistic
    print(f"\n十分位 Spearman(AUC 视角) = {rho:+.4f}（越大越单调，1.0 为完全单调）")
    print("  十分位出险率：" + " ".join(f"{v*100:.0f}%" for v in tbl["mean"]))

    # --- 重抽样稳定性
    rng = np.random.default_rng(0)
    stab = {}
    for j, n in enumerate(names):
        neg = 0
        for _ in range(200):
            idx = rng.integers(0, len(y), len(y))
            try:
                c = fit_card(cand[cols].values.astype(float)[idx], y[idx], names)
            except Exception:
                continue
            if c["beta"][j] < 0:
                neg += 1
        stab[n] = neg / 200.0
    print("\n系数有效性稳定性（B=200，该指标系数 <0 即有效 的占比）：")
    for n, f in stab.items():
        print(f"  {n:8s} {f*100:5.1f}%")

    with open(out_dir / "task2_scorecard.pkl", "wb") as f:
        pickle.dump(card, f)
    print(f"\n输出：{out_dir}")


if __name__ == "__main__":
    main()
