# -*- coding: utf-8 -*-
"""复现四个方法论陷阱。

这个脚本是仓库的核心。它不追求把 AUC 做高，而是**故意做四件错事，
再用正确的做法把错误量化出来**。每个陷阱都回答同一个问题：
"如果你不知道这个坑，你会得出什么错误结论？"

四个陷阱：
    T1 暴露量混杂   —— "数据量"类特征测的是跑了多少，不是开得多危险
    T2 特征选择偏差 —— 用同一批数据既挑特征又评性能，AUC 虚高
    T3 共线性符号反转 —— 多元回归系数与常识反号，不是数据错，是被强指标吸收
    T4 分箱泄漏     —— WOE 分箱用了全样本标签，评估会乐观

运行
----
    python scripts/demo_traps.py

依赖：numpy pandas scikit-learn scipy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from paths import resolve_data_dir  # noqa: E402

# 与 src/ 下的脚本共用同一套路径解析：命令行 --data > 环境变量 DSA_DATA_DIR > 仓库默认。
# 早期版本这里写死 ROOT/"data"/"synthetic"，导致 README 里「不需要改路径」这句话
# 对本脚本并不成立——数据换了目录就得改代码。
DATA = ROOT / "data" / "synthetic"
RNG = np.random.default_rng(42)


def load():
    lab = pd.read_csv(DATA / "labels.csv", dtype={"gpsno": str}).set_index("gpsno")
    cand = pd.read_csv(DATA / "task2_candidates.csv", dtype={"gpsno": str}).set_index("gpsno")
    fev = pd.read_csv(DATA / "features_events.csv", dtype={"gpsno": str}).set_index("gpsno")
    prof = (
        pd.read_csv(DATA / "vehicle_profile.csv", dtype={"设备号": str})
        .rename(columns={"设备号": "gpsno"})
        .set_index("gpsno")
    )
    return lab, cand, fev, prof


# ---------------------------------------------------------------- T1 暴露量混杂


def tray_pure_exposure():
    """纯暴露量特征：轨迹采样点总数。

    这是真实项目里 imu_n_points / traj_n_points 这类特征的合成对应物——
    它不含任何驾驶行为信息，只反映"这台车跑了多少、GPS 采样了多少个点"。
    """
    traj = pd.read_parquet(DATA / "traj_daily.parquet", columns=["gpsno", "n_points", "dist_km"])
    agg = traj.groupby("gpsno").agg(n_points_total=("n_points", "sum"),
                                    km_traj=("dist_km", "sum"))
    return agg


def trap_exposure(lab, cand, fev):
    """'数据量'类特征的高 AUC 有多少来自里程？"""
    y = lab["label"].values
    km = cand["km"].values
    total_events = fev["total_events"].values
    pure = tray_pure_exposure().reindex(lab.index)
    n_points = pure["n_points_total"].fillna(0).values

    print("=" * 74)
    print("T1 暴露量混杂：这个特征测的是'跑得多'还是'开得危险'？")
    print("=" * 74)

    auc_km = roc_auc_score(y, km)

    def residual_auc(feat, use_log=True):
        """把特征对里程回归后取残差的 AUC。

        残差代表"同等里程下这个特征偏高还是偏低"，即剥离暴露量后的
        纯行为信号。残差 AUC 若掉到 0.5 附近，说明原特征只是里程的代理。

        重要：计数类特征随里程近似幂律增长，所以要在**对数—对数**尺度上
        回归，而不是拿原始值去对 log(里程) 回归。后者残差里还留着随里程
        单调变化的曲线，会把 AUC 抬高（本仓库早期版本就踩了这个坑：
        同一份数据用原始值回归得 0.6080，用对数—对数回归得 0.4995）。
        """
        x = np.log(km)
        f = np.log(feat) if use_log else feat
        res = f - np.polyval(np.polyfit(x, f, 1), x)
        return roc_auc_score(y, res)

    print(f"  行驶里程本身对标签 AUC            = {auc_km:.4f}   ← 混杂源")
    print()
    print("  【纯暴露量特征】轨迹采样点总数（不含任何行为信息）")
    r_pure_raw = residual_auc(n_points, use_log=False)
    r_pure_log = residual_auc(n_points, use_log=True)
    print(f"    原始 AUC                        = {roc_auc_score(y, n_points):.4f}")
    print(f"    对数—对数残差化后 AUC            = {r_pure_log:.4f}")
    print(f"    （同一数据若误用原始值回归，会得到 {r_pure_raw:.4f}——")
    print("      残差里残余的里程曲线造成的虚假信号，这是本仓库自己踩过的坑）")
    print()
    print("  【混合特征】事件总条数（暴露量 × 行为率，两者都有）")
    auc_count = roc_auc_score(y, total_events)
    auc_resid = residual_auc(total_events, use_log=True)
    print(f"    原始 AUC                        = {auc_count:.4f}")
    print(f"    对数—对数残差化后 AUC            = {auc_resid:.4f}")
    print(f"    → 损失 {auc_count - auc_resid:+.4f}，但残差仍有预测力，")
    print("      说明事件数里既有暴露量也有真实行为信号，无法靠残差完全分离。")
    print()
    print(f"  Spearman：采样点数 vs 里程 = {stats.spearmanr(n_points, km).statistic:+.4f}"
          f"（几乎共线）；事件数 vs 里程 = {stats.spearmanr(total_events, km).statistic:+.4f}")
    print()

    # 对照：率指标天然不受暴露量影响
    rate_best = None
    for c in [c for c in cand.columns if "|" in c]:
        a = roc_auc_score(y, cand[c].fillna(0).values)
        if rate_best is None or abs(a - 0.5) > abs(rate_best[1] - 0.5):
            rate_best = (c, a)
    print(f"  对照：最强**率**指标 {rate_best[0]} AUC = {rate_best[1]:.4f}")
    print("  率指标 = 事件数 / 里程，暴露量已在分母里约掉——")
    print("  这就是为什么最终评分体系全部改用率而不是计数。")
    print()
    return {"auc_km": auc_km, "auc_count": auc_count, "auc_resid": auc_resid,
            "auc_pure": roc_auc_score(y, n_points),
            "auc_pure_resid_loglog": r_pure_log,
            "auc_pure_resid_rawlog": r_pure_raw,
            "spearman_pure_km": float(stats.spearmanr(n_points, km).statistic),
            "spearman_count_km": float(stats.spearmanr(total_events, km).statistic),
            "best_rate": rate_best[0], "best_rate_auc": float(rate_best[1])}


# ---------------------------------------------------------------- T2 特征选择偏差


def trap_feature_selection(lab, cand, prof):
    """用同一批数据既挑特征又评性能，会虚高多少？

    正确的对照设计：两条路线必须用**同一个最终模型规模**，
    唯一差别是"挑特征时有没有看过验证集"。
    早期版本的嵌套 CV 固定只取 top_k 个特征，而朴素路线用了全部 22 个，
    两者根本不是同一件事，比较出来的差值没有意义。
    这里改为：两条路线都取 top_k，只在"在哪份数据上排重要性"上不同。
    """
    y = lab["label"].values
    X = build_feature_matrix(lab, cand, prof)
    n_features = X.shape[1]
    top_k = 8

    print("=" * 74)
    print("T2 特征选择偏差：既挑特征又评性能，AUC 会虚高吗？")
    print("=" * 74)

    naive = naive_select_auc(X, y, top_k=top_k, seed=0)
    nested = nested_select_auc(X, y, top_k=top_k, seed=0)
    print(f"  朴素（全样本排重要性）  OOF AUC = {naive:.4f}")
    print(f"  嵌套（折内排重要性）    OOF AUC = {nested:.4f}")
    print(f"  → 乐观偏差 {naive - nested:+.4f}")
    print()

    # 特征数量扫描：验证"特征越多越好"这个直觉
    print("  特征数扫描（嵌套口径，每次都在折内重新挑）：")
    scan = {}
    for k in (4, 6, 8, 12, 16, n_features):
        a = nested_select_auc(X, y, top_k=k, seed=0)
        scan[k] = a
        print(f"    top_k={k:2d}  OOF AUC = {a:.4f}")
    best_k = max(scan, key=scan.get)
    spread = max(scan.values()) - min(scan.values())
    print(f"  → 最高 top_k={best_k}（{scan[best_k]:.4f}），但整条曲线只在 "
          f"{spread:.4f} 的范围内来回波动，没有单调趋势。")
    print(f"    {len(y)} 台车、{int(y.sum())} 个正例的样本量下，这个量级的跳动基本是噪声——")
    print("    所以结论是「挑几个特征对泛化没有稳定影响」，")
    print("    而不是「某个 k 最好」。在正例只有几十个时，")
    print("    top_k 扫描的排名本身就是不稳定的，别拿它当结论。")
    print("    特征精简的收益在可解释性和维护成本，性能上别指望它带来提升。")
    print()
    return {"naive": naive, "nested": nested, "scan": scan}


def build_feature_matrix(lab, cand, prof):
    """拼一个中小规模特征矩阵：率指标 + 画像数值列 + 少量噪声列。"""
    cols = [c for c in cand.columns if "|" in c]
    X = cand[cols].fillna(0).values.astype(float)

    num = prof.select_dtypes(include=[np.number])
    Xp = num.fillna(num.median()).values.astype(float)
    # 噪声特征：与标签无关，用来观察"特征越多是否越好"这个直觉是否成立
    noise = RNG.normal(0, 1, (len(lab), 6))
    return np.hstack([X, Xp, noise])


def naive_select_auc(X, y, top_k=8, seed=0):
    """朴素做法（有偏差）：先在**全部样本**上排特征重要性，再交叉验证评估。

    偏差来源：排重要性时已经看过了每个验证折的标签，
    等于让验证集参与了特征选择决定。
    """
    m_all = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    m_all.fit(X, y)
    idx = np.argsort(m_all.feature_importances_)[-top_k:]

    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        m = RandomForestClassifier(n_estimators=120, random_state=seed, n_jobs=-1)
        m.fit(X[tr][:, idx], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te][:, idx])[:, 1]))
    return float(np.mean(aucs))


def nested_select_auc(X, y, top_k=8, seed=0):
    """嵌套 CV（无偏）：每一折都只在训练集里重新排重要性。"""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        m_sel = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        m_sel.fit(X[tr], y[tr])
        idx = np.argsort(m_sel.feature_importances_)[-top_k:]

        m = RandomForestClassifier(n_estimators=120, random_state=seed, n_jobs=-1)
        m.fit(X[tr][:, idx], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te][:, idx])[:, 1]))
    return float(np.mean(aucs))


# ---------------------------------------------------------------- T3 共线性符号反转


def trap_collinearity(lab, cand):
    """系数反号是数据错，还是被强指标吸收了？"""
    y = lab["label"].values
    cols = [c for c in cand.columns if "|" in c]
    short = [c.split("|")[-1] for c in cols]

    print("=" * 74)
    print("T3 共线性符号反转：系数与常识反号时该怀疑什么？")
    print("=" * 74)

    Xstd = standardize(cand[cols].fillna(0).values.astype(float))
    vif = calc_vif(Xstd)
    print("  各指标 VIF（共线性诊断，>5 才需处理）：")
    for name, v in sorted(zip(short, vif), key=lambda t: -t[1]):
        print(f"    {name:10s} VIF = {v:.2f}")

    coef_all = fit_logit(Xstd, y)
    print("\n  全部指标一起回归（标准化系数）：")
    anom = []
    for i, (name, c) in enumerate(zip(short, coef_all)):
        # 吸烟 / 盲区报警在合成数据里被刻意设为负敏感度
        # （复现真实数据的指标方向反常），它们反号是设计使然，不算异常。
        designed_negative = name in ("吸烟", "盲区报警")
        if c < 0 and not designed_negative:
            flag = "  <== 与常识反号（待查）"
            anom.append(i)
        elif c < 0:
            flag = "  （合成数据刻意设为负向）"
        else:
            flag = ""
        print(f"    {name:10s} {c:+.4f}{flag}")

    # 对每个可疑反转的指标：逐个把其它指标单独移除，看系数有没有翻正。
    # 注意——不能只测"与它最相关的那一个"。真实数据里疲劳/分神反转，
    # 移除对象是车道偏移；但"最相关"未必等于"吸收者"，必须看移除了
    # 之后系数到底动没动。早期版本只打印最相关者就下结论，是错的。
    print("\n  逐个排除法（移除单个指标后，看可疑系数是否翻正）：")
    if not anom:
        print("    本次拟合没有出现「与常识反号」的指标，跳过。")
        print("    注意这不代表问题消失了——真实数据里分神系数是 -0.0208。")
        print("    这里没复现出来，本身就是个提示：换个随机种子就可能出现，")
        print("    说明这种反号在几十个正例的样本上本来就不稳定（见下方稳定性检查）。")
    corr = pd.DataFrame(Xstd, columns=short).corr()
    for i in anom:
        name = short[i]
        base = coef_all[i]
        best = None
        for j in range(len(short)):
            if j == i:
                continue
            keep = [k for k in range(len(short)) if k != j]
            after = fit_logit(Xstd[:, keep], y)[keep.index(i)]
            if best is None or after > best[2]:
                best = (short[j], j, after)
        tgt, _, after = best
        moved = "翻正" if after >= 0 else "仍为负"
        delta = after - base
        print(f"    {name}：基准 {base:+.4f}；移除 {tgt} 后 {after:+.4f}"
              f"（Δ{delta:+.4f}，{moved}）")
        print(f"      两者相关 = {corr.loc[name, tgt]:+.4f}")
        if after < 0 and delta < 0.05:
            print("      → 移除任何一个指标都救不回来，说明这个反号不能用"
                  "「被别的指标吸收」解释；")
            print("        更可能是采样噪声，或该指标与事故之间本就存在反向的真实关联。")
        else:
            print("      → 反号来自与强指标的共线性吸收，属于抑制变量现象。")

    print("\n  怎么处理这种反号（按优先级）：")
    print("    1) 先做排除法，确认是不是被强指标吸收；吸收不了的别硬解释。")
    print("    2) 是吸收 → 符号约束回归（风险因素系数强制 >= 0），或按维度分组建模。")
    print("    3) 不是吸收 → 如实披露为「方向待核实」，不要把它翻译成行为建议。")

    # 符号稳定性：系数的正负本身稳不稳？
    # 这一步是本仓库自己踩坑后加的。真实数据里分神系数是 -0.0208，
    # 移除车道偏移后翻到 +0.178——看起来像干净的抑制变量现象。
    # 但在同样大小的合成样本上做重抽样，这个符号并不稳：
    # 换几个随机种子，分神系数有时为正、有时为负。
    # 也就是说，在正例只有几十个的样本量下，"系数是正是负"这件事本身
    # 就有很大抽样波动，不该拿单次拟合的符号去讲故事。
    print("\n  符号稳定性检查（B=200 次车辆重抽样，看系数正负有多稳）：")
    stab = {}
    rng_b = np.random.default_rng(0)
    for i, name in enumerate(short):
        neg = 0
        for _ in range(200):
            idx = rng_b.integers(0, len(y), len(y))
            yy = y[idx]
            if yy.min() == yy.max():
                continue
            try:
                c = fit_logit(Xstd[idx], yy)[i]
            except Exception:
                continue
            if c < 0:
                neg += 1
        frac = neg / 200.0
        stab[name] = frac
        print(f"    {name:10s} 系数为负的重抽样占比 = {frac * 100:5.1f}%")
    # 判定"不稳"用 0.05~0.95 而不是 0<f<1：
    # 后者会把 99.5%（只有 1 次重抽样翻向）也算成不稳，
    # 那是正常的抽样噪声，不是"符号不稳"。
    FLIP_LO, FLIP_HI = 0.05, 0.95
    flip_prone = [n for n, f in stab.items()
                  if FLIP_LO < f < FLIP_HI and n not in ("吸烟", "盲区报警")]
    if flip_prone:
        print(f"  → {('、'.join(flip_prone))} 的系数符号在重抽样中不稳定：")
        print("    这说明在 500 台车、几十个正例的量级下，单个指标的系数符号")
        print("    本身就带很大抽样波动。写结论前必须先看稳定性，")
        print("    别把某一次拟合的符号当成「该指标有害 / 有益」的证据。")
    print()
    return {"vif": dict(zip(short, vif)), "coef": dict(zip(short, coef_all)),
            "stab": stab, "anomalies": [short[i] for i in anom]}


def standardize(X):
    m, s = X.mean(0), X.std(0)
    s[s == 0] = 1.0
    return (X - m) / s


def calc_vif(X):
    """VIF = 1 / (1 - R²)，用每个特征对其余特征做**线性**回归得到。

    注意这里必须用 LinearRegression 而不是 LogisticRegression：
    VIF 的定义基于连续量的线性回归 R²，特征本身是连续值，
    用分类器会直接报 "Unknown label type: continuous"。
    """
    from sklearn.linear_model import LinearRegression

    out = []
    for i in range(X.shape[1]):
        others = np.delete(X, i, axis=1)
        m = LinearRegression().fit(others, X[:, i])
        r2 = m.score(others, X[:, i])
        out.append(1.0 / max(1.0 - r2, 1e-6))
    return out


def fit_logit(X, y):
    m = LogisticRegression(max_iter=5000, C=1.0).fit(X, y)
    return m.coef_[0]


# ---------------------------------------------------------------- T4 分箱泄漏


def trap_binning_leak(lab, cand):
    """WOE 分箱用全样本标签，评估会乐观多少？"""
    y = lab["label"].values
    col = [c for c in cand.columns if "|" in c]
    X = cand[col].fillna(0).values.astype(float)

    print("=" * 74)
    print("T4 分箱泄漏：WOE 分箱偷看了标签，评估会乐观多少？")
    print("=" * 74)

    # 错误做法：用全样本标签算 WOE，然后直接在同一批数据上评估
    Xw_all = np.column_stack([woe_transform(X[:, j], y, make_bins(X[:, j])) for j in range(X.shape[1])])
    leak_auc = cv_auc(Xw_all, y)
    print(f"  全样本分箱（有泄漏）      OOF AUC = {leak_auc:.4f}")

    # 正确做法：每折只用训练集标签分箱，验证集沿用训练集的分箱边界
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    aucs = []
    for tr, te in skf.split(X, y):
        xtr = np.zeros((len(tr), X.shape[1]))
        xte = np.zeros((len(te), X.shape[1]))
        for j in range(X.shape[1]):
            edges = make_bins(X[tr, j])
            woe_map = fit_woe(X[tr, j], y[tr], edges)
            xtr[:, j] = apply_woe(X[tr, j], edges, woe_map)
            xte[:, j] = apply_woe(X[te, j], edges, woe_map)
        m = LogisticRegression(max_iter=5000).fit(xtr, y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(xte)[:, 1]))
    clean_auc = float(np.mean(aucs))
    print(f"  折内分箱（无泄漏）        OOF AUC = {clean_auc:.4f}")
    print(f"  → 乐观偏差 {leak_auc - clean_auc:+.4f}")
    print()
    print("  凡是'用标签做的离散化'（WOE/IV、最优分箱、目标编码）都有这个坑。")
    print("  正确姿势：分箱边界只能从训练折学，验证折照抄。")
    print()
    return {"leak": leak_auc, "clean": clean_auc}


def make_bins(x, n=5):
    """按分位数切箱，返回箱边界（含 -inf/inf）。"""
    qs = np.unique(np.quantile(x, np.linspace(0, 1, n + 1)[1:-1]))
    return np.concatenate([[-np.inf], qs, [np.inf]])


def fit_woe(x, y, edges):
    """在给定样本上算每个箱的 WOE。"""
    idx = np.digitize(x, edges[1:-1])
    total_pos, total_neg = y.sum(), len(y) - y.sum()
    out = {}
    for b in range(len(edges) - 1):
        mask = idx == b
        pos = y[mask].sum()
        neg = mask.sum() - pos
        # 加 0.5 平滑，避免某箱无正例或全正例导致 log(0)
        out[b] = np.log(((pos + 0.5) / (total_pos + 0.5)) / ((neg + 0.5) / (total_neg + 0.5)))
    return out


def apply_woe(x, edges, woe_map):
    idx = np.digitize(x, edges[1:-1])
    default = float(np.mean(list(woe_map.values())))
    return np.array([woe_map.get(b, default) for b in idx])


def woe_transform(x, y, edges):
    return apply_woe(x, edges, fit_woe(x, y, edges))


def cv_auc(X, y, seed=0):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        m = LogisticRegression(max_iter=5000).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


# ---------------------------------------------------------------- main


def main():
    global DATA
    ap = argparse.ArgumentParser(description="四个方法论陷阱的量化复现")
    ap.add_argument("--data", default=None, help="数据目录（默认 data/synthetic）")
    ap.add_argument("--json", default=None,
                    help="把全部指标写成 JSON（用于 CI 比对与画图）")
    args = ap.parse_args()
    DATA = resolve_data_dir(args.data)

    if not (DATA / "labels.csv").exists():
        print(f"找不到合成数据（{DATA}），请先运行：")
        print("  python scripts/generate_synthetic.py")
        sys.exit(1)

    lab, cand, fev, prof = load()
    print()
    print("=" * 74)
    print("合成数据上的四个方法论陷阱复现")
    print(f"样本 {len(lab)} 台车，正例 {int(lab['label'].sum())} 台 "
          f"({lab['label'].mean() * 100:.1f}%)")
    print("全部数字来自虚构数据，只用于演示方法，不代表任何真实车队表现。")
    print("=" * 74)
    print()

    out = {}
    out["T1_exposure"] = trap_exposure(lab, cand, fev)
    out["T3_collinearity"] = trap_collinearity(lab, cand)
    out["T2_feature_selection"] = trap_feature_selection(lab, cand, prof)
    out["T4_binning_leak"] = trap_binning_leak(lab, cand)
    out["meta"] = {
        "n_vehicles": int(len(lab)),
        "n_positive": int(lab["label"].sum()),
        "positive_rate": float(lab["label"].mean()),
        "data_dir": str(DATA),
    }

    if args.json:
        import json
        dst = Path(args.json).expanduser().resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"指标已写入 {dst}")

    print("=" * 74)
    print("四个陷阱的共同点：它们都不会报错。")
    print("代码跑得通、AUC 还挺高、图表也好看——错的是结论。")
    print("=" * 74)
    return out


if __name__ == "__main__":
    main()
