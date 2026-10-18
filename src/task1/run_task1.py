"""任务一：车辆级事故风险预测。

原项目有两个视图（同期特征 / 时序窗口特征），这里保留结构但简化：
用车辆级聚合特征 + 按车辆分组的交叉验证，输出每台车的事故风险概率。

保留的关键设计（都在真实项目里验证过必要性）：
1. 按车辆分组做 CV。若按样本随机分折，同一台车的多个样本会同时出现在
   训练和验证集里，AUC 会明显虚高。
2. 三个模型做秩融合而不是概率平均。各模型的概率校准程度不同，
   直接平均会被"更自信的那个模型"带偏；秩融合只比较相对顺序。
3. 特征全部用「率」而不是「计数」，避免把"跑得多"误判成"开得危险"。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import load_bundle, resolve_data_dir, resolve_out_dir  # noqa: E402

INDICATORS = {
    "疲劳": ["fatigue"],
    "分神": ["distraction"],
    "吸烟": ["smoke"],
    "超速": ["speeding"],
    "车道偏移": ["lane"],
    "碰撞预警": ["collision_warn"],
    "设备异常": ["camera"],
    "盲区报警": ["blind_zone"],
}


def build_features(bundle: dict) -> pd.DataFrame:
    """把日级明细聚合成车辆级特征。

    所有计数类特征都转成每千公里发生率；里程作为独立列保留，
    便于后续做暴露量残差化诊断（见 scripts/demo_traps.py 的 T1）。
    """
    ev = bundle["event_daily"]
    traj = bundle["traj_daily"]
    imu = bundle["imu_daily"]
    imu_ev = bundle["imu_events_daily"]
    labels = bundle["labels"]

    km = traj.groupby("gpsno")["dist_km"].sum().rename("km")
    run_h = traj.groupby("gpsno")["run_time_h"].sum().rename("run_time_h")

    agg = ev.groupby("gpsno").sum(numeric_only=True)
    rows = pd.DataFrame(index=agg.index)

    for name, cols in INDICATORS.items():
        cnt = agg[cols].sum(axis=1)
        rows[f"rate_{name}"] = cnt / km.reindex(rows.index) * 1000.0

    # 行驶特征：速度分位的均值、夜间占比
    spd_cols = [c for c in traj.columns if c.startswith("spd_bin_")]
    spd_center = np.array([float(c.split("_")[-1]) for c in spd_cols])
    spd_sum = traj.groupby("gpsno")[spd_cols].sum()
    rows["speed_mean"] = (spd_sum.values @ spd_center) / np.maximum(spd_sum.values.sum(1), 1)

    hr_all = [c for c in traj.columns if c.startswith("hr_all_")]
    night = [c for c in hr_all if int(c.split("_")[-1]) in (0, 1, 2, 3, 4, 22, 23)]
    hsum = traj.groupby("gpsno")[hr_all].sum()
    rows["night_ratio"] = hsum[night].sum(axis=1) / np.maximum(hsum.sum(axis=1), 1)

    # IMU 特征：只在有 IMU 覆盖的车辆-日上聚合（真实数据覆盖率约 48.7%）
    imu_cnt = imu.groupby("gpsno").size().rename("imu_days")
    imu_agg = imu_ev.groupby("gpsno").sum(numeric_only=True)
    rate_base = km.reindex(imu_agg.index)
    for c in ["harsh_brake", "harsh_accel", "sharp_turn", "swerve"]:
        rows[f"imu_rate_{c}"] = (imu_agg[c] / rate_base * 1000.0).reindex(rows.index)
    rows["imu_coverage"] = (imu_cnt / 61.0).reindex(rows.index)

    rows["km"] = km.reindex(rows.index)
    rows["run_time_h"] = run_h.reindex(rows.index)

    lab = labels.set_index("gpsno")
    rows = rows.join(lab[["label", "accident_cnt", "near_miss_cnt"]], how="inner")
    # 缺失率指标填 0（该车没有这类事件）
    feat_cols = [c for c in rows.columns if c not in ("label", "accident_cnt", "near_miss_cnt")]
    rows[feat_cols] = rows[feat_cols].fillna(0.0)
    return rows.reset_index()


def make_models(seed: int):
    return {
        "RF": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, max_features="sqrt",
            random_state=seed, n_jobs=-1),
        "ET": ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=3, max_features="sqrt",
            random_state=seed, n_jobs=-1),
        "LR-L1": LogisticRegression(
            l1_ratio=1.0, solver="liblinear", C=0.3, max_iter=5000, random_state=seed),
    }


def cv_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int):
    """分层分组交叉验证，返回每个模型的 OOF 概率与 AUC。"""
    models = make_models(seed)
    oof = {k: np.zeros(len(y)) for k in models}
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for name, m in models.items():
            m.fit(Xtr, y[tr])
            oof[name][te] = m.predict_proba(Xte)[:, 1]
    aucs = {k: roc_auc_score(y, v) for k, v in oof.items()}
    return oof, aucs


def rank_fuse(oof: dict) -> np.ndarray:
    """等权秩融合：各模型先转成秩，再取平均。

    概率融合的问题：模型 A 输出集中在 0.1~0.3，模型 B 输出 0.6~0.9，
    直接平均会被 B 主导。秩融合只保留"谁排在前面"，不受校准影响。
    """
    ranks = np.column_stack([stats.rankdata(v) for v in oof.values()])
    return ranks.mean(1)


def main():
    ap = argparse.ArgumentParser(description="任务一：车辆级事故风险预测")
    ap.add_argument("--data", default=None, help="数据目录（默认 data/synthetic）")
    ap.add_argument("--out", default=None, help="输出目录（默认 outputs）")
    ap.add_argument("--seed", type=int, default=20260601)
    args = ap.parse_args()

    data_dir = resolve_data_dir(args.data)
    out_dir = resolve_out_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(data_dir)
    df = build_features(bundle)
    feat_cols = [c for c in df.columns
                 if c not in ("gpsno", "label", "accident_cnt", "near_miss_cnt")]
    X = df[feat_cols].values.astype(float)
    y = df["label"].values.astype(int)
    groups = df["gpsno"].values

    print(f"样本 {len(df)} 台车，特征 {len(feat_cols)} 维，正例 {int(y.sum())} "
          f"({y.mean() * 100:.1f}%)")
    print("说明：全部数字来自合成数据，只用于演示方法。")

    oof, aucs = cv_oof(X, y, groups, args.seed)
    print("\n各模型 OOF AUC（按车辆分组 5 折）：")
    for k, v in sorted(aucs.items(), key=lambda t: -t[1]):
        print(f"  {k:8s} {v:.4f}")

    fused = rank_fuse(oof)
    auc_fused = roc_auc_score(y, fused)
    print(f"  {'秩融合':8s} {auc_fused:.4f}")

    from scipy.stats import rankdata
    # 秩融合的输出是"秩均值"（1.0 ~ n），不是概率。直接叫 risk_prob 会误导，
    # 所以分成两列：risk_score 保留秩分原值，risk_prob 只做线性映射到 [0,1]。
    # risk_prob 的含义是"该车在车队中的相对位置"，不是校准后的出险概率——
    # 要得到真正的概率需要做校准（Platt / isotonic），本仓库没有做。
    n = len(fused)
    risk_prob = (fused - 1.0) / max(n - 1, 1)
    pred = pd.DataFrame({
        "gpsno": df["gpsno"],
        "risk_score": fused,
        "risk_prob": risk_prob,
        "risk_rank": rankdata(-fused, method="ordinal").astype(int),
    }).sort_values("risk_rank")
    # 正类比例与训练集正例率一致
    n_pos = int(round(len(pred) * y.mean()))
    pred["accident"] = (pred["risk_rank"] <= n_pos).astype(int)
    pred.to_csv(out_dir / "task1_forecast_result.csv", index=False, encoding="utf-8-sig")
    print(f"\n正类 {n_pos} 台（按训练集正例率 {y.mean() * 100:.1f}% 折算）")
    print(f"输出：{out_dir / 'task1_forecast_result.csv'}")


if __name__ == "__main__":
    main()
