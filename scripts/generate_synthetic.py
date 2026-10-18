# -*- coding: utf-8 -*-
"""合成驾驶安全数据集生成器。

为什么需要这个文件
------------------
本仓库的方法来自一次真实的车队安全竞赛，原始数据（500 台车 × 61 天、
13 亿行轨迹与 IMU 传感器数据）受数据保密协议约束，不能公开。
为了让任何人 clone 下来就能跑通整条管线，这里按**完全相同的表结构**
生成一份合成数据：列名、类型、行数、取值量级都对齐真实数据，
但每一个数字都是随机生成的，与真实车队没有任何对应关系。

生成的数据伪造了三个真实数据里存在的关键特性，
因为方法的价值恰恰体现在处理这些特性上：

1. **里程混杂** —— 跑得多的车事件更多，若不做暴露量归一化，
   "数据量"类特征会伪装成高预测力特征。
2. **事件密度差异巨大** —— 不同事件类型的频次跨越 5 个数量级
   （车道偏移几十万条 vs 事故几十条），直接建模会被高频事件主导。
3. **标签稀疏且不平衡** —— 正例约 10%，且事故类事件几乎不出现，
   导致正例定义必须组合事故与未遂事故两种信号。

运行
----
    python scripts/generate_synthetic.py --out data/synthetic

输出（全部为虚构数据）
----------------------
    vehicle_profile.csv     500 行车辆画像
    event_daily.parquet     车辆-日 粒度事件计数
    traj_daily.parquet      车辆-日 粒度轨迹聚合
    imu_daily.parquet       车辆-日 粒度 IMU 聚合
    imu_events_daily.parquet 车辆-日 粒度 IMU 事件识别结果
    labels.csv              500 行标签
    task2_candidates.csv    8 个率指标的车辆级聚合
"""

from __future__ import annotations

import argparse
import json
from math import erf, erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 全局参数

SEED = 20260601
N_VEHICLES = 500
START_DATE = "2026-06-01"
N_DAYS = 61

# 标签阈值：与真实口径一致——事故 >= 1 次 或 未遂事故 >= 2 次
ACCIDENT_THRESHOLD = 1
NEARMISS_THRESHOLD = 2

# IMU 设备覆盖不全（真实数据只有约 48.7% 的车辆-日存在 IMU 记录），
# 这个缺陷必须复现，否则"覆盖率混淆"这个技术难点在合成数据上不会出现。
IMU_COVERAGE = 0.487

# 事件类型 -> 每千公里基准发生率（合成值，非真实基准）
# 数值刻意跨越多个数量级，复现真实数据"高频事件主导建模"的问题。
# accident / nearmiss 两档经标定：生成后正例率落在 10%~14%，
# 与真实数据的 12.8% 同量级，否则正例太少无法训练也无法演示分层。
EVENT_RATES_PER_1K_KM = {
    "lane": 45.0,           # 车道偏移 / 压线
    "smoke": 8.0,           # 吸烟
    "distraction": 4.5,     # 分神（低头/打电话/看手机）
    "collision_warn": 3.2,  # 碰撞预警
    "fatigue": 2.0,         # 疲劳（闭眼/打哈欠/困倦）
    "blind_zone": 1.1,      # 盲区报警
    "camera": 0.15,         # 摄像头遮挡 / 角度异常
    "speeding": 0.9,        # 超速
    "nearmiss": 0.016,      # 未遂事故（极稀疏但比事故多两个数量级）
    "accident": 0.0006,     # 事故（极稀疏）
}

# 各事件对潜在风险因子的敏感度。
# 盲区报警与吸烟刻意给负敏感度——复现真实数据中这两个指标
# 方向与直觉相反的现象，使"方向反常"这个技术难点能被演示。
EVENT_RISK_SENSITIVITY = {
    "lane": 0.85,
    "smoke": -0.30,
    "distraction": 0.55,
    "collision_warn": 0.45,
    "fatigue": 0.40,
    "blind_zone": -0.45,
    "camera": 0.30,
    "speeding": 0.60,
    "nearmiss": 0.90,
    "accident": 0.60,
}

SPEED_BINS = list(range(0, 135, 5))  # spd_bin_0 .. spd_bin_130，与真实表 27 档一致

# 汇总成"真实危险倾向"的各指标权重。
# 车道偏移刻意占据主导——它既是真实数据里单变量 AUC 最高的指标（0.8071），
# 又与分神强相关（真实 0.546）。只有让它在"危险倾向"里压倒分神，
# 才能复现真实数据里那个关键现象：分神单变量 AUC 有 0.6768（有信息），
# 但在多元回归中系数被挤成负数（真实值 -0.0208），
# 移除车道偏移后又翻正（真实 -0.138 -> +0.178）。
# 分神权重 0.10 是扫描标定出来的（0.20 以上系数就翻正、抑制现象消失）。
DANGER_WEIGHTS = {
    "lane": 1.60,
    "distraction": 0.10,
    "speeding": 0.55,
    "fatigue": 0.35,
    "camera": 0.25,
    "collision_warn": 0.20,
}
# 危险倾向对事故/未遂事故发生率的放大强度（标定项，见 --report-calibration）
DANGER_KAPPA = 0.95


def _vehicle_ids(n: int, rng: np.random.Generator) -> list[str]:
    """生成外观与真实设备号相似、但纯属虚构的车号。

    真实车号是 8 位数字且集中在特定号段。这里刻意使用不同的号段，
    避免与任何真实设备撞号；可用 anonymize.py 再转成 VEH_XXXX。
    """
    ids, seen = [], set()
    while len(ids) < n:
        candidate = f"90{rng.integers(100000, 999999)}"
        if candidate not in seen:
            seen.add(candidate)
            ids.append(candidate)
    return ids


def _build_latent(n: int, rng: np.random.Generator) -> dict:
    """构造车辆级潜在变量。

    真实车队里风险高的车往往同时"跑得多"，两者相关但不是同义词。
    这里用两个独立因子按 0.55 / 0.83 的权重合成，使
    corr(风险, log 里程) 落在 0.5 左右，既不独立也不等价——
    这正是"必须控制里程才能看出真实行为差异"的场景。
    """
    z_exposure = rng.normal(0, 1, n)
    z_behavior = rng.normal(0, 1, n)

    monthly_km = np.exp(np.log(9000) + 0.55 * z_exposure).clip(1200, 32000)
    log_km_std = (np.log(monthly_km) - np.log(monthly_km).mean()) / np.log(monthly_km).std()

    # 潜在风险因子：一半来自里程，一半来自行为
    z = 0.55 * log_km_std + 0.83 * z_behavior

    night_ratio = np.clip(0.10 + 0.06 * z_exposure + 0.05 * z_behavior, 0.0, 0.55)
    highway_ratio = np.clip(0.20 - 0.10 * z_exposure + rng.normal(0, 0.13, n), 0.0, 0.85)
    stay_per_month = np.clip(60 + 18 * rng.normal(0, 1, n), 8, 200)

    return {
        "z": z,
        "z_exposure": z_exposure,
        "monthly_km": monthly_km,
        "monthly_hours": monthly_km / np.clip(rng.normal(46, 8, n), 20, 75),
        "night_ratio": night_ratio,
        "highway_ratio": highway_ratio,
        "stay_per_month": stay_per_month,
    }


def _daily_exposure(veh_idx: int, lat: dict, rng: np.random.Generator, n_days: int) -> np.ndarray:
    """把月均里程摊到每一天。

    真实数据里车辆并非每天都出车（有停运日），且工作日/周末强度不同。
    这里用车日活跃概率 + 对数正态扰动来复现这个形态。
    """
    daily_km_mean = lat["monthly_km"][veh_idx] / 30.0
    active = rng.random(n_days) < 0.82
    # 星期效应：周末略降
    weekday_factor = np.where(np.arange(n_days) % 7 >= 5, 0.75, 1.0)
    noise = np.exp(rng.normal(0, 0.45, n_days))
    km = daily_km_mean * weekday_factor * noise * active
    # 上界保护：货运车辆单日里程极少超过 1200 km。
    # 不设上界时，低均值车辆偶发的大扰动会算出一天几十小时行驶，
    # 派生出数万个采样点，既不合物理也让生成器慢到不可用。
    return np.round(np.clip(km, 0, 1200), 3)


def build_traj_daily(gpsnos, lat, dates, rng):
    """轨迹日聚合。列结构与真实数据一致（83 列）。

    性能说明：早期版本对每个车辆-日都做一次 n_points 量级的 numpy 抽样
    （速度分布 + 小时分布），500 车 × 61 天 × 数千点会跑十几分钟。
    改为向量化：先把一天的所有采样点压成直方图，再一次性算出
    spd_bin / hr_all / hr_mv，耗时降到秒级且结果分布等价。
    """
    rows = []
    for i, g in enumerate(gpsnos):
        km = _daily_exposure(i, lat, rng, len(dates))
        for d, k in zip(dates, km):
            if k <= 0:
                continue
            # 平均速度随高速占比上升而上升
            mean_spd = np.clip(rng.normal(38, 11) + 25 * lat["highway_ratio"][i], 3, 95)
            run_h = k / max(mean_spd, 3.0)
            # 采样点数：真实车载终端按**里程**触发采样（约每 150 米一个点），
            # 所以点数几乎是里程的线性函数。这一点很关键——它让
            # n_points 成为"纯粹的暴露量代理"，不含任何速度或行为信息，
            # 从而能干净地演示"计数类特征 AUC 高只是因为跑得多"。
            interval_m = float(np.clip(rng.normal(150, 15), 90, 220))
            n_points = int(np.clip(k * 1000.0 / interval_m, 20, 60000))
            n_moving = int(n_points * np.clip(0.80 + rng.normal(0, 0.07), 0.35, 0.99))

            rec = {"gpsno": g, "date": d, "n_points": n_points, "n_moving": n_moving,
                   "dist_km": round(float(k), 3), "run_time_h": round(float(run_h), 4)}

            # 速度分布：以 mean_spd 为均值的对数正态，直接出直方图
            spd = np.clip(rng.lognormal(np.log(max(mean_spd, 4)), 0.35, n_points), 0, 140)
            rec["spd_sum"] = float(spd.sum())
            rec["spd_sq_sum"] = float((spd ** 2).sum())
            idx = np.clip((spd // 5).astype(int), 0, len(SPEED_BINS) - 1)
            speed_counts = np.bincount(idx, minlength=len(SPEED_BINS))
            for j, b in enumerate(SPEED_BINS):
                rec[f"spd_bin_{b}"] = float(speed_counts[j])

            # 小时分布：行驶时段集中在白天，夜间按车辆夜间占比加权
            night_w = float(lat["night_ratio"][i])
            hour_p = np.array([max(night_w, 0.02)] * 6 + [1.0] * 12 + [max(night_w * 3, 0.05)] * 6)
            hour_p = hour_p / hour_p.sum()
            hours = rng.choice(24, size=n_points, p=hour_p)
            moving_mask = rng.random(n_points) < (n_moving / max(n_points, 1))
            h_all = np.bincount(hours, minlength=24)
            h_mv = np.bincount(hours[moving_mask], minlength=24)
            for h in range(24):
                rec[f"hr_all_{h}"] = float(h_all[h])
                rec[f"hr_mv_{h}"] = float(h_mv[h])
            rows.append(rec)
    return pd.DataFrame(rows)


def build_imu_daily(gpsnos, lat, dates, rng):
    """IMU 日聚合（73 列）。覆盖不全 + 大量充分统计量。

    性能说明：与轨迹表同理，早期版本逐日生成 n 个样本点（n 可达数万），
    500 车 × 61 天单是循环就超过十分钟。这里改为直接按统计量解析生成：
    均值、平方和、分箱计数都可以由正态分布参数直接算出，
    不需要真的抽 n 个数。
    """
    # 列清单从 schema 快照读取，保证与真实表结构一致
    schema_path = Path(__file__).resolve().parent.parent / "docs" / "schema_snapshot.json"
    with open(schema_path, encoding="utf-8") as f:
        cols = [c["name"] for c in json.load(f)["imu_daily"]["columns"]]

    dm_edges = np.arange(-0.20, 1.30, 0.05)
    rows = []
    for i, g in enumerate(gpsnos):
        km = _daily_exposure(i, lat, rng, len(dates))
        for d, k in zip(dates, km):
            if k <= 0 or rng.random() > IMU_COVERAGE:
                continue
            n = int(np.clip(k * rng.uniform(900, 1400), 500, 400000))  # 约 1 Hz 采样
            n_stat = int(n * np.clip(rng.normal(0.18, 0.05), 0.05, 0.45))
            n_mov = n - n_stat

            # 重力在 z 轴、动态加速度在 x/y 平面；前向轴符号随机翻转
            # （复现真实数据中 54.4% 车辆前向轴反向的问题）
            flip = rng.choice([-1, 1])
            dv_sigma = 0.9
            accel_fwd_sigma = dv_sigma * rng.uniform(0.7, 1.3)
            accel_lat_sigma = 1.4

            rec = {"gpsno": g, "date": d, "n": n, "n_spd_valid": n, "n_stat": n_stat, "n_mov": n_mov,
                   "dist_proxy_m": round(float(k * 1000), 2),
                   "ems_sum": float(np.clip(rng.normal(48, 10), 5, 110) * n)}

            # 零均值正态序列：sum ≈ N(0, sigma*sqrt(n))，平方和 ≈ n*sigma^2
            def _sum_sq(sigma, cnt):
                s = float(rng.normal(0, sigma * np.sqrt(max(cnt, 1))))
                ss = float(max(cnt, 1) * sigma ** 2 + rng.normal(0, sigma ** 2 * np.sqrt(max(cnt, 1))))
                return s, ss

            for axis, sigma in (("x", accel_fwd_sigma), ("y", accel_lat_sigma), ("z", 0.05)):
                s, ss = _sum_sq(sigma, n_mov)
                rec[f"sum_accel_{axis}_raw"] = s * flip if axis == "x" else s
                rec[f"sq_accel_{axis}_raw"] = ss
            for axis in ("x", "y", "z"):
                s, ss = _sum_sq(40.0, n)
                rec[f"sum_gyro_{axis}_raw"] = s
                rec[f"sq_gyro_{axis}_raw"] = ss
            for axis in ("x", "y", "z"):
                rec[f"stat_sum_accel_{axis}_raw"] = -9.8 * n_stat / 3 if axis == "z" else float(rng.normal(0, 2))
                rec[f"stat_sum_gyro_{axis}_raw"] = float(rng.normal(0, 15))
            for axis, sigma in (("x", accel_fwd_sigma), ("y", accel_lat_sigma), ("z", 0.05)):
                s, _ = _sum_sq(dv_sigma * sigma, n_mov)
                rec[f"dvdt_cross_accel_{axis}_raw"] = s
            rec["dvdt_sq"] = float(n_mov * dv_sigma ** 2)
            # 阈值计数：正态尾概率 × 样本数
            for thr in (0.10, 0.15, 0.20, 0.30, 0.40):
                p = erfc(thr / (dv_sigma * sqrt(2.0)))
                rec[f"dm_gt_{thr:.2f}"] = int(n_mov * p)
            for thr in (15, 25, 40, 60):
                p = erfc(thr / (40.0 * sqrt(2.0)))
                rec[f"gy_gt_{thr}"] = int(n * p)
            rec["stat_gy_mag_sq"] = float(n_stat * 40.0 ** 2)
            rec["spd_gps_ems_absdiff"] = float(rng.uniform(0, 400) * n_mov / 1000)
            rec["n_gps_ems_both"] = int(n_mov * 0.7)
            # dm 直方图：用正态密度的累积分布差估各箱计数
            for b in dm_edges:
                lo, hi = b - 0.025, b + 0.025
                p = 0.5 * (erf(hi / (dv_sigma * sqrt(2.0))) - erf(lo / (dv_sigma * sqrt(2.0))))
                rec[f"dm_bin_{b:.2f}"] = float(n_mov * p)

            # 只保留 schema 里存在的列，且按 schema 顺序
            rows.append({c: rec.get(c, 0) for c in cols})
    return pd.DataFrame(rows, columns=cols)


def build_imu_events_daily(gpsnos, lat, dates, rng):
    """IMU 事件识别结果（12 列）。频次关系遵循官方参考文档的数量级。"""
    rows = []
    for i, g in enumerate(gpsnos):
        km = _daily_exposure(i, lat, rng, len(dates))
        for d, k in zip(dates, km):
            if k <= 0 or rng.random() > IMU_COVERAGE:
                continue
            # 官方参考：急刹/急加 >> 急转弯/猛打方向 >> 疑似碰撞 >> 疑似侧翻
            base = max(k / 1000.0, 0.01)
            risk_mult = float(np.exp(0.35 * lat["z"][i]))
            rows.append({
                "gpsno": g, "date": d,
                "harsh_brake": int(rng.poisson(9.0 * base * risk_mult)),
                "harsh_accel": int(rng.poisson(6.0 * base * risk_mult)),
                "sharp_turn": int(rng.poisson(2.0 * base * risk_mult)),
                "swerve": int(rng.poisson(1.5 * base * risk_mult)),
                "collision_like": int(rng.poisson(0.05 * base * risk_mult)),
                "rollover_like": int(rng.poisson(0.004 * base * risk_mult)),
                "imu_mov_n": int(max(100, k * rng.uniform(900, 1400))),
                "al_sq": float(abs(rng.normal(0, 12))),
                "alat_sq": float(abs(rng.normal(0, 14))),
                "yaw_sq": float(abs(rng.normal(0, 30000))),
            })
    return pd.DataFrame(rows)


def build_event_daily(traj_df, lat, rng, id_to_idx, ind):
    """事件日计数（16 列）。

    关键设计：事件数 ∝ 当日里程 × exp(指标敏感度 × 该指标潜在因子)。
    里程项是"暴露量"，敏感度项是"行为"，两者乘在一起——这就是为什么
    直接用原始计数会把"跑得多"误判成"开得危险"。

    每个指标有自己的潜在因子（共享成分 + 独立成分），
    否则 8 个指标会完全共线，无法演示"被强指标吸收"这个现象。

    事故与未遂事故例外：它们不由自己的独立因子驱动，而是由各行为指标
    汇总出的"真实危险倾向"驱动（见 _danger_latent 的说明）。这样
    "行为差 → 事故多"的因果链在数据里真实存在，指标才具备真实量级的
    区分度；否则标签会退化成纯噪声。
    """
    cls_names = list(EVENT_RATES_PER_1K_KM.keys())
    danger = _danger_latent(ind, len(lat["z"]))
    rows = []
    for rec in traj_df[["gpsno", "date", "dist_km"]].itertuples(index=False):
        i = id_to_idx[rec.gpsno]
        km = float(rec.dist_km)
        out = {"gpsno": rec.gpsno, "date": rec.date}
        for name in cls_names:
            if name in ("accident", "nearmiss"):
                # 由真实危险倾向驱动，而非该事件自身的独立因子
                mult = float(np.exp(DANGER_KAPPA * danger[i]))
            else:
                mult = float(np.exp(EVENT_RISK_SENSITIVITY[name] * ind[name][i]))
            rate = EVENT_RATES_PER_1K_KM[name] * mult
            out[name] = int(rng.poisson(max(rate * km / 1000.0, 0.0)))
        counted = [out[n] for n in cls_names]
        out["ev_night"] = int(round(float(lat["night_ratio"][i]) * sum(counted)))
        out["ev_total"] = int(sum(counted))
        out["ev_speed_mean"] = float(np.clip(rng.normal(52, 14), 0, 120))
        out["ev_speed_max"] = int(np.clip(out["ev_speed_mean"] + rng.uniform(5, 45), 10, 140))
        rows.append(out)
    df = pd.DataFrame(rows)
    # 列顺序对齐真实表
    order = ["gpsno", "date", "accident", "blind_zone", "camera", "collision_warn",
             "distraction", "fatigue", "lane", "nearmiss", "smoke", "speeding",
             "ev_night", "ev_total", "ev_speed_mean", "ev_speed_max"]
    return df[order]


def _indicator_latent(n: int, rng: np.random.Generator, z: np.ndarray, rho: float = 0.55) -> dict:
    """为每个指标生成一个独立的潜在因子。

    为什么需要这一步
    ----------------
    早期版本让所有事件都只由同一个 z 驱动，结果是 8 个指标彼此高度相关，
    VIF 高达 25~31。真实数据的 VIF 全部低于 2，说明各指标既有共同成分
    也有大量各自独立的成分。这里用 rho 控制共享比例：
    与 z 的相关约 rho，指标之间相关约 rho²，由此得到与真实数据同量级的共线性。

    另外刻意让「车道偏移」与「分神」强相关（目标相关约 0.55）。
    真实数据里这两个指标的相关是 0.546，而且车道偏移的区分度明显更强，
    结果是在多元回归中分神的系数被挤成负数（抑制变量现象）。
    没有这一对强相关，"共线性导致系数反号"这个陷阱就无法真实演示。
    """
    s = np.sqrt(max(1.0 - rho ** 2, 1e-6))
    e = {name: rng.normal(0, 1, n) for name in EVENT_RATES_PER_1K_KM}

    # 分神 = 共享成分 + 一部分车道偏移的独立成分 + 自己的独立成分
    # 权重按目标相关 0.55 反解：corr = rho² + s²·w，解得 w ≈ 0.354
    w_lane = 0.354
    w_own = np.sqrt(max(1.0 - w_lane ** 2, 1e-6))
    e["distraction"] = w_lane * e["lane"] + w_own * e["distraction"]

    out = {}
    for name in EVENT_RATES_PER_1K_KM:
        out[name] = rho * z + s * e[name]
    return out


def _danger_latent(ind: dict, n: int) -> np.ndarray:
    """把各行为指标潜在因子汇总成"真实危险倾向"潜在变量。

    为什么标签不能只用自己的独立因子
    --------------------------------
    早期版本让 accident / nearmiss 各自带一个几乎独立的因子，
    结果事故与可观测行为（车道偏移等）几乎无关，最强的车道偏移
    AUC 只有 0.59——而真实数据里它是 0.8071。这不符合现实：
    出事故的车之所以出事故，正是因为它平时就在跑偏、超速、疲劳。

    权重取自真实数据各指标的单变量 AUC 排序（车道偏移最强，其次超速、
    分神、疲劳、设备异常、碰撞预警），所以这里的次序不是随手定的，
    是照真实数据的方向排的；数值本身是合成的。
    """
    d = np.zeros(n)
    total = sum(DANGER_WEIGHTS.values())
    for k, w in DANGER_WEIGHTS.items():
        d += (w / total) * ind[k]
    return (d - d.mean()) / max(d.std(), 1e-9)


def build_labels(event_daily, gpsnos, lat, rng):
    """标签表（500 行）。

    真实口径：事故 >= 1 次 或 未遂事故 >= 2 次 记为正例。
    事故太稀疏（合成后往往只有个位数），所以必须组合未遂事故信号，
    否则正例数会少到无法训练。列名与真实表一致。
    """
    agg = event_daily.groupby("gpsno").agg(
        accident_cnt=("accident", "sum"),
        near_miss_cnt=("nearmiss", "sum"),
    ).reindex(gpsnos).fillna(0).astype(int)

    pos = (agg["accident_cnt"] >= ACCIDENT_THRESHOLD) | (agg["near_miss_cnt"] >= NEARMISS_THRESHOLD)
    df = agg.reset_index()
    df["label"] = pos.values.astype(int)
    return df[["gpsno", "accident_cnt", "near_miss_cnt", "label"]]


def build_profile(gpsnos, lat, rng):
    """车辆画像（背景信息，不含任何模型特征）。"""
    energy = rng.choice(["柴油", "电动", "天然气"], size=len(gpsnos), p=[0.62, 0.28, 0.10])
    n = len(gpsnos)
    return pd.DataFrame({
        "设备号": gpsnos,
        "能源类型": energy,
        "月均行驶里程": np.round(lat["monthly_km"], 1),
        "月均行驶时长": np.round(lat["monthly_hours"], 1),
        "月平均停留次数": np.round(lat["stay_per_month"], 1),
        "高速里程占比": np.round(lat["highway_ratio"], 4),
        "早晨行驶时长占比": np.round(np.clip(rng.normal(0.18, 0.08, n), 0.01, 0.5), 4),
        "黄昏行驶时长占比": np.round(np.clip(rng.normal(0.12, 0.06, n), 0.01, 0.4), 4),
        "夜间行驶时长占比": np.round(lat["night_ratio"], 4),
        "夜间里程占比": np.round(np.clip(lat["night_ratio"] * rng.uniform(0.7, 1.3, n), 0.0, 0.6), 4),
    })


def build_task2_candidates(event_daily, traj_daily, gpsnos, labels):
    """任务二候选指标：8 个**率**指标（每千公里）。

    注意这里全部用率而不是计数——这正是"去暴露量混杂"的结论落地：
    计数类指标会把跑得多的车判为高风险。
    列名与真实表一致（含 km 与 label 两列辅助列）。
    """
    km = traj_daily.groupby("gpsno")["dist_km"].sum().reindex(gpsnos).fillna(0)
    km = km.replace(0, np.nan)

    mapping = {
        "驾驶员状态|疲劳": ["fatigue"],
        "驾驶员状态|分神": ["distraction"],
        "驾驶员状态|吸烟": ["smoke"],
        "驾驶平稳性|超速": ["speeding"],
        "驾驶平稳性|车道偏移": ["lane"],
        "驾驶平稳性|碰撞预警": ["collision_warn"],
        "管理配合度|设备异常": ["camera"],
        "环境道路|盲区报警": ["blind_zone"],
    }
    out = pd.DataFrame({"gpsno": gpsnos})
    out["km"] = km.values.round(5)
    out["label"] = labels.set_index("gpsno").reindex(gpsnos)["label"].values
    for name, cols in mapping.items():
        cnt = event_daily.groupby("gpsno")[cols].sum().sum(axis=1).reindex(gpsnos).fillna(0)
        out[name] = (cnt / km * 1000).round(8).values
    return out


def build_features_events(event_daily, gpsnos):
    """事件总量（只有一个用途：后面演示"数据量类特征是里程代理"这个陷阱）。"""
    total = event_daily.groupby("gpsno")["ev_total"].sum().reindex(gpsnos).fillna(0)
    return pd.DataFrame({"gpsno": gpsnos, "total_events": total.astype(int).values})


def report_calibration(cand, labels):
    """打印生成结果的关键统计，用于对照真实数据的口径。

    这不是装饰——合成数据最大的风险是"看着像，统计性质却完全不同"。
    每次重新标定后都应该跑一遍，确认：
      1) 正例率与真实数据的 12.8% 同量级；
      2) 各指标对标签的区分度方向正确、量级接近真实数据；
      3) 指标间共线性不超过真实数据的水平（真实最大 VIF 1.94）；
      4) 吸烟、盲区报警仍保持方向反常（这是要复现的缺陷，不是 bug）。
    """
    from sklearn.metrics import roc_auc_score

    y = labels.set_index("gpsno").loc[cand["gpsno"], "label"].values
    cols = [c for c in cand.columns if "|" in c]
    print("\n--- 标定自检（合成数据 vs 真实数据口径）---")
    print(f"  正例数 {int(y.sum())} / {len(y)} = {y.mean() * 100:.1f}%   "
          f"（真实 64/500 = 12.8%）")

    print("  各指标单变量 AUC：")
    for c in cols:
        a = roc_auc_score(y, cand[c].fillna(0).values)
        print(f"    {c:20s} AUC = {a:.4f}")

    X = cand[cols].fillna(0).values.astype(float)
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    vmax = 0.0
    for j in range(X.shape[1]):
        others = [k for k in range(X.shape[1]) if k != j]
        A = np.column_stack([np.ones(len(X)), X[:, others]])
        beta, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
        resid = X[:, j] - A @ beta
        r2 = 1 - resid.var() / max(X[:, j].var(), 1e-12)
        vmax = max(vmax, 1.0 / max(1 - r2, 1e-9))
    print(f"  最大 VIF = {vmax:.2f}   （真实数据最大 1.94）")


def main():
    # 默认输出到仓库内的 data/synthetic，而不是「当前工作目录下的 data/synthetic」。
    # 这是个真踩过的坑：早先写成相对路径 "data/synthetic"，
    # 从别的目录调用脚本时数据被悄悄写到了调用者的工作目录里，
    # 脚本本身不报错，只是后面所有步骤都读不到数据。
    repo_root = Path(__file__).resolve().parent.parent
    default_out = repo_root / "data" / "synthetic"

    ap = argparse.ArgumentParser(description="生成合成驾驶安全数据集")
    ap.add_argument("--out", default=str(default_out),
                    help=f"输出目录（默认 {default_out}）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--vehicles", type=int, default=N_VEHICLES)
    ap.add_argument("--report-calibration", action="store_true",
                    help="生成后打印标定自检（AUC / VIF / 正例率）")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    gpsnos = _vehicle_ids(args.vehicles, rng)
    lat = _build_latent(args.vehicles, rng)
    # 每个指标一个潜在因子（共享成分 + 独立成分），控制指标间共线性强度
    ind = _indicator_latent(args.vehicles, rng, lat["z"], rho=0.55)
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D").strftime("%Y-%m-%d").tolist()
    id_to_idx = {g: i for i, g in enumerate(gpsnos)}

    print(f"生成 {len(gpsnos)} 台车 × {len(dates)} 天 ...")

    traj = build_traj_daily(gpsnos, lat, dates, rng)
    print(f"  traj_daily            {traj.shape}")
    imu = build_imu_daily(gpsnos, lat, dates, rng)
    print(f"  imu_daily             {imu.shape}")
    imu_ev = build_imu_events_daily(gpsnos, lat, dates, rng)
    print(f"  imu_events_daily      {imu_ev.shape}")
    ev = build_event_daily(traj, lat, rng, id_to_idx, ind)
    print(f"  event_daily           {ev.shape}")
    labels = build_labels(ev, gpsnos, lat, rng)
    print(f"  labels                {labels.shape}  正例 {int(labels['label'].sum())} "
          f"({labels['label'].mean() * 100:.1f}%)")
    prof = build_profile(gpsnos, lat, rng)
    print(f"  vehicle_profile       {prof.shape}")
    cand = build_task2_candidates(ev, traj, gpsnos, labels)
    print(f"  task2_candidates      {cand.shape}")
    fev = build_features_events(ev, gpsnos)
    print(f"  features_events       {fev.shape}")

    traj.to_parquet(out_dir / "traj_daily.parquet", index=False)
    imu.to_parquet(out_dir / "imu_daily.parquet", index=False)
    imu_ev.to_parquet(out_dir / "imu_events_daily.parquet", index=False)
    ev.to_parquet(out_dir / "event_daily.parquet", index=False)
    labels.to_csv(out_dir / "labels.csv", index=False, encoding="utf-8-sig")
    prof.to_csv(out_dir / "vehicle_profile.csv", index=False, encoding="utf-8-sig")
    cand.to_csv(out_dir / "task2_candidates.csv", index=False, encoding="utf-8-sig")
    fev.to_csv(out_dir / "features_events.csv", index=False, encoding="utf-8-sig")

    print(f"\n输出目录：{out_dir.resolve()}")
    print("注意：以上全部为虚构数据，与任何真实车队无关。")

    if args.report_calibration:
        report_calibration(cand, labels)


if __name__ == "__main__":
    main()
