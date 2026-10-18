"""路径与数据装载。

原项目里几乎所有脚本都把数据路径写死成 `D:\\browserdownload\\...`，
换一台机器就跑不动。这里统一改成「命令行参数 + 环境变量」，
默认值指向仓库内的 data/synthetic。
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "data" / "synthetic"
DEFAULT_OUT = REPO_ROOT / "outputs"


def resolve_data_dir(arg: str | None = None) -> Path:
    """数据目录优先级：命令行参数 > 环境变量 DSA_DATA_DIR > 仓库默认。"""
    if arg:
        return Path(arg).expanduser().resolve()
    if os.environ.get("DSA_DATA_DIR"):
        return Path(os.environ["DSA_DATA_DIR"]).expanduser().resolve()
    return DEFAULT_DATA


def resolve_out_dir(arg: str | None = None) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    if os.environ.get("DSA_OUT_DIR"):
        return Path(os.environ["DSA_OUT_DIR"]).expanduser().resolve()
    return DEFAULT_OUT


def load_bundle(data_dir: Path) -> dict[str, pd.DataFrame]:
    """一次性读入合成数据集的全部分表。

    所有含车号的表都以字符串读入。这一点在真实项目里踩过坑：
    pandas 推断类型会把 8 位车号变成 int64，跨表 merge 时静默错配。
    """
    d = data_dir
    return {
        "labels": pd.read_csv(d / "labels.csv", dtype={"gpsno": str}),
        "cand": pd.read_csv(d / "task2_candidates.csv", dtype={"gpsno": str}),
        "profile": pd.read_csv(d / "vehicle_profile.csv", dtype={"设备号": str}),
        "features_events": pd.read_csv(d / "features_events.csv", dtype={"gpsno": str}),
        "event_daily": pd.read_parquet(d / "event_daily.parquet"),
        "traj_daily": pd.read_parquet(d / "traj_daily.parquet"),
        "imu_daily": pd.read_parquet(d / "imu_daily.parquet"),
        "imu_events_daily": pd.read_parquet(d / "imu_events_daily.parquet"),
    }


def require(bundle: dict, name: str) -> pd.DataFrame:
    if name not in bundle or bundle[name] is None:
        raise FileNotFoundError(f"缺少数据表：{name}，请先运行 scripts/generate_synthetic.py")
    return bundle[name]
