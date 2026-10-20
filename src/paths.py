"""路径与数据装载。

原项目的脚本把数据路径写死在 D:\\browserdownload\\... 里，换台机器就跑不了。
这里改成参数加环境变量，默认指向仓库内的 data/synthetic。
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
    """一次读进全部 8 张分表。

    含车号的列一律按字符串读。真实项目踩过：pandas 推断类型会把 8 位车号变成
    int64，跨表 merge 时静默错配，一条都不报。
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
