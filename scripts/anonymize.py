# -*- coding: utf-8 -*-
"""车号脱敏：真实 8 位设备号换成 VEH_xxxxxxxx。

要求三条：不撞号、同一辆车在任何文件里代号相同、代号不暴露原车号的大小和
先后。三条都是被实测打回之后才定下来的。

第一版用 hash % 500 往 500 个坑里塞，按生日问题必然撞，跑起来直接打「映射
冲突」。第二版改成按哈希排序后依次发 VEH_0001，单文件里确实唯一了，但代号
取决于这份文件有几台车——500 台和 400 台子集跑出来 397 个对不上。评分结果
50 台、明细 500 台，同一个人就有了两个代号。

现在代号直接由 sha256(盐 + 车号) 派生，跟文件内容无关，跨文件自然一致。

碰撞概率：8 位十六进制是 32 位空间，500 台车大约 3e-5。真撞了会报错并提示
加宽位数，不会静默把两个司机并成一个。
"""
import hashlib
import os
import sys

SALT = "dph-public-repo-2026"
WIDTH = 8          # 代号哈希位数；若报碰撞，调大这个值


def code_of(gpsno: str, width: int = WIDTH) -> str:
    """由车号直接算出代号。相同车号永远得到相同代号，与所在文件无关。"""
    digest = hashlib.sha256((SALT + str(gpsno)).encode("utf-8")).hexdigest()
    return "VEH_" + digest[:width]


def build_mapping(gpsnos, width: int = WIDTH):
    """给一组车号建立 原号 -> 代号 的映射。

    唯一性由 assert 兜底：一旦碰撞就报错，而不是让两个司机共用一个代号。
    """
    uniq = sorted({str(g) for g in gpsnos if str(g).strip() and str(g) != "nan"})
    mapping = {g: code_of(g, width) for g in uniq}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(
            f"哈希碰撞：{len(uniq)} 个车号只得到 {len(set(mapping.values()))} 个代号。"
            f"请把 WIDTH 从 {width} 调大（例如 {width + 4}）。"
        )
    return mapping


def main():
    import pandas as pd

    if len(sys.argv) < 3:
        print("用法: python anonymize.py <输入csv> <输出csv>")
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]
    df = pd.read_csv(src, dtype=str)

    col = None
    for cand in ("gpsno", "设备号", "车辆ID", "vehicle_id"):
        if cand in df.columns:
            col = cand
            break
    if col is None:
        print("未找到车号列，实际列名：", list(df.columns))
        sys.exit(1)

    raw = df[col].tolist()
    code_of_one = {str(g): code_of(str(g)) for g in raw if str(g) not in ("nan", "")}
    df[col] = df[col].astype(str).map(code_of_one)

    n_before = len(code_of_one)
    assert df[col].notna().all(), "存在未映射的车号"
    assert df[col].nunique() == n_before, "脱敏后代号数与原车号数不一致"

    df.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"{os.path.basename(src)}: {len(df)} 行, {n_before} 个车号 -> 唯一代号 {df[col].nunique()} 个，无撞号")


if __name__ == "__main__":
    main()
