# -*- coding: utf-8 -*-
"""车号脱敏：把真实 8 位设备号映射为 VEH_xxxxxxxx 形式。

三个设计目标，都是被实测打回后才定下来的：

1. 唯一 —— 不能撞号。
   早期版本用 `hash % 500` 把车号塞进 500 个坑，按生日问题必然冲突，
   实测直接输出「映射冲突」。

2. 跨文件一致 —— 同一辆车在不同文件里必须得到同一代号。
   中间版本改成「按哈希排序后依次分配 VEH_0001、VEH_0002……」，
   单文件内确实唯一，但**代号取决于这份文件里有多少辆车**。
   实测：全量 500 台与子集 400 台跑出来的代号，397 个不一致。
   只要两份文件的车号集合不同（比如评分结果只 50 台、
   明细表 500 台），同一个司机就会有两个代号，跨文件对不上账。
   最终方案：代号直接由 `sha256(盐 + 车号)` 派生，与文件内容无关。

3. 不可逆 —— 代号不能暴露真实车号的大小或先后关系。
   哈希派生天然满足；原本"按哈希排序分配序号"的方案也满足
   （实测真实车号数值与代号序号的 Spearman 为 −0.0168），
   但它被第 2 条否决了。

碰撞概率：8 位十六进制 = 32 位空间，500 台车的碰撞概率约 3e-5。
真的撞了会直接报错并提示加宽，不会静默合并两个司机。
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
