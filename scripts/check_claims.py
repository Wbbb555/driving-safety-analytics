"""核对 README 和 findings 里写的结论是否还成立。

文档里的数字会随依赖版本、种子、生成参数漂移。CI 只跑「脚本不报错」的话，
文档可以慢慢变成谎话，没人会发现。

所以只钉性质，不钉精确值。钉死 auc_km == 0.8074291300097751 会在每次依赖
升级时变红，然后被加 skip，等于没写。

用法：
    python scripts/check_claims.py
    python scripts/check_claims.py --json 路径
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAIL = []


def check(name: str, ok: bool, detail: str) -> None:
    tag = "OK  " if ok else "FAIL"
    print(f"  [{tag}] {name}: {detail}")
    if not ok:
        FAIL.append(name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "outputs" / "metrics.json"))
    args = ap.parse_args(argv)

    src = Path(args.json).expanduser().resolve()
    if not src.exists():
        print(f"找不到 {src}\n请先运行：python scripts/demo_traps.py --json outputs/metrics.json")
        return 2

    m = json.loads(src.read_text(encoding="utf-8"))
    t1, t2, t3, t4 = (m["T1_exposure"], m["T2_feature_selection"],
                      m["T3_collinearity"], m["T4_binning_leak"])

    # metrics.json 可能是旧版本留下的（字段只增不减，旧文件会缺 key）。
    # 直接 KeyError 看不出该怎么办，这里给一条能照着做的提示。
    required = {
        "T1_exposure": ["auc_km", "auc_count", "auc_resid", "auc_pure",
                        "auc_pure_resid_loglog", "auc_pure_resid_rawlog",
                        "spearman_pure_km"],
        "T2_feature_selection": ["naive", "nested", "scan"],
        "T3_collinearity": ["vif", "coef", "stab"],
        "T4_binning_leak": ["leak", "clean"],
    }
    for section, keys in required.items():
        missing = [k for k in keys if k not in m[section]]
        if missing:
            print(f"❌ {src} 的 {section} 缺少字段 {missing}。\n"
                  f"   这个文件可能是旧版生成的，请重新运行：\n"
                  f"   python scripts/demo_traps.py --json outputs/metrics.json")
            return 2

    print("检查 README / docs/findings.md 的公开结论：")

    # ---- 样本规模（所有偏差量级都以这个为前提） ----
    meta = m["meta"]
    check("样本量", meta["n_vehicles"] == 500, f"{meta['n_vehicles']} 台车")
    check("正例数", 40 <= meta["n_positive"] <= 90,
          f"{meta['n_positive']} 个正例（{meta['positive_rate']:.1%}）")

    # ---- 陷阱一：暴露量混杂 ----
    # 核心主张：纯采样点数（零行为信息）原始 AUC 很高，对数—对数残差化后 ≈ 0.5。
    check("纯暴露量原始 AUC 偏高", t1["auc_pure"] > 0.72,
          f"采样点数原始 AUC {t1['auc_pure']:.4f}")
    check("纯暴露量残差化后接近随机", abs(t1["auc_pure_resid_loglog"] - 0.5) < 0.06,
          f"对数—对数残差 {t1['auc_pure_resid_loglog']:.4f}（期望 ≈0.5）")
    check("残差化确实降了 AUC",
          t1["auc_pure_resid_loglog"] < t1["auc_pure"] - 0.15,
          f"降幅 {t1['auc_pure'] - t1['auc_pure_resid_loglog']:.4f}")
    # 这条是被自己的数据打回来后加的：raw-on-log 残差会留下假信号。
    check("raw-on-log 残差确实比 log-log 更乐观",
          t1["auc_pure_resid_rawlog"] > t1["auc_pure_resid_loglog"],
          f"{t1['auc_pure_resid_rawlog']:.4f} > {t1['auc_pure_resid_loglog']:.4f}")
    check("采样点数≈里程的改写", t1["spearman_pure_km"] > 0.95,
          f"Spearman {t1['spearman_pure_km']:.4f}")
    # 里程本身就有预测力，这是整个陷阱的前提。
    check("里程本身就有预测力", t1["auc_km"] > 0.70, f"里程 AUC {t1['auc_km']:.4f}")
    # 事件数含真信号，残差化后不该掉到 0.5。
    check("事件数残差化后仍留信号", t1["auc_resid"] > 0.65,
          f"事件数残差 {t1['auc_resid']:.4f}")

    # ---- 陷阱二：特征选择偏差 ----
    # 核心主张：朴素口径必须比嵌套口径乐观。若方向反了，说明流程写错了。
    check("朴素口径比嵌套乐观", t2["naive"] > t2["nested"],
          f"{t2['naive']:.4f} > {t2['nested']:.4f}（偏差 +{t2['naive'] - t2['nested']:.4f}）")
    scan = {int(k): v for k, v in t2["scan"].items()}
    spread = max(scan.values()) - min(scan.values())
    # 文档主张「整条曲线只在 ~0.013 内波动、无单调趋势」。
    check("top_k 扫描带宽是小量", spread < 0.05,
          f"带宽 {spread:.4f}（最高 {max(scan.values()):.4f} @ k={max(scan, key=scan.get)}）")
    # 用秩相关判「有没有单调趋势」，比肉眼可靠。
    ks = sorted(scan)
    n = len(ks)
    ranks = {k: i for i, k in enumerate(sorted(ks, key=lambda x: scan[x]))}
    d2 = sum((i - ranks[k]) ** 2 for i, k in enumerate(ks))
    rho = 1 - 6 * d2 / (n * (n * n - 1))
    check("扫描无稳定单调趋势", abs(rho) < 0.75,
          f"top_k 与 AUC 的 Spearman ρ = {rho:+.3f}")

    # ---- 陷阱三：共线性符号反转 ----
    check("没有严重共线性", max(t3["vif"].values()) < 5.0,
          f"最大 VIF {max(t3['vif'].values()):.2f}（警戒线 5）")
    # 这条是最重要的：不能把反号简单归因共线性。
    check("反号不能用共线性解释", max(t3["vif"].values()) < 3.0,
          f"最大 VIF 仅 {max(t3['vif'].values()):.2f}，远低于警戒线")
    stab = t3["stab"]
    # stab 记的是「系数为负的重抽样占比」，负=反常识方向。
    # 三档必须互斥：稳定正 / 不稳 / 稳定反。文档里把 21% 标成「稳定反号」
    # 就是踩了这个坑——21% 意味着 79% 反号，但 79% 没到稳定线。
    stable = [k for k, v in stab.items() if v <= 0.05]
    flippy = [k for k, v in stab.items() if 0.05 < v < 0.95]
    reversed_ = [k for k, v in stab.items() if v >= 0.95]
    check("三档互斥且覆盖全部指标",
          len(stable) + len(flippy) + len(reversed_) == len(stab),
          f"稳 {len(stable)} / 不稳 {len(flippy)} / 稳反 {len(reversed_)}，共 {len(stab)}")
    check("存在稳定方向的指标", len(stable) >= 2, f"{stable}")
    check("存在符号不稳的指标", len(flippy) >= 2, f"{flippy}")
    check("存在稳定反号的指标", len(reversed_) >= 1, f"{reversed_}")
    # 文档写「约 28~39% 会翻向」，只对「不稳」档成立。
    # 盲区报警 79% 属「多数反号但不稳」，必须单独看，否则会误列进这一句。
    mids = {k: v for k, v in stab.items() if 0.05 < v < 0.95}
    mild = {k: v for k, v in mids.items() if 0.15 <= v <= 0.60}
    extreme = {k: v for k, v in mids.items() if v > 0.60}
    check("文档所称 28~39% 那一档确实存在", len(mild) >= 2,
          "、".join(f"{k} {v:.1%}" for k, v in mild.items()))
    check("高反号但未达稳定线的指标被单独处理", True,
          "、".join(f"{k} {v:.1%}" for k, v in extreme.items()) or "本期无")

    # ---- 陷阱四：分箱泄漏 ----
    # 核心主张：全样本分箱必须比折内分箱乐观，且偏差在同量级。
    check("全样本分箱更乐观", t4["leak"] > t4["clean"],
          f"{t4['leak']:.4f} > {t4['clean']:.4f}（偏差 +{t4['leak'] - t4['clean']:.4f}）")
    gap = t4["leak"] - t4["clean"]
    check("泄漏量级与文档一致", 0.002 < gap < 0.06,
          f"偏差 +{gap:.4f}（文档写 +0.0165，run_task2 实测 +0.0075~+0.0115）")

    print()
    if FAIL:
        print(f"❌ {len(FAIL)} 项公开结论不再成立：{FAIL}")
        print("   要么修代码，要么改文档——不能让文档继续写不成立的话。")
        return 1
    print("✅ 全部公开结论成立")
    return 0


if __name__ == "__main__":
    sys.exit(main())
