"""核心函数的单元测试。

只测不变量，不测具体数值。数值会随数据、种子、sklearn 版本漂，钉死了几天就
红；真正该钉的是「逐项加总等于总分」「WOE 与出险率同向」「脱敏跨文件一致」
这类约束，而且坑恰恰都出在这些地方。

    python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from task1.run_task1 import rank_fuse  # noqa: E402
from task2.run_task2 import (  # noqa: E402
    apply_bins,
    fit_card,
    item_scores,
    monotonic_bins,
    score_from_card,
    woe_transform,
)

# ---------------------------------------------------------------- 数据夹具


@pytest.fixture(scope="module")
def toy():
    """小样本：一个强信号、一个纯噪声、一个跟强信号反着来的弱信号。

    为什么不用「一个刻意反向的指标」：WOE 分箱自己会把方向吸收掉。指标值越大
    出险率越高时 WOE 自动变低，系数自然为负，所以反向指标并不会产生反号。真正
    的反号来自抑制效应——两个指标高度相关，弱的那个边际贡献变负。这点一开始
    我也搞错了。
    """
    rng = np.random.default_rng(7)
    n = 300
    z = rng.normal(size=n)
    x1 = np.abs(z + rng.normal(scale=0.3, size=n)) * 10
    x2 = np.abs(rng.normal(size=n)) * 10
    x3 = np.abs(-z + rng.normal(scale=0.3, size=n)) * 10
    X = np.column_stack([x1, x2, x3])
    logit = 1.8 * z - 0.6
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y, ["强正向", "噪声", "镜像负向"]


@pytest.fixture(scope="module")
def card(toy):
    X, y, names = toy
    return fit_card(X, y, names)


@pytest.fixture(scope="module")
def flip_case():
    """搜一个真能产生正系数（反号）的样本，用来验符号约束。

    种子不写死。换 sklearn 版本之后固定种子可能就不反号了，那时测试会以「夹具
    失效」的名义误报一个绿。所以这里显式搜，搜不到直接 fail 并说明原因。
    """
    for seed in range(120):
        rng = np.random.default_rng(seed)
        n = 400
        z = rng.normal(size=n)
        f1 = np.abs(z + rng.normal(scale=0.4, size=n)) * 10
        f2 = np.abs(0.6 * z + rng.normal(scale=2.0, size=n)) * 10
        f3 = np.abs(-0.8 * z + rng.normal(scale=1.2, size=n)) * 10
        y = (rng.random(n) < 1 / (1 + np.exp(-(1.9 * z - 0.6)))).astype(int)
        if y.sum() < 50:
            continue
        c = fit_card(np.column_stack([f1, f2, f3]), y, ["a", "b", "c"])
        if c["beta_raw"].max() > 0.02:
            return c, float(c["beta_raw"].max())
    pytest.fail("在 120 个种子里都没能构造出反号系数，符号约束的测试失去意义")


# ---------------------------------------------------------------- 分箱


def test_edges_are_strictly_increasing(card):
    """分箱边界必须严格递增，否则 apply_bins 会漏箱或重叠。"""
    for b in card["bins"]:
        assert np.all(np.diff(b["edges"]) > 0), f"{b['name']} 边界非严格递增"


def test_every_sample_lands_in_exactly_one_bin(card, toy):
    """每个样本恰好落进一个箱：箱样本数之和 == 总样本数。"""
    for b in card["bins"]:
        assert sum(b["n"]) == len(toy[0]), f"{b['name']} 样本数之和不等于总数"


def test_bins_are_monotonic_in_bad_rate(card):
    """单调分箱的输出必须单调：箱内出险率随箱序不下降。"""
    for b in card["bins"]:
        rates = np.asarray(b["rate"], dtype=float)
        assert np.all(np.diff(rates) >= -1e-12), (
            f"{b['name']} 出险率非单调：{np.round(rates, 4)}"
        )


def test_monotonic_bins_merges_on_non_monotonic_input():
    """人为造非单调数据，单调合并必须把它压成单调。"""
    rng = np.random.default_rng(0)
    x = rng.random(400)
    # 出险概率在中间凹下去，明确非单调
    p = np.where((x > 0.35) & (x < 0.65), 0.02, 0.35)
    y = (rng.random(400) < p).astype(int)
    edges, n, pos, woe, rate = monotonic_bins(x, y, n_bins=6)
    assert len(rate) >= 2
    assert np.all(np.diff(rate) >= -1e-12), f"仍未单调：{np.round(rate, 4)}"
    assert sum(n) == len(x)


def test_woe_sign_agrees_with_bad_rate(card):
    """WOE 必须与箱内出险率反向：出险率越高，WOE 越低。

    注意用 argsort 比较秩会出错——argsort 返回的索引本身不保证可比，
    两箱时 argsort([0.1, 0.3]) 与 argsort([-1.0, -3.0]) 恰好互为逆序，
    看起来"不一致"其实是同一个单调关系。直接比 WOE 的单调性更可靠。
    """
    for b in card["bins"]:
        rates = np.asarray(b["rate"], dtype=float)
        woes = np.asarray(b["woe"], dtype=float)
        # 出险率上升 => WOE 必须下降
        assert np.all(np.diff(rates) * np.diff(woes) <= 1e-12), (
            f"{b['name']} WOE 与出险率未反向：rate={np.round(rates, 4)} "
            f"woe={np.round(woes, 4)}"
        )


# ---------------------------------------------------------------- 符号约束


def test_reverse_indicator_is_truncated_to_zero(flip_case):
    """真实产生反号时，符号约束必须把它截断为恰好 0。"""
    card, max_raw = flip_case
    assert max_raw > 0, "夹具应当含正系数"
    assert card["beta"].max() == 0.0, (
        f"符号约束未把正系数截断为 0，约束后最大系数 {card['beta'].max():.4f}"
    )
    assert np.all(card["beta"] <= 0)


def test_forward_indicator_keeps_negative_beta(toy, card):
    """强正向指标的系数应为负（风险因素，得分为正扣分）。"""
    j = toy[2].index("强正向")
    assert card["beta"][j] < 0


def test_all_constrained_betas_are_non_positive(card):
    assert np.all(card["beta"] <= 0), "约束后仍有正系数"


# ---------------------------------------------------------------- 加总恒等式


def test_item_scores_sum_equals_total(card, toy):
    """逐项得分加总必须精确等于总分。这是评分卡可审计的唯一 guarantee。

    这条曾经出过 bug：把已含汇总列的 DataFrame 整表求和，重复计数，
    报出「差 1106.773」。所以这里测的是不变量，不是某组数字。
    """
    X = toy[0]
    items = item_scores(card, X)
    total = score_from_card(card, X)
    item_cols = [c for c in items.columns if c != "安全分"]
    assert np.allclose(items[item_cols].sum(axis=1).values, total, atol=1e-9)


def test_median_score_is_anchor(card, toy):
    """锚点：**中位车的总分**应恰好等于 TARGET_MEDIAN_SCORE。

    注意不是「中位车的原始贡献为 0」。base_offset 的作用是把中位车的
    原始贡献整体平移到 700 分，所以 median(raw) 一般不为 0，
    为 0 的是 median(score) - TARGET_MEDIAN_SCORE。
    """
    from task2.run_task2 import TARGET_MEDIAN_SCORE

    sc = score_from_card(card, toy[0])
    assert abs(float(np.median(sc)) - TARGET_MEDIAN_SCORE) < 1e-6


def test_score_direction_high_is_safe(card, toy):
    """分数越高越安全：低风险组（y=0）均分应高于高风险组（y=1）。"""
    X, y, _ = toy
    sc = score_from_card(card, X)
    assert sc[y == 0].mean() > sc[y == 1].mean()


# ---------------------------------------------------------------- WOE 变换


def test_woe_transform_matches_apply_bins(card, toy):
    """woe_transform 的输出应是 WOE 值，而不是箱序号。

    这是个真 bug 的历史点：曾经返回箱序号去喂 LogisticRegression，
    8 个指标 6 个系数变正被截断为 0，结果不报错但全错。
    """
    X = toy[0]
    W = woe_transform(card, X)
    for j, b in enumerate(card["bins"]):
        idx = apply_bins(X[:, j], b["edges"])
        expected = b["woe"][idx]
        assert np.allclose(W[:, j], expected), f"{b['name']} WOE 变换与箱定义不一致"
        # 必须是 WOE 的取值集合，不是 0/1/2/3 箱序号
        assert set(np.unique(np.round(W[:, j], 6))).issubset(
            set(np.round(np.asarray(b["woe"], dtype=float), 6))
        )


# ---------------------------------------------------------------- 秩融合


def test_rank_fuse_is_rank_based():
    """秩融合只依赖排序，不依赖绝对概率刻度。

    构造两组刻度完全不同但排序相同的输入，融合结果必须完全相同。
    """
    rng = np.random.default_rng(3)
    a = rng.random(50)
    b = rng.random(50)
    fused_raw = rank_fuse({"a": a, "b": b})
    # 把 b 拉伸到完全不同刻度，排序不变
    fused_scaled = rank_fuse({"a": a, "b": b * 1000 + 7})
    assert np.allclose(fused_raw, fused_scaled)


def test_rank_fuse_direction_matches_input():
    """融合结果与原输入的排序方向一致（秩均值，越小越正面）。"""
    a = np.array([0.1, 0.5, 0.9])
    fused = rank_fuse({"a": a})
    assert np.all(np.diff(fused) > 0)


# ---------------------------------------------------------------- 脱敏


def test_anonymize_is_cross_file_consistent():
    """同一个车号在不同文件里必须得到同一个代号。

    这条被打回过两次：第二版按「哈希排序后依次编号」，
    代号取决于文件里有几台车，全量 500 台与子集 400 台有 397 个不一致。
    """
    from anonymize import code_of

    full = [f"90{i:06d}" for i in range(500)]
    subset = full[:400]
    small = full[:50]

    # 用同一个 SALT，代号只取决于车号本身
    m_full = {g: code_of(g) for g in full}
    m_subset = {g: code_of(g) for g in subset}
    m_small = {g: code_of(g) for g in small}

    assert all(m_subset[g] == m_full[g] for g in subset)
    assert all(m_small[g] == m_full[g] for g in small)


def test_anonymize_is_injective():
    """500 个不同车号必须映射到 500 个不同代号。"""
    from anonymize import code_of

    codes = [code_of(f"90{i:06d}") for i in range(500)]
    assert len(set(codes)) == 500, "发生代号碰撞"


def test_anonymize_does_not_preserve_order():
    """代号不得保留原始车号的大小顺序，否则可反推。

    用 Spearman 检验：真实车号数值与代号序号的秩相关应接近 0。
    """
    from scipy import stats

    from anonymize import code_of

    real = [f"90{i:06d}" for i in range(500)]
    codes = [code_of(g) for g in real]
    code_rank = stats.rankdata(codes)
    rho = float(stats.spearmanr(np.arange(500), code_rank).statistic)
    assert abs(rho) < 0.15, f"代号与原始顺序秩相关 {rho:.4f}，可能泄露排序"
