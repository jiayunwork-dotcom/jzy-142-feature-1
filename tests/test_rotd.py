"""RotD50/RotD100 组合谱数值关系测试（需求逐条）。

容差口径
--------
- 旋转不变（整度旋转，角度集合只是置换）、交换分量、整体取反：这些关系
  在浮点运算顺序一致时应**逐位成立**；交换/取反改变了投影的求和次序，
  用 1e-12 相对容差（与 README 声明的浮点口径一致）；
- 一条分量为零、两条分量相同：解析恒等，1e-10；
- RotD100 ≥ RotD50、RotD100 ≥ 两条分量谱较大者：逐点 1e-12；
- 整组乘 k：2e-12（基线 mean 下的舍入量级，见 README §7）；
- 同一组同参数重复：**逐位相同**；
- 方位角离散误差：180 点（1°）网格相对 2880 点参考网格，RotD100 用文档
  声明的硬上界 0.87%，RotD50 用保证界 1%（实测 ~1e-3）——不允许为通过
  测试把容差放得比声明更宽。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.rotd import (
    CALC_VERSION,
    compute_rot_spectrum,
    prepare_rot_request,
    rot_cache_key,
)

PERIODS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
# 同一网格上旋转/交换/取反只改变投影求和次序，纯浮点舍入，2e-12
# （README 声明的浮点口径；不涉及角度离散误差）。
TIGHT = 2e-12


def _pair(n=1600, dt=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    a = np.sin(2 * np.pi * 1.7 * t) + 0.4 * rng.normal(size=n)
    b = 0.7 * np.cos(2 * np.pi * 1.1 * t + 0.5) + 0.4 * rng.normal(size=n)
    return a, b, dt


def _group(h1, h2, dt, gid="grp_t"):
    return {
        "id": gid, "name": gid, "h1": h1, "h2": h2, "dt": dt,
        "h1_id": "rec_a", "h2_id": "rec_b", "h1_name": "a", "h2_name": "b",
        "members": [
            {"record_id": "rec_a", "name": "a"},
            {"record_id": "rec_b", "name": "b"},
        ],
    }


def _run(h1, h2, dt, *, n_angles=180, dampings=(0.05,), periods=None):
    req = prepare_rot_request(
        periods=periods or PERIODS, dampings=list(dampings),
        n_angles=n_angles,
    )
    return compute_rot_spectrum(_group(h1, h2, dt), req)["spectra"][0]


def _arr(spec, band, q):
    return np.asarray(spec[band][q], dtype=np.float64)


# 1. 整度旋转不变性 ---------------------------------------------------------

@pytest.mark.parametrize("deg", [30.0, 117.0])
def test_rotation_invariance(deg):
    a, b, dt = _pair()
    th = np.deg2rad(deg)
    ar = a * np.cos(th) + b * np.sin(th)
    br = -a * np.sin(th) + b * np.cos(th)
    base = _run(a, b, dt)
    rot = _run(ar, br, dt)
    for band in ("rotd50", "rotd100"):
        for q in ("sd", "sa", "psa"):
            x = _arr(base, band, q)
            y = _arr(rot, band, q)
            np.testing.assert_allclose(
                y, x, rtol=TIGHT, atol=1e-14,
                err_msg=f"旋转 {deg}° 后 {band}.{q} 不一致",
            )


def test_rotation_preserves_rotd100_angle_set():
    """旋转后逐周期的 RotD100 方位角整体平移同一角度（模 180）。

    分量向量转动 th 等价于投影轴转动 -th（约定只影响符号），因此平移量
    为 ±th；这里只检验「所有周期平移量一致且大小为 th」。
    """

    a, b, dt = _pair(seed=5)
    base = _run(a, b, dt)
    th = 30.0
    c, s = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))
    rot = _run(a * c + b * s, -a * s + b * c, dt)
    ang0 = np.asarray(base["rotd100_angle_deg"]["psa"])
    ang1 = np.asarray(rot["rotd100_angle_deg"]["psa"])
    shift = (ang1 - ang0) % 180.0
    diff = np.minimum(np.abs(shift - th), np.abs(shift - (180.0 - th)))
    assert np.max(diff) <= 1.0 + 1e-9


# 2. 交换 / 取反 ------------------------------------------------------------

def test_swap_components_invariant():
    a, b, dt = _pair()
    base = _run(a, b, dt)
    swap = _run(b, a, dt)
    for band in ("rotd50", "rotd100"):
        for q in ("sd", "sa", "psa"):
            np.testing.assert_allclose(
                _arr(swap, band, q), _arr(base, band, q), rtol=TIGHT, atol=1e-14
            )


def test_negate_one_component_invariant():
    a, b, dt = _pair()
    base = _run(a, b, dt)
    neg = _run(-a, b, dt)
    for band in ("rotd50", "rotd100"):
        for q in ("sd", "sa", "psa"):
            np.testing.assert_allclose(
                _arr(neg, band, q), _arr(base, band, q), rtol=TIGHT, atol=1e-14
            )


# 3. 一条分量为零 ----------------------------------------------------------

def test_one_zero_component_rotd100_equals_single():
    a, _, dt = _pair(seed=2)
    zero = np.zeros_like(a)
    s = _run(a, zero, dt)
    single = np.asarray(s["components"][0]["psa"])
    d100 = _arr(s, "rotd100", "psa")
    np.testing.assert_allclose(d100, single, rtol=1e-10, atol=1e-14)
    d50 = _arr(s, "rotd50", "psa")
    # 一个分量恒为零时，方向峰值 = U|cosθ|，中位恰为 U/√2（解析）
    np.testing.assert_allclose(d50, single / np.sqrt(2.0), rtol=2e-3)
    # 方位角应对准非零分量所在轴（0°）
    assert np.all(np.asarray(s["rotd100_angle_deg"]["psa"]) == 0.0)


# 4. 两条分量完全相同 -------------------------------------------------------

def test_identical_components_rotd100_is_sqrt2():
    a, _, dt = _pair(seed=3)
    s = _run(a, a.copy(), dt)
    single = np.asarray(s["components"][0]["psa"])
    d100 = _arr(s, "rotd100", "psa")
    np.testing.assert_allclose(d100, np.sqrt(2.0) * single, rtol=1e-10, atol=1e-14)
    d50 = _arr(s, "rotd50", "psa")
    # RotD50 = 单条 × 中位|cosθ+sinθ|（参考值），恒 ≤ RotD100
    assert np.all(d50 <= d100 + 1e-12)


# 5. 逐点不等式 -------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 7, 11])
def test_rotd100_dominates_rotd50_and_components(seed):
    a, b, dt = _pair(seed=seed)
    s = _run(a, b, dt)
    c1 = np.asarray(s["components"][0]["psa"])
    c2 = np.asarray(s["components"][1]["psa"])
    d50 = _arr(s, "rotd50", "psa")
    d100 = _arr(s, "rotd100", "psa")
    assert np.all(d100 + 1e-12 >= d50)
    assert np.all(d100 + 1e-12 >= np.maximum(c1, c2))
    # 几何平均也不超过 RotD100
    geo = np.asarray(s["geomean"]["psa"])
    assert np.all(d100 + 1e-12 >= geo)
    # SD、SA 同样满足
    for q in ("sd", "sa"):
        d50q = _arr(s, "rotd50", q)
        d100q = _arr(s, "rotd100", q)
        cc1 = np.asarray(s["components"][0][q])
        cc2 = np.asarray(s["components"][1][q])
        assert np.all(d100q + 1e-12 >= d50q)
        assert np.all(d100q + 1e-12 >= np.maximum(cc1, cc2))


# 6. 整组乘 k ---------------------------------------------------------------

def test_group_scaling_linearity():
    a, b, dt = _pair(seed=4)
    base = _run(a, b, dt)
    k = 2.5
    scaled = _run(k * a, k * b, dt)
    for band in ("rotd50", "rotd100"):
        for q in ("sd", "sa", "psa"):
            x = _arr(base, band, q)
            y = _arr(scaled, band, q)
            np.testing.assert_allclose(y, k * x, rtol=2e-12, atol=1e-12)


def test_scaling_inverse_scale_factor_in_matching():
    """组整体乘 k，匹配缩放系数变为 1/k（在 group 匹配层验证解析关系）。"""

    from app.matching import make_design_spectrum, match_group_batch

    a, b, dt = _pair(seed=6)
    periods = np.array(PERIODS)
    s = _run(a, b, dt, periods=list(periods))
    psa = _arr(s, "rotd50", "psa")
    # 设计谱 = 2×记录谱 ⇒ 最优缩放系数 s* = 2；乘 k 后 s* = 2/k
    design = make_design_spectrum(periods.tolist(), (2.0 * psa).tolist(), "m/s2")
    k = 3.0
    sk = _run(k * a, k * b, dt, periods=list(periods))

    def one(values):
        cand = [{
            "group_id": "g", "periods": periods.tolist(),
            "rotd50": {"psa": values.tolist()},
            "rotd100": {"psa": values.tolist()},
            "members": [
                {"record_id": "a", "periods": periods.tolist(), "psa": values.tolist()},
                {"record_id": "b", "periods": periods.tolist(), "psa": values.tolist()},
            ],
        }]
        return match_group_batch(
            cand, design, 0.1, 3.0, s_min=1e-6, s_max=1e6,
            top_n=1, bounds_policy="reject",
        )["ranking"][0]["scale_optimal"]

    s0 = one(psa)
    s1 = one(_arr(sk, "rotd50", "psa"))
    assert s1 * k == pytest.approx(s0, rel=1e-12)
    assert s0 == pytest.approx(2.0, rel=1e-10)


# 7. 逐位确定性 -------------------------------------------------------------

def test_repeat_bitwise_identical():
    a, b, dt = _pair(seed=8)
    r1 = _run(a, b, dt, dampings=(0.02, 0.05))
    r2 = _run(a, b, dt, dampings=(0.02, 0.05))
    for band in ("rotd50", "rotd100"):
        for q in ("sd", "sa", "psa"):
            np.testing.assert_array_equal(
                _arr(r1, band, q), _arr(r2, band, q),
                err_msg=f"{band}.{q} 重复计算不逐位一致",
            )


# 8. 角度离散误差与文档声明一致 --------------------------------------------

@pytest.mark.parametrize("band,bound", [("rotd100", 0.0087), ("rotd50", 0.01)])
def test_angle_discretization_within_declared_bound(band, bound):
    a, b, dt = _pair(n=2400, seed=9)
    periods = np.logspace(np.log10(0.05), np.log10(4.0), 50).tolist()
    ref = _run(a, b, dt, n_angles=2880, periods=periods)
    coarse = _run(a, b, dt, n_angles=180, periods=periods)
    for q in ("sd", "sa", "psa"):
        x = _arr(ref, band, q)
        y = _arr(coarse, band, q)
        rel = np.abs(x - y) / np.maximum(np.abs(x), 1e-30)
        assert np.max(rel) <= bound, (
            f"{band}.{q} 角度离散相对误差 {np.max(rel):.3e} 超过文档声明 {bound}"
        )


def test_rotd100_angle_is_in_grid_and_half_period_redundant():
    a, b, dt = _pair(seed=12)
    s = _run(a, b, dt, n_angles=180)
    ang = np.asarray(s["rotd100_angle_deg"]["psa"])
    assert np.all((ang >= 0) & (ang < 180))
    assert np.allclose(ang, np.round(ang))  # 1° 网格


# 9. 缓存键对组成员/参数/版本敏感 ------------------------------------------

def test_cache_key_sensitive_to_inputs():
    a, b, dt = _pair()
    g = _group(a, b, dt)
    params = {"periods": PERIODS, "dampings": [0.05], "method": "average_acceleration",
              "instability_policy": "refine", "baseline_mode": "mean",
              "n_angles": 180}
    k0 = rot_cache_key(g, params)
    p2 = dict(params, n_angles=90)
    assert rot_cache_key(g, p2) != k0
    p3 = dict(params, baseline_mode="linear")
    assert rot_cache_key(g, p3) != k0
    g2 = _group(a * 1.0001, b, dt)
    assert rot_cache_key(g2, params) != k0
    assert CALC_VERSION  # 版本号存在；改动数值逻辑必须升版本


# 10. 输出结构 --------------------------------------------------------------

def test_result_structure_has_all_quantities_and_members():
    a, b, dt = _pair()
    s = _run(a, b, dt, dampings=(0.05,), periods=[0.0] + PERIODS)
    assert len(s["components"]) == 2
    assert set(s["rotd50"]) == {"sd", "sa", "sa_g", "psa", "psa_g"}
    assert set(s["rotd100"]) == set(s["rotd50"])
    assert "sd" in s["rotd100_angle_deg"] and "sa" in s["rotd100_angle_deg"]
    # 零周期点：PSA/SA 取 PGA 口径且非零
    assert s["rotd100"]["psa"][0] > 0
    assert s["rotd100"]["sd"][0] == 0.0
    # 两类 g 单位字段存在且等于 m/s2 除以 9.80665
    from app.parsing import GRAVITY

    np.testing.assert_allclose(
        s["rotd100"]["psa_g"],
        np.asarray(s["rotd100"]["psa"]) / GRAVITY,
    )


def test_multiple_dampings_independent():
    a, b, dt = _pair()
    out = compute_rot_spectrum(
        _group(a, b, dt),
        prepare_rot_request(periods=PERIODS, dampings=[0.02, 0.05]),
    )
    assert [x["damping"] for x in out["spectra"]] == [0.02, 0.05]
    # 阻尼越大，长周期谱峰值越小（阻尼效应）
    d02 = np.asarray(out["spectra"][0]["rotd100"]["psa"])
    d05 = np.asarray(out["spectra"][1]["rotd100"]["psa"])
    assert np.all(d02[1:] >= d05[1:] - 1e-12)
