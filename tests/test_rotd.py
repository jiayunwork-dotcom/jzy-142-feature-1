"""RotD50 / RotD100 组合谱的数值关系测试。

需求中列出的每一条关系都有独立用例；容差全部取自
:mod:`app.rotd` 声明的方位角离散误差上界，不为测试另放宽：

- 网格整数角（30°、117°）旋转：纯角度重排，理论严格不变；
- 非网格角（30.5°）旋转：受方位角离散影响，按声明上界给容差；
- 交换分量、整条取反：同一角度集合的重排/等价，严格不变；
- 一条全零、两条相同、整体乘 k：解析关系，严格容差；
- RotD100 ≥ RotD50、RotD100 ≥ 两条分量谱较大者：逐周期成立；
- 重复提交逐位相同：确定性（另在 API 层端到端验证）。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.rotd import (
    DEFAULT_N_ANGLES,
    ROTD100_COMPARE_TOL,
    ROTD50_COMPARE_TOL,
    compute_rotd_spectrum,
    rotd_fingerprint,
    validate_n_angles,
)
from app.errors import ValidationError

DT = 0.01
PERIODS = np.array([0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0])
DAMPING = 0.05
N_ANG = DEFAULT_N_ANGLES


@pytest.fixture()
def pair():
    rng = np.random.default_rng(20261002)
    n = 4000
    t = np.arange(n) * DT
    a1 = (np.sin(2 * np.pi * t / 0.47) * np.exp(-t / 7.0)
          + 0.35 * np.cos(2 * np.pi * t / 1.13) * np.exp(-t / 9.0)
          + 0.15 * rng.standard_normal(n))
    a2 = (0.8 * np.sin(2 * np.pi * t / 0.62 + 1.1) * np.exp(-t / 6.0)
          + 0.3 * np.cos(2 * np.pi * t / 0.91 + 0.4) * np.exp(-t / 11.0)
          + 0.15 * rng.standard_normal(n))
    return a1, a2


def _rotd(a1, a2, *, n_angles=N_ANG, periods=PERIODS, baseline_mode="mean"):
    return compute_rotd_spectrum(
        a1, a2, DT, periods, DAMPING, n_angles=n_angles,
        baseline_mode=baseline_mode,
    )


def _arr(r, which, q):
    return np.asarray(r[which][q], dtype=np.float64)


# ---------------- 旋转不变性 ----------------

@pytest.mark.parametrize("deg", [30.0, 117.0])
def test_grid_angle_rotation_invariant_rotd100(pair, deg):
    """整数网格角旋转（30°、117°）：角度集合纯重排，逐位不变。"""

    a1, a2 = pair
    r0 = _rotd(a1, a2)
    th = np.deg2rad(deg)
    b1 = a1 * np.cos(th) + a2 * np.sin(th)
    b2 = -a1 * np.sin(th) + a2 * np.cos(th)
    r1 = _rotd(b1, b2)
    for q in ("sd", "sa", "psa"):
        np.testing.assert_allclose(
            _arr(r1, "rotd100", q), _arr(r0, "rotd100", q),
            rtol=1e-12, atol=0.0,
            err_msg=f"网格角 {deg}° 旋转后 RotD100 {q} 应逐位不变",
        )


@pytest.mark.parametrize("deg", [30.5, 117.3])
def test_offgrid_rotation_within_declared_bound(pair, deg):
    """非网格整数角旋转：偏差不得超过文档声明的 RotD100 上界（1e-4）。"""

    a1, a2 = pair
    r0 = _rotd(a1, a2)
    th = np.deg2rad(deg)
    b1 = a1 * np.cos(th) + a2 * np.sin(th)
    b2 = -a1 * np.sin(th) + a2 * np.cos(th)
    r1 = _rotd(b1, b2)
    for q in ("sd", "sa", "psa"):
        v0 = _arr(r0, "rotd100", q)
        v1 = _arr(r1, "rotd100", q)
        rel = np.max(np.abs(v1 - v0) / np.maximum(np.abs(v0), 1e-300))
        assert rel <= ROTD100_COMPARE_TOL, (
            f"非网格角 {deg}° 旋转 RotD100 {q} 相对偏差 {rel:.3e} 超过"
            f"声明容差 {ROTD100_COMPARE_TOL:g}"
        )


def test_offgrid_rotation_rotd50_within_bound(pair):
    """RotD50 非网格旋转比较按 RotD50 声明上界（3e-3 容差）。"""

    a1, a2 = pair
    r0 = _rotd(a1, a2)
    th = np.deg2rad(30.5)
    b1 = a1 * np.cos(th) + a2 * np.sin(th)
    b2 = -a1 * np.sin(th) + a2 * np.cos(th)
    r1 = _rotd(b1, b2)
    for q in ("sd", "sa", "psa"):
        v0 = _arr(r0, "rotd50", q)
        v1 = _arr(r1, "rotd50", q)
        rel = np.max(np.abs(v1 - v0) / np.maximum(np.abs(v0), 1e-300))
        assert rel <= ROTD50_COMPARE_TOL


def test_rotation_converges_to_fine_grid(pair):
    """误差收敛性：默认 1° 网格相对 0.1° 参考网格的 RotD100 偏差在
    声明上界 5e-5 内（证明文档上界站得住，不是为测试放宽）。"""

    a1, a2 = pair
    fine = _rotd(a1, a2, n_angles=1800)
    coarse = _rotd(a1, a2, n_angles=180)
    for q in ("sd", "sa", "psa"):
        vf = _arr(fine, "rotd100", q)
        vc = _arr(coarse, "rotd100", q)
        rel = np.max(np.abs(vc - vf) / np.maximum(np.abs(vf), 1e-300))
        assert rel <= 5.0e-5, f"RotD100 {q} 网格偏差 {rel:.3e} 超声明上界"


# ---------------- 分量顺序 / 符号 ----------------

def test_swap_components_invariant(pair):
    a1, a2 = pair
    r0 = _rotd(a1, a2)
    r1 = _rotd(a2, a1)
    for which in ("rotd50", "rotd100"):
        for q in ("sd", "sv", "sa", "psv", "psa"):
            np.testing.assert_allclose(
                _arr(r1, which, q), _arr(r0, which, q), rtol=1e-12, atol=0
            )


def test_negate_one_component_invariant(pair):
    a1, a2 = pair
    r0 = _rotd(a1, a2)
    r1 = _rotd(-a1, a2)
    r2 = _rotd(a1, -a2)
    for rr in (r1, r2):
        for which in ("rotd50", "rotd100"):
            for q in ("sd", "sv", "sa", "psv", "psa"):
                np.testing.assert_allclose(
                    _arr(rr, which, q), _arr(r0, which, q),
                    rtol=1e-12, atol=0,
                )


# ---------------- 极限/解析关系 ----------------

def test_zero_component_rotd100_equals_single(pair):
    # 这是纯线性代数关系（投影后只有一个非零方向），在 baseline_mode=none
    # 下检验，避免与基线处理混在一起。
    a1, _ = pair
    zero = np.zeros_like(a1)
    r = _rotd(a1, zero, baseline_mode="none")
    single = np.asarray(r["components"][0]["psa"])
    np.testing.assert_allclose(
        _arr(r, "rotd100", "psa"), single, rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        _arr(r, "rotd100", "sa"),
        np.asarray(r["components"][0]["sa"]), rtol=1e-12, atol=0,
    )
    # 一条分量全零时逐角度峰值为 |cosθ|·S1：
    # RotD50 = |cosθ| 在 [0,π) 的中位 (√2/2) × 单条谱。
    np.testing.assert_allclose(
        _arr(r, "rotd50", "psa"), single * np.sqrt(0.5),
        rtol=2e-3, atol=0,
    )


def test_identical_components_rotd100_sqrt2(pair):
    a1, _ = pair
    r = _rotd(a1, a1, baseline_mode="none")
    single = np.asarray(r["components"][0]["psa"])
    v100 = _arr(r, "rotd100", "psa")
    pos = np.asarray(PERIODS) > 0
    np.testing.assert_allclose(
        v100[pos], single[pos] * np.sqrt(2.0), rtol=1e-12, atol=0
    )
    # 注意 RotD50 此时**不是** √2：两条相同时程投影为
    # (cosθ+sinθ)·u(t)，各方位角响应峰值不同（45° 处取 √2 即
    # RotD100），中位数取的是别的角度——这是物理正确的行为。


def test_rotd100_dominates_rotd50_and_components(pair):
    a1, a2 = pair
    r = _rotd(a1, a2)
    v100 = _arr(r, "rotd100", "psa")
    v50 = _arr(r, "rotd50", "psa")
    c1 = np.asarray(r["components"][0]["psa"])
    c2 = np.asarray(r["components"][1]["psa"])
    np.testing.assert_array_less(v50 - 1e-12, v100 + 1e-9)
    np.testing.assert_array_less(np.maximum(c1, c2) - 1e-12, v100 + 1e-9)
    # SA 也成立
    s100 = _arr(r, "rotd100", "sa")
    s50 = _arr(r, "rotd50", "sa")
    assert np.all(s100 >= s50 - 1e-12)
    assert np.all(s100 >= np.maximum(
        np.asarray(r["components"][0]["sa"]),
        np.asarray(r["components"][1]["sa"])) - 1e-12)


def test_whole_group_scaling_linear(pair):
    """整组乘 k：组合谱随之乘 k（含 RotD100 方位角不变）。"""

    a1, a2 = pair
    k = 3.7
    r0 = _rotd(a1, a2)
    rk = _rotd(a1 * k, a2 * k)
    for which in ("rotd50", "rotd100"):
        for q in ("sd", "sv", "sa", "psv", "psa"):
            np.testing.assert_allclose(
                _arr(rk, which, q), _arr(r0, which, q) * k,
                rtol=1e-10, atol=0,
            )
    np.testing.assert_array_equal(
        _arr(rk, "rotd100", "angle_deg"),
        _arr(r0, "rotd100", "angle_deg"),
    )
    np.testing.assert_array_equal(
        np.asarray(rk["rotd100"]["angle_deg_sa"]),
        np.asarray(r0["rotd100"]["angle_deg_sa"]),
    )


def test_deterministic_bitwise(pair):
    a1, a2 = pair
    r1 = _rotd(a1, a2)
    r2 = _rotd(a1, a2)
    import json

    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


# ---------------- 零周期点与方位角 ----------------

def test_zero_period_is_pga(pair):
    a1, a2 = pair
    r = _rotd(a1, a2)
    # T=0：体系刚度无穷大，SA=PSA=水平面内地面加速度的 RotD 统计量。
    # 实现对整条（基线校正后的）地面加速度做投影，这里用 NumPy 的
    # median/max 独立逐角度核对。
    a1c = a1 - a1.mean()
    a2c = a2 - a2.mean()
    angles = np.arange(N_ANG) * (np.pi / N_ANG)
    proj = a1c[:, None] * np.cos(angles) + a2c[:, None] * np.sin(angles)
    pga_angles = np.max(np.abs(proj), axis=0)
    pga100 = float(pga_angles.max())
    pga50 = float(np.median(pga_angles))
    assert _arr(r, "rotd100", "sa")[0] == pytest.approx(pga100, rel=1e-12)
    assert _arr(r, "rotd100", "psa")[0] == pytest.approx(pga100, rel=1e-12)
    assert _arr(r, "rotd50", "sa")[0] == pytest.approx(pga50, rel=1e-12)
    assert _arr(r, "rotd50", "sd")[0] == 0.0
    assert _arr(r, "rotd100", "angle_deg_sa")[0] == \
        pytest.approx(np.degrees(angles[np.argmax(pga_angles)]))


def test_rotd100_angle_in_range(pair):
    a1, a2 = pair
    r = _rotd(a1, a2)
    for key in ("angle_deg", "angle_deg_sd", "angle_deg_sv", "angle_deg_sa"):
        ang = np.asarray(r["rotd100"][key])
        assert np.all(ang >= 0.0) and np.all(ang < 180.0)


# ---------------- 参数校验 ----------------

@pytest.mark.parametrize("bad", [3, 7, 1441, 1.5, "x", -4])
def test_invalid_n_angles(bad):
    with pytest.raises(ValidationError):
        validate_n_angles(bad)


def test_component_spectra_match_single_implementation(pair):
    """组合结果里两条分量各自的谱，必须与单条谱
    :func:`app.spectrum.compute_spectrum` 逐位一致（同一套 Newmark 递推，
    组合计算不得另搞一套数值口径），含 mean 与 none 两种基线模式。
    """

    from app.spectrum import compute_spectrum, prepare_spectrum_request

    a1, a2 = pair
    for mode in ("mean", "none"):
        req = prepare_spectrum_request(periods=PERIODS, dampings=DAMPING,
                                       baseline_mode=mode)
        s1 = compute_spectrum(a1, DT, req)["spectra"][0]
        s2 = compute_spectrum(a2, DT, req)["spectra"][0]
        r = _rotd(a1, a2, baseline_mode=mode)
        for q in ("sd", "sv", "sa", "psv", "psa"):
            np.testing.assert_allclose(
                r["components"][0][q], s1[q], atol=1e-12,
                err_msg=f"{mode} 下分量1 {q} 与单条谱不一致")
            np.testing.assert_allclose(
                r["components"][1][q], s2[q], atol=1e-12,
                err_msg=f"{mode} 下分量2 {q} 与单条谱不一致")


def test_fingerprint_distinguishes_params(pair):
    base = dict(
        group_content_key="abc", periods=[0.1, 1.0], dampings=[0.05],
        method="average_acceleration", instability_policy="refine",
        baseline_mode="mean", n_angles=180,
    )
    fp0 = rotd_fingerprint(**base)
    assert rotd_fingerprint(**base) == fp0
    for key, val in (("group_content_key", "abd"),
                    ("method", "linear_acceleration"),
                    ("baseline_mode", "linear"),
                    ("n_angles", 360),
                    ("instability_policy", "reject")):
        changed = dict(base)
        changed[key] = val
        assert rotd_fingerprint(**changed) != fp0, f"{key} 变化指纹必须变"
