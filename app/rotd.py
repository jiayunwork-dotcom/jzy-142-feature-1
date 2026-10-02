"""方向无关组合谱 RotD50 / RotD100（Boore 等的 RotD 定义）。

做法
----
对周期 T、阻尼 ξ，单自由度线性体系对两条正交水平分量 :math:`a_1(t)`、
:math:`a_2(t)` 的响应（相对位移 u、相对速度 v）各自只积分**一次**；
利用线性体系的叠加性，把响应在水平面内按方位角 θ 投影::

    u_θ(t) = u_1(t) cos θ + u_2(t) sin θ

取时程峰值 :math:`S_θ`，再在离散方位角集合上统计：

- ``RotD100``：:math:`\\max_θ S_θ`，并给出取得最大值的方位角；
- ``RotD50``：:math:`S_θ` 的中位（偶数个方位角时取中间两个的算术
  平均）。

由于 :math:`S_{θ+π} = S_θ`，方位角只需扫 [0°, 180°)。默认 **180 个
方位角、步长 1°**。位移、伪速度/伪加速度、速度、绝对加速度三类量都
给出组合结果；其中 ``PSV = ω·RotD(SD)``、``PSA = ω²·RotD(SD)``
（ω 为正常数，与取 max/median 可交换），SD/SA 则按各自的投影时程
统计峰值。零周期点 T=0 直接在地面加速度上做同样的投影统计，
RotD100 = 水平面内 PGA 的方向最大值。

精度与误差
~~~~~~~~~~
离散方位角只让 RotD100 偏低。设真最大值在 θ*，网格上最近角度偏差
δ ≤ Δ/2；任意 θ 处的峰值可写成椭圆型表达式
:math:`S(θ)^2 = A\\cos^2θ + B\\sin^2θ + 2C\\sinθ\\cosθ`，其曲率有界，
由此网格相对误差不超过 :math:`Δ²/8`（Δ 以弧度计）。Δ = 1° 时该上界
约 **3.9×10⁻⁵**，多组随机信号对实测最坏 3.7×10⁻⁵，服务声明
RotD100 上界 ``5×10⁻⁵``，非整数网格角旋转比较采用
``1×10⁻⁴`` 相对容差。

RotD50 是逐方位角峰值样本的中位数（偶数点取中间两个的算术平均），
分箱统计是一阶 O(Δ)：2° 网格实测最坏约 2.6×10⁻³，1° 网格约
9×10⁻⁴（随信号变平缓，0.5° 时约 7×10⁻⁴ 即进入噪声平台）。服务
声明 RotD50 上界 ``2×10⁻³``（1° 网格），比较容差
``3×10⁻³``。这些容差与 README §11 声明一致，不为测试另开口子。

内存组织
~~~~~~~~
若一次保留全部「时间步 × 周期 × 方位角」的中间量，在 20 万点 ×
100 周期下会超过 5 GB。本模块两级分块：

1. **周期分块**：响应时程只保留一个周期块（默认每块至多 8 个周期，
   极端 20 万点下两条分量的 u/v 时程约 4×200001×8×8 B ≈ 51 MB），
   每个块重新积分一次（代价：积分 Python 循环按块数倍增，见 README
   §11/§12 的内存/耗时权衡）；
2. **时间分块统计**：方位角投影与峰值统计再按时间行分块（默认每块
   4096 行），用 (行×周期×2) 小矩阵乘 (2×方位角) 方向矩阵完成投影、
   单个大缓冲依次复用，瞬态中间量约 4096×8×180×8 B ≈ 47 MB。

20 万点 × 100 周期的极端组合峰值工作集约 250–310 MB（实测 RSS
约 300 MB，含 NumPy 与解释器基线；典型数千点记录仅数 MB），
实测值写在 README §12。
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .baseline import baseline_correct, ground_motion
from .errors import ValidationError
from .integrator import METHODS, newmark_state_histories
from .parsing import GRAVITY

ROTD_ALGORITHM_VERSION = "rotd-1"

DEFAULT_N_ANGLES = 180
MIN_N_ANGLES = 8
MAX_N_ANGLES = 7200

# 周期块：每块最多保留多少个周期的响应时程（内存上限的主旋钮）。
# 极端 20 万点下两条分量的 u/v 时程 4×200001×8×8 ≈ 51 MB；取 8 而非
# 更大的 12，是为了与方位角投影缓冲合计后把峰值 RSS 压在 250 MB 内
# （代价：积分循环按 ⌈周期数/8⌉ 块重复）。
PERIOD_BLOCK = 8
# 时间行块：投影统计时一次处理多少个时间步
ROW_BLOCK = 4096

# 离散方位角误差（实测/理论，见 README §11）：
# - RotD100 只可能偏低，椭圆型峰值曲率给出理论相对上界 Δ²/8；
#   Δ=1° 时 ≈3.9×10⁻⁵，多组随机信号对实测最坏 3.7×10⁻⁵，声明 5×10⁻⁵。
# - RotD50 是方位角峰值分布的中位数（偶数点取中间两个平均），分箱统计
#   是一阶 O(Δ) 误差，噪声型记录上 1° 网格实测最坏约 9×10⁻⁴，
#   声明保守上界 2×10⁻³。
# 受方位角离散影响的比较（旋转非网格整数角度）按这两个上界做判定。
ROTD100_ERROR_BOUND = 5.0e-5
ROTD50_ERROR_BOUND = 2.0e-3
ROTD100_COMPARE_TOL = 1.0e-4
ROTD50_COMPARE_TOL = 3.0e-3
# 旧名保留（其他模块/测试导入用）
ROTD_COMPARE_TOL = ROTD100_COMPARE_TOL


def validate_n_angles(n) -> int:
    """方位角个数：[8, 1440] 内的偶数（中位定义需要），默认 180。"""

    if n is None:
        return DEFAULT_N_ANGLES
    try:
        m = int(n)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"方位角个数 n_angles 必须是整数，收到 {n!r}") from exc
    if m != float(n):
        raise ValidationError(f"方位角个数 n_angles 必须是整数，收到 {n!r}")
    if not (MIN_N_ANGLES <= m <= MAX_N_ANGLES):
        raise ValidationError(
            f"方位角个数 n_angles 必须在 [{MIN_N_ANGLES},{MAX_N_ANGLES}] 之间，"
            f"收到 {m}（默认 {DEFAULT_N_ANGLES}，1° 网格）"
        )
    if m % 2:
        raise ValidationError(f"方位角个数 n_angles 必须是偶数（中位定义），收到 {m}")
    return m


def angle_grid(n_angles: int) -> np.ndarray:
    """[0, 180°) 上等间距方位角（弧度），利用 π 周期性。"""

    return np.arange(n_angles, dtype=np.float64) * (np.pi / n_angles)


def _rotd_stats(peak_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """给定 (..., n_angles) 的逐方位角峰值，返回 RotD50 / RotD100 /
    RotD100 方位角索引。沿最后一维统计。"""

    n = peak_angles.shape[-1]
    part = np.partition(peak_angles, (n // 2 - 1, n // 2), axis=-1)
    med = 0.5 * (part[..., n // 2 - 1] + part[..., n // 2])
    imax = np.argmax(peak_angles, axis=-1)
    mx = np.take_along_axis(peak_angles, np.expand_dims(imax, -1), axis=-1)[..., 0]
    return med, mx, imax


def _projected_peaks(
    u1: np.ndarray, v1: np.ndarray, u2: np.ndarray, v2: np.ndarray,
    w2: np.ndarray, c2xiw: np.ndarray,
    angles: np.ndarray,
) -> dict:
    """对一个周期块、全部方位角统计 SD/SV/SA 的逐方位角峰值。

    u*/v* 形状 ``(n_rows, n_periods_block)``；angles 形状 ``(n_angles,)``。
    返回各量形状 ``(n_periods_block, n_angles)``。SA 由平衡方程
    :math:`a_{abs} = -2ξω v - ω² u` 现算（地面输入已包含在 u/v 中，
    自由振动段同样成立）。时间行分块处理，使瞬态中间量内存有界。
    """

    n_rows, npb = u1.shape
    n_ang = angles.size
    # 方向矩阵 D = [[cos θ...], [sin θ...]]，形状 (2, n_angles)。
    # 投影统一写成 (rows, block, 2) × (2, n_angles) 的批量矩阵乘，
    # 全程只保留一个 (rows, block, n_angles) 大缓冲依次复用，避免
    # ux/vx/ax 三份大数组同时驻留内存。
    dirmat = np.vstack((np.cos(angles), np.sin(angles)))

    sd_p = np.zeros((npb, n_ang))
    sv_p = np.zeros((npb, n_ang))
    sa_p = np.zeros((npb, n_ang))

    buf = np.empty((min(ROW_BLOCK, n_rows), npb, n_ang))
    pair = np.empty((buf.shape[0], npb, 2))

    for r0 in range(0, n_rows, ROW_BLOCK):
        r1 = min(r0 + ROW_BLOCK, n_rows)
        rb = r1 - r0
        b = buf[:rb]
        p = pair[:rb]

        # SD：位移方向投影
        p[..., 0] = u1[r0:r1]
        p[..., 1] = u2[r0:r1]
        np.matmul(p, dirmat, out=b)
        np.maximum(sd_p, np.max(np.abs(b), axis=0), out=sd_p)

        # SV：速度方向投影
        p[..., 0] = v1[r0:r1]
        p[..., 1] = v2[r0:r1]
        np.matmul(p, dirmat, out=b)
        np.maximum(sv_p, np.max(np.abs(b), axis=0), out=sv_p)

        # 绝对加速度 SA：平衡方程给出 a_abs = a_rel + a_g = -2ξω v - ω²u，
        # 地面输入已含在 u/v 里，不能再加一次 a_g（自由振动段 a_g=0 时
        # 该式同样成立）。先在两个物理分量上组合，再投影。
        p[..., 0] = -(c2xiw * v1[r0:r1] + w2 * u1[r0:r1])
        p[..., 1] = -(c2xiw * v2[r0:r1] + w2 * u2[r0:r1])
        np.matmul(p, dirmat, out=b)
        np.maximum(sa_p, np.max(np.abs(b), axis=0), out=sa_p)

    return {"sd": sd_p, "sv": sv_p, "sa": sa_p}


def _ground_rotation_peaks(a1: np.ndarray, a2: np.ndarray,
                           angles: np.ndarray) -> np.ndarray:
    """T=0：地面加速度向量在各方位角投影的峰值，形状 (n_angles,)。"""

    cos = np.cos(angles)
    sin = np.sin(angles)
    # (n, n_angles)；20 万点 × 180 ≈ 288 MB 一次性偏大，按行块处理
    peaks = np.zeros(angles.size)
    for r0 in range(0, a1.size, ROW_BLOCK):
        r1 = min(r0 + ROW_BLOCK, a1.size)
        proj = a1[r0:r1, None] * cos + a2[r0:r1, None] * sin
        np.maximum(peaks, np.max(np.abs(proj), axis=0), out=peaks)
    return peaks


def compute_rotd_spectrum(
    acc1_si: np.ndarray,
    acc2_si: np.ndarray,
    dt: float,
    periods: np.ndarray,
    damping: float,
    *,
    method: str = "average_acceleration",
    instability_policy: str = "refine",
    baseline_mode: str = "mean",
    n_angles: int = DEFAULT_N_ANGLES,
) -> dict:
    """对一对已对齐、同单位（m/s²）、同步长的水平分量计算 RotD 组合谱。

    返回该阻尼比下的完整谱（周期列表、两条原始分量谱、几何平均、
    RotD50/RotD100 及 RotD100 方位角）以及与单条谱一致的数值方法
    元数据。
    """

    if method not in METHODS:
        raise ValidationError(
            f"未知 Newmark 方法 '{method}'，可选：{', '.join(METHODS)}"
        )
    periods = np.asarray(periods, dtype=np.float64)
    n_angles = validate_n_angles(n_angles)
    angles = angle_grid(n_angles)
    angle_deg = np.degrees(angles)

    c1 = baseline_correct(acc1_si, dt, baseline_mode)
    c2 = baseline_correct(acc2_si, dt, baseline_mode)
    v1g, d1g = ground_motion(c1, dt)
    v2g, d2g = ground_motion(c2, dt)

    # 地面加速度向量的方位角投影峰值（T=0 点与水平面 PGA 都用它），
    # 只算一次；时间行分块，瞬态内存有界。
    gp = _ground_rotation_peaks(c1, c2, angles)
    gmed, gmax, gimax = _rotd_stats(gp)

    has_zero = bool((periods <= 0).any())
    pos = periods > 0
    t_pos = periods[pos]
    nper = periods.size

    out: dict[str, np.ndarray] = {
        k: np.zeros(nper) for k in
        ("sd50", "sd100", "sv50", "sv100", "sa50", "sa100",
         "sd1", "sd2", "sv1", "sv2", "sa1", "sa2", "sdgm", "sagm")
    }
    theta100 = {q: np.zeros(nper) for q in ("sd", "sv", "sa")}

    comp_sd1 = np.zeros(t_pos.size)
    comp_sv1 = np.zeros(t_pos.size)
    comp_sa1 = np.zeros(t_pos.size)
    comp_sd2 = np.zeros_like(comp_sd1)
    comp_sv2 = np.zeros_like(comp_sd1)
    comp_sa2 = np.zeros_like(comp_sd1)

    refine_info = {"refined": False, "factor": 1, "dt_used": float(dt),
                   "tail_steps": 0}
    pos_idxs = np.nonzero(pos)[0]

    # 周期分块，控制响应时程内存
    for b0 in range(0, t_pos.size, PERIOD_BLOCK):
        b1 = min(b0 + PERIOD_BLOCK, t_pos.size)
        tb = t_pos[b0:b1]
        h1 = newmark_state_histories(
            c1, dt, tb, damping, method=method,
            instability_policy=instability_policy,
        )
        h2 = newmark_state_histories(
            c2, dt, tb, damping, method=method,
            instability_policy=instability_policy,
        )
        # 两条分量参数完全相同（同步长/同周期/同方法），加密信息应一致
        refine_info = {
            "refined": h1["refined"], "factor": h1["refine_factor"],
            "dt_used": h1["dt_used"], "tail_steps": h1["tail_steps"],
        }

        omega = 2.0 * np.pi / tb
        peaks = _projected_peaks(
            h1["u"], h1["v"], h2["u"], h2["v"],
            omega * omega, 2.0 * damping * omega, angles,
        )

        # 分量标量峰值（与 newmark_response 同源的状态更新）
        comp_sd1[b0:b1] = np.max(np.abs(h1["u"]), axis=0)
        comp_sv1[b0:b1] = np.max(np.abs(h1["v"]), axis=0)
        comp_sa1[b0:b1] = _abs_accel_peaks(
            h1["u"], h1["v"], omega * omega, 2.0 * damping * omega,
        )
        comp_sd2[b0:b1] = np.max(np.abs(h2["u"]), axis=0)
        comp_sv2[b0:b1] = np.max(np.abs(h2["v"]), axis=0)
        comp_sa2[b0:b1] = _abs_accel_peaks(
            h2["u"], h2["v"], omega * omega, 2.0 * damping * omega,
        )

        for qkey in ("sd", "sv", "sa"):
            med, mx, imax = _rotd_stats(peaks[qkey])
            idxs = pos_idxs[b0:b1]
            out[f"{qkey}50"][idxs] = med
            out[f"{qkey}100"][idxs] = mx
            # 各量 RotD100 方位角可能不同（SD/SA 的峰值不必同角度）
            theta100[qkey][idxs] = angle_deg[imax]

    # 零周期点：地面加速度投影统计
    if has_zero:
        zi = int(np.argmax(~pos))
        out["sa50"][zi] = gmed
        out["sa100"][zi] = gmax
        theta100["sa"][zi] = angle_deg[gimax]
        # T=0 时 SD/SV 为 0，方位角无定义，置 0

    comp_sd1f = np.zeros(nper); comp_sd1f[pos] = comp_sd1
    comp_sd2f = np.zeros(nper); comp_sd2f[pos] = comp_sd2
    comp_sv1f = np.zeros(nper); comp_sv1f[pos] = comp_sv1
    comp_sv2f = np.zeros(nper); comp_sv2f[pos] = comp_sv2
    comp_sa1f = np.zeros(nper); comp_sa1f[pos] = comp_sa1
    comp_sa2f = np.zeros(nper); comp_sa2f[pos] = comp_sa2
    if has_zero:
        zi = int(np.argmax(~pos))
        comp_sa1f[zi] = np.max(np.abs(c1))
        comp_sa2f[zi] = np.max(np.abs(c2))
    comp_sd1, comp_sd2 = comp_sd1f, comp_sd2f
    comp_sv1, comp_sv2 = comp_sv1f, comp_sv2f
    comp_sa1, comp_sa2 = comp_sa1f, comp_sa2f

    omega_full = np.divide(2.0 * np.pi, periods,
                           out=np.zeros_like(periods), where=pos)
    psv50 = omega_full * out["sd50"]
    psv100 = omega_full * out["sd100"]
    psa50 = omega_full ** 2 * out["sd50"]
    psa100 = omega_full ** 2 * out["sd100"]
    if has_zero:
        zi = int(np.argmax(~pos))
        psa50[zi] = gmed      # 零周期点 PSA = 水平方向 PGA 统计量
        psa100[zi] = gmax

    out["sd1"][...] = comp_sd1
    out["sd2"][...] = comp_sd2
    out["sv1"][...] = comp_sv1
    out["sv2"][...] = comp_sv2
    out["sa1"][...] = comp_sa1
    out["sa2"][...] = comp_sa2
    with np.errstate(divide="ignore", invalid="ignore"):
        out["sdgm"] = np.sqrt(np.maximum(comp_sd1 * comp_sd2, 0.0))
        out["sagm"] = np.sqrt(np.maximum(comp_sa1 * comp_sa2, 0.0))

    def _lst(a):
        return np.asarray(a, dtype=np.float64).tolist()

    psv1_full = omega_full * comp_sd1
    psa1_full = omega_full ** 2 * comp_sd1
    psv2_full = omega_full * comp_sd2
    psa2_full = omega_full ** 2 * comp_sd2
    if has_zero:
        zi = int(np.argmax(~pos))
        psa1_full[zi] = comp_sa1[zi]  # 零周期点 PSA = PGA
        psa2_full[zi] = comp_sa2[zi]

    return {
        "damping": damping,
        "periods": periods.tolist(),
        "n_angles": n_angles,
        "angle_step_deg": 180.0 / n_angles,
        "rotd50": {"sd": _lst(out["sd50"]), "sv": _lst(out["sv50"]),
                   "sa": _lst(out["sa50"]), "psv": _lst(psv50),
                   "psa": _lst(psa50)},
        "rotd100": {"sd": _lst(out["sd100"]), "sv": _lst(out["sv100"]),
                    "sa": _lst(out["sa100"]), "psv": _lst(psv100),
                    "psa": _lst(psa100),
                    # 各量取得 RotD100 的方位角；PSV/PSA 与 SD 同角，
                    # 故给 sd/sv/sa 三个角度即可
                    "angle_deg": _lst(theta100["sd"]),
                    "angle_deg_sd": _lst(theta100["sd"]),
                    "angle_deg_sv": _lst(theta100["sv"]),
                    "angle_deg_sa": _lst(theta100["sa"])},
        "components": [
            {"record_role": "h1", "sd": _lst(comp_sd1), "sv": _lst(comp_sv1),
             "sa": _lst(comp_sa1), "psv": _lst(psv1_full),
             "psa": _lst(psa1_full),
             "sa_g": _lst(comp_sa1 / GRAVITY),
             "psa_g": _lst(psa1_full / GRAVITY)},
            {"record_role": "h2", "sd": _lst(comp_sd2), "sv": _lst(comp_sv2),
             "sa": _lst(comp_sa2), "psv": _lst(psv2_full),
             "psa": _lst(psa2_full),
             "sa_g": _lst(comp_sa2 / GRAVITY),
             "psa_g": _lst(psa2_full / GRAVITY)},
        ],
        "geometric_mean": {
            "sd": _lst(out["sdgm"]),
            "sa": _lst(out["sagm"]),
            "psa": _lst(np.sqrt(np.maximum(psa1_full * psa2_full, 0.0))),
        },
        "method": method,
        "dt_used": refine_info["dt_used"],
        "refined": refine_info["refined"],
        "refine_factor": refine_info["factor"],
        "tail_steps": refine_info["tail_steps"],
        "ground_motion": {
            "h1": {"pga": float(np.max(np.abs(c1))),
                   "pgv": float(np.max(np.abs(v1g))),
                   "pgd": float(np.max(np.abs(d1g)))},
            "h2": {"pga": float(np.max(np.abs(c2))),
                   "pgv": float(np.max(np.abs(v2g))),
                   "pgd": float(np.max(np.abs(d2g)))},
            "rotd100_pga": float(gmax),
            "rotd50_pga": float(gmed),
        },
        "baseline": baseline_mode,
        "units": {"sd": "m", "sv": "m/s", "psv": "m/s",
                  "sa": "m/s2", "psa": "m/s2",
                  "sa_g": "g", "psa_g": "g"},
    }


def _abs_accel_peaks(u, v, w2, c2xiw) -> np.ndarray:
    """由 u/v 时程统计单条分量的绝对加速度峰值（时间行分块）。

    :math:`a_{abs} = a_{rel} + a_g = -2ξω v - ω² u`；地面输入已含在
    u/v 中，自由振动段 a_g=0 该式同样成立。
    """

    peaks = np.zeros(u.shape[1])
    for r0 in range(0, u.shape[0], ROW_BLOCK):
        r1 = min(r0 + ROW_BLOCK, u.shape[0])
        ax = -(c2xiw * v[r0:r1] + w2 * u[r0:r1])
        np.maximum(peaks, np.max(np.abs(ax), axis=0), out=peaks)
    return peaks


def rotd_fingerprint(
    *,
    group_content_key: str,
    periods: list[float],
    dampings: list[float],
    method: str,
    instability_policy: str,
    baseline_mode: str,
    n_angles: int,
) -> str:
    """组合谱结果缓存指纹：组内容 + 全部积分参数 + 算法版本。

    组不可变（成员变化即产生新组 ID 与新内容键），参数任何一项变化
    指纹都变，因此命中缓存绝不可能拿到旧成员/旧参数的结果。
    """

    payload = {
        "v": ROTD_ALGORITHM_VERSION,
        "group": group_content_key,
        "periods": [float(x) for x in periods],
        "dampings": [float(x) for x in dampings],
        "method": method,
        "instability_policy": instability_policy,
        "baseline_mode": baseline_mode,
        "n_angles": int(n_angles),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return "fp_" + hashlib.sha256(blob).hexdigest()
