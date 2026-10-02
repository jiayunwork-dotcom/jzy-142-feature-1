"""方向无关组合谱 RotD50 / RotD100 的计算引擎（独立于单条谱逻辑）。

做法
----
对每个周期、每个阻尼比，把两条水平分量记为地面加速度向量
:math:`\\mathbf a_g(t)=(a_1(t),a_2(t))`。对方位角 θ 方向的单自由度
体系（运动方程 :math:`\\ddot x+2ξω\\dot x+ω²x=-\\mathbf a_g\\cdot e_θ`），
先分别积分两条分量的响应：

    xθ(t) = x1(t)·cosθ + x2(t)·sinθ   （线性系统满足叠加原理）

即**每个分量只积分一次**，方位角投影用分量响应的线性组合完成，而不是
逐方位角重新积分（后者耗时随角度数成倍增长）。每个 θ 取时程峰值后：

- RotD100(θ 网格) = max_θ max_t |xθ(t)|，并记录取得最大值的方位角；
- RotD50 = median_θ max_t |xθ(t)|。

方位角离散与误差上限（README 有完整推导与实测）
------------------------------------------------
θ 在 [0°,180°) 内均匀取 ``n_angles`` 个（默认 180，即 1° 一格；
θ 与 θ+180° 只相差正负号，峰值相同，故只需半周）。组合谱是集合
:math:`K=\\mathrm{conv}\\{\\pm\\mathbf r(t)\\}` 的支撑函数，其角度网格
近似有可证明的包络界：网格最大相对真最大的低估不超过

    1 − cos(Δθ/2)/(1 + sin(Δθ/2))

Δθ=1° 时约 **0.87%**（RotD100 的硬上界，只会低估不会高估）；RotD50
取中位值，文档声明的保证容差为 **1%**，180 点网格对典型强震记录的
实测偏差在 1e-4 量级（测试中以 2880 点网格做参考逐条核对）。1° 网格
还保证旋转任意整数度数后角度集合只是置换（见旋转不变性测试）。

内存有界
--------
分量响应按时间块流式积分（``TIME_BLOCK`` 步一块），角度投影也按角度
批（``ANGLE_BATCH``）在块内做收缩，峰值累加器只有「周期 × 角度」大小
（100×180×8B ≈ 144KB）。不在任何时刻保存「时间 × 周期 × 角度」的全量
中间结果。最坏情形（200 000 点、线性加速度法触发 5× 加密）峰值内存
≈ 150–200 MB（README 给测量值），不会撑爆容器。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .baseline import BASELINE_MODES, baseline_correct, ground_motion
from .errors import IntegrationError, ValidationError
from .integrator import (
    LINEAR_DT_OVER_T_LIMIT,
    METHODS,
    tail_step_count,
)
from .parsing import GRAVITY
from .spectrum import default_periods, validate_damping, validate_periods

# 计算版本：改变本模块任何数值逻辑都必须升版本号，使旧缓存自动失效。
CALC_VERSION = "rotd-1.0"

DEFAULT_N_ANGLES = 180
MIN_N_ANGLES = 8
MAX_N_ANGLES = 3600

# 流式块大小（时间步 / 角度批）：乘积决定投影临时数组的峰值占用。
# 512 × 100 周期 × 32 角度 × 8B ≈ 13 MB（表达式求值含一个同形临时量，
# 峰值约 2 倍），换内存不换结果。
TIME_BLOCK = 512
ANGLE_BATCH = 32

# 线性加速度法自动加密后，强迫段步数的硬上限：2×5,000,000×8B ≈ 80MB。
MAX_REFINED_STEPS = 5_000_000


@dataclass
class RotRequest:
    periods: np.ndarray
    dampings: tuple[float, ...]
    method: str = "average_acceleration"
    instability_policy: str = "refine"
    baseline_mode: str = "mean"
    n_angles: int = DEFAULT_N_ANGLES


def prepare_rot_request(
    periods=None,
    dampings=0.05,
    method: str = "average_acceleration",
    instability_policy: str = "refine",
    baseline_mode: str = "mean",
    n_angles: int = DEFAULT_N_ANGLES,
) -> RotRequest:
    if method not in METHODS:
        raise ValidationError(
            f"未知 Newmark 方法 '{method}'，可选：{', '.join(METHODS)}"
        )
    if instability_policy not in ("refine", "reject"):
        raise ValidationError(
            "instability_policy 必须是 refine 或 reject，"
            f"收到 {instability_policy!r}"
        )
    if baseline_mode not in BASELINE_MODES:
        raise ValidationError(
            f"基线处理方式必须是 {', '.join(BASELINE_MODES)} 之一，收到 "
            f"{baseline_mode!r}"
        )
    t = default_periods() if periods is None else validate_periods(periods)
    zs = validate_damping(dampings, allow_list=True)
    if not isinstance(zs, list):
        zs = [zs]
    try:
        na = int(n_angles)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"方位角数 n_angles 必须是整数，收到 {n_angles!r}") from exc
    if not (MIN_N_ANGLES <= na <= MAX_N_ANGLES):
        raise ValidationError(
            f"n_angles 必须在 [{MIN_N_ANGLES}, {MAX_N_ANGLES}] 之间，收到 {na}"
        )
    return RotRequest(
        periods=t, dampings=tuple(zs), method=method,
        instability_policy=instability_policy, baseline_mode=baseline_mode,
        n_angles=na,
    )


# --------------------------------------------------------------------------
# 流式 Newmark 积分（与 app.integrator 同一套递推，但按时间块产出分量
# 响应，供上层在块内做角度收缩；状态跨块保持）
# --------------------------------------------------------------------------

def _refine_series(h1, h2, dt, t_min, method, instability_policy):
    """两条分量共用同一加密倍数（只取决于 dt 与最短周期）。"""

    from .integrator import LINEAR_DT_OVER_T_LIMIT, refine_acceleration, required_subdivision

    factor = 1
    unstable = False
    if method == "linear_acceleration":
        unstable = (dt / t_min) > LINEAR_DT_OVER_T_LIMIT
        factor = required_subdivision(dt, t_min) if unstable else 1
        if factor > 1 and instability_policy == "reject":
            raise IntegrationError(
                f"线性加速度法在步长/周期比 Δt/T={dt / t_min:.4f} 超过稳定上限 "
                f"{LINEAR_DT_OVER_T_LIMIT:.5f}（最短周期 {t_min:g}s，Δt={dt:g}s）"
                "时失稳，策略为 reject：请改用平均加速度法或允许自动加密"
            )
    if factor > 1:
        total = (h1.size - 1) * factor + 1
        if total > MAX_REFINED_STEPS:
            raise IntegrationError(
                f"线性加速度法自动加密 {factor} 倍后强迫段共 {total} 步，超过 "
                f"{MAX_REFINED_STEPS} 步的内存保护上限：请缩短记录、加大最短"
                "周期或改用无条件稳定的平均加速度法"
            )
        h1, dt1 = refine_acceleration(h1, dt, factor)
        h2, _ = refine_acceleration(h2, dt, factor)
        dt = dt1
    return h1, h2, dt, factor, unstable


def _stream_component(
    ag: np.ndarray,
    dt: float,
    w2: np.ndarray,
    c2xiw: np.ndarray,
    gamma: float,
    beta: float,
    t_max: float,
    block: int = TIME_BLOCK,
):
    """生成器：逐块产出 (位移块, 绝对加速度块)，形状各为 (≤block, n_osc)。

    强迫段（ag[0] 用于初值）之后紧接自由振动段（地面输入置 0），
    与单条谱积分器的步数口径完全一致。
    """

    a1 = 1.0 / (beta * dt * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 0.5 / beta - 1.0
    c1 = gamma / (beta * dt)
    c2 = gamma / beta - 1.0
    c3 = dt * (gamma / (2.0 * beta) - 1.0)
    one_m_gamma_dt = (1.0 - gamma) * dt
    gamma_dt = gamma * dt
    keff = w2 + c2xiw * c1 + a1

    u = np.zeros_like(w2)
    v = np.zeros_like(w2)
    a = np.full_like(w2, -float(ag[0]))

    n_forced = ag.size - 1
    n_total = n_forced + tail_step_count(t_max, dt)

    u_buf = np.empty((block, w2.size), dtype=np.float64)
    q_buf = np.empty((block, w2.size), dtype=np.float64)
    k = 0
    for step in range(n_total):
        gk = float(ag[step + 1]) if step < n_forced else 0.0
        pred = c1 * u + c2 * v + c3 * a
        rhs = -gk + a1 * u + a2 * v + a3 * a + c2xiw * pred
        u_new = rhs / keff
        a_new = a1 * (u_new - u) - a2 * v - a3 * a
        v_new = v + one_m_gamma_dt * a + gamma_dt * a_new
        u_buf[k] = u_new
        q_buf[k] = a_new + gk          # 绝对加速度
        u, v, a = u_new, v_new, a_new
        k += 1
        if k == block:
            yield u_buf, q_buf
            k = 0
    if k:
        yield u_buf[:k], q_buf[:k]


# --------------------------------------------------------------------------
# 角度网格与块内收缩
# --------------------------------------------------------------------------

def angle_grid(n_angles: int) -> np.ndarray:
    """[0, π) 上等间距方位角（弧度）。"""

    return np.arange(n_angles, dtype=np.float64) * (math.pi / n_angles)


def _run_angle_contraction(stream1, stream2, cos_a, sin_a, n_period: int):
    na = cos_a.size
    sd_peaks = np.zeros((n_period, na))
    sa_peaks = np.zeros((n_period, na))
    comp_sd1 = np.zeros(n_period)
    comp_sd2 = np.zeros(n_period)
    comp_sa1 = np.zeros(n_period)
    comp_sa2 = np.zeros(n_period)
    for (u1, q1), (u2, q2) in zip(stream1, stream2):
        np.maximum(comp_sd1, np.max(np.abs(u1), axis=0), out=comp_sd1)
        np.maximum(comp_sd2, np.max(np.abs(u2), axis=0), out=comp_sd2)
        np.maximum(comp_sa1, np.max(np.abs(q1), axis=0), out=comp_sa1)
        np.maximum(comp_sa2, np.max(np.abs(q2), axis=0), out=comp_sa2)
        for js in range(0, na, ANGLE_BATCH):
            je = min(js + ANGLE_BATCH, na)
            c = cos_a[js:je][None, None, :]
            s = sin_a[js:je][None, None, :]
            pu = u1[:, :, None] * c + u2[:, :, None] * s
            np.abs(pu, out=pu)
            np.maximum(sd_peaks[:, js:je], np.max(pu, axis=0),
                       out=sd_peaks[:, js:je])
            pq = q1[:, :, None] * c + q2[:, :, None] * s
            np.abs(pq, out=pq)
            np.maximum(sa_peaks[:, js:je], np.max(pq, axis=0),
                       out=sa_peaks[:, js:je])
    return {
        "sd_peaks": sd_peaks, "sa_peaks": sa_peaks,
        "sd1": comp_sd1, "sd2": comp_sd2,
        "sa1": comp_sa1, "sa2": comp_sa2,
    }


def _reduce_rotd(peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """方位角维：RotD50（中位）、RotD100（最大）及 RotD100 方位角（度）。"""

    d50 = np.median(peaks, axis=1)
    idx = np.argmax(peaks, axis=1)
    d100 = np.take_along_axis(peaks, idx[:, None], axis=1).ravel()
    n_angles = peaks.shape[1]
    angles = idx * (180.0 / n_angles)
    return d50, d100, angles


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------

def compute_rot_spectrum(group: dict, req: RotRequest) -> dict:
    """计算一个已对齐分量组的 RotD50/RotD100 组合谱。

    ``group`` 需含 ``h1``/``h2``（m/s² 等步长序列）、``dt`` 及成员信息。
    """

    h1 = np.asarray(group["h1"], dtype=np.float64)
    h2 = np.asarray(group["h2"], dtype=np.float64)
    dt = float(group["dt"])
    if h1.shape != h2.shape or h1.size < 2:
        raise IntegrationError(
            f"分量组 {group.get('id')} 的两条对齐序列长度异常：{h1.shape}/{h2.shape}"
        )

    h1 = baseline_correct(h1, dt, req.baseline_mode)
    h2 = baseline_correct(h2, dt, req.baseline_mode)
    v1, d1g = ground_motion(h1, dt)
    v2, d2g = ground_motion(h2, dt)

    pga1, pga2 = float(np.max(np.abs(h1))), float(np.max(np.abs(h2)))
    pgv1, pgv2 = float(np.max(np.abs(v1))), float(np.max(np.abs(v2)))
    pgd1, pgd2 = float(np.max(np.abs(d1g))), float(np.max(np.abs(d2g)))

    periods = req.periods
    pos = periods > 0
    t_pos = periods[pos]
    angles_rad = angle_grid(req.n_angles)
    cos_a, sin_a = np.cos(angles_rad), np.sin(angles_rad)

    # 零周期点：质量与地面同加速度，直接对地面加速度向量做角度收缩
    ground_peaks, gp1v, gp2v = _ground_peaks(h1, h2, cos_a, sin_a)
    g50, g100, g100a = _reduce_rotd(ground_peaks)

    member_meta = group.get("members") or [
        {"record_id": group.get("h1_id"), "name": group.get("h1_name")},
        {"record_id": group.get("h2_id"), "name": group.get("h2_name")},
    ]

    gamma, beta = METHODS[req.method]
    # 加密倍数只取决于 dt 与最短周期，所有阻尼共用同一条加密序列
    rh1, rh2, dt_eff, factor, _ = _refine_series(
        h1, h2, dt, float(t_pos.min()), req.method, req.instability_policy
    )

    out_per_damping = []
    for zeta in req.dampings:
        omega = 2.0 * np.pi / t_pos
        w2 = omega * omega
        c2xiw = 2.0 * zeta * omega

        contracted = _run_angle_contraction(
            _stream_component(rh1, dt_eff, w2, c2xiw, gamma, beta,
                              float(t_pos.max())),
            _stream_component(rh2, dt_eff, w2, c2xiw, gamma, beta,
                              float(t_pos.max())),
            cos_a, sin_a, t_pos.size,
        )
        sd50, sd100, sd100_ang = _reduce_rotd(contracted["sd_peaks"])
        sa50, sa100, sa100_ang = _reduce_rotd(contracted["sa_peaks"])
        sd1, sd2 = contracted["sd1"], contracted["sd2"]
        sa1, sa2 = contracted["sa1"], contracted["sa2"]

        # 装入完整周期网格（含 T=0）
        def _full(pos_vals, zero=0.0):
            out = np.zeros(periods.size)
            out[pos] = pos_vals
            if periods.size and periods[0] == 0.0:
                out[0] = zero
            return out

        omega_full = np.divide(
            2.0 * np.pi, periods,
            out=np.zeros_like(periods), where=pos,
        )
        comp_specs = []
        sd_full1, sd_full2 = _full(sd1), _full(sd2)
        sa_full1, sa_full2 = _full(sa1, pga1), _full(sa2, pga2)
        for sd_f, sa_f, pga, pgv, pgd, meta in zip(
            (sd_full1, sd_full2), (sa_full1, sa_full2),
            (pga1, pga2), (pgv1, pgv2), (pgd1, pgd2), member_meta,
        ):
            psa = omega_full * omega_full * sd_f
            if periods.size and periods[0] == 0.0:
                psa[0] = pga
            comp_specs.append({
                "record_id": meta.get("record_id"),
                "name": meta.get("name"),
                "sd": sd_f.tolist(),
                "sa": sa_f.tolist(),
                "psa": psa.tolist(),
                "psv": (omega_full * sd_f).tolist(),
                "ground_motion": {"pga": pga, "pgv": pgv, "pgd": pgd,
                                  "pga_g": pga / GRAVITY},
            })

        def _pack(d50, d100, ang, zero50=0.0, zero100=0.0):
            f50, f100 = _full(d50, zero50), _full(d100, zero100)
            fang = np.zeros(periods.size)
            fang[pos] = ang
            return f50, f100, fang

        sd50_f, sd100_f, sd_ang_f = _pack(sd50, sd100, sd100_ang)
        sa50_f, sa100_f, sa_ang_f = _pack(
            sa50, sa100, sa100_ang, zero50=g50[0], zero100=g100[0]
        )
        # 零周期角度用地面加速度收缩结果（SA 口径）
        if periods.size and periods[0] == 0.0:
            sa_ang_f[0] = g100a[0]
            sd_ang_f[0] = g100a[0]
        psa50_f = omega_full * omega_full * sd50_f
        psa100_f = omega_full * omega_full * sd100_f
        if periods.size and periods[0] == 0.0:
            psa50_f[0] = g50[0]
            psa100_f[0] = g100[0]

        geo_sd = np.sqrt(np.maximum(sd_full1 * sd_full2, 0.0))
        geo_sa = np.sqrt(np.maximum(sa_full1 * sa_full2, 0.0))
        geo_psa = omega_full * omega_full * geo_sd
        if periods.size and periods[0] == 0.0:
            geo_psa[0] = math.sqrt(pga1 * pga2)

        unstable_full = np.zeros(periods.size, dtype=bool)
        if req.method == "linear_acceleration":
            unstable_full[pos] = (dt / t_pos) > LINEAR_DT_OVER_T_LIMIT

        out_per_damping.append({
            "damping": zeta,
            "components": comp_specs,
            "geomean": {
                "sd": geo_sd.tolist(),
                "sa": geo_sa.tolist(),
                "psa": geo_psa.tolist(),
            },
            "rotd50": {
                "sd": sd50_f.tolist(),
                "sa": sa50_f.tolist(),
                "sa_g": (sa50_f / GRAVITY).tolist(),
                "psa": psa50_f.tolist(),
                "psa_g": (psa50_f / GRAVITY).tolist(),
            },
            "rotd100": {
                "sd": sd100_f.tolist(),
                "sa": sa100_f.tolist(),
                "sa_g": (sa100_f / GRAVITY).tolist(),
                "psa": psa100_f.tolist(),
                "psa_g": (psa100_f / GRAVITY).tolist(),
            },
            "rotd100_angle_deg": {
                "sd": sd_ang_f.tolist(),
                "sa": sa_ang_f.tolist(),
                "psa": sd_ang_f.tolist(),
            },
            "method": req.method,
            "dt_used": dt_eff,
            "refined": factor > 1,
            "refine_factor": factor,
            "unstable_at_input_dt": unstable_full.tolist(),
            "tail_steps": tail_step_count(float(t_pos.max()), dt_eff),
        })

    return {
        "group_id": group.get("id"),
        "group_name": group.get("name"),
        "periods": periods.tolist(),
        "dampings": list(req.dampings),
        "n_angles": req.n_angles,
        "angle_step_deg": 180.0 / req.n_angles,
        "rot_definition": {
            "rotd50": "全部方位角时程峰值的中位值",
            "rotd100": "全部方位角时程峰值的最大值（上包络）",
            "angles_deg": (angles_rad * 180.0 / math.pi).tolist(),
            "guaranteed_rotd100_underestimate": 0.0087,
        },
        "members": [
            {"record_id": m.get("record_id"), "name": m.get("name")}
            for m in member_meta
        ],
        "spectra": out_per_damping,
        "baseline": req.baseline_mode,
        "ground_motion": [
            {"record_id": m.get("record_id"), "name": m.get("name"),
             "pga": pga, "pgv": pgv, "pgd": pgd, "pga_g": pga / GRAVITY}
            for m, pga, pgv, pgd in zip(
                member_meta, (pga1, pga2), (pgv1, pgv2), (pgd1, pgd2),
            )
        ],
        "alignment": group.get("alignment"),
        "units": {
            "time": "s", "sd": "m", "sa": "m/s2", "psa": "m/s2",
            "sa_g": "g", "psa_g": "g",
        },
        "calc_version": CALC_VERSION,
    }


def _ground_peaks(h1, h2, cos_a, sin_a):
    """T=0 点：把地面加速度向量按角度收缩，取逐角度时程峰值。"""

    # 复用块收缩逻辑：构造「每块就是一段原始地面加速度」的伪响应流，
    # ground=True 取绝对加速度（此处即地面加速度自身）
    n = h1.size
    peaks = np.zeros((1, cos_a.size))
    buf = np.empty((TIME_BLOCK, 1, ANGLE_BATCH))
    for start in range(0, n, TIME_BLOCK):
        b1 = h1[start:start + TIME_BLOCK][:, None]
        b2 = h2[start:start + TIME_BLOCK][:, None]
        nb = b1.shape[0]
        for js in range(0, cos_a.size, ANGLE_BATCH):
            je = min(js + ANGLE_BATCH, cos_a.size)
            view = buf[:nb, :, : je - js]
            np.multiply(b1[:, :, None], cos_a[None, None, js:je], out=view)
            view += b2[:, :, None] * sin_a[None, None, js:je]
            np.abs(view, out=view)
            np.maximum(peaks[:, js:je], np.max(view, axis=0),
                       out=peaks[:, js:je])
    # 分量自身 PGA
    pga1 = np.max(np.abs(h1))
    pga2 = np.max(np.abs(h2))
    return peaks, pga1, pga2


def rot_cache_key(group: dict, params: dict) -> str:
    """组合谱缓存键：组身份（含对齐后序列哈希）+ 全部参数 + 计算版本。

    组一旦建立不可变（成员/对齐都进了组 ID 与组行），积分参数任何一项
    变化都会改变键；计算逻辑升级升 ``CALC_VERSION``，旧键自然失配，
    因此缓存命中不可能拿到旧结果。
    """

    h = hashlib.sha256()
    h.update(group["id"].encode())
    h.update(np.ascontiguousarray(group["h1"], dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(group["h2"], dtype=np.float64).tobytes())
    h.update(repr(float(group["dt"])).encode())
    h.update(CALC_VERSION.encode())
    h.update(repr(sorted(params.items())).encode("utf-8", "replace"))
    return "rot_" + h.hexdigest()[:24]
