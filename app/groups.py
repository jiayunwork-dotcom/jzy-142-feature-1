"""分量组：两条水平分量的单位统一、对齐与建组。

一组（component group）由同一台站、同一次地震的**两条水平分量**组成，
另可挂一条竖向分量（只保留引用与原始记录，不参与水平组合）。两条水平
分量必须先统一到同一单位、同一时间网格后才能做方位角投影组合。

对齐策略（本服务的明确取舍，详见 README「分量对不齐时怎么办」）
----------------------------------------------------------------
- **单位**：必须声明（m/s² 或 g），建组时一律换算到 m/s²；单位缺失的
  记录上传时即 error，不可能进入组。
- **起始时刻**：从文件头 ``start_time`` / ``t0`` 等字段读取（读不到按
  0 处理）。两段时间窗**完全不重叠**直接拒绝并给出可读原因；起始时刻
  仅差舍入量级时按同一时刻处理。
- **步长不一致**：优先取两条步长的有理小整数公步长（分母 ≤ 200，
  相对容差 1e-6，例如 0.01s 与 0.005s → 0.005s、40Hz 与 50Hz → 0.001s
  网格），此时各分量只落在网格整数点上，**不引入插值误差**；找不到
  这种公共网格时，以较细步长为网格、对较粗分量做线性插值重采样，
  加密倍数超过 100 直接拒绝（插值过粗没有物理意义）。
- **时长/点数不同**：默认 ``trim="union"``——取两条时间窗的并集，
  各自覆盖不到的网格点**补零**（等价于该方向在那段时间无地面输入），
  leading/trailing 补零点数逐分量写进 ``alignment.zero_padding``；
  显式给 ``trim="intersection"`` 时只保留重叠段、两端截齐，重叠段为空
  或不足一个步长则拒绝。
- 对齐后点数仍受单条上限 200 000 点约束，超出直接拒绝。

为什么默认补零而不是一律截齐：水平投影在每个时刻都要求两条分量同时
有值，截齐会丢掉先到/后到段里真实存在的地震动能量；而强震记录的有效
持时通常远小于台站触发时差，补零段不产生强迫响应（自由振动段由积分器
另行统一补算）。代价是：若调用方把两条触发时刻相差整段持时的记录错配
成组，补零会掩盖错配——因此起始时差超过较短记录持时的一半会在
``alignment.warnings`` 里明确告警，完全不重叠直接拒绝。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from .errors import ValidationError
from .parsing import GRAVITY
from .storage import RecordRow, Storage

MAX_GROUP_POINTS = 200_000
MAX_STEPS_PER_INTERVAL = 100

# 起始时刻在文件头里可能出现的写法（小写匹配，见 parsing._METADATA_KEYMAP）
_START_TIME_KEYS = ("start_time", "t0", "start", "origin_time")


def _start_time(metadata: dict) -> float:
    """从文件头元数据读取起始时刻（秒），读不到按 0 处理。"""

    for key in _START_TIME_KEYS:
        raw = metadata.get(key)
        if raw is None:
            continue
        m = re.search(r"[-+0-9.eE]+", str(raw))
        if m:
            try:
                return float(m.group(0))
            except ValueError:  # pragma: no cover - 正则已保证可转
                continue
    return 0.0


@dataclass
class AlignedGroup:
    """对齐结果：两条序列同为 m/s²、同一等步长网格。"""

    h1: np.ndarray
    h2: np.ndarray
    dt: float
    alignment: dict
    station: str | None
    event: str | None


def _common_grid(dt1: float, dt2: float) -> tuple[float, int, int, str]:
    """求公共网格与两条分量各自的采样倍数。

    以**较粗步长为基准**寻找小整数细分（而不是以较细步长为基准求公倍）：
    找 (p,q) 使 ``p·dt1 ≈ q·dt2``，公共步长 ``dt0 = dt1/p = dt2/q``，
    粗者在其每个原始间隔内被细分少量整数倍，细者步长不变或同样细分。
    这样 0.01s/0.02s → (p,q)=(1,2)、dt0=0.01；40Hz/50Hz
    （dt=0.025/0.02）→ (p,q)=(5,4)、dt0=0.005，双方都只落在整数网格点
    上，**不插值**。

    返回 ``(dt0, m1, m2, mode)``，``m1/m2`` 为每条分量「每个原始间隔在
    公共网格上的步数」；``mode='lattice'`` 表示严格有理网格，
    ``'interp'`` 表示找不到小整数网格、直接以较细步长为网格、粗者做
    线性插值（``m`` 即加密倍数）。
    """

    coarse, fine = max(dt1, dt2), min(dt1, dt2)
    ratio = coarse / fine
    best = None
    for p in range(1, 201):
        q = int(round(p / ratio))
        if q < 1:
            continue
        cand = abs(p / q - ratio) / ratio
        if best is None or cand < best[0]:
            best = (cand, p, q)
    if best is not None and best[0] <= 1e-6:
        _, p, q = best
        dt0 = coarse / p  # 同时 ≈ fine/q
        m_coarse, m_fine = p, q
        if max(m_coarse, m_fine) > MAX_STEPS_PER_INTERVAL:
            raise ValidationError(
                f"两条分量步长 {dt1:g}s/{dt2:g}s 的有理公共网格需要把间隔细分 "
                f"{max(m_coarse, m_fine)} 倍（超过 {MAX_STEPS_PER_INTERVAL} 倍"
                "上限）：请确认两条记录是否同源，或先在外部重采样后重新上传"
            )
        if dt1 >= dt2:
            return dt0, m_coarse, m_fine, "lattice"
        return dt0, m_fine, m_coarse, "lattice"

    f = math_ceil_safe(coarse / fine)
    if f > MAX_STEPS_PER_INTERVAL:
        raise ValidationError(
            f"两条分量步长 {dt1:g}s/{dt2:g}s 找不到分母 ≤200 的有理网格，"
            f"且按较细步长重采样需要 {f} 倍加密，超过 "
            f"{MAX_STEPS_PER_INTERVAL} 倍上限：请确认两条记录是否同源，"
            "或先在外部重采样后重新上传"
        )
    if dt1 >= dt2:
        return fine, f, 1, "interp"
    return fine, 1, f, "interp"


def math_ceil_safe(x: float) -> int:
    return int(np.ceil(x - 1e-9))


def align_records(
    r1: RecordRow,
    r2: RecordRow,
    *,
    trim: str = "union",
) -> AlignedGroup:
    """把两条已入库的 ready 水平记录对齐到公共网格（m/s²）。"""

    if trim not in ("union", "intersection"):
        raise ValidationError(
            f"对齐方式 trim 只支持 union/intersection，收到 {trim!r}"
        )
    if r1.id == r2.id:
        raise ValidationError("分量组的两条水平分量必须是不同记录，不能引用同一条")
    a1 = r1.acc * (GRAVITY if r1.unit_input == "g" else 1.0)
    a2 = r2.acc * (GRAVITY if r2.unit_input == "g" else 1.0)

    t01, t02 = _start_time(r1.metadata), _start_time(r2.metadata)
    end1, end2 = t01 + r1.duration, t02 + r2.duration

    # 时间窗完全不重叠 → 无法组成一对
    ov_start, ov_end = max(t01, t02), min(end1, end2)
    shorter = min(r1.duration, r2.duration)
    if ov_end - ov_start < min(r1.dt, r2.dt):
        raise ValidationError(
            f"两条分量的时间窗不重叠：{r1.name} 为 "
            f"[{t01:g}, {end1:g}]s，{r2.name} 为 [{t02:g}, {end2:g}]s，"
            "无法作为同一台站同一次地震的水平分量对，请核对起始时刻或记录选择"
        )

    dt0, p1, p2, mode = _common_grid(r1.dt, r2.dt)

    # 用整数网格下标表达时间，避免浮点累积误差
    base = min(t01, t02)

    def grid_index(t: float) -> int:
        idx = int(round((t - base) / dt0))
        if abs((t - base) - idx * dt0) > max(1e-9, 1e-6 * dt0):
            raise ValidationError(
                f"起始时刻 {t:g}s 对不到公共步长 {dt0:g}s 的整数网格上"
                "（残差超过 1e-6 相对容差），请检查文件头起始时间或先在外部对齐"
            )
        return idx

    i01, i02 = grid_index(t01), grid_index(t02)
    n1, n2 = a1.size, a2.size
    ie1 = i01 + (n1 - 1) * p1
    ie2 = i02 + (n2 - 1) * p2

    warnings_list: list[str] = []
    if abs(t01 - t02) > 0.5 * shorter:
        warnings_list.append(
            f"两条分量起始时刻相差 {abs(t01 - t02):g}s，超过较短记录持时 "
            f"{shorter:g}s 的一半：已按并集补零对齐，但请确认不是错配的两条记录"
        )

    if trim == "union":
        lo, hi = min(i01, i02), max(ie1, ie2)
    else:
        # 重叠段在公共网格上的整数下标范围
        lo = max(i01, i02)
        hi = min(ie1, ie2)
        if hi - lo < 1:
            raise ValidationError(
                "trim=intersection 时两条分量没有一个步长以上的重叠段，无法截齐"
            )
    npts = hi - lo + 1
    if npts > MAX_GROUP_POINTS:
        raise ValidationError(
            f"对齐后共 {npts} 点（dt={dt0:g}s），超过单组最多 "
            f"{MAX_GROUP_POINTS} 点的限制：请截短记录或改用更粗的公共步长"
        )

    grid_idx = np.arange(lo, hi + 1, dtype=np.float64)

    def place(acc, i0, n, p):
        # 分量自身样本在公共网格中的下标；网格点超出自身范围时补零
        x = np.full(npts, 0.0, dtype=np.float64)
        own = grid_idx - i0
        inside = (own >= 0) & (own <= (n - 1) * p)
        xi = own[inside] / p  # 分量原始样本序号坐标
        # lattice 模式下 xi 全为整数，np.interp 取端点值不产生插值误差；
        # interp 模式下非整数坐标做线性插值
        x[inside] = np.interp(xi, np.arange(n, dtype=np.float64), acc)
        return x

    h1 = place(a1, i01, n1, p1)
    h2 = place(a2, i02, n2, p2)

    def pad_counts(i0, n, p):
        # 分量自身覆盖下标 [i0, i0+(n-1)p]；网格 [lo,hi] 内该范围之外、
        # 需要补零的点数（落在网格外的成员样本是被截掉，不算补零）
        ie = i0 + (n - 1) * p
        return {"leading": int(max(0, i0 - lo)),
                "trailing": int(max(0, hi - ie))}

    alignment = {
        "strategy": "common_grid",
        "trim": trim,
        "dt": float(dt0),
        "npts": int(npts),
        "time_start": float(base + lo * dt0),
        "time_end": float(base + hi * dt0),
        "grid_mode": mode,
        "members": [
            {
                "record_id": r1.id,
                "name": r1.name,
                "input_dt": float(r1.dt),
                "input_npts": int(n1),
                "start_time": float(t01),
                "steps_per_interval": int(p1),
                "resampled": bool(mode == "interp" and p1 > 1),
                "zero_padding": pad_counts(i01, n1, p1),
            },
            {
                "record_id": r2.id,
                "name": r2.name,
                "input_dt": float(r2.dt),
                "input_npts": int(n2),
                "start_time": float(t02),
                "steps_per_interval": int(p2),
                "resampled": bool(mode == "interp" and p2 > 1),
                "zero_padding": pad_counts(i02, n2, p2),
            },
        ],
        "warnings": warnings_list,
    }
    station = r1.metadata.get("station") or r2.metadata.get("station")
    event = r1.metadata.get("event") or r2.metadata.get("event")
    return AlignedGroup(h1=h1, h2=h2, dt=float(dt0), alignment=alignment,
                        station=station, event=event)


def group_identity(
    r1_id: str, r2_id: str,
    h1: np.ndarray, h2: np.ndarray, dt: float,
    vertical_id: str | None,
) -> str:
    """组 ID 内容寻址：成员（排序后）+ 对齐后字节 + 竖向引用。

    两条水平分量交换顺序得到同一 ID（组合谱本身与顺序无关）。
    """

    a, b = sorted((r1_id, r2_id))
    ha, hb = (h1, h2) if r1_id == a else (h2, h1)
    h = hashlib.sha256()
    h.update(a.encode())
    h.update(b.encode())
    h.update(np.ascontiguousarray(ha, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(hb, dtype=np.float64).tobytes())
    h.update(repr(float(dt)).encode())
    h.update((vertical_id or "").encode())
    return "grp_" + h.hexdigest()[:16]


def build_group(
    storage: Storage,
    h1_id: str,
    h2_id: str,
    *,
    vertical_id: str | None = None,
    name: str | None = None,
    trim: str = "union",
) -> tuple[str, AlignedGroup, RecordRow, RecordRow, RecordRow | None]:
    """校验、对齐并落库一个分量组，返回 (group_id, aligned, r1, r2, vert)。"""

    r1 = _require_ready(storage, h1_id)
    r2 = _require_ready(storage, h2_id)
    vert = _require_ready(storage, vertical_id) if vertical_id else None
    aligned = align_records(r1, r2, trim=trim)

    gid = group_identity(h1_id, h2_id, aligned.h1, aligned.h2, aligned.dt,
                         vertical_id)
    if storage.get_group(gid) is None:
        storage.create_group(
            gid,
            name=name or f"{r1.name}+{r2.name}",
            station=aligned.station,
            event=aligned.event,
            h1_id=h1_id,
            h2_id=h2_id,
            vertical_id=vertical_id,
            dt=aligned.dt,
            alignment=aligned.alignment,
            h1=aligned.h1,
            h2=aligned.h2,
        )
    return gid, aligned, r1, r2, vert


def _require_ready(storage: Storage, record_id: str) -> RecordRow:
    rec = storage.get_record(record_id)
    if rec is None:
        raise ValidationError(f"记录 {record_id} 不存在，可能已被删除")
    if rec.status != "ready" or rec.acc.size == 0:
        raise ValidationError(
            f"记录 {rec.name}({record_id}) 不可用：{rec.error or rec.status}"
        )
    if rec.unit_input not in ("m/s2", "g"):
        raise ValidationError(
            f"记录 {rec.name}({record_id}) 缺少已确认的加速度单位，无法统一单位"
        )
    return rec
