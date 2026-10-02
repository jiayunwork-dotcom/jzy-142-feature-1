"""水平分量组：建组、对齐、配对建议。

一个分量组由**同一台站同一次地震的两条水平分量**构成；竖向分量可以
挂在组里保留，但不参与任何水平组合。

建组时完成三件事：

1. **单位统一**：两条水平分量（及竖向分量）都换算到 m/s²（g 按
   :data:`app.parsing.GRAVITY` 换算）；
2. **起始时刻对齐**：时间–加速度两列格式会保留首点时间
   （``metadata.time_start``），单列/序号格式起始时刻按 0 处理；
   两条分量时间轴取**并集**，各自不在自己覆盖区间内的采样一律补零；
3. **步长统一**：步长不一致时，**线性插值重采样到较细的步长**
   （若细步长不是粗步长的整数分点，粗步长记录的时间点会被线性插值，
   误差由记录本身的带限性质决定，线性加速度法的输入假设也是区间内
   线性，与该处理一致）。

策略取舍（详见 README §11）：不做「截齐」——截掉较长记录的真实地震
动会系统性改变谱值；只在对齐后总点数超过 20 万上限时拒绝建组并给出
可读原因。实际采用的每一步操作都写入组信息的 ``alignment`` 字段。

配对建议只根据上传时文件头里抽取的 ``station / event / component``
等信息给出候选，**绝不自动建组**，由调用方显式确认。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

from .errors import GroupError
from .parsing import GRAVITY

MAX_POINTS_ALIGNED = 200_000

# 判定两个方位角「正交」的容差（度）；文件头写 91° 之类仍认作水平对
_ORTHOGONAL_TOL = 12.0

# 分量方向关键字 -> 语义
_DIR_TOKENS = {
    "n": "N", "north": "N", "ns": "NS", "南北": "NS",
    "e": "E", "east": "E", "ew": "EW", "we": "EW", "东西": "EW",
    "z": "Z", "up": "Z", "down": "Z", "vertical": "Z", "ud": "Z",
    "vert": "Z", "竖向": "Z", "垂直": "Z",
    "x": "X", "y": "Y",
}


def record_time_start(rec) -> float:
    """记录起始时刻（秒）：两列时间格式取首点时间，其余默认 0。"""

    if rec.fmt == "two_column_time":
        try:
            return float(rec.metadata.get("time_start", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


@dataclass
class AlignedComponent:
    """一条分量在组公共时间轴上的对齐结果。"""

    record_id: str
    name: str
    acc_si: np.ndarray                 # 已换算到 m/s² 的对齐加速度
    dt: float
    t0: float                          # 公共轴起始时刻
    npts: int
    unit_input: str | None
    orientation: float | None = None   # 文件头解析出的方位角（度，0=北，顺时针）
    direction_label: str | None = None
    resampled: bool = False            # 步长不一致，做了线性插值
    zero_padded: bool = False          # 起止缺口补了零
    pad_start: int = 0
    pad_end: int = 0


@dataclass
class AlignedGroup:
    """建组对齐产物：两条水平分量共享同一条公共时间轴。"""

    dt: float
    t0: float
    npts: int
    h1: AlignedComponent
    h2: AlignedComponent
    vertical: AlignedComponent | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.dt * (self.npts - 1)


def _unit_factor(rec) -> float:
    return GRAVITY if rec.unit_input == "g" else 1.0


def _resample_to(acc: np.ndarray, dt_old: float, t0_old: float,
                 grid_t: np.ndarray) -> tuple[np.ndarray, bool, int, int]:
    """把记录重采样到公共时间轴 ``grid_t``：区间内线性插值，区间外补零。

    返回 (对齐序列, 是否补零, 前补零点数, 后补零点数)。
    """

    start = t0_old
    end = t0_old + dt_old * (acc.size - 1)
    out = np.zeros(grid_t.size, dtype=np.float64)
    inside = (grid_t >= start) & (grid_t <= end)
    # np.interp 要求旧时间点严格递增
    old_t = t0_old + dt_old * np.arange(acc.size, dtype=np.float64)
    out[inside] = np.interp(grid_t[inside], old_t, acc)
    pad_start = int(np.count_nonzero(grid_t < start))
    pad_end = int(np.count_nonzero(grid_t > end))
    return out, bool(pad_start or pad_end), pad_start, pad_end


def align_group_records(h1, h2, vertical=None) -> AlignedGroup:
    """把两条水平记录（可选竖向）对齐到公共时间轴，单位统一为 m/s²。

    任何一步失败抛 :class:`~app.errors.GroupError`，消息面向调用方。
    """

    for tag, rec in (("第一水平分量", h1), ("第二水平分量", h2)):
        if rec is None or rec.acc is None or rec.acc.size < 2:
            raise GroupError(f"{tag}记录不可用（不存在或数据为空），无法建组")
        if rec.status != "ready":
            raise GroupError(
                f"{tag}记录 {rec.name}({rec.id}) 不可用，状态为 {rec.status}："
                f"{rec.error or '无错误信息'}"
            )
        if not np.all(np.isfinite(rec.acc)):
            raise GroupError(f"{tag}记录 {rec.name}({rec.id}) 含 NaN/Inf")
    if h1.id == h2.id:
        raise GroupError("两条水平分量不能是同一条记录（record_id 相同）")

    dt1, dt2 = float(h1.dt), float(h2.dt)
    if not (dt1 > 0 and dt2 > 0):
        raise GroupError(f"记录步长非法：{dt1:g} / {dt2:g} s")
    t01, t02 = record_time_start(h1), record_time_start(h2)

    # 公共步长取较细者；步长一致（相对容差 1e-9）时直接用之
    if abs(dt1 - dt2) <= 1e-9 * max(dt1, dt2):
        dt_c = min(dt1, dt2)
    else:
        dt_c = min(dt1, dt2)
    t_start = min(t01, t02)

    end1 = t01 + dt1 * (h1.acc.size - 1)
    end2 = t02 + dt2 * (h2.acc.size - 1)
    t_end = max(end1, end2)
    npts = int(round((t_end - t_start) / dt_c)) + 1
    if npts > MAX_POINTS_ALIGNED:
        raise GroupError(
            f"两条分量按起始时刻 {t_start:g}s 与较细步长 {dt_c:g}s 取并集后共 "
            f"{npts} 点，超过单条 {MAX_POINTS_ALIGNED} 点（20 万点）上限；"
            "请截短记录或改用步长更粗的分量后再建组"
            "（本服务不自动截齐，以免丢失真实地震动）"
        )
    grid = t_start + dt_c * np.arange(npts, dtype=np.float64)

    notes = []
    if abs(dt1 - dt2) > 1e-9 * max(dt1, dt2):
        notes.append(
            f"两条分量步长不一致（{dt1:g}s / {dt2:g}s），已线性插值重采样到"
            f"较细步长 {dt_c:g}s"
        )

    def _build(rec, tag: str) -> AlignedComponent:
        acc_si = np.asarray(rec.acc, dtype=np.float64) * _unit_factor(rec)
        if abs(float(rec.dt) - dt_c) <= 1e-9 * max(float(rec.dt), dt_c):
            # 同步长：平移后补零即可，不插值
            shift = round((record_time_start(rec) - t_start) / dt_c)
            out = np.zeros(npts, dtype=np.float64)
            seg = acc_si[: min(acc_si.size, npts - shift)]
            out[shift: shift + seg.size] = seg
            pad_start, pad_end = shift, npts - shift - seg.size
            padded = bool(pad_start or pad_end)
            if padded:
                notes.append(
                    f"{tag} {rec.name} 起止时刻与公共轴不对齐，"
                    f"已前补零 {pad_start} 点、后补零 {pad_end} 点"
                )
            resampled = False
        else:
            out, padded, pad_start, pad_end = _resample_to(
                acc_si, float(rec.dt), record_time_start(rec), grid
            )
            resampled = True
            if padded:
                notes.append(
                    f"{tag} {rec.name} 在重采样时按起始时刻对齐，"
                    f"区间外前补零 {pad_start} 点、后补零 {pad_end} 点"
                )
        azi, label = parse_orientation(rec.metadata)
        return AlignedComponent(
            record_id=rec.id, name=rec.name, acc_si=out, dt=dt_c,
            t0=t_start, npts=npts, unit_input=rec.unit_input,
            orientation=azi, direction_label=label,
            resampled=resampled, zero_padded=padded,
            pad_start=pad_start, pad_end=pad_end,
        )

    a1 = _build(h1, "第一水平分量")
    a2 = _build(h2, "第二水平分量")

    vcomp = None
    if vertical is not None:
        if vertical.id in (h1.id, h2.id):
            raise GroupError("竖向分量不能与水平分量是同一条记录")
        if vertical.status != "ready" or vertical.acc.size < 2:
            raise GroupError(
                f"竖向记录 {vertical.name}({vertical.id}) 不可用："
                f"{vertical.error or vertical.status}"
            )
        vcomp = _build(vertical, "竖向分量")
        if vcomp.orientation is not None and vcomp.direction_label != "Z":
            notes.append(
                f"竖向分量 {vertical.name} 文件头方位角为 {vcomp.orientation:g}°，"
                "调用方仍指定其为竖向，已按竖向保留（不参与水平组合）"
            )

    return AlignedGroup(
        dt=dt_c, t0=t_start, npts=npts, h1=a1, h2=a2,
        vertical=vcomp, notes=notes,
    )


def component_dict(c: AlignedComponent) -> dict:
    return {
        "record_id": c.record_id,
        "name": c.name,
        "unit_input": c.unit_input,
        "orientation": c.orientation,
        "direction_label": c.direction_label,
        "resampled": c.resampled,
        "zero_padded": c.zero_padded,
        "pad_start": c.pad_start,
        "pad_end": c.pad_end,
    }


def group_identity_key(horizontal_ids: tuple[str, str],
                       vertical_id: str | None) -> str:
    """组的内容键：两条水平分量无序、竖向分量有序参与。

    交换两条水平分量得到**同一个组**（组合谱对分量顺序不敏感）；
    竖向分量不同则视为不同组。
    """

    h = sorted(horizontal_ids)
    raw = "H:" + "|".join(h) + (";V:" + vertical_id if vertical_id else ";V:")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def group_id(horizontal_ids: tuple[str, str],
             vertical_id: str | None = None) -> str:
    return "grp_" + group_identity_key(horizontal_ids, vertical_id)[:16]


# ---------------------------------------------------------------- 配对建议

def _norm_key(v) -> str | None:
    if v is None:
        return None
    s = re.sub(r"\s+", " ", str(v).strip().lower())
    return s or None


def parse_orientation(metadata: dict) -> tuple[float | None, str | None]:
    """从文件头元数据解析分量方位。

    返回 (方位角度数 0–360（0=北，顺时针）, 语义标签 N/E/Z/X/Y/None)。
    识别不了时返回 (None, None)——配对建议此时不假设其方向。
    """

    if not isinstance(metadata, dict):
        return None, None
    comp = metadata.get("component") or metadata.get("channel") or ""
    chan = str(metadata.get("channel") or "")
    text = f"{comp} {chan}".strip().lower()
    if not text:
        return None, None

    # 1) 显式角度：30 / 30.0 / 30deg / HN? 中的数字（仅当看起来像方位角）
    m = re.search(r"(?<![A-Za-z0-9.])([0-9]{1,3}(?:\.[0-9]+)?)\s*(?:deg|°)?", text)
    angle = None
    if m:
        val = float(m.group(1))
        # 0–360 且不是年份之类；要求文本里出现 component/orientation/deg 等线索，
        # 避免把 "channel 10" 误当方位角
        if 0.0 <= val <= 360.0 and re.search(
            r"(comp|orient|deg|°|分量|方向|az)", text
        ):
            angle = val

    # 2) 方向关键字：按切出的 token 精确比对，避免 "component" 里的 n
    label = None
    tokens = set(re.findall(r"[A-Za-z一-鿿]+", text))
    for token, canon in _DIR_TOKENS.items():
        if token in tokens:
            if canon == "Z":
                return (angle if angle is not None else 0.0), "Z"
            if label is None:
                label = canon
    if label in ("NS", "N"):
        label = "N"
    if label in ("EW", "E"):
        label = "E"
    if angle is None:
        angle = {"N": 0.0, "E": 90.0}.get(label)
    return angle, label


def _is_vertical(rec) -> bool:
    meta = rec.get("metadata") if isinstance(rec, dict) else rec.metadata
    angle, label = parse_orientation(meta)
    if label == "Z":
        return True
    return False


def _angular_delta(a: float, b: float) -> float:
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def suggest_pairs(records: list[dict]) -> dict:
    """根据文件头元数据给出水平分量配对建议（不建组、不落库）。

    ``records`` 为 :meth:`Storage.list_records` 风格的 dict 列表（需含
    metadata 且 status=ready）。返回 ``{"suggestions": [...],
    "unpaired": [...]}``。同一 (station, event) 簇内：

    - 两条水平分量方位角差在 90°±12° 内：``confidence="orthogonal"``；
    - 其余水平两两组合：``confidence="candidate"``，由调用方自行确认；
    - 无法判定方向（角度/标签都缺）时仍给候选，confidence=candidate 且
      在 reason 里写明缺少方向信息。
    """

    clusters: dict[tuple[str | None, str | None], list[dict]] = {}
    for r in records:
        if r.get("status") != "ready":
            continue
        meta = r.get("metadata") or {}
        key = (_norm_key(meta.get("station")), _norm_key(meta.get("event")))
        if key == (None, None):
            # 既无台站也无事件，无法做配对建议
            continue
        clusters.setdefault(key, []).append(r)

    suggestions: list[dict] = []
    unpaired: list[dict] = []
    for (station, event), members in sorted(
        clusters.items(), key=lambda kv: tuple(x or "" for x in kv[0])
    ):
        horiz, verts = [], []
        for r in members:
            (verts if _is_vertical(r) else horiz).append(r)
        horiz.sort(key=lambda r: r["id"])
        if len(horiz) < 2:
            for r in horiz:
                unpaired.append({
                    "record_id": r["id"], "name": r["name"],
                    "station": station, "event": event,
                    "reason": "同台站同事件下没有找到另一条水平分量",
                })
            continue
        # 先尝试正交配对（每条分量只进一个最优正交对）
        used: set[str] = set()
        angles = {r["id"]: parse_orientation(r.get("metadata") or {})
                  for r in horiz}
        for i, r in enumerate(horiz):
            if r["id"] in used:
                continue
            best, best_delta = None, None
            for q in horiz[i + 1:]:
                if q["id"] in used:
                    continue
                a1, l1 = angles[r["id"]]
                a2, l2 = angles[q["id"]]
                if a1 is not None and a2 is not None:
                    d = _angular_delta(a1, a2)
                    if abs(d - 90.0) <= _ORTHOGONAL_TOL and (
                        best_delta is None or abs(d - 90.0) < abs(best_delta - 90.0)
                    ):
                        best, best_delta = q, d
                elif {l1, l2} == {"N", "E"} or {l1, l2} == {"X", "Y"}:
                    best, best_delta = q, 90.0
            if best is not None:
                used.add(r["id"])
                used.add(best["id"])
                suggestions.append(_suggestion_entry(
                    station, event, r, best, angles,
                    confidence="orthogonal",
                    reason=("文件头方位角近似正交（差 "
                            f"{best_delta:.0f}°），建议组成水平分量对"),
                ))
        # 剩余水平分量给候选对（数据来源通常是三分量以上的台阵或方向缺失）
        rest = [r for r in horiz if r["id"] not in used]
        if len(rest) == 2:
            a1, _ = angles[rest[0]["id"]]
            a2, _ = angles[rest[1]["id"]]
            reason = "同台站同事件的两条水平分量，但无法确认是否正交，请人工确认"
            if a1 is None or a2 is None:
                reason = ("同台站同事件的两条水平分量，文件头缺少可解析的方位"
                          "信息，无法确认是否正交，请人工确认")
            suggestions.append(_suggestion_entry(
                station, event, rest[0], rest[1], angles,
                confidence="candidate", reason=reason,
            ))
        elif len(rest) > 2:
            for r in unpaired:  # 不应该出现；防御
                pass
            for r in rest:
                unpaired.append({
                    "record_id": r["id"], "name": r["name"],
                    "station": station, "event": event,
                    "reason": f"同台站同事件下有 {len(rest)} 条水平分量，"
                              "无法自动确定唯一配对",
                })
        for v in verts:
            unpaired.append({
                "record_id": v["id"], "name": v["name"],
                "station": station, "event": event,
                "reason": "判定为竖向分量，可在建组时挂到 vertical_record_id",
            })

    suggestions.sort(key=lambda s: (s["station"] or "", s["event"] or "",
                                    s["record_ids"]))
    unpaired.sort(key=lambda u: (u["station"] or "", u["event"] or "",
                                 u["record_id"]))
    return {"suggestions": suggestions, "unpaired": unpaired}


def _suggestion_entry(station, event, r1, r2, angles, *, confidence, reason):
    def _ori(r):
        a, label = angles[r["id"]]
        return {"record_id": r["id"], "name": r.get("name"),
                "orientation": a, "direction_label": label}

    return {
        "station": station,
        "event": event,
        "confidence": confidence,
        "reason": reason,
        "record_ids": sorted([r1["id"], r2["id"]]),
        "components": [_ori(r1), _ori(r2)],
    }
