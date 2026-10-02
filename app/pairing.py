"""按文件头元数据给出分量配对建议（只建议，不自动建组）。

建议仅依据上传时抽取的 ``station / event / component(channel/orientation)``
文件头信息：

- 先按 ``(station, event)`` 分桶（event 缺失时只按 station，跨事件的分量
  绝不配到一起；两者都缺失的记录进 ``unclassified``，不给建议）；
- 每条记录从 ``component/channel/orientation`` 字段判断水平 / 竖向及
  方位角（N/S/E/W、度数、HNN/HNE、东/南/西/北、Z/UD/UP/竖向 等）；
- 桶内按「方位角差 ≈ 90°（mod 180）」优先成对（confidence=preferred）；
  方位角读不出来但桶内恰有两条水平分量时也给建议（possible），由调用方
  自行核对；读不出角度又超过两条的，全部进 unpaired 并写明原因；
- 桶内恰有一条竖向分量时，把它作为 ``vertical_id`` 附在每条建议上
  （竖向不参与组合，只是挂在组里保留）；多条竖向则不附、列入 unpaired。

服务**绝不**根据建议自动建组：建议必须由调用方显式确认后再 POST 建组。
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .storage import Storage

# 竖向分量的识别词（小写包含匹配）
_VERTICAL_TOKENS = (
    "vertical", "vert", "up", "down", "ud", "-z", ".z", "zcomp",
    "竖向", "竖直", "垂直", "上下",
)
# 方位角关键字 → 度（从北顺时针；模 180 后只关心水平投影方向）
_COMPASS = {
    "n": 0.0, "north": 0.0, "北": 0.0,
    "s": 180.0, "south": 180.0, "南": 180.0,
    "e": 90.0, "east": 90.0, "东": 90.0,
    "w": 270.0, "west": 270.0, "西": 270.0,
}
_DEG_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:deg|°|度)?\b")


@dataclass
class Orientation:
    axis: str           # 'h' | 'v' | 'unknown'
    angle: float | None  # 水平方位角（度，模 180），未知为 None
    source: str         # 判定依据，用于建议说明


def parse_orientation(metadata: dict) -> Orientation:
    """从一条记录的文件头元数据判断分量轴向与水平方位角。"""

    raw = " ".join(
        str(metadata.get(k, ""))
        for k in ("component", "channel", "orientation")
    ).strip().lower()
    if not raw:
        return Orientation("unknown", None, "文件头无分量方向字段")

    compact = re.sub(r"[^a-z0-9一-鿿.]+", "", raw)
    for tok in _VERTICAL_TOKENS:
        if tok.replace("-", "").replace(".", "") in compact or tok in raw.lower():
            return Orientation("v", None, f"竖向标识 {raw!r}")
    if compact.endswith("z") or re.search(r"\bz\b", raw.lower()):
        return Orientation("v", None, f"通道尾缀 Z：{raw!r}")

    # 通道码尾缀优先（HNN/HNE/HN1/HN2/BH1/BH2…），避免把 "HNE" 里的
    # 子串 "n" 错判成北向
    m_tail = re.search(r"([a-z]{1,3})([12]|n|e|w|s|z)$", compact)
    if m_tail:
        tail = m_tail.group(2)
        tail_angle = {"n": 0.0, "e": 90.0, "s": 180.0, "w": 270.0}.get(tail)
        if tail_angle is not None:
            return Orientation("h", tail_angle % 180.0, f"通道码尾缀 {raw!r}")
        if tail in ("1", "2"):
            return Orientation("h", None, f"水平通道码尾缀 {tail}（方位角未知）：{raw!r}")

    # 罗盘方位（N-S、EW 等写法）：要求独立词或结尾
    for word, deg in _COMPASS.items():
        if re.search(rf"(?:^|[^a-z]){re.escape(word)}(?:[^a-z]|$)", compact) \
                or compact.endswith(word):
            return Orientation("h", deg % 180.0, f"罗盘方位 {raw!r}")

    m = _DEG_RE.search(raw)
    if m:
        deg = float(m.group(1)) % 180.0
        return Orientation("h", deg, f"显式方位角 {m.group(1)}°")
    return Orientation("h", None, f"判定为水平但方位角未知（{raw!r}）")


def _angular_distance_mod180(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


@dataclass
class _Record:
    id: str
    name: str
    station: str | None
    event: str | None
    ori: Orientation


def suggest_pairs(storage: Storage, record_ids: list[str] | None = None) -> dict:
    """扫描 ready 记录（或限定 id 列表），产出配对建议与配不上的清单。"""

    listed = storage.list_records()
    by_id = {r["id"]: r for r in listed}
    ids = record_ids if record_ids is not None else [r["id"] for r in listed]

    records: list[_Record] = []
    missing_or_bad: list[dict] = []
    for rid in ids:
        row = by_id.get(rid)
        if row is None:
            missing_or_bad.append({"record_id": rid, "name": None,
                                   "reason": "记录不存在"})
            continue
        if row["status"] != "ready":
            missing_or_bad.append(
                {"record_id": rid, "name": row["name"],
                 "reason": f"记录不可用：{row['error'] or row['status']}"}
            )
            continue
        meta = row["metadata"]
        records.append(_Record(
            id=rid, name=row["name"],
            station=meta.get("station"), event=meta.get("event"),
            ori=parse_orientation(meta),
        ))

    buckets: dict[tuple[str, str | None], list[_Record]] = defaultdict(list)
    unclassified: list[dict] = []
    for r in records:
        if not r.station:
            unclassified.append(
                {"record_id": r.id, "name": r.name,
                 "reason": "文件头缺少 station 信息，无法确认同一台站，未参与自动建议"}
            )
            continue
        buckets[(r.station, r.event)].append(r)

    suggestions: list[dict] = []
    unpaired: list[dict] = list(unclassified)
    for (station, event), members in sorted(buckets.items()):
        horizontals = [m for m in members if m.ori.axis != "v"]
        verticals = [m for m in members if m.ori.axis == "v"]
        vert_id = verticals[0].id if len(verticals) == 1 else None
        for v in (verticals if len(verticals) != 1 else verticals[1:]):
            unpaired.append({"record_id": v.id, "name": v.name,
                             "reason": "该台站/事件存在多条竖向分量，未自动挂接"})

        used: set[str] = set()
        # 先生成全部候选对并打分：正交性偏差（°）优先，其次 id 保序确定性
        pairs = []
        for i, a in enumerate(horizontals):
            for b in horizontals[i + 1:]:
                if a.ori.angle is not None and b.ori.angle is not None:
                    dev = abs(_angular_distance_mod180(a.ori.angle, b.ori.angle)
                              - 90.0)
                    confidence = "preferred" if dev <= 15.0 else "possible"
                    reason = (f"方位角 {a.ori.angle:g}°/{b.ori.angle:g}°，"
                              f"正交偏差 {dev:g}°")
                else:
                    dev = 90.0  # 未知角度的候选排在已知角度之后
                    confidence = None
                    reason = (f"{a.ori.source}；{b.ori.source}，"
                              "无法从文件头确认是否正交，请人工核对")
                pairs.append((dev, a.id, b.id, a, b, confidence, reason))
        pairs.sort(key=lambda x: (x[0], x[1], x[2]))

        for _, _, _, a, b, confidence, reason in pairs:
            if a.id in used or b.id in used:
                continue
            # 桶内只有两条水平分量时，即使角度未知也值得建议（possible）；
            # 超过两条且角度未知时不猜，留给人工。
            if confidence is None:
                if len(horizontals) != 2:
                    continue
                confidence = "possible"
            used.add(a.id)
            used.add(b.id)
            suggestions.append({
                "station": station,
                "event": event,
                "h1_id": a.id,
                "h2_id": b.id,
                "h1_name": a.name,
                "h2_name": b.name,
                **({"vertical_id": vert_id} if vert_id else {}),
                "confidence": confidence,
                "reason": reason,
            })

        for m in horizontals:
            if m.id not in used:
                unpaired.append({
                    "record_id": m.id, "name": m.name,
                    "reason": "同一台站/事件下没有找到可与之正交配对的水平分量",
                })

    unpaired.extend(missing_or_bad)
    suggestions.sort(key=lambda s: (s["station"] or "", s["event"] or "",
                                    s["h1_id"], s["h2_id"]))
    unpaired.sort(key=lambda u: (u["record_id"],))
    return {
        "suggestions": suggestions,
        "unpaired": unpaired,
        "n_suggestions": len(suggestions),
        "n_unpaired": len(unpaired),
        "note": "以上仅为按文件头信息给出的建议，服务不会自动建组；"
                "请确认后显式调用 POST /api/groups",
    }
