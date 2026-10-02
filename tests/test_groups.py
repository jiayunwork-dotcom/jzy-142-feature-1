"""分量组：单位统一、对齐（步长/起始时刻）、配对建议、组 ID。"""

from __future__ import annotations

import numpy as np
import pytest

from app.errors import GroupError
from app.groups import (
    MAX_POINTS_ALIGNED,
    align_group_records,
    group_id,
    group_identity_key,
    parse_orientation,
    suggest_pairs,
)
from app.parsing import GRAVITY


class _Rec:
    def __init__(self, rid, acc, dt, *, t0=0.0, unit="m/s2",
                 status="ready", error=None, fmt="single_column",
                 metadata=None, name=None):
        self.id = rid
        self.name = name or f"{rid}.txt"
        self.acc = np.asarray(acc, dtype=np.float64)
        self.dt = dt
        self.fmt = fmt
        self.unit_input = unit
        self.status = status
        self.error = error
        self.metadata = metadata or {}
        if t0:
            self.metadata["time_start"] = t0


def test_units_normalized_to_si():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    g = align_group_records(
        _Rec("r1", a, 0.01, unit="g"),
        _Rec("r2", a * 2.0, 0.01, unit="m/s2"),
    )
    np.testing.assert_allclose(g.h1.acc_si, a * GRAVITY)
    np.testing.assert_allclose(g.h2.acc_si, a * 2.0)
    assert g.alignment_notes() if hasattr(g, "alignment_notes") else True


def test_same_dt_offsets_zero_pad():
    a1 = np.ones(100)
    a2 = np.full(100, 2.0)
    # 第二条晚 10 个步长开始（公共轴 0..1.09s，共 110 点）
    g = align_group_records(
        _Rec("r1", a1, 0.01, t0=0.0, fmt="two_column_time"),
        _Rec("r2", a2, 0.01, t0=0.1, fmt="two_column_time"),
    )
    assert g.npts == 110
    assert g.h2.pad_start == 10 and g.h2.pad_end == 0
    assert g.h2.zero_padded and not g.h2.resampled
    np.testing.assert_array_equal(g.h2.acc_si[:10], 0.0)
    np.testing.assert_array_equal(g.h2.acc_si[10:], 2.0)
    # 第一条前 100 点为 1，后 10 点补零
    np.testing.assert_array_equal(g.h1.acc_si[:100], 1.0)
    np.testing.assert_array_equal(g.h1.acc_si[100:], 0.0)
    assert any("补零" in n for n in g.notes)


def test_different_dt_resample_to_finer():
    # 同一斜坡、同一覆盖时长：dt=0.02 与 dt=0.01 两个版本
    t_fine = np.arange(0.0, 1.0, 0.01)
    a2 = t_fine.copy()                      # 100 点，0..0.99
    t_coarse = np.arange(0.0, 1.0, 0.02)
    a1 = t_coarse.copy()                    # 50 点，覆盖到 0.98
    g = align_group_records(_Rec("r1", a1, 0.02), _Rec("r2", a2, 0.01))
    assert g.dt == 0.01
    assert g.npts == 100
    assert g.h1.resampled and not g.h2.resampled
    # 区间内（前 99 个细网格点）线性插值精确还原；末点 0.99s 超出粗
    # 序列覆盖，按策略补零
    np.testing.assert_allclose(g.h1.acc_si[:-1], a2[:-1], atol=1e-12)
    assert g.h1.acc_si[-1] == 0.0 and g.h1.zero_padded
    assert any("重采样" in n for n in g.notes)


def test_start_offset_with_different_dt():
    a1 = np.ones(50)
    a2 = np.full(80, 3.0)
    g = align_group_records(
        _Rec("r1", a1, 0.02, t0=0.0, fmt="two_column_time"),    # 0..0.98
        _Rec("r2", a2, 0.01, t0=0.05, fmt="two_column_time"),  # 0.05..0.84
    )
    assert g.dt == 0.01
    assert g.t0 == 0.0
    # union: 0..0.98 -> 99 points
    assert g.npts == 99
    assert g.h1.resampled and g.h2.pad_start == 5
    np.testing.assert_array_equal(g.h2.acc_si[:5], 0.0)


def test_rejects_when_union_exceeds_limit():
    a1 = np.ones(MAX_POINTS_ALIGNED)
    a2 = np.ones(MAX_POINTS_ALIGNED)
    with pytest.raises(GroupError, match="20 万"):
        align_group_records(
            _Rec("r1", a1, 0.01, t0=0.0, fmt="two_column_time"),
            _Rec("r2", a2, 0.01, t0=2000.0, fmt="two_column_time"),
        )


def test_rejects_same_record_and_bad_status():
    r = _Rec("r1", np.ones(10), 0.01)
    with pytest.raises(GroupError, match="同一条记录"):
        align_group_records(r, r)
    bad = _Rec("rb", np.ones(2), 0.01, status="error", error="x")
    with pytest.raises(GroupError, match="不可用"):
        align_group_records(r, bad)


def test_vertical_attached_but_marked():
    a1, a2 = np.ones(50), np.zeros(50)
    v = np.full(50, 0.2)
    g = align_group_records(
        _Rec("r1", a1, 0.01), _Rec("r2", a2, 0.01),
        _Rec("rv", v, 0.01, metadata={"component": "Z"}),
    )
    assert g.vertical is not None
    assert g.vertical.record_id == "rv"
    np.testing.assert_array_equal(g.vertical.acc_si, 0.2)


def test_group_id_order_independent_vertical_sensitive():
    k1 = group_id(("a", "b"), None)
    k2 = group_id(("b", "a"), None)
    assert k1 == k2
    assert group_id(("a", "b"), "c") != k1
    # 内容键本身也无序
    assert group_identity_key(("a", "b"), None) == \
        group_identity_key(("b", "a"), None)


def test_orientation_parsing():
    assert parse_orientation({"component": "N"}) == (0.0, "N")
    assert parse_orientation({"component": "E-W"}) == (90.0, "E")
    assert parse_orientation({"component": "north 0 deg"})[0] == 0.0
    angle, label = parse_orientation(
        {"component": "horizontal orientation 37 deg"})
    assert angle == pytest.approx(37.0)
    assert parse_orientation({"component": "vertical Z"})[1] == "Z"
    assert parse_orientation({}) == (None, None)
    # 不能把普通 channel 编号当方位角
    assert parse_orientation({"channel": "10"}) == (None, None)


def _dict_rec(rid, station="S1", event="E1", component=None):
    meta = {"station": station, "event": event}
    if component:
        meta["component"] = component
    return {"id": rid, "name": rid, "status": "ready", "metadata": meta}


def test_suggest_orthogonal_pair_and_vertical():
    recs = [_dict_rec("a", component="N"), _dict_rec("b", component="E"),
            _dict_rec("z", component="Z"),
            _dict_rec("c", station="S2", component="N")]
    out = suggest_pairs(recs)
    assert len(out["suggestions"]) == 1
    s = out["suggestions"][0]
    assert s["confidence"] == "orthogonal"
    assert s["record_ids"] == ["a", "b"]
    unpaired_ids = {u["record_id"] for u in out["unpaired"]}
    assert unpaired_ids == {"z", "c"}


def test_suggest_angle_based_orthogonal_pair():
    recs = [_dict_rec("a", component="horizontal 10 deg"),
            _dict_rec("b", component="horizontal 100 deg")]
    out = suggest_pairs(recs)
    assert out["suggestions"][0]["confidence"] == "orthogonal"


def test_suggest_candidate_when_direction_missing():
    recs = [_dict_rec("a", component=None), _dict_rec("b", component=None)]
    out = suggest_pairs(recs)
    assert out["suggestions"][0]["confidence"] == "candidate"
    assert "人工确认" in out["suggestions"][0]["reason"]


def test_suggest_requires_station_or_event():
    recs = [{"id": "a", "name": "a", "status": "ready", "metadata": {}},
            {"id": "b", "name": "b", "status": "ready",
             "metadata": {"component": "N"}}]
    out = suggest_pairs(recs)
    assert out["suggestions"] == []
