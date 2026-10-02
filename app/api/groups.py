"""分量组建组、查询与配对建议接口。

POST /api/groups/suggestions  按文件头台站/事件/分量方向给出配对建议
                              （只读，绝不自动建组）
POST /api/groups              显式点名两条水平分量（可附竖向）建组
GET  /api/groups              组列表
GET  /api/groups/<id>         组详情（含实际对齐处理说明）
"""

from __future__ import annotations

from flask import Blueprint, current_app, request

from ..errors import GroupError, SeismicError, ValidationError
from ..groups import (
    align_group_records,
    component_dict,
    group_id as make_group_id,
    group_identity_key,
    suggest_pairs,
)
from .helpers import error_body, ok, require_record_limit

bp = Blueprint("groups", __name__)


def _payload():
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        raise ValidationError("请求体必须是 JSON 对象")
    return data


def _get_ready(storage, rid, tag):
    rec = storage.get_record(rid)
    if rec is None:
        raise ValidationError(f"{tag}记录 {rid} 不存在")
    if rec.status != "ready":
        raise ValidationError(
            f"{tag}记录 {rec.name}({rid}) 不可用：{rec.error or rec.status}"
        )
    return rec


@bp.post("/api/groups/suggestions")
def suggestions():
    storage = current_app.extensions["storage"]
    try:
        data = request.get_json(force=True, silent=True)
        ids = None
        if data is not None:
            if not isinstance(data, dict):
                raise ValidationError("请求体必须是 JSON 对象或为空")
            ids = data.get("record_ids")
            if ids is not None:
                if not isinstance(ids, list) or not ids:
                    raise ValidationError("record_ids 必须是非空数组")
                require_record_limit(len(ids))
        records = storage.list_records()
        if ids is not None:
            wanted = set(ids)
            missing = [x for x in ids if x not in {r["id"] for r in records}]
            if missing:
                raise ValidationError(f"以下记录不存在：{', '.join(missing)}")
            records = [r for r in records if r["id"] in wanted]
        result = suggest_pairs(records)
    except SeismicError as exc:
        return ok(error_body(str(exc)), 400)
    return ok({
        "suggested": len(result["suggestions"]),
        "suggestions": result["suggestions"],
        "unpaired": result["unpaired"],
        "note": "以上仅为配对建议，服务不会自动建组；确认后请调用 POST /api/groups",
    })


@bp.post("/api/groups")
def create_group():
    storage = current_app.extensions["storage"]
    try:
        data = _payload()
        h1id = data.get("horizontal1") or data.get("h1")
        h2id = data.get("horizontal2") or data.get("h2")
        if not isinstance(h1id, str) or not isinstance(h2id, str):
            raise ValidationError(
                "必须给出两条水平分量记录 id：horizontal1、horizontal2"
            )
        vid = data.get("vertical_record_id") or data.get("vertical")
        if vid is not None and not isinstance(vid, str):
            raise ValidationError("vertical_record_id 必须是字符串或省略")
        if h1id == h2id:
            raise ValidationError("两条水平分量不能是同一条记录")
        h1 = _get_ready(storage, h1id, "第一水平分量")
        h2 = _get_ready(storage, h2id, "第二水平分量")
        hv = _get_ready(storage, vid, "竖向分量") if vid else None

        aligned = align_group_records(h1, h2, hv)
        gid = make_group_id((h1id, h2id), vid)
        key = group_identity_key((h1id, h2id), vid)
        name = str(data.get("name") or f"{h1.name}+{h2.name}")
        alignment = {
            "policy": "union_resample_zeropad",
            "policy_description": (
                "单位统一为 m/s²；起始时刻取并集，缺口补零；步长不一致时"
                "线性插值重采样到较细步长；不截齐，对齐后点数超 20 万则拒绝"
            ),
            "dt": aligned.dt,
            "t0": aligned.t0,
            "npts": aligned.npts,
            "duration": aligned.duration,
            "unit_normalized": "m/s2",
            "components": [component_dict(aligned.h1),
                           component_dict(aligned.h2)]
            + ([component_dict(aligned.vertical)] if aligned.vertical else []),
            "notes": aligned.notes,
        }
        storage.upsert_group(
            gid,
            horizontal1=h1id, horizontal2=h2id, vertical=vid,
            content_key=key, name=name,
            dt=aligned.dt, t0=aligned.t0, npts=aligned.npts,
            duration=aligned.duration, alignment=alignment,
        )
    except GroupError as exc:
        return ok(error_body(str(exc)), 400)
    except SeismicError as exc:
        return ok(error_body(str(exc)), 400)

    existing = storage.get_group(gid)
    return ok({
        "group_id": gid,
        "status": "ready",
        "name": name,
        "horizontal1": h1id,
        "horizontal2": h2id,
        "vertical": vid,
        "dt": aligned.dt,
        "t0": aligned.t0,
        "npts": aligned.npts,
        "duration": aligned.duration,
        "alignment": alignment,
        "created_at": existing["created_at"] if existing else None,
    }, 201)


@bp.get("/api/groups")
def list_groups():
    storage = current_app.extensions["storage"]
    return ok({"groups": storage.list_groups()})


@bp.get("/api/groups/<gid>")
def get_group(gid: str):
    storage = current_app.extensions["storage"]
    g = storage.get_group(gid)
    if g is None:
        return ok(error_body(f"分量组 {gid} 不存在"), 404)
    return ok({"group": g})
