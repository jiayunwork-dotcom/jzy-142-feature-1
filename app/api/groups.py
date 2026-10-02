"""分量组接口：配对建议、显式建组、查询；以及以组为单位的两类作业。

GET  /api/groups                         分量组列表（不含对齐序列 BLOB）
GET  /api/groups/<id>                    单个分量组详情（含对齐说明）
POST /api/groups                         显式点名两条水平分量建组
POST /api/groups/suggestions             按文件头给出配对建议（不自动建）
POST /api/jobs/rot-spectrum              组合谱作业（RotD50/RotD100）
POST /api/jobs/match-group               以组为单位的谱匹配作业

建组**只**接受调用方显式点名；suggestions 仅返回建议，必须由调用方确认后
再来 POST。
"""

from __future__ import annotations

from flask import Blueprint, current_app, request

from ..errors import SeismicError, ValidationError
from ..groups import build_group
from ..matching import make_design_spectrum
from ..pairing import suggest_pairs
from ..rotd import prepare_rot_request
from .helpers import error_body, job_id, ok, require_record_limit

bp = Blueprint("groups", __name__)


def _payload():
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        raise ValidationError("请求体必须是 JSON 对象")
    return data


def _resolve_group_ids(storage, group_ids):
    if not isinstance(group_ids, list) or not group_ids:
        raise ValidationError("group_ids 必须是非空数组")
    require_record_limit(len(group_ids))
    seen, ordered = set(), []
    for x in group_ids:
        if not isinstance(x, str) or not x:
            raise ValidationError(f"group_ids 中存在非法项：{x!r}")
        if x not in seen:
            seen.add(x)
            ordered.append(x)
    if len(ordered) != len(group_ids):
        raise ValidationError("group_ids 中存在重复项，请去重后再提交")
    missing = [g for g in ordered if storage.get_group(g) is None]
    if missing:
        raise ValidationError(f"以下分量组不存在：{', '.join(missing)}")
    not_ready = []
    for g in ordered:
        grp = storage.get_group(g)
        if grp["status"] != "ready":
            not_ready.append(f"{grp['name']}({g}): {grp.get('error') or grp['status']}")
    if not_ready:
        raise ValidationError("以下分量组不可用：" + "；".join(not_ready))
    return ordered


def _rot_params(data) -> dict:
    req = prepare_rot_request(
        periods=data.get("periods"),
        dampings=data.get("dampings", 0.05),
        method=data.get("method", "average_acceleration"),
        instability_policy=data.get("instability_policy", "refine"),
        baseline_mode=data.get("baseline_mode", "mean"),
        n_angles=data.get("n_angles", 180),
    )
    return {
        "periods": req.periods.tolist(),
        "dampings": list(req.dampings),
        "method": req.method,
        "instability_policy": req.instability_policy,
        "baseline_mode": req.baseline_mode,
        "n_angles": req.n_angles,
    }


def _create_job(job_type: str, params: dict, group_ids: list[str]):
    storage = current_app.extensions["storage"]
    scheduler = current_app.extensions["scheduler"]
    jid = job_id()
    storage.create_job(jid, job_type, params, total=len(group_ids))
    storage.add_job_items(
        jid, [(None, gid, seq) for seq, gid in enumerate(group_ids)]
    )
    scheduler.notify()
    return ok({
        "job_id": jid,
        "type": job_type,
        "status": "queued",
        "total": len(group_ids),
    }, 201)


# ---------------- 分量组 ----------------

@bp.get("/api/groups")
def list_groups():
    storage = current_app.extensions["storage"]
    return ok({"groups": storage.list_groups()})


@bp.get("/api/groups/<gid>")
def get_group(gid: str):
    storage = current_app.extensions["storage"]
    grp = storage.get_group(gid)
    if grp is None:
        return ok(error_body(f"分量组 {gid} 不存在"), 404)
    # 详情不回传 BLOB 序列（体量可能很大），只给元信息与对齐说明
    grp.pop("h1", None)
    grp.pop("h2", None)
    return ok({"group": grp})


@bp.post("/api/groups")
def create_group():
    storage = current_app.extensions["storage"]
    try:
        data = _payload()
        h1_id = data.get("h1_id")
        h2_id = data.get("h2_id")
        if not h1_id or not h2_id:
            raise ValidationError("必须显式给出两条水平分量 h1_id、h2_id")
        vertical_id = data.get("vertical_id")
        name = data.get("name")
        if name is not None and not isinstance(name, str):
            raise ValidationError("name 必须是字符串")
        trim = data.get("trim", "union")
        gid, aligned, r1, r2, vert = build_group(
            storage, str(h1_id), str(h2_id),
            vertical_id=str(vertical_id) if vertical_id else None,
            name=name, trim=trim,
        )
        created = storage.get_group(gid)
        return ok({
            "group_id": gid,
            "name": created["name"],
            "status": "ready",
            "station": aligned.station,
            "event": aligned.event,
            "h1_id": r1.id,
            "h2_id": r2.id,
            "vertical_id": vert.id if vert else None,
            "dt": aligned.dt,
            "npts": aligned.h1.size,
            "duration": aligned.dt * (aligned.h1.size - 1),
            "alignment": aligned.alignment,
        }, 201)
    except SeismicError as exc:
        return ok(error_body(str(exc)), 400)


@bp.post("/api/groups/suggestions")
def pairing_suggestions():
    storage = current_app.extensions["storage"]
    data = request.get_json(force=True, silent=True)
    record_ids = None
    if data is not None:
        if not isinstance(data, dict):
            return ok(error_body("请求体必须是 JSON 对象"), 400)
        record_ids = data.get("record_ids")
        if record_ids is not None and not isinstance(record_ids, list):
            return ok(error_body("record_ids 必须是数组"), 400)
    return ok(suggest_pairs(storage, record_ids))


# ---------------- 以组为单位的作业 ----------------

@bp.post("/api/jobs/rot-spectrum")
def create_rot_spectrum_job():
    storage = current_app.extensions["storage"]
    try:
        data = _payload()
        group_ids = _resolve_group_ids(storage, data.get("group_ids"))
        params = _rot_params(data)
    except SeismicError as exc:
        return ok(error_body(str(exc)), 400)
    return _create_job("rot_spectrum", params, group_ids)


@bp.post("/api/jobs/match-group")
def create_match_group_job():
    storage = current_app.extensions["storage"]
    try:
        data = _payload()
        group_ids = _resolve_group_ids(storage, data.get("group_ids"))
        params = _rot_params(data)

        ds = data.get("design_spectrum")
        if not isinstance(ds, dict):
            raise ValidationError("缺少 design_spectrum：{periods, values, unit}")
        design = make_design_spectrum(
            ds.get("periods"), ds.get("values"), ds.get("unit", "g")
        )
        try:
            t1 = float(data["t1"])
            t2 = float(data["t2"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("必须给出数值型匹配区间 t1、t2（秒）") from exc
        if not (0 < t1 < t2):
            raise ValidationError(f"匹配区间非法：需 0 < t1 < t2，收到 {t1}, {t2}")
        if t1 < design.periods[0] or t2 > design.periods[-1]:
            raise ValidationError(
                f"匹配区间 [{t1}, {t2}] 超出设计谱范围 "
                f"[{design.periods[0]}, {design.periods[-1]}]"
            )
        td = float(data.get("target_damping", 0.05))
        if not (0.0 <= td <= 1.0):
            raise ValidationError(f"target_damping 必须在 0 到 1 之间，收到 {td}")
        s_min = float(data.get("s_min", 0.5))
        s_max = float(data.get("s_max", 3.0))
        if not (0 < s_min <= s_max):
            raise ValidationError(
                f"缩放系数上下限非法：需 0 < s_min <= s_max，收到 {s_min}, {s_max}"
            )
        top_n = int(data.get("top_n", min(7, len(group_ids))))
        if top_n < 1:
            raise ValidationError("top_n 至少为 1")
        quantity = data.get("quantity", "psa")
        if quantity not in ("psa", "sa"):
            raise ValidationError("quantity 只支持 psa 或 sa")
        rotd_band = data.get("rotd_band", "rotd50")
        if rotd_band not in ("rotd50", "rotd100"):
            raise ValidationError("rotd_band 只支持 rotd50 或 rotd100")
        bounds_policy = data.get("bounds_policy", "clamp")
        if bounds_policy not in ("clamp", "reject"):
            raise ValidationError("bounds_policy 只支持 clamp 或 reject")
        if ds.get("unit", "g") not in ("g", "m/s2"):
            raise ValidationError("设计谱单位只支持 g 或 m/s2")

        params.update({
            "design_spectrum": {
                "periods": design.periods.tolist(),
                "values": design.values.tolist(),
                "unit": design.unit,
            },
            "t1": t1, "t2": t2, "target_damping": td,
            "s_min": s_min, "s_max": s_max, "top_n": top_n,
            "quantity": quantity, "rotd_band": rotd_band,
            "bounds_policy": bounds_policy,
        })
    except SeismicError as exc:
        return ok(error_body(str(exc)), 400)
    return _create_job("match_group", params, group_ids)
