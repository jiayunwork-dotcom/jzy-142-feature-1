"""分量组作业的执行逻辑：组合谱作业与组匹配作业。

沿用单条作业的机制（后台串行、可查进度/取消、单组失败只标记该组）：

- ``rot_spectrum``：逐组计算 RotD50/RotD100 组合谱。每个组、每套参数的
  结果以内容寻址键（``rot_cache_key``）写入 ``rot_spectra`` 缓存表，
  重复提交直接命中、逐位一致；组成员或任何积分参数变化都会得到不同的键，
  绝无可能读到旧结果（见 README「组合谱的缓存复用」）。
- ``match_group``：逐组先（按缓存）取得目标阻尼的组合谱，再用与单条匹配
  同一套口径（:func:`app.matching.match_group_batch`）按 RotD50/RotD100
  对设计谱，同一组两条分量共用一个缩放系数。
"""

from __future__ import annotations

from .errors import SeismicError
from .matching import make_design_spectrum, match_group_batch
from .parsing import GRAVITY
from .rotd import compute_rot_spectrum, prepare_rot_request, rot_cache_key
from .storage import Storage


def _load_ready_group(storage: Storage, group_id: str | None):
    if not group_id:
        return None, "缺少分量组 id"
    grp = storage.get_group(group_id)
    if grp is None:
        return None, f"分量组 {group_id} 不存在，可能已被删除"
    if grp["status"] != "ready" or grp.get("h1") is None or grp["h1"].size == 0:
        return None, f"分量组 {grp['name']}({group_id}) 不可用：{grp.get('error') or grp['status']}"
    return grp, None


def _rot_params(params: dict) -> tuple:
    periods = params.get("periods")
    dampings = params.get("dampings", 0.05)
    req = prepare_rot_request(
        periods=periods,
        dampings=dampings,
        method=params.get("method", "average_acceleration"),
        instability_policy=params.get("instability_policy", "refine"),
        baseline_mode=params.get("baseline_mode", "mean"),
        n_angles=params.get("n_angles", 180),
    )
    return req, {
        "periods": req.periods.tolist(),
        "dampings": list(req.dampings),
        "method": req.method,
        "instability_policy": req.instability_policy,
        "baseline_mode": req.baseline_mode,
        "n_angles": req.n_angles,
    }


def _group_view(grp: dict, r1, r2) -> dict:
    """组装 rotd 计算所需的 group 视图（含成员名）。"""

    return {
        "id": grp["id"],
        "name": grp["name"],
        "h1": grp["h1"],
        "h2": grp["h2"],
        "dt": grp["dt"],
        "h1_id": grp["h1_id"],
        "h2_id": grp["h2_id"],
        "h1_name": r1.name,
        "h2_name": r2.name,
        "members": [
            {"record_id": grp["h1_id"], "name": r1.name},
            {"record_id": grp["h2_id"], "name": r2.name},
        ],
        "alignment": grp["alignment"],
    }


def _compute_or_cached(storage: Storage, grp: dict, param_dict: dict):
    """命中缓存直接返回；否则计算并落缓存。返回 (结果 dict, 是否命中)。"""

    key = rot_cache_key(grp, param_dict)
    cached = storage.get_rot_spectrum(key)
    if cached is not None:
        return cached, True
    req = prepare_rot_request(
        periods=param_dict["periods"], dampings=param_dict["dampings"],
        method=param_dict["method"],
        instability_policy=param_dict["instability_policy"],
        baseline_mode=param_dict["baseline_mode"],
        n_angles=param_dict["n_angles"],
    )
    result = compute_rot_spectrum(grp, req)
    storage.put_rot_spectrum(key, grp["id"], param_dict, result)
    return result, False


def run_rot_spectrum_job(job_id: str, storage: Storage, is_cancelled) -> dict:
    params = storage.get_job_params(job_id)
    req, param_dict = _rot_params(params)
    items = storage.list_job_items(job_id)
    groups_out = []
    errors = 0
    skipped = 0
    cache_hits = 0

    for i, item in enumerate(items):
        if is_cancelled():
            storage.finish_cancelled(job_id)
            return {"cancelled": True}
        storage.set_item_running(job_id, item["seq"])
        gid = item["group_id"]
        grp, load_err = _load_ready_group(storage, gid)
        if load_err:
            storage.set_item_skipped(job_id, item["seq"], load_err)
            groups_out.append({"seq": item["seq"], "group_id": gid,
                               "status": "skipped", "error": load_err})
            skipped += 1
            storage.set_job_progress(job_id, i + 1)
            continue
        try:
            r1 = storage.get_record(grp["h1_id"])
            r2 = storage.get_record(grp["h2_id"])
            view = _group_view(grp, r1, r2)
            result, hit = _compute_or_cached(storage, view, param_dict)
            cache_hits += int(hit)
            summary = {
                "seq": item["seq"],
                "group_id": gid,
                "name": grp["name"],
                "status": "done",
                "cache_hit": hit,
                "npts": grp["npts"],
                "dt": grp["dt"],
                "alignment": grp["alignment"],
                "result": result,
            }
            storage.set_item_result(job_id, item["seq"], summary)
            groups_out.append(summary)
        except SeismicError as exc:
            msg = f"分量组 {grp['name']}({gid}) 组合谱计算失败：{exc}"
            storage.set_item_error(job_id, item["seq"], msg)
            groups_out.append({"seq": item["seq"], "group_id": gid,
                               "name": grp["name"], "status": "error",
                               "error": str(exc)})
            errors += 1
        storage.set_job_progress(job_id, i + 1)

    return {
        "cancelled": False,
        "config": param_dict,
        "groups": groups_out,
        "summary": {
            "total": len(items),
            "done": len(items) - errors - skipped,
            "error": errors,
            "skipped": skipped,
            "cache_hits": cache_hits,
        },
    }


def run_match_group_job(job_id: str, storage: Storage, is_cancelled) -> dict:
    params = storage.get_job_params(job_id)
    target_damping = float(params["target_damping"])
    req, param_dict = _rot_params(params)
    if target_damping not in req.dampings:
        req.dampings = tuple(sorted(set(req.dampings) | {target_damping}))
        param_dict["dampings"] = list(req.dampings)

    dp = params["design_spectrum"]
    design = make_design_spectrum(dp["periods"], dp["values"], dp.get("unit", "g"))
    quantity = params.get("quantity", "psa")
    rotd_band = params.get("rotd_band", "rotd50")

    items = storage.list_job_items(job_id)
    candidates = []
    per_group = []
    errors = 0
    skipped = 0
    cache_hits = 0

    for i, item in enumerate(items):
        if is_cancelled():
            storage.finish_cancelled(job_id)
            return {"cancelled": True}
        storage.set_item_running(job_id, item["seq"])
        gid = item["group_id"]
        grp, load_err = _load_ready_group(storage, gid)
        if load_err:
            storage.set_item_skipped(job_id, item["seq"], load_err)
            per_group.append({"seq": item["seq"], "group_id": gid,
                              "status": "skipped", "error": load_err})
            skipped += 1
            storage.set_job_progress(job_id, i + 1)
            continue
        try:
            r1 = storage.get_record(grp["h1_id"])
            r2 = storage.get_record(grp["h2_id"])
            view = _group_view(grp, r1, r2)
            rot, hit = _compute_or_cached(storage, view, param_dict)
            cache_hits += int(hit)
            chosen = next(
                s for s in rot["spectra"]
                if abs(s["damping"] - target_damping) < 1e-12
            )

            def _unit(vals):
                # 设计谱以 g 给出时，谱量统一换算到 g
                return [v / GRAVITY for v in vals] if design.unit == "g" else vals

            members = []
            for m in chosen["components"]:
                members.append({
                    "record_id": m["record_id"],
                    "name": m.get("name"),
                    "periods": rot["periods"],
                    quantity: _unit(m[quantity]),
                })
            candidates.append({
                "group_id": gid,
                "name": grp["name"],
                "periods": rot["periods"],
                "rotd50": {quantity: _unit(chosen["rotd50"][quantity])},
                "rotd100": {quantity: _unit(chosen["rotd100"][quantity])},
                "members": members,
            })
            entry = {
                "seq": item["seq"],
                "group_id": gid,
                "name": grp["name"],
                "status": "done",
                "cache_hit": hit,
                "target_damping": target_damping,
                "result": rot,
            }
            storage.set_item_result(job_id, item["seq"], entry)
            per_group.append(entry)
        except SeismicError as exc:
            msg = f"分量组 {grp['name']}({gid}) 计算失败：{exc}"
            storage.set_item_error(job_id, item["seq"], msg)
            per_group.append({"seq": item["seq"], "group_id": gid,
                              "name": grp["name"], "status": "error",
                              "error": str(exc)})
            errors += 1
        storage.set_job_progress(job_id, i + 1)

    matching = match_group_batch(
        candidates,
        design,
        t1=float(params["t1"]),
        t2=float(params["t2"]),
        s_min=float(params.get("s_min", 0.5)),
        s_max=float(params.get("s_max", 3.0)),
        top_n=int(params.get("top_n", min(7, max(1, len(candidates))))),
        quantity=quantity,
        rotd_band=rotd_band,
        bounds_policy=params.get("bounds_policy", "clamp"),
    )

    return {
        "cancelled": False,
        "config": {
            **param_dict,
            "target_damping": target_damping,
            "rotd_band": rotd_band,
        },
        "matching": matching,
        "groups": per_group,
        "summary": {
            "total": len(items),
            "spectra_done": len(candidates),
            "error": errors,
            "skipped": skipped,
            "cache_hits": cache_hits,
        },
    }
