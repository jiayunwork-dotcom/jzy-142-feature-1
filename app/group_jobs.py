"""分量组作业执行逻辑：组合谱（rotd）作业与组匹配（match_group）作业。

- ``rotd``：逐组调用 :mod:`app.rotd` 计算 RotD50/RotD100 组合谱，单组
  失败只标记该作业项；结果以「组内容 + 全部积分参数 + 算法版本」指纹
  写入 ``rotd_cache``，后续同参数作业直接复用（组不可变：成员一变就是
  新组 ID/内容键，绝不会拿到旧结果）。
- ``match_group``：以组为单位，两条水平分量**共用一个缩放系数**，
  误差按 RotD100 与设计谱比较；越限处理、排序、前 N 组平均谱、逐点
  比值全部复用单条匹配的口径（:func:`app.matching.match_batch`），
  输出额外给出每组缩放后两条分量各自的谱。
"""

from __future__ import annotations

from .errors import GroupError, SeismicError
from .groups import align_group_records, component_dict
from .matching import make_design_spectrum, match_batch
from .rotd import (
    compute_rotd_spectrum,
    rotd_fingerprint,
    validate_n_angles,
)
from .spectrum import prepare_spectrum_request
from .storage import Storage


def _load_aligned(storage: Storage, group_id: str):
    g = storage.get_group(group_id)
    if g is None:
        return None, None, f"分量组 {group_id} 不存在，可能未建组或已删除"
    h1 = storage.get_record(g["horizontal1"])
    h2 = storage.get_record(g["horizontal2"])
    hv = storage.get_record(g["vertical"]) if g["vertical"] else None
    try:
        aligned = align_group_records(h1, h2, hv)
    except GroupError as exc:
        return None, g, f"分量组 {g['name'] or group_id} 无法对齐：{exc}"
    return aligned, g, None


def _config_from_params(params: dict, *, with_angles: bool):
    req = prepare_spectrum_request(
        periods=params.get("periods"),
        dampings=params.get("dampings", 0.05),
        method=params.get("method", "average_acceleration"),
        instability_policy=params.get("instability_policy", "refine"),
        baseline_mode=params.get("baseline_mode", "mean"),
    )
    n_angles = None
    if with_angles:
        n_angles = validate_n_angles(params.get("n_angles", 180))
    return req, n_angles


def _group_result_body(seq, group_id, g, aligned, spectra, *, source):
    return {
        "seq": seq,
        "group_id": group_id,
        "name": g["name"] if g else "",
        "status": "done",
        "result_source": source,       # computed / cache
        "npts": aligned.npts,
        "dt": aligned.dt,
        "t0": aligned.t0,
        "duration": aligned.duration,
        "alignment": g["alignment"] if g else None,
        "components": [component_dict(aligned.h1), component_dict(aligned.h2)]
        + ([component_dict(aligned.vertical)] if aligned.vertical else []),
        "spectra": spectra,
    }


def _compute_group_spectra(aligned, req, n_angles):
    spectra = []
    for zeta in req.dampings:
        spectra.append(
            compute_rotd_spectrum(
                aligned.h1.acc_si, aligned.h2.acc_si,
                aligned.dt, req.periods, zeta,
                method=req.method,
                instability_policy=req.instability_policy,
                baseline_mode=req.baseline_mode,
                n_angles=n_angles,
            )
        )
    return spectra


def run_rotd_job(job_id: str, storage: Storage, is_cancelled) -> dict:
    """执行一条 rotd 作业。"""

    params = storage.get_job_params(job_id)
    req, n_angles = _config_from_params(params, with_angles=True)
    items = storage.list_job_items(job_id)
    submitted_ids = params.get("group_ids")
    groups_out = []
    errors = 0
    skipped = 0
    cache_hits = 0

    cache_params = {
        "periods": req.periods.tolist(),
        "dampings": list(req.dampings),
        "method": req.method,
        "instability_policy": req.instability_policy,
        "baseline_mode": req.baseline_mode,
        "n_angles": n_angles,
    }

    for i, item in enumerate(items):
        if is_cancelled():
            storage.finish_cancelled(job_id)
            return {"cancelled": True}
        storage.set_item_running(job_id, item["seq"])
        gid = item["group_id"] or (
            submitted_ids[item["seq"]]
            if submitted_ids and item["seq"] < len(submitted_ids) else None)
        aligned, g, load_err = _load_aligned(storage, gid)
        if load_err:
            storage.set_item_skipped(job_id, item["seq"], load_err)
            groups_out.append({"seq": item["seq"], "group_id": gid,
                               "status": "skipped", "error": load_err})
            skipped += 1
            storage.set_job_progress(job_id, i + 1)
            continue

        fp = rotd_fingerprint(group_content_key=g["content_key"], **cache_params)
        cached = storage.get_rotd_cache(fp)
        try:
            if cached is not None:
                spectra = cached["spectra"]
                source = "cache"
                cache_hits += 1
            else:
                spectra = _compute_group_spectra(aligned, req, n_angles)
                source = "computed"
                storage.put_rotd_cache(fp, gid, cache_params, {"spectra": spectra})
            body = _group_result_body(
                item["seq"], gid, g, aligned, spectra, source=source
            )
            storage.set_item_result(job_id, item["seq"], body)
            groups_out.append(body)
        except SeismicError as exc:
            msg = f"分量组 {g['name'] or gid} 组合谱计算失败：{exc}"
            storage.set_item_error(job_id, item["seq"], msg)
            groups_out.append({"seq": item["seq"], "group_id": gid,
                               "name": g["name"] if g else "",
                               "status": "error", "error": str(exc)})
            errors += 1
        storage.set_job_progress(job_id, i + 1)

    return {
        "cancelled": False,
        "config": {
            "periods": req.periods.tolist(),
            "dampings": list(req.dampings),
            "method": req.method,
            "instability_policy": req.instability_policy,
            "baseline_mode": req.baseline_mode,
            "n_angles": n_angles,
            "angle_step_deg": 180.0 / n_angles,
        },
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
    """执行一条组匹配作业：逐组算组合谱（可命中缓存），再整体匹配。"""

    params = storage.get_job_params(job_id)
    req, n_angles = _config_from_params(params, with_angles=True)
    target_damping = float(params["target_damping"])
    if target_damping not in req.dampings:
        req.dampings = tuple(sorted(set(req.dampings) | {target_damping}))

    dp = params["design_spectrum"]
    design = make_design_spectrum(dp["periods"], dp["values"], dp.get("unit", "g"))
    quantity = params.get("quantity", "psa")

    cache_params = {
        "periods": req.periods.tolist(),
        "dampings": list(req.dampings),
        "method": req.method,
        "instability_policy": req.instability_policy,
        "baseline_mode": req.baseline_mode,
        "n_angles": n_angles,
    }

    items = storage.list_job_items(job_id)
    submitted_ids = params.get("group_ids")
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
        gid = item["group_id"] or (
            submitted_ids[item["seq"]]
            if submitted_ids and item["seq"] < len(submitted_ids) else None)
        aligned, g, load_err = _load_aligned(storage, gid)
        if load_err:
            storage.set_item_skipped(job_id, item["seq"], load_err)
            per_group.append({"seq": item["seq"], "group_id": gid,
                              "status": "skipped", "error": load_err})
            skipped += 1
            storage.set_job_progress(job_id, i + 1)
            continue

        fp = rotd_fingerprint(group_content_key=g["content_key"], **cache_params)
        cached = storage.get_rotd_cache(fp)
        try:
            if cached is not None:
                spectra = cached["spectra"]
                source = "cache"
                cache_hits += 1
            else:
                spectra = _compute_group_spectra(aligned, req, n_angles)
                source = "computed"
                storage.put_rotd_cache(fp, gid, cache_params, {"spectra": spectra})

            chosen_spec = next(
                s for s in spectra
                if abs(s["damping"] - target_damping) < 1e-12
            )
            values = chosen_spec["rotd100"][quantity]
            if design.unit == "g":
                from .parsing import GRAVITY
                values = [v / GRAVITY for v in values]
            candidates.append({
                "group_id": gid,
                "name": g["name"] or _auto_group_name(g),
                "periods": req.periods.tolist(),
                quantity: values,
            })
            entry = {
                "seq": item["seq"],
                "group_id": gid,
                "name": g["name"] or "",
                "status": "done",
                "result_source": source,
                "alignment": g["alignment"],
                "spectrum": chosen_spec,
            }
            storage.set_item_result(job_id, item["seq"], entry)
            per_group.append(entry)
        except SeismicError as exc:
            msg = f"分量组 {g['name'] or gid} 组合谱计算失败：{exc}"
            storage.set_item_error(job_id, item["seq"], msg)
            per_group.append({"seq": item["seq"], "group_id": gid,
                              "name": g["name"] if g else "",
                              "status": "error", "error": str(exc)})
            errors += 1
        storage.set_job_progress(job_id, i + 1)

    # 所有组都失败时不让整作业失败：给出空匹配结果并注明原因
    if not candidates:
        matching = {
            "entity": "group",
            "matched_quantity": quantity,
            "matched_spectrum": "rotd100",
            "ranking": [],
            "excluded": [],
            "average_spectrum": None,
            "ratios": None,
            "n_candidates": 0,
            "n_excluded": 0,
            "note": "没有任何组成功算出组合谱，匹配未执行（详见各组错误）",
        }
    else:
        matching = match_batch(
            candidates,
            design,
            t1=float(params["t1"]),
            t2=float(params["t2"]),
            s_min=float(params.get("s_min", 0.5)),
            s_max=float(params.get("s_max", 3.0)),
            top_n=int(params.get("top_n", 7)),
            quantity=quantity,
            bounds_policy=params.get("bounds_policy", "clamp"),
            id_key="group_id",
            entity="group",
        )

    # 入榜每组：给出缩放后 RotD100 / RotD50 以及两条分量各自的缩放谱
    from .parsing import GRAVITY
    spec_by_gid = {e["group_id"]: e["spectrum"]
                   for e in per_group if e.get("status") == "done"}
    scaled_groups = []
    for rank in matching["ranking"]:
        gid = rank["group_id"]
        s = rank["scale"]
        spec = spec_by_gid[gid]
        unit_div = GRAVITY if design.unit == "g" else 1.0

        def _scaled(part, key):
            return [v * s / unit_div for v in part[key]]

        scaled_groups.append({
            "rank": rank["rank"],
            "group_id": gid,
            **({"name": rank["name"]} if "name" in rank else {}),
            "scale": s,
            "periods": spec["periods"],
            "matched_quantity": quantity,
            "rotd100_scaled": {
                q: _scaled(spec["rotd100"], q)
                for q in ("sd", "sv", "sa", "psv", "psa")
            },
            "rotd50_scaled": {
                q: _scaled(spec["rotd50"], q)
                for q in ("sd", "sv", "sa", "psv", "psa")
            },
            "components_scaled": [
                {
                    "record_role": c["record_role"],
                    **{q: [v * s / unit_div for v in c[q]]
                       for q in ("sd", "sv", "sa", "psv", "psa")},
                }
                for c in spec["components"]
            ],
        })
    matching["scaled_groups"] = scaled_groups

    return {
        "cancelled": False,
        "config": {
            "periods": req.periods.tolist(),
            "dampings": list(req.dampings),
            "method": req.method,
            "instability_policy": req.instability_policy,
            "baseline_mode": req.baseline_mode,
            "n_angles": n_angles,
            "angle_step_deg": 180.0 / n_angles,
            "target_damping": target_damping,
            "matched_spectrum": "rotd100",
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


def _auto_group_name(g: dict) -> str:
    return f"{g['horizontal1']}+{g['horizontal2']}"
