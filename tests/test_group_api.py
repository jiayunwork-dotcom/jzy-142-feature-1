"""分量组 / rotd 作业 / 组匹配的 HTTP 端到端测试。"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from app.rotd import ROTD100_COMPARE_TOL, ROTD50_COMPARE_TOL


def _record_text(acc, *, station, event, component):
    lines = [f"# station: {station}", f"# event: {event}",
             f"# component: {component}"]
    lines += [f"{v:.12g}" for v in acc]
    return "\n".join(lines)


@pytest.fixture()
def signals():
    rng = np.random.default_rng(42)
    n = 2500
    dt = 0.01
    t = np.arange(n) * dt
    a1 = (np.sin(2 * np.pi * t / 0.5) * np.exp(-t / 6)
          + 0.2 * rng.standard_normal(n))
    a2 = (np.cos(2 * np.pi * t / 0.7) * np.exp(-t / 7)
          + 0.2 * rng.standard_normal(n))
    return a1, a2, dt


def _upload(client, name, acc, dt, component, station="ST1", event="EV1"):
    rv = client.post("/api/records", json={"records": [{
        "name": name, "unit": "m/s2", "dt": dt,
        "content": _record_text(
            acc, station=station, event=event, component=component),
    }]})
    return rv.get_json()["records"][0]["record_id"]


def _wait(client, jid, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{jid}").get_json()
        st = body["job"]["status"]
        if st in ("completed", "failed", "cancelled", "interrupted"):
            return body["job"]
        time.sleep(0.03)
    raise AssertionError(f"作业 {jid} 超时未结束")


PERIODS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
DESIGN = {"periods": [0.05, 0.1, 0.5, 1.0, 4.0],
          "values": [0.1, 0.25, 0.25, 0.15, 0.05], "unit": "g"}


def _make_group(client, signals, r1meta=("N",), r2meta=("E",),
                vertical=False, scale=1.0, suffix=""):
    a1, a2, dt = signals
    rid1 = _upload(client, f"n{suffix}.txt", a1 * scale, dt, r1meta[0])
    rid2 = _upload(client, f"e{suffix}.txt", a2 * scale, dt, r2meta[0])
    payload = {"horizontal1": rid1, "horizontal2": rid2}
    if vertical:
        rv = _upload(client, f"z{suffix}.txt", a1 * 0.3, dt, "Z")
        payload["vertical_record_id"] = rv
    rv = client.post("/api/groups", json=payload)
    assert rv.status_code == 201, rv.get_json()
    return rv.get_json()["group_id"], rid1, rid2


def test_group_lifecycle_and_suggestions(client, signals):
    a1, a2, dt = signals
    rid_n = _upload(client, "rec-n.txt", a1, dt, "N")
    rid_e = _upload(client, "rec-e.txt", a2, dt, "E")
    rid_z = _upload(client, "rec-z.txt", a1 * 0.3, dt, "Z")

    # 建议只读，不自动建组
    s = client.post("/api/groups/suggestions", json={}).get_json()
    assert s["suggestions"][0]["confidence"] == "orthogonal"
    assert client.get("/api/groups").get_json()["groups"] == []

    rv = client.post("/api/groups", json={
        "horizontal1": rid_n, "horizontal2": rid_e,
        "vertical_record_id": rid_z})
    assert rv.status_code == 201
    g = rv.get_json()
    gid = g["group_id"]
    assert g["npts"] == a1.size
    assert g["alignment"]["policy"] == "union_resample_zeropad"
    assert g["alignment"]["components"][2]["direction_label"] == "Z"

    # 点名不存在的记录 / 两条相同 -> 400
    assert client.post("/api/groups", json={
        "horizontal1": rid_n, "horizontal2": "rec_nope"}).status_code == 400
    assert client.post("/api/groups", json={
        "horizontal1": rid_n, "horizontal2": rid_n}).status_code == 400

    # 组落库，可查
    assert client.get(f"/api/groups/{gid}").get_json()["group"]["id"] == gid
    assert client.get("/api/groups/nope").status_code == 404


def test_swap_horizontal_order_same_group(client, signals):
    a1, a2, dt = signals
    r1 = _upload(client, "aa.txt", a1, dt, "N")
    r2 = _upload(client, "ab.txt", a2, dt, "E")
    g1 = client.post("/api/groups",
                     json={"horizontal1": r1, "horizontal2": r2}).get_json()
    g2 = client.post("/api/groups",
                     json={"horizontal1": r2, "horizontal2": r1}).get_json()
    assert g1["group_id"] == g2["group_id"]


def _submit_rotd(client, gid, *, periods=PERIODS, n_angles=180,
                 dampings=(0.05,)):
    rv = client.post("/api/jobs/rotd", json={
        "group_ids": [gid], "periods": periods, "dampings": list(dampings),
        "n_angles": n_angles})
    assert rv.status_code == 201, rv.get_json()
    return rv.get_json()["job_id"]


def test_rotd_job_end_to_end_and_cache(wclient, signals):
    client = wclient
    gid, _, _ = _make_group(client, signals)
    jid = _submit_rotd(client, gid)
    job = _wait(client, jid)
    assert job["status"] == "completed"
    res = job["result"]
    assert res["summary"]["done"] == 1
    spec = res["groups"][0]["spectra"][0]
    assert spec["n_angles"] == 180
    for key in ("rotd50", "rotd100"):
        for q in ("sd", "sv", "sa", "psv", "psa"):
            assert len(spec[key][q]) == len(PERIODS)
    assert len(spec["rotd100"]["angle_deg"]) == len(PERIODS)
    assert len(spec["components"]) == 2
    assert "psa" in spec["geometric_mean"]

    # items 执行中可查，完成后可查
    items = client.get(f"/api/jobs/{jid}/items").get_json()
    assert items["items"][0]["status"] == "done"

    # 再提交一次：命中缓存且逐位相同
    jid2 = _submit_rotd(client, gid)
    job2 = _wait(client, jid2)
    assert job2["result"]["summary"]["cache_hits"] == 1
    # 第二次结果体（只含 spectra）与第一次的 spectra 部分逐位一致
    assert json.dumps(job["result"]["groups"][0]["spectra"],
                      sort_keys=True) == \
        json.dumps(job2["result"]["groups"][0]["spectra"], sort_keys=True)


def test_rotd_invariants_over_http(wclient, signals):
    """需求里的旋转/交换/取反/缩放关系，通过上传新记录、建新组走全链路。"""

    client = wclient
    a1, a2, dt = signals
    gid0, _, _ = _make_group(client, signals)
    j0 = _wait(client, _submit_rotd(client, gid0))
    base = j0["result"]["groups"][0]["spectra"][0]
    b100 = np.asarray(base["rotd100"]["psa"])
    b50 = np.asarray(base["rotd50"]["psa"])

    def rotd_for_pair(acc1, acc2, tag, comp1="N", comp2="E"):
        r1 = _upload(client, f"{tag}-1.txt", acc1, dt, comp1)
        r2 = _upload(client, f"{tag}-2.txt", acc2, dt, comp2)
        gid = client.post("/api/groups",
                          json={"horizontal1": r1, "horizontal2": r2}
                          ).get_json()["group_id"]
        job = _wait(client, _submit_rotd(client, gid))
        return job["result"]["groups"][0]["spectra"][0]

    # 30° 与 117° 旋转（网格整数角）
    for deg in (30.0, 117.0):
        th = np.deg2rad(deg)
        rr = rotd_for_pair(a1 * np.cos(th) + a2 * np.sin(th),
                           -a1 * np.sin(th) + a2 * np.cos(th), f"rot{deg}")
        np.testing.assert_allclose(rr["rotd100"]["psa"], b100, rtol=1e-10)

    # 非网格角 30.5°：按声明容差
    th = np.deg2rad(30.5)
    rr = rotd_for_pair(a1 * np.cos(th) + a2 * np.sin(th),
                       -a1 * np.sin(th) + a2 * np.cos(th), "rot30.5")
    rel = np.max(np.abs(np.asarray(rr["rotd100"]["psa"]) - b100)
                 / np.maximum(np.abs(b100), 1e-300))
    assert rel <= ROTD100_COMPARE_TOL
    rel50 = np.max(np.abs(np.asarray(rr["rotd50"]["psa"]) - b50)
                   / np.maximum(np.abs(b50), 1e-300))
    assert rel50 <= ROTD50_COMPARE_TOL

    # 交换两条 / 一条取反
    rs = rotd_for_pair(a2, a1, "swap")
    np.testing.assert_allclose(rs["rotd100"]["psa"], b100, rtol=1e-10)
    rn = rotd_for_pair(-a1, a2, "neg", comp1="E", comp2="N")
    np.testing.assert_allclose(rn["rotd100"]["psa"], b100, rtol=1e-10)


def _submit_match(client, gid, *, s_min=1e-6, s_max=1e6, quantity="psa"):
    rv = client.post("/api/jobs/match-group", json={
        "group_ids": [gid], "periods": PERIODS, "dampings": [0.05],
        "n_angles": 180, "design_spectrum": DESIGN,
        "t1": 0.1, "t2": 2.0, "quantity": quantity,
        "s_min": s_min, "s_max": s_max, "bounds_policy": "reject"})
    assert rv.status_code == 201, rv.get_json()
    return _wait(client, rv.get_json()["job_id"])


def test_group_match_common_scale_and_inverse_scaling(wclient, signals):
    client = wclient
    gid, _, _ = _make_group(client, signals)
    job = _submit_match(client, gid)
    assert job["status"] == "completed"
    m = job["result"]["matching"]
    assert m["matched_spectrum"] == "rotd100"
    scale0 = m["ranking"][0]["scale"]

    # 输出含缩放后两条分量各自的谱
    sg = m["scaled_groups"][0]
    assert len(sg["components_scaled"]) == 2
    for comp in sg["components_scaled"]:
        assert len(comp["psa"]) == len(PERIODS)
    assert "arithmetic_mean" in m["average_spectrum"]
    assert "ratio_average_over_design" in m["ratios"]

    # 整组乘 k -> 缩放系数变 1/k
    k = 4.0
    gid_k, _, _ = _make_group(client, signals, scale=k, suffix="-k")
    job_k = _submit_match(client, gid_k)
    scale_k = job_k["result"]["matching"]["ranking"][0]["scale"]
    assert scale_k * k == pytest.approx(scale0, rel=1e-8)


def test_group_match_bounds_reject_and_clamp(wclient, signals):
    client = wclient
    gid, _, _ = _make_group(client, signals)
    # 极窄可行区间 -> 最优系数越限被剔除
    rv = client.post("/api/jobs/match-group", json={
        "group_ids": [gid], "periods": PERIODS, "dampings": [0.05],
        "n_angles": 180, "design_spectrum": DESIGN,
        "t1": 0.1, "t2": 2.0, "s_min": 0.99, "s_max": 1.01,
        "bounds_policy": "reject"})
    job = _wait(client, rv.get_json()["job_id"])
    m = job["result"]["matching"]
    assert m["ranking"] == []
    assert m["n_excluded"] == 1
    assert "scale_optimal" in m["excluded"][0]


def test_one_bad_group_does_not_fail_batch(wclient, signals):
    client = wclient
    gid, _, _ = _make_group(client, signals)
    rv = client.post("/api/jobs/rotd", json={
        "group_ids": [gid, "grp_deadbeefdead"], "periods": PERIODS,
        "dampings": [0.05], "n_angles": 180})
    assert rv.status_code == 201
    job = _wait(client, rv.get_json()["job_id"])
    assert job["status"] == "completed"
    assert job["result"]["summary"]["done"] == 1
    assert job["result"]["summary"]["skipped"] == 1


def test_match_group_all_missing_completes_empty(wclient):
    """所有组都不存在（全部 skipped）时，匹配作业仍 completed，
    返回空排序而不是整作业失败。"""

    client = wclient
    rv = client.post("/api/jobs/match-group", json={
        "group_ids": ["grp_absent0000001", "grp_absent0000002"],
        "periods": PERIODS, "dampings": [0.05], "n_angles": 180,
        "design_spectrum": DESIGN, "t1": 0.1, "t2": 2.0,
        "s_min": 0.1, "s_max": 10.0})
    assert rv.status_code == 201
    job = _wait(client, rv.get_json()["job_id"])
    assert job["status"] == "completed"
    m = job["result"]["matching"]
    assert m["ranking"] == []
    assert job["result"]["summary"]["skipped"] == 2


def test_rotd_batch_limit(client, signals):
    rv = client.post("/api/jobs/rotd",
                     json={"group_ids": [f"g{i}" for i in range(201)]})
    assert rv.status_code == 400
    assert "200" in rv.get_json()["error"]


def test_rotd_validation_errors(client, signals):
    gid, _, _ = _make_group(client, signals)
    rv = client.post("/api/jobs/rotd", json={
        "group_ids": [gid], "periods": PERIODS, "n_angles": 9})
    assert rv.status_code == 400 and "偶数" in rv.get_json()["error"]
    rv = client.post("/api/jobs/rotd", json={"group_ids": []})
    assert rv.status_code == 400


def test_cancel_rotd_job(wclient, signals):
    client = wclient
    # 造多个大组，给取消留出窗口
    gids = []
    for i in range(3):
        rng = np.random.default_rng(100 + i)
        big = rng.standard_normal(60000)
        rid1 = _upload(client, f"cb1-{i}.txt", big, 0.01, "N")
        rid2 = _upload(client, f"cb2-{i}.txt", big * 0.8, 0.01, "E")
        gids.append(client.post("/api/groups", json={
            "horizontal1": rid1, "horizontal2": rid2}).get_json()["group_id"])
    rv = client.post("/api/jobs/rotd", json={
        "group_ids": gids,
        "periods": np.linspace(0.05, 6, 100).tolist(),
        "dampings": [0.05], "n_angles": 180})
    jid = rv.get_json()["job_id"]
    time.sleep(0.2)
    assert client.post(f"/api/jobs/{jid}/cancel").status_code == 200
    job = _wait(client, jid)
    assert job["status"] == "cancelled"


def test_group_persists_across_restart(tmp_path, signals):
    """组、已完成的 rotd 作业结果重启后仍可读。"""

    from app import create_app

    db_path = str(tmp_path / "persist.db")
    app1 = create_app(db_path, start_worker=True)
    c1 = app1.test_client()
    a1, a2, dt = signals
    r1 = _upload(c1, "p1.txt", a1, dt, "N")
    r2 = _upload(c1, "p2.txt", a2, dt, "E")
    gid = c1.post("/api/groups",
                  json={"horizontal1": r1, "horizontal2": r2}
                  ).get_json()["group_id"]
    jid = _submit_rotd(c1, gid)
    job1 = _wait(c1, jid)
    assert job1["status"] == "completed"
    app1.extensions["scheduler"].stop()
    app1.extensions["storage"].close()

    app2 = create_app(db_path, start_worker=False)
    c2 = app2.test_client()
    g = c2.get(f"/api/groups/{gid}").get_json()["group"]
    assert g["horizontal1"] == r1
    job2 = c2.get(f"/api/jobs/{jid}").get_json()["job"]
    assert job2["status"] == "completed"
    assert job2["result"]["summary"]["done"] == 1
    app2.extensions["scheduler"].stop()
    app2.extensions["storage"].close()
