"""分量组 / 组合谱作业 / 组匹配作业的端到端 HTTP 测试。"""

from __future__ import annotations

import io
import json
import time

import numpy as np
import pytest

from tests.conftest import wait_for_status


def _record_text(n=1000, dt=0.01, seed=0, f=1.5, *, station="ST1",
                 event="EV1", component=None, extra_headers=""):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    a = np.sin(2 * np.pi * f * t) + 0.3 * rng.normal(size=n)
    head = f"# Station: {station}\n# Event: {event}\n"
    if component:
        head += f"# Component: {component}\n"
    return head + extra_headers + "\n".join(f"{x:.6f}" for x in a)


def _upload(client, text, name, unit="m/s2", dt=0.01):
    return client.post("/api/records", data={
        "files": (io.BytesIO(text.encode()), name),
        "unit": unit, "dt": str(dt),
    }, content_type="multipart/form-data").get_json()["records"][0]["record_id"]


def _wait(client, jid):
    return wait_for_status(client, jid)


@pytest.fixture()
def group_ids(app_worker):
    client = app_worker.test_client()
    h1 = _upload(client, _record_text(seed=0, component="HNN"), "a.HNN.txt")
    h2 = _upload(client, _record_text(seed=1, component="HNE"), "b.HNE.txt")
    vz = _upload(client, _record_text(seed=2, component="HNZ", f=2.2),
                 "c.HNZ.txt")
    rv = client.post("/api/groups", json={
        "h1_id": h1, "h2_id": h2, "vertical_id": vz, "name": "ST1/EV1",
    })
    assert rv.status_code == 201
    gid = rv.get_json()["group_id"]
    return client, [gid], (h1, h2, vz)


def test_suggestions_do_not_auto_create(client):
    _upload(client, _record_text(seed=0, component="HNN"), "a.HNN.txt")
    _upload(client, _record_text(seed=1, component="HNE"), "b.HNE.txt")
    rv = client.post("/api/groups/suggestions")
    body = rv.get_json()
    assert body["n_suggestions"] == 1
    s = body["suggestions"][0]
    assert s["confidence"] == "preferred"
    assert client.get("/api/groups").get_json()["groups"] == []
    # 必须显式确认后才建组
    rv = client.post("/api/groups", json={"h1_id": s["h1_id"],
                                          "h2_id": s["h2_id"]})
    assert rv.status_code == 201
    assert len(client.get("/api/groups").get_json()["groups"]) == 1


def test_create_group_explicit_and_query(group_ids):
    client, ids, _ = group_ids
    gid = ids[0]
    rv = client.get(f"/api/groups/{gid}")
    assert rv.status_code == 200
    g = rv.get_json()["group"]
    assert g["npts"] == 1000
    assert g["vertical_id"] is not None
    assert g["alignment"]["grid_mode"] == "lattice"
    listed = client.get("/api/groups").get_json()["groups"]
    assert listed[0]["id"] == gid and "h1" not in listed[0]


def test_create_group_validation(client):
    h1 = _upload(client, _record_text(seed=0, component="HNN"), "a.txt")
    # 缺一条
    rv = client.post("/api/groups", json={"h1_id": h1})
    assert rv.status_code == 400
    # 同一条记录引用两次
    rv = client.post("/api/groups", json={"h1_id": h1, "h2_id": h1})
    assert rv.status_code == 400 and "不同记录" in rv.get_json()["error"]
    # 不存在
    rv = client.post("/api/groups", json={"h1_id": h1, "h2_id": "rec_nope"})
    assert rv.status_code == 400 and "不存在" in rv.get_json()["error"]
    rv = client.get("/api/groups/grp_nope")
    assert rv.status_code == 404


def test_rot_spectrum_job_end_to_end(group_ids):
    client, ids, _ = group_ids
    jid = client.post("/api/jobs/rot-spectrum", json={
        "group_ids": ids,
        "periods": [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0],
        "dampings": [0.02, 0.05],
    }).get_json()["job_id"]
    body = _wait(client, jid)
    assert body["job"]["status"] == "completed"
    res = body["job"]["result"]
    assert res["summary"]["done"] == 1
    item = res["groups"][0]
    spec = item["result"]["spectra"]
    assert [s["damping"] for s in spec] == [0.02, 0.05]
    s = spec[1]
    assert len(s["components"]) == 2
    for q in ("sd", "sa", "psa"):
        assert q in s["rotd50"] and q in s["rotd100"]
        d100 = np.asarray(s["rotd100"][q])
        d50 = np.asarray(s["rotd50"][q])
        assert np.all(d100 + 1e-12 >= d50)
    # 位移/伪加速度/绝对加速度三类齐全
    assert s["rotd100"]["psa_g"]
    assert "angle" in json.dumps(s)
    items = client.get(f"/api/jobs/{jid}/items").get_json()
    assert items["counts"]["done"] == 1


def test_rot_job_repeat_is_bitwise_identical_and_cached(group_ids):
    client, ids, _ = group_ids
    payload = {"group_ids": ids,
               "periods": [0.05, 0.1, 0.3, 1.0, 3.0], "dampings": [0.05]}
    outs = []
    for _ in range(2):
        jid = client.post("/api/jobs/rot-spectrum", json=payload).get_json()["job_id"]
        outs.append(_wait(client, jid)["job"]["result"])
    # 组合结果（逐位）相同；cache_hit 这类作业元信息允许不同
    r0 = outs[0]["groups"][0]["result"]
    r1 = outs[1]["groups"][0]["result"]
    assert json.dumps(r0, sort_keys=True) == json.dumps(r1, sort_keys=True)
    # 第一次实算、第二次命中缓存
    assert outs[0]["groups"][0]["cache_hit"] is False
    assert outs[1]["groups"][0]["cache_hit"] is True
    assert outs[1]["summary"]["cache_hits"] == 1


def test_one_bad_group_does_not_fail_batch(app_worker):
    client = app_worker.test_client()
    h1 = _upload(client, _record_text(seed=0, component="HNN"), "a.txt")
    h2 = _upload(client, _record_text(seed=1, component="HNE"), "b.txt")
    gid = client.post("/api/groups",
                      json={"h1_id": h1, "h2_id": h2}).get_json()["group_id"]
    # 手工在库里放一个「ready 但数据损坏」的组，绕过建组时校验，
    # 用来验证执行期单组失败只标记该组
    storage = app_worker.extensions["storage"]
    bad_blob = np.zeros(0, dtype=np.float64)
    storage.create_group(
        "grp_broken", name="broken", station=None, event=None,
        h1_id=h1, h2_id=h2, vertical_id=None, dt=0.01,
        alignment={"reason": "构造的损坏组"},
        h1=bad_blob, h2=bad_blob,
    )

    jid = client.post("/api/jobs/rot-spectrum", json={
        "group_ids": [gid, "grp_broken"],
        "periods": [0.1, 0.5, 1.0],
    }).get_json()["job_id"]
    body = _wait(client, jid)
    assert body["job"]["status"] == "completed"
    summ = body["job"]["result"]["summary"]
    assert summ["done"] == 1 and summ["skipped"] == 1
    statuses = {g["group_id"]: g["status"]
                for g in body["job"]["result"]["groups"]}
    assert statuses[gid] == "done" and statuses["grp_broken"] == "skipped"


def test_match_group_job_end_to_end(group_ids):
    client, ids, _ = group_ids
    design = {"periods": [0.05, 0.1, 0.5, 1.0, 3.0],
              "values": [0.15, 0.2, 0.2, 0.12, 0.06], "unit": "g"}
    jid = client.post("/api/jobs/match-group", json={
        "group_ids": ids,
        "periods": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0],
        "dampings": [0.05], "target_damping": 0.05,
        "design_spectrum": design,
        "t1": 0.1, "t2": 2.0, "s_min": 0.01, "s_max": 100.0, "top_n": 1,
    }).get_json()["job_id"]
    body = _wait(client, jid)
    assert body["job"]["status"] == "completed"
    m = body["job"]["result"]["matching"]
    rank = m["ranking"][0]
    assert rank["group_id"] == ids[0]
    # 同一组两条分量共用一个缩放系数，输出含各自缩放后谱
    ms = rank["members_scaled"]
    assert len(ms) == 2
    assert ms[0]["scale"] == ms[1]["scale"] == rank["scale"]
    assert len(ms[0]["psa"]) == 7
    assert m["ratios"] and len(m["ratios"]["ratio_average_over_design"]) == 5
    # 平均谱 = 唯一入选组的组合谱×scale（逐点）
    avg = np.asarray(m["average_spectrum"]["arithmetic_mean"])
    # 平均谱周期点与提交一致
    assert m["average_spectrum"]["periods"][:3] == [0.05, 0.1, 0.2]
    assert np.all(np.isfinite(avg))


def test_match_group_scaled_group_factor_divides_by_k(app_worker):
    """整组乘 k，组匹配得到的缩放系数变为 1/k（端到端）。"""

    client = app_worker.test_client()

    def make_group(amp, tag):
        h1 = _upload(client, _record_text(seed=0, component="HNN"),
                     f"{tag}a.txt")
        h2 = _upload(client, _record_text(seed=1, component="HNE"),
                     f"{tag}b.txt")
        return client.post("/api/groups",
                           json={"h1_id": h1, "h2_id": h2}).get_json()["group_id"]

    g_base = make_group(1.0, "base_")

    # 用 base 组放大 k 倍的记录另建一组（数值乘 k 后重新上传）
    dt, n, k = 0.01, 1000, 2.0

    def upload_scaled(seed, comp, name):
        rng = np.random.default_rng(seed)
        t = np.arange(n) * dt
        a = k * (np.sin(2 * np.pi * 1.5 * t) + 0.3 * rng.normal(size=n))
        txt = f"# Station: ST2\n# Event: EV1\n# Component: {comp}\n"
        txt += "\n".join(f"{x:.6f}" for x in a)
        return _upload(client, txt, name, dt=dt)

    s1 = upload_scaled(0, "HNN", "ka.txt")
    s2 = upload_scaled(1, "HNE", "kb.txt")
    g_k = client.post("/api/groups",
                      json={"h1_id": s1, "h2_id": s2}).get_json()["group_id"]

    design = {"periods": [0.05, 0.1, 0.5, 1.0, 3.0],
              "values": [0.12, 0.18, 0.18, 0.10, 0.05], "unit": "g"}

    def run(gid):
        jid = client.post("/api/jobs/match-group", json={
            "group_ids": [gid],
            "periods": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0],
            "dampings": [0.05], "target_damping": 0.05,
            "design_spectrum": design,
            "t1": 0.1, "t2": 2.0, "s_min": 0.001, "s_max": 1000.0,
            "top_n": 1,
        }).get_json()["job_id"]
        body = _wait(client, jid)
        return body["job"]["result"]["matching"]["ranking"][0]["scale_optimal"]

    s0, sk = run(g_base), run(g_k)
    assert sk * k == pytest.approx(s0, rel=1e-4)


def test_cancel_rot_job(app_worker):
    client = app_worker.test_client()
    n = 120000
    big1 = _record_text(n=n, seed=9, component="HNN")
    big2 = _record_text(n=n, seed=10, component="HNE")
    h1 = _upload(client, big1, "big1.txt")
    h2 = _upload(client, big2, "big2.txt")
    gid = client.post("/api/groups",
                      json={"h1_id": h1, "h2_id": h2}).get_json()["group_id"]
    jid = client.post("/api/jobs/rot-spectrum", json={
        "group_ids": [gid],
        "periods": np.linspace(0.05, 6, 120).tolist(),
    }).get_json()["job_id"]
    assert client.post(f"/api/jobs/{jid}/cancel").status_code == 200
    body = _wait(client, jid)
    assert body["job"]["status"] == "cancelled"


def test_invalid_params_rejected_at_creation(client):
    h1 = _upload(client, _record_text(seed=0, component="HNN"), "a.txt")
    h2 = _upload(client, _record_text(seed=1, component="HNE"), "b.txt")
    gid = client.post("/api/groups",
                      json={"h1_id": h1, "h2_id": h2}).get_json()["group_id"]
    rv = client.post("/api/jobs/rot-spectrum", json={
        "group_ids": [gid], "periods": [0.1, 1.0], "n_angles": 3,
    })
    assert rv.status_code == 400 and "n_angles" in rv.get_json()["error"]
    rv = client.post("/api/jobs/match-group", json={
        "group_ids": [gid], "design_spectrum": {"periods": [0.1], "values": [0.1]},
        "t1": 0.1, "t2": 0.2, "rotd_band": "rotd99",
    })
    assert rv.status_code == 400


def test_group_batch_limit(client):
    h1 = _upload(client, _record_text(seed=0, component="HNN"), "a.txt")
    h2 = _upload(client, _record_text(seed=1, component="HNE"), "b.txt")
    gid = client.post("/api/groups",
                      json={"h1_id": h1, "h2_id": h2}).get_json()["group_id"]
    rv = client.post("/api/jobs/rot-spectrum",
                     json={"group_ids": [gid] * 201})
    assert rv.status_code == 400 and "200" in rv.get_json()["error"]


def test_groups_survive_restart(tmp_path):
    path = str(tmp_path / "g.db")
    from app import create_app

    app1 = create_app(path, start_worker=False)
    c1 = app1.test_client()
    h1 = _upload(c1, _record_text(seed=0, component="HNN"), "a.txt")
    h2 = _upload(c1, _record_text(seed=1, component="HNE"), "b.txt")
    gid = c1.post("/api/groups",
                  json={"h1_id": h1, "h2_id": h2}).get_json()["group_id"]
    app1.extensions["storage"].close()

    app2 = create_app(path, start_worker=False)
    c2 = app2.test_client()
    g = c2.get(f"/api/groups/{gid}").get_json()["group"]
    assert g["status"] == "ready" and g["npts"] == 1000
    app2.extensions["storage"].close()
