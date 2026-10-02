"""旧版本库文件升级启动测试（需求硬性要求一条）。

用**旧版本结构**（升级前的三张表、jobs.type 的旧 CHECK、job_items 无
group_id 列）手工建一个库，写入记录、完成的/排队的/执行中的作业与逐条
结果，然后：

1. 用新版 :class:`app.storage.Storage` 直接打开同一库文件，不报错、
   老记录/老作业/老逐条结果字段与值原样可读；
2. running 作业照常被标记 interrupted、queued 照常排队；
3. 新版能力（分量组、组合谱缓存、两类组作业）能在升级后的库里正常使用；
4. 旧版单条作业接口返回结构不变。
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np

# 升级前的建表语句（与旧版本 storage.py 完全一致，作为升级测试基线）
LEGACY_SCHEMA = """
CREATE TABLE records (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, dt REAL NOT NULL,
    npts INTEGER NOT NULL, duration REAL NOT NULL, unit_input TEXT,
    fmt TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}',
    acc BLOB NOT NULL, acc_dtype TEXT NOT NULL DEFAULT 'float64',
    status TEXT NOT NULL DEFAULT 'ready', error TEXT, created_at TEXT NOT NULL
);
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('spectrum', 'match')),
    status TEXT NOT NULL, params TEXT NOT NULL DEFAULT '{}',
    progress INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL DEFAULT 0,
    result TEXT, error TEXT, created_at TEXT NOT NULL,
    started_at TEXT, finished_at TEXT
);
CREATE TABLE job_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    record_id TEXT REFERENCES records(id) ON DELETE SET NULL,
    seq INTEGER NOT NULL, status TEXT NOT NULL, result TEXT, error TEXT,
    UNIQUE(job_id, seq)
);
CREATE INDEX idx_items_job ON job_items(job_id, seq);
CREATE INDEX idx_jobs_status ON jobs(status);
"""


def _build_legacy_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_SCHEMA)
    acc = np.sin(np.linspace(0, 8, 400)).astype(np.float64)
    conn.execute(
        """INSERT INTO records(id,name,dt,npts,duration,unit_input,fmt,metadata,
                               acc,acc_dtype,status,created_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
        ("rec_legacy1", "old.txt", 0.02, 400, 0.02 * 399, "m/s2",
         "single_column", json.dumps({"station": "OLD"}),
         acc.tobytes(), "float64", "ready", "2025-01-01T00:00:00Z"),
    )
    # 已完成的旧 spectrum 作业，含大结果体
    conn.execute(
        """INSERT INTO jobs(id,type,status,params,progress,total,result,
                            created_at,started_at,finished_at)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        ("job_old_done", "spectrum", "completed",
         json.dumps({"periods": [0.1, 1.0], "dampings": [0.05]}), 1, 1,
         json.dumps({"summary": {"done": 1},
                     "records": [{"record_id": "rec_legacy1",
                                  "spectra": [{"damping": 0.05,
                                               "psa": [0.1, 0.2]}]}]}),
         "2025-01-01T00:01:00Z", "2025-01-01T00:01:01Z",
         "2025-01-01T00:01:02Z"),
    )
    conn.execute(
        """INSERT INTO job_items(job_id,record_id,seq,status,result)
           VALUES(?,?,?,?,?)""",
        ("job_old_done", "rec_legacy1", 0, "done",
         json.dumps({"spectra": [{"psa": [0.1, 0.2]}]})),
    )
    # 执行中的旧作业（重启应 interrupted）
    conn.execute(
        """INSERT INTO jobs(id,type,status,params,progress,total,created_at,
                            started_at)
           VALUES('job_old_run','spectrum','running','{}',0,1,
                  '2025-01-01T00:02:00Z','2025-01-01T00:02:01Z')""",
    )
    conn.execute(
        "INSERT INTO job_items(job_id,record_id,seq,status) "
        "VALUES('job_old_run','rec_legacy1',0,'running')"
    )
    # 排队中的旧 match 作业（重启后应能继续被领取）
    conn.execute(
        """INSERT INTO jobs(id,type,status,params,progress,total,created_at)
           VALUES('job_old_queued','match','queued','{}',0,1,
                  '2025-01-01T00:03:00Z')""",
    )
    conn.execute(
        "INSERT INTO job_items(job_id,record_id,seq,status) "
        "VALUES('job_old_queued','rec_legacy1',0,'pending')"
    )
    conn.commit()
    conn.close()


def test_upgrade_keeps_legacy_data_and_enables_new_features(tmp_path):
    path = str(tmp_path / "legacy.db")
    _build_legacy_db(path)

    # 新版代码直接打开旧库
    from app.storage import Storage

    storage = Storage(path)
    interrupted = storage.recover_interrupted()
    assert interrupted == ["job_old_run"]

    # 老记录：含 BLOB，字段值不变
    rec = storage.get_record("rec_legacy1")
    assert rec is not None
    assert rec.name == "old.txt"
    assert rec.dt == 0.02 and rec.npts == 400
    assert rec.metadata == {"station": "OLD"}
    assert rec.acc.shape == (400,)
    np.testing.assert_allclose(rec.acc, np.sin(np.linspace(0, 8, 400)))

    # 老作业：状态机与结果原样可读
    done = storage.get_job("job_old_done")
    assert done["type"] == "spectrum"
    assert done["status"] == "completed"
    assert done["params"] == {"periods": [0.1, 1.0], "dampings": [0.05]}
    assert done["result"]["summary"] == {"done": 1}
    items = storage.list_job_items("job_old_done")
    assert items[0]["status"] == "done"
    assert items[0]["group_id"] is None  # 新列在老数据上为空
    assert storage.get_job("job_old_run")["status"] == "interrupted"
    # 老 queued 作业在新调度器下可被领取
    assert storage.claim_next_queued() == "job_old_queued"

    # 升级后新结构可用：建组 + 组合谱缓存写入同库
    h2 = (np.cos(np.linspace(0, 6, 400))).astype(np.float64)
    storage.upsert_record(
        "rec_legacy2", name="old2.txt", dt=0.02, unit_input="m/s2",
        fmt="single_column", metadata={"station": "OLD", "component": "HNE"},
        acc=h2, status="ready",
    )
    from app.groups import build_group

    gid, *_ = build_group(storage, "rec_legacy1", "rec_legacy2")
    assert storage.get_group(gid)["status"] == "ready"
    storage.put_rot_spectrum("rot_dummy", gid, {"n_angles": 180}, {"ok": True})
    assert storage.get_rot_spectrum("rot_dummy") == {"ok": True}

    # 再开一次（模拟又一次重启），老数据与新数据都还在
    storage.close()
    s2 = Storage(path)
    assert s2.get_group(gid)["npts"] == 400
    assert s2.get_rot_spectrum("rot_dummy") == {"ok": True}
    assert s2.get_job("job_old_done")["status"] == "completed"
    s2.close()


def test_upgrade_via_flask_client_legacy_endpoints_unchanged(tmp_path):
    path = str(tmp_path / "legacy2.db")
    _build_legacy_db(path)

    from app import create_app

    app = create_app(path, start_worker=False)
    client = app.test_client()
    # 老接口路径、返回结构不变
    rv = client.get("/api/records")
    rec = next(r for r in rv.get_json()["records"] if r["id"] == "rec_legacy1")
    assert rec["name"] == "old.txt" and rec["status"] == "ready"

    rv = client.get("/api/jobs/job_old_done")
    job = rv.get_json()["job"]
    assert job["type"] == "spectrum" and job["status"] == "completed"
    assert job["result"]["records"][0]["spectra"][0]["psa"] == [0.1, 0.2]

    items = client.get("/api/jobs/job_old_done/items").get_json()
    assert items["items"][0]["record_id"] == "rec_legacy1"
    assert items["counts"]["done"] == 1

    # 老作业依然允许新建单条 spectrum 作业（旧 CHECK 已放宽但不破坏旧值）
    rv = client.post("/api/jobs/spectrum", json={
        "record_ids": ["rec_legacy1"], "periods": [0.1, 1.0],
    })
    assert rv.status_code == 201
    app.extensions["scheduler"].stop()
    app.extensions["storage"].close()
