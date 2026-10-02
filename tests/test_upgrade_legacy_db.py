"""旧版本（v0）SQLite 库文件在线升级测试。

用**旧版本的原始建表语句**手工建一个老库（不含 group_id 列、jobs 的
CHECK 只允许 spectrum/match、没有 component_groups/rotd_cache 表），
塞入记录、两种作业、作业项及已完成结果，再用新版 Storage 直接打开，
验证：

- 升级自动完成，无报错；
- 老记录 / 老作业（含完成结果、时间戳）/ 老作业项字段原样可读；
- 老作业项的新增字段 group_id 为 None；
- 升级后仍可正常创建新的 rotd / match_group 作业并执行；
- 二次启动不重复迁移、数据不丢。
"""

from __future__ import annotations

import json
import time

import numpy as np

from app.scheduler import Scheduler
from app.storage import SCHEMA, Storage

# 与升级前版本完全一致的建表脚本（摘自升级前 storage.py）
LEGACY_SCHEMA = """
CREATE TABLE records (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    dt          REAL NOT NULL,
    npts        INTEGER NOT NULL,
    duration    REAL NOT NULL,
    unit_input  TEXT,
    fmt         TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    acc         BLOB NOT NULL,
    acc_dtype   TEXT NOT NULL DEFAULT 'float64',
    status      TEXT NOT NULL DEFAULT 'ready',
    error       TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE jobs (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL CHECK(type IN ('spectrum', 'match')),
    status      TEXT NOT NULL,
    params      TEXT NOT NULL DEFAULT '{}',
    progress    INTEGER NOT NULL DEFAULT 0,
    total       INTEGER NOT NULL DEFAULT 0,
    result      TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT
);
CREATE TABLE job_items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    record_id   TEXT REFERENCES records(id) ON DELETE SET NULL,
    seq         INTEGER NOT NULL,
    status      TEXT NOT NULL,
    result      TEXT,
    error       TEXT,
    UNIQUE(job_id, seq)
);
CREATE INDEX idx_items_job ON job_items(job_id, seq);
CREATE INDEX idx_jobs_status ON jobs(status);
"""


def _build_legacy_db(path: str) -> dict:
    import sqlite3

    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_SCHEMA)
    acc = np.linspace(0.0, 1.0, 200).astype(np.float64)
    conn.execute(
        """INSERT INTO records(id,name,dt,npts,duration,unit_input,fmt,
                               metadata,acc,acc_dtype,status,error,created_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        ("rec_old1", "old.txt", 0.01, 200, 1.99, "m/s2", "single_column",
         json.dumps({"station": "OLD"}), acc.tobytes(), "float64", "ready",
         None, "2026-09-01T00:00:00Z"),
    )
    # 一条已完成的 spectrum 作业，带完整结果
    spec_result = {"summary": {"done": 1, "error": 0, "skipped": 0},
                   "records": [{"record_id": "rec_old1",
                                "spectra": [{"damping": 0.05,
                                             "psa": [1.0, 2.0]}]}]}
    conn.execute(
        """INSERT INTO jobs(id,type,status,params,progress,total,result,
                           error,created_at,started_at,finished_at)
           VALUES('job_old_done','spectrum','completed',
                  '{"periods":[0.1,1.0],"dampings":[0.05]}',1,1,?,
                  NULL,'2026-09-01T00:01:00Z','2026-09-01T00:01:01Z',
                  '2026-09-01T00:01:02Z')""",
        (json.dumps(spec_result),),
    )
    # 一条中断态 match 作业（升级后 recovery 应标记 interrupted）
    conn.execute(
        """INSERT INTO jobs(id,type,status,params,progress,total,result,
                           error,created_at,started_at,finished_at)
           VALUES('job_old_run','match','running',
                  '{"t1":0.1,"t2":1.0}',0,1,NULL,NULL,
                  '2026-09-01T00:02:00Z','2026-09-01T00:02:01Z',NULL)""",
    )
    conn.execute(
        """INSERT INTO job_items(job_id,record_id,seq,status,result,error)
           VALUES('job_old_done','rec_old1',0,'done','{"sd":[0.1]}',NULL)""",
    )
    conn.execute(
        """INSERT INTO job_items(job_id,record_id,seq,status,result,error)
           VALUES('job_old_run','rec_old1',0,'running',NULL,NULL)""",
    )
    conn.commit()
    conn.close()
    return {"result": spec_result}


def test_legacy_db_opens_and_data_preserved(tmp_path):
    path = str(tmp_path / "legacy.db")
    expected = _build_legacy_db(path)

    # 旧库文件确为旧结构：没有 component_groups 表、job_items 无 group_id
    import sqlite3

    conn = sqlite3.connect(path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "component_groups" not in tables
    cols = {r[1] for r in conn.execute("PRAGMA table_info(job_items)")}
    assert "group_id" not in cols
    conn.close()

    storage = Storage(path)
    assert storage.get_job("job_old_done") is not None

    # 老记录完整
    rec = storage.get_record("rec_old1")
    assert rec.npts == 200 and rec.dt == 0.01
    assert rec.metadata["station"] == "OLD"
    np.testing.assert_array_equal(rec.acc, np.linspace(0, 1, 200))

    # 老作业：状态/字段/结果/时间戳不变
    done = storage.get_job("job_old_done")
    assert done["type"] == "spectrum"
    assert done["status"] == "completed"
    assert done["created_at"] == "2026-09-01T00:01:00Z"
    assert done["result"] == expected["result"]
    assert done["params"]["dampings"] == [0.05]

    items = storage.list_job_items("job_old_done")
    assert items[0]["record_id"] == "rec_old1"
    assert items[0]["group_id"] is None   # 新字段对老数据为空
    assert items[0]["result"] == {"sd": [0.1]}

    # running 的老作业在调度器恢复后标记 interrupted
    sch = Scheduler(storage)
    sch.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            runj = storage.get_job("job_old_run")
            if runj["status"] == "interrupted":
                break
            time.sleep(0.05)
        assert storage.get_job("job_old_run")["status"] == "interrupted"
    finally:
        sch.stop()

    # 升级后新功能可用：建组 + rotd 作业执行完成
    gid = _create_minimal_group(storage, acc=rec.acc)
    jid = "job_new_rotd"
    storage.create_job(jid, "rotd", {"periods": [0.1, 0.5, 1.0],
                                     "dampings": [0.05], "n_angles": 180},
                       total=1)
    storage.add_job_items(jid, [(None, gid, 0)])
    sch2 = Scheduler(storage)
    sch2.start()
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            j = storage.get_job(jid)
            if j["status"] == "completed":
                break
            assert j["status"] in ("queued", "running")
            time.sleep(0.05)
        assert storage.get_job(jid)["status"] == "completed"
        assert storage.get_job(jid)["result"]["summary"]["done"] == 1
    finally:
        sch2.stop()

    storage.close()

    # 二次启动：不再迁移，所有数据仍在
    s2 = Storage(path)
    assert s2.get_job("job_old_done")["status"] == "completed"
    assert s2.get_job(jid)["status"] == "completed"
    assert s2.get_group(gid) is not None
    s2.close()


def _create_minimal_group(storage: Storage, *, acc: np.ndarray) -> str:
    storage.upsert_record(
        "rec_ng1", name="g1.txt", dt=0.01, unit_input="m/s2",
        fmt="single_column", metadata={"component": "N"},
        acc=acc, status="ready",
    )
    storage.upsert_record(
        "rec_ng2", name="g2.txt", dt=0.01, unit_input="m/s2",
        fmt="single_column", metadata={"component": "E"},
        acc=acc * 0.5, status="ready",
    )
    from app.groups import group_id, group_identity_key

    gid = group_id(("rec_ng1", "rec_ng2"), None)
    key = group_identity_key(("rec_ng1", "rec_ng2"), None)
    storage.upsert_group(
        gid, horizontal1="rec_ng1", horizontal2="rec_ng2", vertical=None,
        content_key=key, name="g1+g2", dt=0.01, t0=0.0, npts=acc.size,
        duration=0.01 * (acc.size - 1), alignment={"policy": "legacy_test"},
    )
    return gid
