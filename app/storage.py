"""SQLite 持久化：记录、作业、逐条作业项。

库文件放在挂载卷（环境变量 ``SEISMIC_DB``，默认 ``/data/seismic.db``）。
采用 WAL 模式配合单进程 gunicorn + 后台调度线程；每个线程使用独立
连接（``check_same_thread=False`` + 线程内独占连接），写操作串行化。

作业状态机：``queued -> running -> completed / failed / cancelled``；
异常中断（进程退出）后重启时由 :meth:`Storage.recover_interrupted`
把 ``running`` 标记为 ``interrupted``，``queued`` 保持排队等待重跑。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
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

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL CHECK(type IN ('spectrum', 'match',
                                             'rot_spectrum', 'match_group')),
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

-- 分量组：两条水平分量对齐后的序列直接落库（BLOB），重启后无需重新对齐，
-- 也保证重复计算逐位一致。竖向分量只存引用，不参与水平组合。
CREATE TABLE IF NOT EXISTS component_groups (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    station     TEXT,
    event       TEXT,
    h1_id       TEXT NOT NULL REFERENCES records(id),
    h2_id       TEXT NOT NULL REFERENCES records(id),
    vertical_id TEXT REFERENCES records(id),
    dt          REAL NOT NULL,
    npts        INTEGER NOT NULL,
    duration    REAL NOT NULL,
    alignment   TEXT NOT NULL DEFAULT '{}',
    h1          BLOB NOT NULL,
    h2          BLOB NOT NULL,
    status      TEXT NOT NULL DEFAULT 'ready',
    error       TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    record_id   TEXT REFERENCES records(id) ON DELETE SET NULL,
    group_id    TEXT REFERENCES component_groups(id) ON DELETE SET NULL,
    seq         INTEGER NOT NULL,
    status      TEXT NOT NULL,
    result      TEXT,
    error       TEXT,
    UNIQUE(job_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_items_job ON job_items(job_id, seq);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

-- 组合谱结果缓存：rot_key 由分量组（对齐后序列）+ 全部积分参数 +
-- 计算版本内容寻址，组成员或参数变化必然得到不同 key，绝不会拿到旧结果。
CREATE TABLE IF NOT EXISTS rot_spectra (
    rot_key     TEXT PRIMARY KEY,
    group_id    TEXT NOT NULL REFERENCES component_groups(id),
    params      TEXT NOT NULL DEFAULT '{}',
    result      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rot_group ON rot_spectra(group_id);
"""


def _now() -> str:
    # 统一 UTC ISO8601，毫秒；仅用于展示，不参与结果内容
    import datetime as _dt

    return (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def json_dumps(obj: Any) -> str:
    """JSON 序列化，原生支持 NumPy 标量/数组，键排序保证确定性。"""

    return json.dumps(obj, default=_json_default, sort_keys=True, ensure_ascii=False)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"不可 JSON 序列化的类型：{type(o)!r}")


def _encode_acc(acc: np.ndarray) -> tuple[bytes, str]:
    a = np.ascontiguousarray(acc, dtype=np.float64)
    return a.tobytes(), str(a.dtype)


def _decode_acc(blob: bytes, dtype: str = "float64") -> np.ndarray:
    return np.frombuffer(blob, dtype=np.dtype(dtype)).copy()


@dataclass
class RecordRow:
    id: str
    name: str
    dt: float
    npts: int
    duration: float
    unit_input: str | None
    fmt: str
    metadata: dict
    acc: np.ndarray
    status: str
    error: str | None
    created_at: str


class Storage:
    """线程安全的 SQLite 封装（单进程使用）。"""

    def __init__(self, path: str) -> None:
        self.path = path
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        self._lock = threading.RLock()
        self._tls = threading.local()
        with self._lock:
            conn = self._connect()
            self._migrate(conn)
            conn.executescript(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """旧版本库文件（无 component_groups 表）的增量升级。

        只做加法、不改写任何历史数据：老表字段保持原样，jobs 的
        type 约束放宽到新增的两类作业（重建表、拷数据），job_items
        新增可空 group_id 列。老作业/老结果升级后照常可查。
        """

        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "component_groups" in tables:
            return  # 已是新版本结构（新库由 SCHEMA 直接建全）

        # 老库：先建分量组表（job_items.group_id 外键依赖它）
        conn.execute(
            """CREATE TABLE IF NOT EXISTS component_groups (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, station TEXT, event TEXT,
                h1_id TEXT NOT NULL REFERENCES records(id),
                h2_id TEXT NOT NULL REFERENCES records(id),
                vertical_id TEXT REFERENCES records(id),
                dt REAL NOT NULL, npts INTEGER NOT NULL, duration REAL NOT NULL,
                alignment TEXT NOT NULL DEFAULT '{}',
                h1 BLOB NOT NULL, h2 BLOB NOT NULL,
                status TEXT NOT NULL DEFAULT 'ready', error TEXT,
                created_at TEXT NOT NULL)"""
        )
        # jobs：放宽 type 的 CHECK 约束（SQLite 不能直接改约束，重建表）
        has_new_check = False
        if "jobs" in tables:
            sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='jobs'"
            ).fetchone()[0]
            has_new_check = "rot_spectrum" in sql
        if "jobs" in tables and not has_new_check:
            # 重建 jobs 时其外键被 job_items 引用：重命名旧表会让旧外键
            # 残留在被 DROP 的 jobs_legacy 上，故 jobs 与 job_items 必须
            # 一起重建拷数据。整个过程关闭外键强制。
            self.conn.execute("PRAGMA foreign_keys=OFF")
            self.conn.execute("BEGIN")
            self.conn.execute("ALTER TABLE jobs RENAME TO jobs_legacy")
            self.conn.execute("ALTER TABLE job_items RENAME TO job_items_legacy")
            self.conn.execute(
                """CREATE TABLE jobs (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL CHECK(type IN ('spectrum','match',
                                                      'rot_spectrum','match_group')),
                    status TEXT NOT NULL, params TEXT NOT NULL DEFAULT '{}',
                    progress INTEGER NOT NULL DEFAULT 0,
                    total INTEGER NOT NULL DEFAULT 0, result TEXT, error TEXT,
                    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT)"""
            )
            self.conn.execute(
                """INSERT INTO jobs(id, type, status, params, progress, total,
                                    result, error, created_at, started_at,
                                    finished_at)
                   SELECT id, type, status, params, progress, total, result,
                          error, created_at, started_at, finished_at
                   FROM jobs_legacy"""
            )
            self.conn.execute(
                """CREATE TABLE job_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    record_id TEXT REFERENCES records(id) ON DELETE SET NULL,
                    group_id TEXT REFERENCES component_groups(id) ON DELETE SET NULL,
                    seq INTEGER NOT NULL, status TEXT NOT NULL,
                    result TEXT, error TEXT, UNIQUE(job_id, seq))"""
            )
            self.conn.execute(
                """INSERT INTO job_items(id, job_id, record_id, group_id, seq,
                                         status, result, error)
                   SELECT id, job_id, record_id, NULL, seq, status, result, error
                   FROM job_items_legacy"""
            )
            self.conn.execute("DROP TABLE job_items_legacy")
            self.conn.execute("DROP TABLE jobs_legacy")
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_items_job ON job_items(job_id, seq)"
            )
            self.conn.execute("COMMIT")
            self.conn.execute("PRAGMA foreign_keys=ON")
        elif "job_items" in tables:
            cols = {r[1] for r in self.conn.execute("PRAGMA table_info(job_items)")}
            if "group_id" not in cols:
                self.conn.execute(
                    "ALTER TABLE job_items ADD COLUMN group_id TEXT "
                    "REFERENCES component_groups(id) ON DELETE SET NULL"
                )

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._tls, "conn", None)
        if c is None:
            c = self._connect()
            self._tls.conn = c
        return c

    # ---------------- 记录 ----------------

    def upsert_record(
        self,
        record_id: str,
        *,
        name: str,
        dt: float,
        unit_input: str | None,
        fmt: str,
        metadata: dict,
        acc: np.ndarray | None,
        status: str,
        error: str | None = None,
    ) -> None:
        blob, dtype = (b"", "float64") if acc is None else _encode_acc(acc)
        with self._lock:
            self.conn.execute(
                """INSERT INTO records(id, name, dt, npts, duration, unit_input,
                                      fmt, metadata, acc, acc_dtype, status,
                                      error, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     name=excluded.name, dt=excluded.dt, npts=excluded.npts,
                     duration=excluded.duration, unit_input=excluded.unit_input,
                     fmt=excluded.fmt, metadata=excluded.metadata,
                     acc=excluded.acc, acc_dtype=excluded.acc_dtype,
                     status=excluded.status, error=excluded.error""",
                (
                    record_id, name, float(dt), 0 if acc is None else acc.size,
                    0.0 if acc is None else float(dt * (acc.size - 1)),
                    unit_input, fmt, json_dumps(metadata or {}),
                    blob, dtype, status, error, _now(),
                ),
            )
            self.conn.commit()

    def get_record(self, record_id: str) -> RecordRow | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM records WHERE id=?", (record_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_records(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """SELECT id, name, dt, npts, duration, unit_input, fmt,
                          metadata, status, error, created_at
                   FROM records ORDER BY id"""
            ).fetchall()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "dt": r["dt"],
                "npts": r["npts"],
                "duration": r["duration"],
                "unit_input": r["unit_input"],
                "fmt": r["fmt"],
                "metadata": json.loads(r["metadata"]),
                "status": r["status"],
                "error": r["error"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def _row_to_record(self, row: sqlite3.Row) -> RecordRow:
        return RecordRow(
            id=row["id"],
            name=row["name"],
            dt=row["dt"],
            npts=row["npts"],
            duration=row["duration"],
            unit_input=row["unit_input"],
            fmt=row["fmt"],
            metadata=json.loads(row["metadata"]),
            acc=_decode_acc(row["acc"], row["acc_dtype"]) if row["acc"] else np.zeros(0),
            status=row["status"],
            error=row["error"],
            created_at=row["created_at"],
        )

    # ---------------- 作业 ----------------

    def create_job(
        self,
        job_id: str,
        job_type: str,
        params: dict,
        total: int,
    ) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT INTO jobs(id, type, status, params, progress, total,
                                    created_at)
                   VALUES(?,?, 'queued', ?, 0, ?, ?)""",
                (job_id, job_type, json_dumps(params), int(total), _now()),
            )
            self.conn.commit()

    def add_job_items(self, job_id: str, entries: Iterable[tuple]) -> None:
        """entries: ``(record_id 或 group_id, seq)`` 或
        ``(record_id, group_id, seq)``。由作业类型决定外键落在哪一列。
        """

        rows = []
        for e in entries:
            rid, gid, seq = e if len(e) == 3 else (e[0], None, e[1])
            rows.append((job_id, rid, gid, seq))
        with self._lock:
            self.conn.executemany(
                """INSERT INTO job_items(job_id, record_id, group_id, seq, status)
                   VALUES(?, ?, ?, ?, 'pending')""",
                rows,
            )
            self.conn.commit()

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        return self._job_dict(row) if row else None

    def get_job_params(self, job_id: str) -> dict:
        with self._lock:
            row = self.conn.execute(
                "SELECT params FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return json.loads(row["params"])

    def list_jobs(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [self._job_dict(r, with_result=False) for r in rows]

    def _job_dict(self, row: sqlite3.Row, *, with_result: bool = True) -> dict:
        d = {
            "id": row["id"],
            "type": row["type"],
            "status": row["status"],
            "params": json.loads(row["params"]),
            "progress": row["progress"],
            "total": row["total"],
            "error": row["error"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        }
        if with_result:
            d["result"] = json.loads(row["result"]) if row["result"] else None
        return d

    def claim_next_queued(self) -> str | None:
        """原子领取一个排队作业并置为 running；无则返回 None。"""

        with self._lock:
            row = self.conn.execute(
                """SELECT id FROM jobs WHERE status='queued'
                   ORDER BY created_at ASC, id ASC LIMIT 1"""
            ).fetchone()
            if row is None:
                return None
            self.conn.execute(
                "UPDATE jobs SET status='running', started_at=? WHERE id=?",
                (_now(), row["id"]),
            )
            self.conn.commit()
            return row["id"]

    def request_cancel(self, job_id: str) -> bool:
        """请求取消：queued 直接取消，running 由执行线程轮询后收尾。

        返回 True 表示作业存在且状态被更新（已完成的返回 False）。
        """

        with self._lock:
            row = self.conn.execute(
                "SELECT status FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None:
                return False
            if row["status"] in ("completed", "failed", "cancelled", "interrupted"):
                return False
            self.conn.execute(
                "UPDATE jobs SET status='cancelled', finished_at=? WHERE id=?",
                (_now(), job_id),
            )
            self.conn.commit()
            return True

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT status FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        return row is not None and row["status"] == "cancelled"

    def complete_job(self, job_id: str, result: dict) -> None:
        with self._lock:
            self.conn.execute(
                """UPDATE jobs SET status='completed', progress=total,
                                   result=?, finished_at=? WHERE id=?""",
                (json_dumps(result), _now(), job_id),
            )
            self.conn.commit()

    def fail_job(self, job_id: str, error: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE jobs SET status='failed', error=?, finished_at=? WHERE id=?",
                (error, _now(), job_id),
            )
            self.conn.commit()

    def finish_cancelled(self, job_id: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE jobs SET finished_at=? WHERE id=? AND status='cancelled'",
                (_now(), job_id),
            )
            self.conn.commit()

    def set_job_progress(self, job_id: str, progress: int) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE jobs SET progress=? WHERE id=?", (int(progress), job_id)
            )
            self.conn.commit()

    def recover_interrupted(self) -> list[str]:
        """进程启动时调用：running -> interrupted，返回受影响作业 id。"""

        with self._lock:
            rows = self.conn.execute(
                "SELECT id FROM jobs WHERE status='running'"
            ).fetchall()
            ids = [r["id"] for r in rows]
            self.conn.executemany(
                "UPDATE jobs SET status='interrupted', error=?, finished_at=? WHERE id=?",
                [
                    ("进程重启，作业在执行中中断，结果不可用；请重新提交", _now(), i)
                    for i in ids
                ],
            )
            self.conn.commit()
        return ids

    # ---------------- 作业项 ----------------

    def list_job_items(self, job_id: str) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """SELECT seq, record_id, group_id, status, result, error
                   FROM job_items WHERE job_id=? ORDER BY seq""",
                (job_id,),
            ).fetchall()
        return [
            {
                "seq": r["seq"],
                "record_id": r["record_id"],
                "group_id": r["group_id"],
                "status": r["status"],
                "result": json.loads(r["result"]) if r["result"] else None,
                "error": r["error"],
            }
            for r in rows
        ]

    def set_item_running(self, job_id: str, seq: int) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE job_items SET status='running' WHERE job_id=? AND seq=?",
                (job_id, seq),
            )
            self.conn.commit()

    def set_item_result(self, job_id: str, seq: int, result: dict) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE job_items SET status='done', result=? WHERE job_id=? AND seq=?",
                (json_dumps(result), job_id, seq),
            )
            self.conn.commit()

    def set_item_error(self, job_id: str, seq: int, error: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE job_items SET status='error', error=? WHERE job_id=? AND seq=?",
                (error, job_id, seq),
            )
            self.conn.commit()

    def set_item_skipped(self, job_id: str, seq: int, error: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE job_items SET status='skipped', error=? WHERE job_id=? AND seq=?",
                (error, job_id, seq),
            )
            self.conn.commit()

    def counts_by_status(self, job_id: str) -> dict[str, int]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT status, COUNT(*) c FROM job_items WHERE job_id=? GROUP BY status",
                (job_id,),
            ).fetchall()
        counts = {r["status"]: r["c"] for r in rows}
        counts.setdefault("pending", 0)
        counts.setdefault("running", 0)
        counts.setdefault("done", 0)
        counts.setdefault("error", 0)
        counts.setdefault("skipped", 0)
        return counts

    # ---------------- 分量组 ----------------

    def create_group(
        self,
        group_id: str,
        *,
        name: str,
        station: str | None,
        event: str | None,
        h1_id: str,
        h2_id: str,
        vertical_id: str | None,
        dt: float,
        alignment: dict,
        h1: np.ndarray,
        h2: np.ndarray,
    ) -> None:
        """新建分量组；对齐后的两条水平序列（m/s²、公共网格）直接落库。"""

        with self._lock:
            self.conn.execute(
                """INSERT INTO component_groups(
                       id, name, station, event, h1_id, h2_id, vertical_id,
                       dt, npts, duration, alignment, h1, h2, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    group_id, name, station, event, h1_id, h2_id, vertical_id,
                    float(dt), int(h1.size), float(dt * (h1.size - 1)),
                    json_dumps(alignment),
                    _encode_acc(h1)[0], _encode_acc(h2)[0], _now(),
                ),
            )
            self.conn.commit()

    def get_group(self, group_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM component_groups WHERE id=?", (group_id,)
            ).fetchone()
        return self._group_row(row, with_acc=True) if row else None

    def list_groups(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """SELECT id, name, station, event, h1_id, h2_id, vertical_id,
                          dt, npts, duration, alignment, status, error, created_at
                   FROM component_groups ORDER BY created_at, id"""
            ).fetchall()
        return [self._group_row(r, with_acc=False) for r in rows]

    def _group_row(self, row: sqlite3.Row, *, with_acc: bool) -> dict:
        d = {
            "id": row["id"],
            "name": row["name"],
            "station": row["station"],
            "event": row["event"],
            "h1_id": row["h1_id"],
            "h2_id": row["h2_id"],
            "vertical_id": row["vertical_id"],
            "dt": row["dt"],
            "npts": row["npts"],
            "duration": row["duration"],
            "alignment": json.loads(row["alignment"]),
            "status": row["status"],
            "error": row["error"],
            "created_at": row["created_at"],
        }
        if with_acc:
            keys = row.keys()
            if "h1" in keys and row["npts"]:
                d["h1"] = _decode_acc(row["h1"])
                d["h2"] = _decode_acc(row["h2"])
            else:
                d["h1"] = np.zeros(0)
                d["h2"] = np.zeros(0)
        return d

    # ---------------- 组合谱缓存 ----------------

    def get_rot_spectrum(self, rot_key: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT result FROM rot_spectra WHERE rot_key=?", (rot_key,)
            ).fetchone()
        return json.loads(row["result"]) if row else None

    def put_rot_spectrum(
        self, rot_key: str, group_id: str, params: dict, result: dict
    ) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT INTO rot_spectra(rot_key, group_id, params, result,
                                            created_at)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(rot_key) DO UPDATE SET result=excluded.result""",
                (rot_key, group_id, json_dumps(params),
                 json_dumps(result), _now()),
            )
            self.conn.commit()

    def close(self) -> None:
        c = getattr(self._tls, "conn", None)
        if c is not None:
            c.close()
            self._tls.conn = None
