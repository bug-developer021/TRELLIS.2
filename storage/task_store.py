import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    status: str
    params: Dict[str, Any]
    created_at: float
    result_path: Optional[str]
    owner: str
    preview_size: int
    preview_mode: str
    canceled: bool


class TaskStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    params TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    result_path TEXT,
                    owner TEXT NOT NULL,
                    preview_size INTEGER NOT NULL,
                    preview_mode TEXT NOT NULL,
                    canceled INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, check_same_thread=False)

    def create_task(
        self,
        task_id: str,
        status: str,
        params: Dict[str, Any],
        owner: str,
        preview_size: int,
        preview_mode: str,
    ) -> TaskRecord:
        created_at = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    task_id, status, params, created_at, result_path, owner, preview_size, preview_mode, canceled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    status,
                    json.dumps(params),
                    created_at,
                    None,
                    owner,
                    preview_size,
                    preview_mode,
                    0,
                ),
            )
            conn.commit()
        return TaskRecord(
            task_id=task_id,
            status=status,
            params=params,
            created_at=created_at,
            result_path=None,
            owner=owner,
            preview_size=preview_size,
            preview_mode=preview_mode,
            canceled=False,
        )

    def update_status(self, task_id: str, status: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE tasks SET status = ? WHERE task_id = ?",
                (status, task_id),
            )
            conn.commit()

    def mark_canceled(self, task_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE tasks SET status = ?, canceled = 1 WHERE task_id = ?",
                ("canceled", task_id),
            )
            conn.commit()

    def set_result_path(self, task_id: str, result_path: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE tasks SET result_path = ? WHERE task_id = ?",
                (result_path, task_id),
            )
            conn.commit()

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return TaskRecord(
            task_id=row["task_id"],
            status=row["status"],
            params=json.loads(row["params"]),
            created_at=row["created_at"],
            result_path=row["result_path"],
            owner=row["owner"],
            preview_size=row["preview_size"],
            preview_mode=row["preview_mode"],
            canceled=bool(row["canceled"]),
        )

    def delete_task(self, task_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            conn.commit()

    def list_expired(self, ttl_seconds: int, now: Optional[float] = None) -> Iterable[TaskRecord]:
        threshold = (now or time.time()) - ttl_seconds
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM tasks WHERE created_at < ?",
                (threshold,),
            ).fetchall()
        for row in rows:
            yield TaskRecord(
                task_id=row["task_id"],
                status=row["status"],
                params=json.loads(row["params"]),
                created_at=row["created_at"],
                result_path=row["result_path"],
                owner=row["owner"],
                preview_size=row["preview_size"],
                preview_mode=row["preview_mode"],
                canceled=bool(row["canceled"]),
            )

    def count_active_by_owner(self, owner: str) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM tasks
                WHERE owner = ? AND status IN ("queued", "running")
                """,
                (owner,),
            ).fetchone()
        return int(row[0]) if row else 0
