"""SQLite persistence for vFL experiments.

Schema is multi-user from day one; every row of experiment data is scoped by
``user_id``. Location defaults to ``./.velocity/experiments.db`` (gitignored).
Override with ``VFL_DB_PATH``.

Separation of concerns:
  - This module owns **experiment episodic memory** (runs, rounds, attacks,
    hypotheses, agent_actions).
  - ``velocity.memory`` owns **per-user semantic memory** (profile, recipes,
    style) as transparent markdown files plus an event ledger.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(".velocity") / "experiments.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,
    display_name TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(user_id),
    git_sha      TEXT,
    strategy     TEXT NOT NULL,
    model_id     TEXT NOT NULL,
    dataset      TEXT,
    seed         INTEGER,
    min_clients  INTEGER,
    rounds       INTEGER,
    config_json  TEXT,
    status       TEXT NOT NULL DEFAULT 'running',
    started_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_runs_user_time ON runs(user_id, started_at DESC);

CREATE TABLE IF NOT EXISTS rounds (
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    round_num    INTEGER NOT NULL,
    global_loss  REAL,
    num_clients  INTEGER,
    duration_ms  INTEGER,
    timestamp    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, round_num)
);

CREATE TABLE IF NOT EXISTS attacks (
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    round_num    INTEGER NOT NULL,
    attack_type  TEXT NOT NULL,
    params_json  TEXT,
    result_json  TEXT,
    PRIMARY KEY (run_id, round_num, attack_type)
);

CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL REFERENCES users(user_id),
    statement     TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'active',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_hyp_user ON hypotheses(user_id, status);

CREATE TABLE IF NOT EXISTS hypothesis_run_link (
    hypothesis_id INTEGER NOT NULL REFERENCES hypotheses(hypothesis_id),
    run_id        TEXT    NOT NULL REFERENCES runs(run_id),
    relationship  TEXT    NOT NULL,  -- 'tests' | 'supports' | 'refutes'
    PRIMARY KEY (hypothesis_id, run_id)
);

CREATE TABLE IF NOT EXISTS agent_actions (
    action_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        TEXT NOT NULL REFERENCES users(user_id),
    session_id     TEXT,
    timestamp      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_prompt    TEXT,
    tool           TEXT,
    args_json      TEXT,
    result_summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_actions_user_time ON agent_actions(user_id, timestamp DESC);
"""


def db_path() -> Path:
    return Path(os.environ.get("VFL_DB_PATH", DEFAULT_DB_PATH))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def init_db(path: Path | None = None) -> Path:
    path = path or db_path()
    _ensure_parent(path)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
    return path


_LOCAL = threading.local()


def _shared_connection() -> sqlite3.Connection:
    conn: sqlite3.Connection | None = getattr(_LOCAL, "conn", None)
    if conn is not None:
        return conn
    path = db_path()
    _ensure_parent(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # WAL enables concurrent readers + one writer; much less fsync churn than
    # the default rollback journal. isolation_level=None hands transaction
    # control to our context manager (explicit commit / rollback below).
    conn.isolation_level = None
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    _LOCAL.conn = conn
    return conn


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    # Explicit path → short-lived connection (tests / migrations).
    if path is not None:
        _ensure_parent(path)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
        return

    conn = _shared_connection()
    conn.execute("BEGIN")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def ensure_user(user_id: str, display_name: str | None = None) -> None:
    with connect() as c:
        c.execute(
            "INSERT OR IGNORE INTO users(user_id, display_name) VALUES (?, ?)",
            (user_id, display_name or user_id),
        )


def start_run(user_id: str, config: dict[str, Any]) -> str:
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    ensure_user(user_id)
    with connect() as c:
        c.execute(
            """INSERT INTO runs(run_id, user_id, git_sha, strategy, model_id,
                                dataset, seed, min_clients, rounds, config_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                user_id,
                config.get("git_sha"),
                config["strategy"],
                config["model_id"],
                config.get("dataset"),
                config.get("seed"),
                config.get("min_clients"),
                config.get("rounds"),
                json.dumps(config),
            ),
        )
    return run_id


def record_round(run_id: str, summary: dict[str, Any]) -> None:
    with connect() as c:
        c.execute(
            """INSERT OR REPLACE INTO rounds(run_id, round_num, global_loss,
                                             num_clients, duration_ms)
               VALUES (?, ?, ?, ?, ?)""",
            (
                run_id,
                summary["round"],
                summary.get("global_loss"),
                summary.get("num_clients"),
                summary.get("duration_ms"),
            ),
        )
        for a in summary.get("attack_results") or []:
            c.execute(
                """INSERT OR REPLACE INTO attacks(run_id, round_num, attack_type,
                                                  params_json, result_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    run_id,
                    summary["round"],
                    a["attack_type"],
                    json.dumps(a.get("params")),
                    json.dumps(a),
                ),
            )


def complete_run(run_id: str, status: str = "complete") -> None:
    with connect() as c:
        c.execute(
            "UPDATE runs SET status = ?, completed_at = CURRENT_TIMESTAMP WHERE run_id = ?",
            (status, run_id),
        )


def log_action(
    user_id: str,
    session_id: str | None,
    tool: str,
    args: dict[str, Any],
    user_prompt: str | None = None,
    result_summary: str | None = None,
) -> None:
    ensure_user(user_id)
    with connect() as c:
        c.execute(
            """INSERT INTO agent_actions(user_id, session_id, user_prompt,
                                         tool, args_json, result_summary)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, session_id, user_prompt, tool, json.dumps(args), result_summary),
        )


def recent_runs(user_id: str, limit: int = 10) -> list[dict[str, Any]]:
    with connect() as c:
        rows = c.execute(
            """SELECT run_id, strategy, model_id, status, started_at, completed_at
                 FROM runs WHERE user_id = ? ORDER BY started_at DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def run_history(run_id: str) -> list[dict[str, Any]]:
    with connect() as c:
        rows = c.execute(
            "SELECT round_num, global_loss, num_clients FROM rounds "
            "WHERE run_id = ? ORDER BY round_num",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def active_hypotheses(user_id: str) -> list[dict[str, Any]]:
    with connect() as c:
        rows = c.execute(
            "SELECT hypothesis_id, statement, created_at FROM hypotheses "
            "WHERE user_id = ? AND status = 'active' ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
