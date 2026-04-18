"""Unit tests for `velocity.db` — the SQLite experiment store.

Each test runs against a fresh tmp_path DB. The module caches a connection in a
threading.local; we reset it per-test so `VFL_DB_PATH` actually takes effect.
"""

from __future__ import annotations

import sqlite3

import pytest
from velocity import db


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    monkeypatch.setenv("VFL_DB_PATH", str(tmp_path / "experiments.db"))
    if hasattr(db._LOCAL, "conn"):
        try:
            db._LOCAL.conn.close()
        except sqlite3.Error:
            pass
        del db._LOCAL.conn
    yield
    if hasattr(db._LOCAL, "conn"):
        try:
            db._LOCAL.conn.close()
        except sqlite3.Error:
            pass
        del db._LOCAL.conn


def test_db_path_honours_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VFL_DB_PATH", str(tmp_path / "x.db"))
    assert db.db_path() == tmp_path / "x.db"


def test_init_db_creates_file_and_schema(tmp_path):
    path = db.init_db(tmp_path / "init.db")
    assert path.exists()
    with sqlite3.connect(path) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"users", "runs", "rounds", "attacks", "hypotheses", "agent_actions"} <= names


def test_ensure_user_is_idempotent():
    db.ensure_user("alice", "Alice")
    db.ensure_user("alice", "ignored second display name")
    with db.connect() as c:
        rows = c.execute("SELECT user_id, display_name FROM users").fetchall()
    assert len(rows) == 1
    assert rows[0]["display_name"] == "Alice"


def test_start_run_returns_unique_id_and_persists_config():
    cfg = {
        "strategy": "FedAvg",
        "model_id": "demo/model",
        "dataset": "demo/data",
        "seed": 7,
        "min_clients": 3,
        "rounds": 5,
        "git_sha": "deadbeef",
    }
    a = db.start_run("alice", cfg)
    b = db.start_run("alice", cfg)
    assert a != b
    assert a.startswith("run-") and len(a) == len("run-") + 12
    with db.connect() as c:
        row = c.execute("SELECT * FROM runs WHERE run_id = ?", (a,)).fetchone()
    assert row["strategy"] == "FedAvg"
    assert row["seed"] == 7
    assert row["status"] == "running"
    assert row["git_sha"] == "deadbeef"


def test_record_round_persists_round_and_attacks():
    run_id = db.start_run("alice", {"strategy": "FedAvg", "model_id": "m"})
    db.record_round(
        run_id,
        {
            "round": 1,
            "global_loss": 0.42,
            "num_clients": 4,
            "duration_ms": 12,
            "attack_results": [
                {"attack_type": "gaussian_noise", "params": {"std_dev": 0.1}, "magnitude": 0.05},
            ],
        },
    )
    with db.connect() as c:
        round_row = c.execute("SELECT * FROM rounds WHERE run_id = ?", (run_id,)).fetchone()
        attack_row = c.execute("SELECT * FROM attacks WHERE run_id = ?", (run_id,)).fetchone()
    assert round_row["global_loss"] == pytest.approx(0.42)
    assert round_row["num_clients"] == 4
    assert attack_row["attack_type"] == "gaussian_noise"


def test_record_round_idempotent_replace():
    run_id = db.start_run("alice", {"strategy": "FedAvg", "model_id": "m"})
    db.record_round(run_id, {"round": 1, "global_loss": 1.0, "num_clients": 2})
    db.record_round(run_id, {"round": 1, "global_loss": 0.5, "num_clients": 2})
    history = db.run_history(run_id)
    assert len(history) == 1
    assert history[0]["global_loss"] == pytest.approx(0.5)


def test_complete_run_updates_status_and_timestamp():
    run_id = db.start_run("alice", {"strategy": "FedAvg", "model_id": "m"})
    db.complete_run(run_id, status="failed")
    with db.connect() as c:
        row = c.execute(
            "SELECT status, completed_at FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    assert row["status"] == "failed"
    assert row["completed_at"] is not None


def test_log_action_writes_provenance_row():
    db.log_action(
        "alice",
        session_id="sess-1",
        tool="run_demo",
        args={"rounds": 3},
        user_prompt="kick off a demo",
        result_summary="ok",
    )
    with db.connect() as c:
        row = c.execute("SELECT * FROM agent_actions WHERE user_id = 'alice'").fetchone()
    assert row["tool"] == "run_demo"
    assert row["session_id"] == "sess-1"
    assert '"rounds": 3' in row["args_json"]


def test_recent_runs_is_user_scoped():
    # SQLite CURRENT_TIMESTAMP is 1-s resolution, so DESC ordering is undefined
    # for inserts in the same second. The user-scoping is the load-bearing
    # behaviour worth asserting here.
    cfg = {"strategy": "FedAvg", "model_id": "m"}
    a1 = db.start_run("alice", cfg)
    a2 = db.start_run("alice", cfg)
    db.start_run("bob", cfg)
    runs = db.recent_runs("alice", limit=10)
    ids = {r["run_id"] for r in runs}
    assert ids == {a1, a2}


def test_recent_runs_respects_limit():
    cfg = {"strategy": "FedAvg", "model_id": "m"}
    for _ in range(5):
        db.start_run("alice", cfg)
    assert len(db.recent_runs("alice", limit=2)) == 2


def test_active_hypotheses_filters_status_and_user():
    db.ensure_user("alice")
    db.ensure_user("bob")
    with db.connect() as c:
        c.execute("INSERT INTO hypotheses(user_id, statement) VALUES ('alice', 'h1-active')")
        c.execute(
            "INSERT INTO hypotheses(user_id, statement, status) "
            "VALUES ('alice', 'h2-resolved', 'resolved')"
        )
        c.execute("INSERT INTO hypotheses(user_id, statement) VALUES ('bob', 'h3-bob')")
    out = db.active_hypotheses("alice")
    statements = [h["statement"] for h in out]
    assert statements == ["h1-active"]


def test_foreign_keys_block_orphan_round():
    with pytest.raises(sqlite3.IntegrityError):
        with db.connect() as c:
            c.execute(
                "INSERT INTO rounds(run_id, round_num, num_clients) VALUES (?, ?, ?)",
                ("does-not-exist", 1, 2),
            )


def test_explicit_path_uses_short_lived_connection(tmp_path):
    p = tmp_path / "explicit.db"
    with db.connect(path=p) as c:
        c.execute("INSERT INTO users(user_id, display_name) VALUES ('x', 'X')")
    with sqlite3.connect(p) as conn:
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1


def test_connect_rolls_back_on_exception():
    db.ensure_user("alice")
    with pytest.raises(RuntimeError):
        with db.connect() as c:
            c.execute(
                "INSERT INTO users(user_id, display_name) VALUES ('temp', 'Temp')",
            )
            raise RuntimeError("boom")
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM users WHERE user_id = 'temp'").fetchone()[0] == 0
