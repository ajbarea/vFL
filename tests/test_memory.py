"""Unit tests for `velocity.memory` — per-user transparent file memory."""

from __future__ import annotations

import json

import pytest
from velocity import memory


@pytest.fixture(autouse=True)
def _isolate_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("VFL_MEMORY_DIR", str(tmp_path))
    memory._BOOTSTRAPPED.clear()
    yield
    memory._BOOTSTRAPPED.clear()


def test_memory_root_honours_env(tmp_path):
    assert memory.memory_root() == tmp_path


def test_default_user_id_prefers_env(monkeypatch):
    monkeypatch.setenv("VFL_USER_ID", "explicit-user")
    assert memory.default_user_id() == "explicit-user"


def test_default_user_id_falls_back_to_shell_user(monkeypatch):
    monkeypatch.delenv("VFL_USER_ID", raising=False)
    monkeypatch.setattr(memory.getpass, "getuser", lambda: "shell-user")
    assert memory.default_user_id() == "shell-user"


def test_user_dir_creates_directory(tmp_path):
    d = memory.user_dir("alice")
    assert d == tmp_path / "alice"
    assert d.is_dir()


def test_bootstrap_creates_scaffolding_and_is_idempotent():
    memory.bootstrap("alice")
    memory.bootstrap("alice")  # second call is a no-op
    d = memory.user_dir("alice")
    assert (d / "MEMORY.md").exists()
    for f in ("profile.md", "style.md", "hypotheses.md", "recent_runs.md", "recipes.md", "preferences.md"):
        assert (d / f).exists()
    # only one bootstrap event recorded
    bootstrap_events = [e for e in memory.events("alice") if e["action"] == "bootstrap"]
    assert len(bootstrap_events) == 1


def test_bootstrap_preserves_existing_memory_md():
    d = memory.user_dir("alice")
    (d / "MEMORY.md").write_text("# my own index\n", encoding="utf-8")
    memory.bootstrap("alice")
    assert (d / "MEMORY.md").read_text(encoding="utf-8") == "# my own index\n"


def test_write_entry_persists_and_logs():
    memory.write_entry("alice", "profile.md", "# AJ\n\nFL researcher.\n", summary="initial profile")
    assert memory.read_entry("alice", "profile.md") == "# AJ\n\nFL researcher.\n"
    log = memory.events("alice")
    assert log[-1]["action"] == "write"
    assert log[-1]["file"] == "profile.md"
    assert log[-1]["summary"] == "initial profile"


def test_write_entry_rejects_non_writable_file():
    with pytest.raises(ValueError, match="not in the writable memory set"):
        memory.write_entry("alice", "secrets.txt", "x", summary="nope")


def test_append_entry_separates_blocks_with_newline():
    memory.append_entry("alice", "recipes.md", "## first", summary="a")
    memory.append_entry("alice", "recipes.md", "## second", summary="b")
    body = memory.read_entry("alice", "recipes.md")
    assert body == "## first\n\n## second\n"


def test_append_entry_rejects_non_writable_file():
    with pytest.raises(ValueError):
        memory.append_entry("alice", ".events.jsonl", "x", summary="nope")


def test_read_entry_returns_empty_for_missing():
    assert memory.read_entry("alice", "profile.md") == ""


def test_forget_entry_deletes_and_logs():
    memory.write_entry("alice", "style.md", "terse", summary="set tone")
    memory.forget_entry("alice", "style.md", reason="user asked to forget")
    assert memory.read_entry("alice", "style.md") == ""
    assert memory.events("alice")[-1] == {
        **memory.events("alice")[-1],
        "action": "delete",
        "file": "style.md",
        "summary": "user asked to forget",
    }


def test_forget_entry_logs_even_when_file_absent():
    memory.forget_entry("alice", "preferences.md", reason="precautionary")
    log = memory.events("alice")
    assert log[-1]["action"] == "delete"


def test_events_returns_last_n():
    for i in range(5):
        memory.write_entry("alice", "profile.md", f"v{i}", summary=f"step {i}")
    last_three = memory.events("alice", limit=3)
    assert [e["summary"] for e in last_three] == ["step 2", "step 3", "step 4"]


def test_events_returns_empty_for_fresh_user():
    assert memory.events("nobody") == []


def test_list_files_excludes_hidden_ledger():
    memory.write_entry("alice", "profile.md", "hi", summary="seed")
    files = memory.list_files("alice")
    assert "profile.md" in files
    assert ".events.jsonl" not in files


def test_events_payload_is_valid_jsonl():
    memory.write_entry("alice", "profile.md", "x", summary="seed")
    raw = memory._events_path("alice").read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in raw]
    for event in parsed:
        assert {"ts", "action", "file", "summary"} <= event.keys()


def test_users_are_isolated():
    memory.write_entry("alice", "profile.md", "alice content", summary="a")
    memory.write_entry("bob", "profile.md", "bob content", summary="b")
    assert memory.read_entry("alice", "profile.md") == "alice content"
    assert memory.read_entry("bob", "profile.md") == "bob content"
