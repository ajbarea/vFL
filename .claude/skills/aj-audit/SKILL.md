---
name: aj-audit
description: Run the repo's make targets in dependency order (setup → fix → granular lint → granular test → end-to-end) and verify each command's terminal output against its `logs/dev-<ts>-<cmd>.log` archive. Supports a full 13-step audit or a fast 5-step variant (setup + `make ci`). Use when the user wants end-to-end validation that the toolchain is clean and the archives match reality.
disable-model-invocation: true
allowed-tools: Bash Glob Read Grep Write
---

# Audit

Run the full `make` audit in phases. Each phase builds on the previous; the ordering matters because `scripts/dev.py` writes one archive per invocation, and granular runs must happen *before* their combined counterparts so you keep diffable per-tool logs.

## Modes

- **Full audit (default)** — all 13 commands below. Run when the user says "run the full audit" or didn't specify.
- **Fast audit** — `check-env` → `clean` → `sync` → `build` → `ci`. Five commands, ~25s end-to-end. Run when the user says "quick check", "fast audit", or only wants to know if CI will pass. Skips the granular lint/test phases; the `ci` archive covers the same ground for pass/fail, just without per-tool isolation for log-diffing.

Pick the mode up front and stick with it — don't silently promote a fast audit to a full one mid-run.

## Run order (full audit)

### Phase 1 — Setup (prove the environment before touching code)

1. `make check-env` — uv / cargo / rustc on PATH. If this fails, stop.
2. `make clean` — wipes `target/`, `__pycache__`, `.ruff_cache`, `.pytest_cache`, stale `_core` artifacts. Starts log history from a known-zero state.
3. `make sync` — `uv sync` resolves the Python env. Required before any Python-side tool.
4. `make build` — `maturin develop` compiles the Rust crate and installs `velocity._core` into `.venv`. **Must run before any Python lint/test** — otherwise `ty` can't resolve the native module and reports phantom import failures.

### Phase 2 — Fix (one-way door, before any checker)

5. `make fix` — every auto-fixer in one pass (ruff format, ruff check --fix, cargo fmt, cargo clippy --fix). Do this once so subsequent lint runs measure *intent*, not trivial formatting noise.

### Phase 3 — Granular lint (each writes its own archive)

6. `make lint-rs` — cargo fmt --check + clippy -- -D warnings.
7. `make lint-py` — ruff format --check, ruff check, ty. Depends on step 4.
8. `make lint` — combined. Redundant with 6+7 but gives a merged archive to diff against the granular ones (catches ordering-dependent failures).

### Phase 4 — Granular test

9. `make test-rs` — cargo test --all.
10. `make test-py` — pytest. Depends on step 4.
11. `make test` — combined. Same rationale as step 8.

### Phase 5 — End-to-end gates

12. `make validate` — fast lint + test-py. The "am I ready to push" probe.
13. `make ci` — full pipeline mirror (sync → build → lint → test). **The single most valuable audit artifact** — one archive showing the exact sequence CI will run.

### Independent, skip by default

- `make docs` — `zensical serve` on :8000. Blocks the terminal (long-running server). Only run if the user explicitly asks; never include in the automated audit.

Run sequentially, not in parallel — the commands share state (`.venv`, `target/`, log archives) and archives need clean separation.

## Per-command verification

For each command:

1. Record the terminal exit code.
2. `Glob` `logs/dev-*-<command>.log`, sorted by mtime, take the newest — that's this run's archive.
3. `Read` the tail (~30 lines) for the `SUMMARY` block.
4. Confirm:
   - Terminal exit code = 0
   - `overall rc    : 0` in SUMMARY
   - `steps failed : 0` in SUMMARY
5. Note the `total elapsed` and `steps run` counts from SUMMARY — they go into the matrix.

If any check fails, mark the row FAIL and pull the failing step name(s) from the `per-step:` block.

**Do not read `logs/dev-latest.log`.** It is overwritten every invocation and only reflects the most recent command, so after step 13 it shows `ci` output exclusively. The per-command archives (`dev-<ts>-<cmd>.log`) are what you want — and the glob above already skips `dev-latest.log` because it lacks a timestamp.

**Timing sanity check.** After collecting all elapsed times, compare `ci` total against the sum of Phase-3 + Phase-4 granulars. If `ci` is significantly shorter (e.g. < 60% of the sum), something cached between runs and the granular numbers aren't independent measurements. Not a failure — but mention it in the verdict so the user knows the timing matrix is warm-cache, not cold.

## Cross-archive sweep

After all runs (13 in full mode, 5 in fast mode), one `Grep` across the fresh archives for error markers:

```
Grep pattern="\[ERR|FAIL|exit 1|error\[" path="logs/" glob="dev-<today>*"
```

Zero hits = clean. Any hits = dump file:line for triage.

## Output format

A markdown table, then the CI per-step block, then a verdict:

```
| # | Command | Terminal result | Archive SUMMARY | Steps | Elapsed |
|---|---|---|---|---|---|
| 1 | `make check-env` | uv/cargo/rustc all ok | rc=0 | 0 | — |
| ... |

## make ci per-step (from logs/dev-<ts>-ci.log)

  PASS  rc=0   0.06s  uv sync
  PASS  rc=0   6.08s  uv run maturin develop --uv
  ...

Full 13-step audit clean.
```

Pick the verdict line to match the mode: `Full 13-step audit clean.` or `Fast 5-step audit clean.` No preamble, no narration. If rows fail, the verdict names which: `"N failures — rows X, Y — see logs/dev-<ts>-<cmd>.log"`.

## Stop-early rules

- If Phase 1 fails (check-env / clean / sync / build), stop and report. The rest won't produce meaningful results.
- If a later phase fails, keep going — the user wants the full matrix even with some red rows.

## Scope

- Runs `make` targets defined in the repo's Makefile only.
- Reads only files under `logs/`.
- Never edits source, config, or docs.
- Never commits, pushes, or changes git state.
