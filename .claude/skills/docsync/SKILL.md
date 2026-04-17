---
name: docsync
description: Check a documentation file for drift against the actual codebase — CLI commands, file paths, config keys, function signatures, version numbers, environment variables — and propose corrections. Use when the user wants to audit README.md, docs/*.md, or similar for claims that no longer match reality. Drift usually comes from refactors that forgot to update the docs.
disable-model-invocation: true
allowed-tools: Bash Glob Grep Read Edit Agent
---

# Docsync

Compare documentation against the code it describes. Find where the docs say one thing and the code says another. Propose fixes grounded in current code, not in what the docs *should* have said.

## Checkable claims

- **Commands** — `uv sync`, `make lint`, `cargo test --all`. Verify the target exists in the Makefile / CLI source / workflow.
- **File paths** — `python/velocity/server.py`, `docs/api.md`. Verify the path exists.
- **Function / class signatures** — `VelocityServer(model_id: str, ...)`. Verify against source.
- **Config keys** — `tool.maturin.module-name = "velocity._core"`. Verify key + value against the config file.
- **Environment variables** — `NO_COLOR`, `PYTHONPATH`. Verify the code actually reads them.
- **Version constraints** — Python / Rust / dependency pins. Verify against `pyproject.toml`, `Cargo.toml`, workflow files.
- **Internal links / anchors** — `[see API](api.md#velocityserver)`. Verify target exists.
- **Exit codes and error messages** — Verify against the source that produces them.

## Ignore

- Prose / marketing copy ("VelocityFL provides high-performance FL orchestration"). Not a checkable claim.
- Roadmap language ("we plan to", "coming soon"). Aspirational, not drift.
- Code examples that are intentionally illustrative.
- Screenshots, images, external URLs.

## Workflow

1. **Scope.** Input is one doc file or a directory. Default if the user didn't specify: `README.md` + every tracked file under `docs/**/*.md`. For directory scope, fan out (see [Fan-out pattern](#fan-out-pattern)).

   Out of scope — hand off to `/docs-site` instead: `zensical.toml` config drift, `.github/workflows/docs.yml` deploy wiring, `docs/style.css` / `docs/javascripts/` asset maintenance, rendered-site link checks. `/docsync` is claim verification inside markdown prose; docs-site infrastructure is a different skill.

2. **Extract claims.** For the doc in scope, list every checkable claim with its exact string and line number. Categorize by the list above.

3. **Verify.** For each claim, run the appropriate check:
   - Command → `Grep` the Makefile / CLI source / workflow for the target.
   - Path → `Glob` or `Read`.
   - Signature → `Grep` for the definition, compare.
   - Config key → `Read` the config file, compare value.
   - Env var → `Grep` for `os.environ` / `env::var` / equivalent.

4. **Report drift.** One entry per mismatch:

   ```
   README.md:42
     claim:   `pip install -e .[dev]`
     reality: dev deps now live under [dependency-groups] in pyproject.toml;
              install is `uv sync`
     fix:     replace with `uv sync`
   ```

5. **Apply with care.**
   - Single-token swaps (command names, paths, signatures, config values) are safe to batch-apply on confirmation.
   - Prose rewrites get shown as a full before/after diff first — never silently reword a sentence.

## Fan-out pattern

For a whole `docs/` directory, spawn one `Explore` subagent per doc file. Each returns a drift report for its file. Main agent consolidates and presents grouped by file, then asks `apply all / apply selected / skip?`.

## Don't touch

- Doc files the user didn't name or imply.
- Generated doc output (`site/`).
- Changelog / release notes — they're historical; drift there is a feature, not a bug.
- Third-party API documentation mirrored into this repo.

## While you're in there

A doc update is also a chance to trim slop. If a fix requires rewriting a sentence, apply the `.claude/skills/_shared/hate-words.md` filter before you commit to the new wording — don't let the rewrite reintroduce "robust", "seamless", "effortlessly" and friends. If the file has heavy slop unrelated to the drift, mention it once and suggest running `/deslop` after — don't quietly expand scope.

## Why this skill is quiet

Output is the drift report and, on confirmation, the edits. No narration of the verification process — just `claim → reality → fix`.
