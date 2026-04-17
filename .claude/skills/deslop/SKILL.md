---
name: deslop
description: Scan the codebase for AI-generated slop in comments and docstrings — temporal markers, self-referential AI framing, narrative WHAT-comments, marketing padding — and propose tightened rewrites. Use when the user wants to audit pending changes or the whole codebase for verbose, low-value commentary left by other assistants (Copilot, Gemini, GPT, etc.).
disable-model-invocation: true
allowed-tools: Bash Glob Grep Read Edit Agent
---

# Deslop

Find and remove low-value, AI-generated commentary in source files. Keep comments that explain a non-obvious *why*; trim or delete everything else.

## What counts as slop

**Cut:**

- **Temporal markers** — "(April 2026 best-practice order)", "as of 2024", "latest version". These rot.
- **Self-referential AI framing** — "designed so an AI assistant who did not see the live terminal can reconstruct", "AI-DEBUG HINTS", "helps LLMs understand", "for model consumption".
- **Narrative WHAT-comments** — `# Now we iterate through the list`, `# Return the result`. If the identifier already tells you, the comment is dead weight.
- **Marketing language** — "robust", "comprehensive", "elegant", "best-practice", "production-ready", "seamless", "powerful", "effortlessly", "with ease", "painlessly", "simply", "just" (as filler), "out of the box", "blazingly", "lightning-fast", "battle-tested", "state-of-the-art", "cutting-edge" when describing your own code. See `.claude/skills/_shared/hate-words.md` for the canonical cross-skill list.
- **Task-context rot** — "added for issue #123", "fix for the auth bug", `TODO(copilot):`. Belongs in the PR description, not the source.
- **Signature restatement** — docstrings that only repeat the type annotations in prose.
- **Excessive hedging** — "we might want to", "in some cases this could potentially", "it's important to note that".
- **Step-by-step narration in docstrings** — "First we do X. Then we do Y. Finally we do Z." when the function body already shows that.
- **Emoji decoration** — emoji in source comments/docstrings unless the user asked for it.

**Keep:**

- **Why-comments** — hidden constraints, subtle invariants, workarounds for specific bugs, behavior that would surprise a reader.
- **Short public-API docstrings** — one or two sentences on what a function is for, because callers don't read the body.
- **Non-obvious references** — RFC links, load-bearing issue links, citations, license headers.

Rule of thumb: if deleting the comment wouldn't confuse a future reader, delete it.

## Grep seed patterns

The canonical pattern list lives at **`.claude/skills/_shared/hate-words.md`** — update *that* file when adding new patterns, not this one. Subagents should `Grep` every section of the shared glossary (case-insensitive, across the scoped paths) to build a candidate list fast, then filter each hit against the Cut/Keep rules above.

A hit is *not* automatically slop — it's just a cheap starting point. "Robust" inside a user-facing error message is fine; "robust implementation" in a docstring is slop.

User-specific calibration — patterns the user flags most often from their own prompt-engineering flow:

- `best[- ]practice` — "based on April 2026 best practice"
- `PHASE ?\d` — "PHASE 1-A", "Phase 2 of the refactor"

## Workflow

1. **Scope.** Default to the whole repo minus vendored/generated paths:
   - Skip: `target/`, `.venv/`, `node_modules/`, `dist/`, `build/`, `site/`, `__pycache__/`, `.ruff_cache/`, `.pytest_cache/`, `uv.lock`, `Cargo.lock`, `docs/assets/`, `logs/`
   - If the user passed a path (e.g. `/deslop scripts/`), restrict to that.

2. **Fan out with Explore subagents in parallel.** One subagent per area, all dispatched in a single message. Typical split for this repo:
   - Rust sources: `vfl-core/src/**/*.rs`
   - Python package: `python/**/*.py`
   - Scripts and tests: `scripts/**/*.py`, `tests/**/*.py`
   - Config/build: `pyproject.toml`, `Makefile`, `.github/workflows/**`, `zensical.toml`, `.vscode/**`
   - Docs (only if user asks — docs prose is a different genre): `docs/**/*.md`

   Brief each subagent with the slop/keep lists, the grep seed patterns, and the calibration examples below. Tell it to start with the grep pass for recall, then read surrounding context to confirm each hit against the Cut/Keep filter before reporting. Ask for a compact report: `file:line` + the offending text + a proposed replacement (or `delete`). Cap each report at ~30 findings so context stays cheap.

3. **Consolidate.** Merge findings grouped by file. Drop duplicates and obvious false positives. If two subagents disagree on the same line, prefer the less aggressive edit.

4. **Present, then apply on request.** Default: show the list, then ask `apply all / apply selected / skip?`. If the user invoked the skill with `--apply` or clearly said "just fix them", skip the confirmation and run `Edit` directly.

## Don't touch

- Comments the user has already reviewed and kept in prior turns.
- License headers, SPDX identifiers, shebangs, `# noqa`/`# type: ignore`/`# pragma` directives.
- Test fixtures that intentionally contain sample text.
- `.pyi` stub files — terse headers are expected there.
- Anything under the skip paths listed above.

## Calibration examples (from this repo)

Use these to tune your threshold — the user and I have already agreed on these calls.

**Slop (removed):**

- `Tooling (April 2026 best-practice order):` → `Tooling:`
- `"The log is designed so an AI assistant who did not see the live terminal can reconstruct what happened..."` → trimmed to a factual description of what the log contains
- `AI-DEBUG HINTS` section header → `DEBUG HINTS`, body shortened
- `"so AI debuggers can recognise the 'not on PATH' case unambiguously"` → `"Missing-binary failures are surfaced as rc=127."`

**Kept (not slop):**

- `# The handle intentionally outlives this method (one file per session), so a context-manager pattern doesn't fit — atexit closes it instead.` — non-obvious design choice.
- `# Fixers may legitimately return non-zero when nothing can be fixed.` — explains why `except`-continue is correct.
- `# ty has no auto-fix; it runs in the check phase only.` — explains asymmetric tool behavior.

## Output format

When presenting findings, group by file, show line number, offending text, and proposed edit:

```
scripts/dev.py
  L55   # ANSI colours; Windows 10+ terminals handle these fine, but fall back gracefully.
    →   # ANSI colour codes with a no-color fallback.

python/velocity/server.py
  L12   # ---------------------------------------------------------------------------
  L13   # Lazy import of the Rust extension so that the pure-Python package is still
  L14   # importable even if the native extension has not been compiled yet.
  L15   # ---------------------------------------------------------------------------
    →   delete (explained by the function name _load_rust_core)
```

Then one line: `Apply all / apply selected (say which) / skip?`

## Why this skill is quiet

The output is the findings and the edits. Don't narrate "I looked at the file and noticed…" — show `file:line → replacement`. If a finding is borderline, one short phrase next to it ("borderline — kept for now") is enough. No preamble, no summary paragraph.
