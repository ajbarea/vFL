# Claude Code Skills

vFL contributors use the [techne plugin](https://github.com/ajbarea/techne) — a Claude Code skill collection for repo hygiene, audits, and doc/code drift. Skills are not part of this repo; they install once at the user level and apply across every linked sister repo.

```bash
# In Claude Code
/plugin marketplace add ajbarea/techne
/plugin install techne@techne
```

The plugin reads per-repo configuration from `.claude/skill-context.md` (in this repo, that file pins the toolchain audit phases for vFL specifically — `make build` before any lint/test since the native `_core` extension must materialize first).

## The skills

| Command | Purpose |
|---|---|
| `/techne:audit` | Runs the repo's `make` targets in dependency order and reconciles terminal output against `logs/dev-*.log` archives. |
| `/techne:ci-audit` | Audits GitHub Actions runs on the current branch/PR: warnings, failures, deprecation notices; fixes what's fixable in-repo. |
| `/techne:auto-commit` | Groups working-tree changes into a structured `COMMITS.md` plan for staged review before anything lands. |
| `/techne:deslop` | Scans comments and docstrings for AI-generated slop and proposes tightened rewrites. |
| `/techne:reslop` | Rewrites docstrings grounded in the implementation rather than deleting them outright. |
| `/techne:docsync` | Verifies documentation claims (CLI commands, paths, config keys, signatures) against the actual code. |
| `/techne:docs-site` | Maintains the Zensical-powered docs site: config, deploy pipeline, theming, link integrity. |
| `/techne:sisters` | Cross-repo drift audit across the sister repos listed in `~/.claude/techne.toml`. |
| `/techne:theoros` | Observed live dev session — Claude drives the REPL in a named `tmux` session; you spectate read-only via `tmux attach -r`. Not vFL-relevant today (no interactive REPL); listed for completeness. |

`/techne:deslop` removes slop; `/techne:reslop` rewrites it grounded in the code; `/techne:docsync` audits prose claims against code; `/techne:docs-site` audits the site as a deployed artifact. `/techne:audit` checks local `make` output; `/techne:ci-audit` checks GitHub Actions — sibling skills for the two rings of validation.

## Typical workflow

1. Edit code, run `make validate` locally.
2. `/techne:auto-commit` → review the generated `COMMITS.md`, commit per group, open a PR.
3. `/techne:audit` before merge to confirm the full matrix is green and the archives agree with terminal output.
4. `/techne:ci-audit` once GitHub Actions finishes to triage warnings / errors / deprecation notices; apply fixes, then commit + push yourself.
5. `/techne:docsync` if any user-facing prose needs a pass.
6. `/techne:docs-site` before a docs release if `zensical.toml` or workflow paths have changed.

## Per-repo configuration

`.claude/skill-context.md` is the contract between the techne skills and this repo's specifics — required setup phases, lint commands, test entrypoints, the maturin-build prerequisite, and the slop-glossary scope. Update it when toolchain, paths, or tooling change; the skills read it at invocation via `!cat .claude/skill-context.md`.

Cross-repo policy (toolchain pin floors, shared `pyproject.toml` conventions) is enforced by `/techne:sisters` against the active sister list in `~/.claude/techne.toml`.

## Adding new skills

New skills land in the techne plugin repo, not here. Submit a PR there with a `SKILL.md` containing:

```markdown
---
name: <name>
description: <one-line description for the skill registry>
disable-model-invocation: true
allowed-tools: <space-separated tool list>
---

# <Title>

<prose explaining scope, workflow, output format, and what not to touch>
```

Existing slop glossaries (e.g. `_shared/hate-words.md` inside the techne plugin) are referenced, not duplicated.

## Why this changed

Skills used to ship in-repo at `.claude/skills/aj-*/SKILL.md` (gitignored, but documented here). The techne plugin migration moved them to a centralized install so every linked sister repo (listed in `~/.claude/techne.toml`) shares one canonical copy. The old `.claude/skills/` directory is still gitignored locally for backward-compat with anyone running the pre-migration copies, but the canonical surface is now `~/.claude/plugins/cache/techne/techne/<sha>/skills/`.
