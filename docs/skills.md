# Claude Code Skills

VelocityFL ships a personal [Claude Code](https://docs.claude.com/en/docs/claude-code) skill library at `.claude/skills/`. These are user-invocable slash-commands for routine dev work — not end-user features, but worth documenting because they shape the contributor workflow and every CI-adjacent artifact (`COMMITS.md`, `logs/dev-*.log`) comes out of them.

All skills use `disable-model-invocation: true`, so they fire only when you type `/name` or explicitly ask for them — never on model judgment.

## Where they live

```text
.claude/skills/
├── _shared/
│   └── hate-words.md             # canonical slop glossary (shared by /aj-deslop, /aj-reslop, /aj-docsync)
├── aj-audit/SKILL.md
├── aj-auto-commit/SKILL.md
├── aj-ci-audit/SKILL.md
├── aj-deslop/SKILL.md
├── aj-docs-site/SKILL.md
├── aj-docsync/SKILL.md
└── aj-reslop/SKILL.md
```

Edit `.claude/skills/_shared/hate-words.md` — not the individual skills — when adding or adjusting slop patterns.

## The skills

| Command | Purpose | Writes | Reads |
|---|---|---|---|
| `/aj-auto-commit` | Group pending git changes into a conventional-commit plan | `COMMITS.md` (with staleness header + `## Notes` preservation) | `git status`, `git diff`, `git log` |
| `/aj-audit` | Run the make-target matrix (full 13-step or fast 5-step) and verify each archive | — (read-only) | `logs/dev-*-<cmd>.log` |
| `/aj-ci-audit` | Audit GitHub Actions runs on the current branch/PR and fix workflow/config/source issues (no commit/push) | Edits on confirmation | `gh run view`, `.github/workflows/**` |
| `/aj-deslop` | Find AI-generated slop in comments and docstrings | Edits on confirmation | `python/**`, `vfl-core/**`, `scripts/**`, `tests/**` |
| `/aj-reslop` | Rewrite docstrings grounded in the actual implementation, call sites, and tests | Edits on confirmation | Same scope as `/aj-deslop` |
| `/aj-docsync` | Verify prose claims in `README.md` + `docs/**/*.md` against the code | Edits on confirmation | Docs + source |
| `/aj-docs-site` | Maintain zensical config, docs workflow, assets, internal link integrity | Edits on confirmation | `zensical.toml`, `.github/workflows/docs.yml`, `docs/**` |

`/aj-deslop` removes slop; `/aj-reslop` rewrites it grounded in the code; `/aj-docsync` audits claims; `/aj-docs-site` audits the site as a deployed artifact. `/aj-audit` checks local `make` output; `/aj-ci-audit` checks GitHub Actions output — sibling skills for the two rings of validation. They're designed to hand off to each other rather than overlap.

## Typical workflow

1. Edit code, run `make validate` locally.
2. `/aj-auto-commit` → review the generated `COMMITS.md`, commit per group, open a PR.
3. `/aj-audit` before merge to confirm the full matrix is green and the archives agree with terminal output.
4. `/aj-ci-audit` once GitHub Actions finishes to triage warnings/errors/deprecations; apply fixes, then commit + push yourself.
5. `/aj-docsync` if any user-facing prose needs a pass.
6. `/aj-docs-site` before a docs release if `zensical.toml` or workflow paths have changed.

## Why this is tracked in the repo

Skills are markdown — small, diff-friendly, reviewable. They live in Git alongside the code they automate. Contributors cloning the repo get the same toolchain, and changes to a skill show up in `git log` the same way a source change does. No separate storage layer, no Xet, no specialized hub — the right primitive for this kind of artifact is just Git.

## Adding your own

Drop a new directory under `.claude/skills/<name>/` with a `SKILL.md` containing:

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

Keep skills quiet — the output IS the response. No preamble, no summary paragraph. If an existing slop glossary or workflow fits, reference it instead of duplicating.
