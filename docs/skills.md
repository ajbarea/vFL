# Claude Code Skills

VelocityFL ships a personal [Claude Code](https://docs.claude.com/en/docs/claude-code) skill library at `.claude/skills/`. These are user-invocable slash-commands for routine dev work — not end-user features, but worth documenting because they shape the contributor workflow and every CI-adjacent artifact (`COMMITS.md`, `logs/dev-*.log`) comes out of them.

All skills use `disable-model-invocation: true`, so they fire only when you type `/name` or explicitly ask for them — never on model judgment.

## Where they live

```text
.claude/skills/
├── _shared/
│   └── hate-words.md          # canonical slop glossary (shared by /deslop, /reslop, /docsync)
├── audit/SKILL.md
├── auto-commit/SKILL.md
├── deslop/SKILL.md
├── docs-site/SKILL.md
├── docsync/SKILL.md
└── reslop/SKILL.md
```

Edit `.claude/skills/_shared/hate-words.md` — not the individual skills — when adding or adjusting slop patterns.

## The skills

| Command | Purpose | Writes | Reads |
|---|---|---|---|
| `/auto-commit` | Group pending git changes into a conventional-commit plan | `COMMITS.md` (with staleness header + `## Notes` preservation) | `git status`, `git diff`, `git log` |
| `/audit` | Run the make-target matrix (full 13-step or fast 5-step) and verify each archive | — (read-only) | `logs/dev-*-<cmd>.log` |
| `/deslop` | Find AI-generated slop in comments and docstrings | Edits on confirmation | `python/**`, `vfl-core/**`, `scripts/**`, `tests/**` |
| `/reslop` | Rewrite docstrings grounded in the actual implementation, call sites, and tests | Edits on confirmation | Same scope as `/deslop` |
| `/docsync` | Verify prose claims in `README.md` + `docs/**/*.md` against the code | Edits on confirmation | Docs + source |
| `/docs-site` | Maintain zensical config, docs workflow, assets, internal link integrity | Edits on confirmation | `zensical.toml`, `.github/workflows/docs.yml`, `docs/**` |

`/deslop` removes slop; `/reslop` rewrites it grounded in the code; `/docsync` audits claims; `/docs-site` audits the site as a deployed artifact. They're designed to hand off to each other rather than overlap.

## Typical workflow

1. Edit code, run `make validate` locally.
2. `/auto-commit` → review the generated `COMMITS.md`, commit per group, open a PR.
3. `/audit` before merge to confirm the full matrix is green and the archives agree with terminal output.
4. `/docsync` if any user-facing prose needs a pass.
5. `/docs-site` before a docs release if `zensical.toml` or workflow paths have changed.

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
