<!--
auto-commit-claude
scanned-at: 2026-04-17T19:02:30Z
head-sha:   8ab0c19
tree-hash:  29e2c63eda88
branch:     feat/dev-ergonomics
-->

Suggested branch: `feat/dev-ergonomics`

---

chore(dev): add cross-platform dev runner, Makefile, and VS Code workspace

- Added `scripts/dev.py`, a logged fix-then-check runner that wraps uv/cargo/ruff/clippy/ty/pytest with per-step summaries written to `logs/dev-<ts>-<cmd>.log`
- Added a thin `Makefile` delegating every target to the dev runner so `make` and `uv run python scripts/dev.py` stay equivalent
- Committed `.vscode/extensions.json` and `.vscode/settings.json` with Rust/Python formatters, pytest config, and watcher/search excludes

Files: .vscode/extensions.json, .vscode/settings.json, Makefile, scripts/dev.py

---

refactor(core): rename PyO3 module to `velocity._core` and add type stubs

- Renamed the `#[pymodule]` entry point from `vfl_core` to `_core` to match `module-name = "velocity._core"`
- Added `python/velocity/_core.pyi` so ty/pyright/mypy can reason about the compiled extension without introspecting the `.so`
- Wrapped the lazy `_core` import in `_load_rust_core()` and dropped the unused `AttackResult` re-export in `server.py`

Files: python/velocity/_core.pyi, python/velocity/server.py, vfl-core/src/lib.rs

---

style(core): apply rustfmt and clippy fixes in vfl-core

- Reformatted function signatures and JSON/error expressions to rustfmt defaults
- Converted the FedMedian inner loop to `iter_mut().enumerate()` and used `is_multiple_of` for parity checks
- Allowed `clippy::enum_variant_names` on `Strategy` since variant names intentionally mirror FL literature

Files: vfl-core/src/orchestrator.rs, vfl-core/src/strategy.rs

---

style(python): apply ruff fixes across velocity sources and tests

- Dropped unused imports (`json`, `pytest`, `Optional`) and reformatted long ValueError/comprehension lines
- Replaced forward-reference string annotations in `flows.py` with direct `VelocityServer` references under `from __future__ import annotations`
- Narrowed the frozen-dataclass test to `FrozenInstanceError` instead of bare `Exception`

Files: python/velocity/attacks.py, python/velocity/cli.py, python/velocity/flows.py, tests/test_attacks.py, tests/test_cli.py, tests/test_server.py, tests/test_strategy.py

---

docs(site): expand reference pages and restyle Zensical theme

- Rewrote api/architecture/attacks/cli/configuration/getting-started/index/strategies with deeper reference tables, diagrams, and decision guides
- Added the hero image asset, a `reveal.js` scroll-reveal script, and wired them through `zensical.toml`
- Recolored the palette (deep-purple / purple) and expanded `docs/style.css` to support the new hero and reveal animations

Files: docs/api.md, docs/architecture.md, docs/assets/velocity-hero.png, docs/attacks.md, docs/cli.md, docs/configuration.md, docs/getting-started.md, docs/index.md, docs/javascripts/reveal.js, docs/strategies.md, docs/style.css, zensical.toml

---

feat(skills): add Claude Code skill library with shared hate-words glossary

- Added six sibling skills under `.claude/skills/`: `/auto-commit` (this plan generator), `/audit` (full/fast make-target matrix with archive verification), `/deslop` (find AI-generated slop), `/reslop` (rewrite grounded in the code), `/docsync` (verify doc claims against code), `/docs-site` (zensical config, deploy workflow, asset integrity)
- Shared `_shared/hate-words.md` as the canonical slop glossary — referenced by `/deslop`, `/reslop`, and `/docsync` so pattern updates happen in one place
- All skills use `disable-model-invocation: true` so they fire only on explicit `/name` or user instruction
- Included `COMMITS.md` as the living output of `/auto-commit`

Files: .claude/skills/_shared/hate-words.md, .claude/skills/audit/SKILL.md, .claude/skills/auto-commit/SKILL.md, .claude/skills/deslop/SKILL.md, .claude/skills/docs-site/SKILL.md, .claude/skills/docsync/SKILL.md, .claude/skills/reslop/SKILL.md, COMMITS.md

---

## Notes

- **Commit order** (dependency-respecting, smallest blast radius first):
  1. `chore(dev)` — tooling foundation; everything else relies on this
  2. `refactor(core)` — PyO3 rename (before the Rust style fixes that touch the same files)
  3. `style(core)` — rustfmt/clippy
  4. `style(python)` — ruff
  5. `docs(site)` — isolated, safe anywhere after 1
  6. `feat(skills)` — last; contains COMMITS.md which references everything above
- **`COMMITS.md` long-term**: consider adding to `.gitignore` after this merge. It's a regeneratable artifact of `/auto-commit`; tracking it long-term means every future skill invocation dirties the tree. Committing it *once* here (so the skill library is self-documenting at landing) is fine; keeping it tracked forever is noise.
- **Branch strategy**: `feat/dev-ergonomics` → regular merge (not squash) to preserve the 6-commit structure. Squash would collapse this into one mega-commit and make `git log` unreadable later.
