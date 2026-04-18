---
name: aj-auto-commit
description: Analyze pending git changes and write a structured, conventional-commit plan to COMMITS.md so the user can review and stage commits in batches before committing. Use whenever the user wants to group working-tree or staged changes into sensible commits, draft commit messages for a dirty working tree, or prepare a commit plan from a diff.
disable-model-invocation: true
allowed-tools: Bash(git status) Bash(git diff *) Bash(git log *) Bash(git rev-parse *) Bash(git hash-object *) Read Write
---

# Auto-Commit Generator

Analyze pending git changes and group them into a structured commit plan written to `COMMITS.md` at the repo root. The user will read that file, decide what they want, and run the actual commits themselves. This skill does not commit anything.

## Workflow

1. **Preflight — check for a prior plan.** Before scanning the diff, check whether `COMMITS.md` already exists at the repo root. If it does, parse its header comment (see [Header format](#header-format)) and compare `head-sha` + `tree-hash` to the current state:
   - **Exact match** → the existing plan is still current. Tell the user in one sentence (`COMMITS.md is current (HEAD @ <sha>, working tree unchanged) — delete or edit it if you want a regenerate.`) and stop. Do not overwrite.
   - **HEAD matches, tree drifted** → partial staleness. Preserve any user-added `## Notes` section (see step 5), then regenerate.
   - **No header, or header unparseable** → treat as a hand-written file. Do not overwrite silently; tell the user it exists without a header and ask whether to regenerate or bail.
   - **File absent** → normal path, continue.

2. **Analyze.** Run these in parallel:
   - `git status` (to see what's changed and in what state)
   - `git diff` (unstaged changes)
   - `git diff --cached` (staged changes — don't skip this; staged work must be included in the plan)
   - `git log -20 --oneline` (to match the repo's existing commit style, e.g. scope names already in use)
   - `git rev-parse HEAD` (for the staleness header)
   - `git diff HEAD | git hash-object --stdin` (fingerprint of the working tree for the staleness header)

3. **Trust `.gitignore`.** Do not hardcode directory filters (no "skip `.claude/`", "skip `designs/`"). `git status` already respects `.gitignore`; if a file shows up, the user wants it tracked. The only filter is: if `git status` reports a clean tree with nothing staged, say so in one sentence and don't write the file.

4. **Group.** Decide the grouping that best reflects the actual work. See [Grouping](#grouping).

5. **Preserve notes.** If the prior `COMMITS.md` had a `## Notes` section (heading at column 0, anywhere after the last `---` separator), copy it verbatim into the regenerated file. That's where the user scribbles between sessions ("group 3 needs a split"); losing it is a papercut.

6. **Suggest a branch name.** Derive one from the largest (or most central) group's scope: `<type>/<scope>-<short-slug>` — e.g. `feat/auth-refresh-rotation`, `chore/dev-toolchain`. Place it in the header comment and as a single line above the first group.

7. **Write.** Use the Write tool to create `COMMITS.md` at the repo root, overwriting any existing file. Then stop. Don't narrate, don't summarize, don't offer next steps in chat.

## Grouping

The whole point of this skill is good grouping, so this is where to spend thought.

Decide what each commit *means* based on the diff, not on file types or directories. One commit should represent one coherent change a reviewer could understand on its own. A refactor that touches a source file and its test belongs together. A doc update that's incidental to a feature can ride along in that feature's commit rather than becoming its own trivial `docs()` entry.

Use these conventional-commit prefixes:

- `feat(scope)`: new functionality
- `fix(scope)`: bug fixes
- `docs(scope)`: documentation-only changes
- `test(scope)`: test-only changes
- `refactor(scope)`: restructuring without behavior change
- `style(scope)`: formatting, whitespace, no logic change
- `perf(scope)`: performance improvements
- `chore(scope)`: config, deps, maintenance
- `ci(scope)`: CI/CD changes
- `build(scope)`: build-system changes

Always replace `scope` with a real module, package, or area name drawn from the actual file paths (e.g. `auth`, `api`, `parser`). Match the style of scopes already used in `git log` if the repo has a convention. Never leave the literal word `scope`.

If a change genuinely spans multiple areas, pick the most specific scope that still covers it, or use a broader one like the package name.

Each file appears in exactly one group. If you're tempted to split a file across groups, the grouping is probably wrong — rethink it.

## Header format

Prepend this HTML comment to `COMMITS.md` (hidden from rendered previews, but greppable and parseable):

```
<!--
aj-auto-commit
scanned-at: 2026-04-17T14:20:00Z
head-sha:   8ab0c19
tree-hash:  a1b2c3d4e5f6
branch:     feat/dev-toolchain-upgrade
-->
```

`tree-hash` is the short (12-char) output of `git diff HEAD | git hash-object --stdin`. `scanned-at` is UTC ISO-8601.

## Output format

After the header comment and the branch-suggestion line, write the commit groups. The blank lines shown here are intentional and required (between headline and bullets, before `Files:`, and around the `---` separator). What you must *not* do is insert blank lines *inside* the bullet list, because that breaks the visual grouping when the user scans the file.

```
<!--
aj-auto-commit
scanned-at: 2026-04-17T14:20:00Z
head-sha:   8ab0c19
tree-hash:  a1b2c3d4e5f6
branch:     feat/auth-refresh
-->

Suggested branch: `feat/auth-refresh`

---

feat(auth): add JWT refresh token rotation

- Added rotating refresh tokens with 7-day expiry
- Wired new endpoint into the auth router
- Covered rotation edge cases in tests

Files: src/auth/tokens.py, src/auth/routes.py, tests/auth/test_tokens.py

---

fix(parser): handle trailing whitespace in CSV headers

- Stripped whitespace before column-name comparison
- Added regression test for the original bug

Files: src/parser/csv.py, tests/parser/test_csv.py
```

If a preserved `## Notes` section exists, append it after the last group, separated by a `---`.

Tense: headlines are imperative present (`add`, `handle`, `remove`), bullets describe what was done in past tense (`added`, `stripped`, `removed`). This matches standard conventional-commit style and reads naturally when the user pastes a headline into `git commit -m`.

## Why this is silent

The output is `COMMITS.md`. The user will open that file and copy from it. Any chat preamble ("I analyzed your changes and found…") is noise they have to scroll past, and it makes the file feel like a summary of a conversation rather than the authoritative plan. Write the file, and that's the response.

The exceptions — one sentence each, no file written:
- Clean working tree with nothing staged.
- Existing `COMMITS.md` already current (HEAD + tree-hash both match).
- Existing `COMMITS.md` without a recognizable header (user-authored) — ask before overwriting.
