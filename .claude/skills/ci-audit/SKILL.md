---
name: ci-audit
description: Audit the latest GitHub Actions workflow runs on the current branch/PR for warnings, errors, failures, deprecation notices, and other log noise — then fix what's fixable in-repo (workflow YAML, configs, source, tests). Hands commit + push back to the developer. Use after CI finishes and before merge, or whenever the user says "audit the PR / workflow logs".
disable-model-invocation: true
allowed-tools: Bash(gh *) Bash(git branch --show-current) Bash(git rev-parse *) Bash(git log *) Glob Grep Read Edit Write
---

# CI Audit

Audit the GitHub Actions runs on the current branch. Classify every warning / error / failure / deprecation, fix what can be fixed in-repo, and leave `git add` / `commit` / `push` to the developer.

## Scope

- **In scope:** workflow YAML under `.github/workflows/`, any config referenced from it (`pyproject.toml`, `Cargo.toml`, `Makefile`, `scripts/dev.py`), and source/tests when a failure traces to code.
- **Out of scope:** committing, pushing, opening PRs, merging, re-running workflows. The developer drives git.

## Where runs live

- Current branch: `git branch --show-current`.
- Associated PR (if any): `gh pr view --json number,url,statusCheckRollup`. `statusCheckRollup[].detailsUrl` gives the run URL; the numeric id after `/runs/` is the run id you need.
- If there's no PR yet, fall back to `gh run list --branch <branch> --limit 5 --json databaseId,headSha,status,conclusion,workflowName,createdAt`.

Pick the **latest run per workflow** on the current head SHA. Don't mix SHAs — an older run against an older commit is noise.

## What to pull

For each run id:

```
gh run view <id> --json jobs,conclusion,status,workflowName,headSha
gh run view <id> --log
```

`--log` is the full concatenated log across all jobs/steps. It's the canonical source — prefer it over the web UI.

For runs with many jobs, `gh run view <id> --log-failed` narrows to failed steps only. Use it when the run failed but don't rely on it for warning-sweep (successful-but-noisy steps are exactly what you're auditing).

## Triage

Grep the log for these markers, in roughly this order of signal:

| Marker | What it usually means |
|---|---|
| `##[error]` | Actions runner reporting a step failure |
| `##[warning]` | Runner-level warning — often deprecation notices worth fixing |
| `Error:` / `error[` / `error:` | Tool-side error (rustc, pytest, ruff, etc.) |
| `FAIL`, `failed`, `exit 1` | Test / lint failure |
| `deprecat` (case-insensitive) | Action or API deprecation — actionable if replacement exists |
| `DeprecationWarning` | Node/Python runtime warnings — usually upstream, note but defer |
| `WARNING:` / `warning -` | Tool warnings (Codecov CLI, pip, etc.) |

A tight first pass:

```
gh run view <id> --log | grep -iE "warning|error|fail|deprecat" | head -100
```

Expand context (`grep -A3 -B1`) around any hit you can't classify from one line.

## Classify every finding

Put every finding into one of four buckets:

1. **Fix now, in this repo.** Workflow YAML referencing a deprecated action, a lint/test failure, a missing file, a misconfigured flag. Edit and verify.
2. **Fix now, upstream-version bump.** An action is deprecated and the repo pins an old major — bump the pin after checking the upstream changelog.
3. **Defer (upstream / cosmetic).** Node version deprecation notices tied to actions that haven't published a Node 24 release yet. Library teardown noise (e.g. Prefect rich-console I/O error during test shutdown). `punycode` warnings from transitive Node deps. Cite it, say why it's deferred.
4. **Expected.** `continue-on-error` steps that fail by design (non-blocking `ty check`). Note that they ran and moved on — they are not findings.

Don't silently drop findings into bucket 3. Every deferred item gets one line in the verdict with *why* so the user can overrule.

## Fixing

For bucket 1/2 items:

- **Deprecated action** → check the action's README / release notes for the replacement (WebFetch the repo's README). Migrate inputs carefully; parameter names sometimes change (`report-type` vs `report_type`, `file` vs `files`). Verify the new input names against current docs, don't guess.
- **Version pin bump** → prefer the latest stable major. Don't pin to floating `@main` or `@v<major>` without a SHA unless the rest of the repo already does.
- **Lint/test failure** → read the traceback, find the file, fix the root cause. Don't suppress unless the user asks.
- **Workflow misconfig** → keep the diff minimal. A missing `fail_ci_if_error`, a wrong `files:` path, a stray newline at EOF. One concern per edit.

After each edit, if the change is locally verifiable (a pytest change, a ruff config tweak, a `Makefile` target), run it. Workflow YAML changes can only be verified by the next CI run — say so explicitly in the verdict.

## Output format

A single block, no preamble:

```
## CI audit — <branch> @ <short-sha>

Runs reviewed:
- <workflowName> #<id> → <conclusion>  (<detailsUrl>)
- ...

### Findings

1. [FIX] <one-line summary> — <file:line> → edited.  Verified: <local check or "next CI run">.
2. [BUMP] <action> <old>→<new> — <file:line> → edited.
3. [DEFER] <one-line summary>.  Why: <upstream / cosmetic / expected reason>.
4. [EXPECTED] <step name> — `continue-on-error: true`; ran as designed.

### Verdict

<"N fixes applied, M deferred. Ready for commit + push." | "All runs clean; N items deferred with reasons.">
```

Numbered findings, one per item. Keep each line tight — link to the file, not the log.

## Rules

- Do not run `git add`, `git commit`, `git push`, `gh pr merge`, or `gh run rerun`. That's the developer's call.
- Do not edit `logs/` — those are `/audit`'s territory and come from local make runs, not CI.
- Do not open or modify PRs / issues. Reading PR metadata via `gh pr view` is fine; writing is not.
- If a finding requires a destructive or high-blast-radius change (removing a check, disabling a gate, force-pushing), stop and ask. Don't paper over a real failure.
- If the latest run is still `in_progress`, say so and stop — don't audit a partial log.
