---
name: docs-site
description: Maintain the Zensical-powered documentation site — nav ordering in zensical.toml, the docs GitHub Pages workflow, style.css and reveal.js assets, and link/anchor integrity across docs/**/*.md. Sibling of /docsync (which only verifies prose claims against code). Use when the user wants the site itself audited — config, deploy pipeline, theming, assets, cross-page links — rather than content accuracy.
disable-model-invocation: true
allowed-tools: Bash Glob Grep Read Edit Agent
---

# Docs-Site

Audit and maintain the documentation site as a build/deploy artifact. `/docsync` answers "does this claim match the code?"; `/docs-site` answers "does this site build, deploy, render, and link correctly?"

## Surface

- **`zensical.toml`** — nav ordering, palette, theme features, extra css/js, site URL, social links.
- **`.github/workflows/docs.yml`** — trigger paths, `astral-sh/setup-uv` action, `zensical build --clean` invocation, Pages upload/deploy actions and versions.
- **`docs/*.md`** — presence matches `nav`, internal links resolve, anchors exist, no orphan files.
- **`docs/style.css`** — selectors reference classes that actually appear in rendered output.
- **`docs/javascripts/*.js`** — listed in `zensical.toml` `extra_javascript`, no dead files.
- **`docs/assets/`** — every asset is referenced somewhere; no orphan binaries inflating the repo.

## Checks

Run as many in parallel as the scope calls for:

1. **Nav vs filesystem.** Parse `zensical.toml` `nav`; `Glob` `docs/*.md`. Every nav entry must exist as a file; every top-level `.md` (except intentional sub-pages) should appear in nav. Report missing-in-nav and missing-on-disk separately.

2. **Internal links.** `Grep` markdown link targets (`](foo.md)`, `](foo.md#bar)`) across `docs/**/*.md`. For each: check the file exists, and if an anchor is specified, `Grep` the target file for a heading that would produce that slug (lowercase, spaces → `-`, strip punctuation).

3. **Assets referenced.** For each file under `docs/assets/` and `docs/javascripts/`, `Grep` the rest of the repo for its basename. Unreferenced assets are candidates for deletion.

4. **`zensical.toml` sanity.**
   - `extra_css` and `extra_javascript` paths exist on disk.
   - `site_url` matches the GitHub Pages URL for the repo (`https://<owner>.github.io/<repo>/`).
   - Palette entries are valid theme colors (warn on typos).
   - `nav` has no duplicate labels.

5. **Workflow sanity (`docs.yml`).**
   - Trigger `paths` covers `docs/**`, `zensical.toml`, and the workflow itself.
   - Action versions are pinned to a tag (not `@main`), and major versions look current.
   - `zensical build --clean` matches what the local `make docs` toolchain expects.
   - `pages write` permission present; `id-token: write` present for OIDC deploy.

6. **Local build smoke test (optional, on request).** `uv tool run zensical build --clean` and diff against the committed state. Never commit the resulting `site/` — it's gitignored.

## Report format

Group findings by surface area. One line per issue: `file:line` — problem — proposed fix.

```
zensical.toml:11
  nav entry "Strategies" = "strategies.md"; file exists — ok
zensical.toml:15
  nav entry "Attacks" → docs/attacks.md:42 missing anchor #label-flip referenced from docs/strategies.md:88
    fix: add "## Label flip" heading to docs/attacks.md, or retarget the link

docs/assets/unused-diagram.png
  orphan — no references in docs/ or zensical.toml
    fix: delete, or wire into a page

.github/workflows/docs.yml:27
  astral-sh/setup-uv@v6 — pinned, current
.github/workflows/docs.yml:42
  actions/configure-pages@v6.0.0 — pinned, current
```

End with a one-line verdict and an `apply all / apply selected / skip?` prompt if any fixes are straightforward edits.

## Fan-out pattern

For a full audit (all six checks), dispatch one `Explore` subagent per check so they run in parallel. Each returns its findings grouped by file. Main agent consolidates and presents.

For a narrow request ("just check the nav"), do it inline without subagents.

## Don't touch

- `site/` (generated build output, gitignored).
- Content of `docs/**/*.md` — that's `/docsync` territory. The only content touch allowed here is fixing a broken anchor reference or adding a missing heading when the link is clearly the authoritative side.
- Third-party workflows (anything not under `.github/workflows/docs.yml` unless it also affects docs deploy).

## Hand-offs

- Prose drift or stale commands in docs pages → `/docsync`.
- Slop in doc prose → `/deslop` with `docs/` scope.
- Rewriting unclear prose → `/reslop`.

## Why this skill is quiet

Output is the findings grouped by surface, then the edits on confirmation. No narration of the check process — the grep-and-verify loop is boring; the findings are the product.
