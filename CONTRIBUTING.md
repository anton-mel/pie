# Maintaining the Pie book

The `main` branch assembles and publishes the book. Tutorial branch READMEs remain the
narrative source; `chapters.json` pins each chapter to a commit and holds its
contents-page description and recommended-reading flag. The generated `docs/`
content and `book/` output are ignored by Git.

Install mdBook 0.5.4, then run from the repository root:

```sh
git fetch origin
python3 scripts/import-book.py
python3 scripts/validate-book.py
mdbook build
mdbook serve --open
```

To update a chapter, edit its tutorial branch README/code, fetch the new
commit, and update `commit` in `chapters.json` on `main`. To add a chapter,
append an entry with the next consecutive number, its tutorial branch and
pinned commit, a unique slug, description, recommended-reading flag, and
aliases. Rebuild and validate the book before opening a PR targeting `main`.
Keep `number` and `slug` stable
so published URLs survive updates. Add an `aliases` entry when an inline file
reference is ambiguous, mapping the text in backticks to a complete source
path. The importer resolves exact paths, unique suffixes, and directory entry
files. Symbols remain inline code. All tracked UTF-8 source files except the
README and Cargo.lock are exported per chapter and available in the viewer's
file selector. Git commits must be available locally; CI fetches full history.

The chapter 15 layout illustration is a fenced block and is preserved. Add
explicit source links in its branch README if a guided read order is desired.
Future line-range annotations and draggable panel sizing are enhancements;
the initial reader uses a fixed desktop width and full-screen mobile drawer.

The introduction uses the maintained `templates/introduction.md` template.
Edit its opening, recommended path, and reading guidance there. The importer
fills its chapter index from `chapters.json`; section headings are shared with
the sidebar in `scripts/import-book.py`. Contents-page styling lives in
`theme/contents.css`. Keep the recommended path consistent with the manifest
flags when changing recommendations.
Do not hand-edit generated pages or source snapshots. The raw `.txt` links
remain usable when JavaScript is disabled.

The standalone **Inspecting an Inferlet** chapter is maintained in
`templates/inspecting-an-inferlet.md`. Its manifest entry specifies a
`source_dir` of checked-in reference files, their upstream repository and
commit, and SHA-256 checksums. This lets CI build without a sibling Pie
checkout. To update it, copy the listed files from a clean checkout of the
new upstream commit and update the commit and checksums together.
`templates/inspecting-an-inferlet.json` defines inclusive source line ranges
for each `{{compare:name}}` block. The importer renders Rust, Python, and
JavaScript columns from those files, omits whole comment lines, and removes common
indentation. Update
the ranges and explanations when changing the sources, then inspect the
rendered chapter. Complete files remain available in the source panel.
Comparison styling lives in `theme/inspection.css` and stacks on mobile.

## GitHub Pages

In repository Settings → Pages, select **GitHub Actions** as the build source.
Allow `main` in the `github-pages` environment's deployment branch rules.
This is a one-time repository setup; the workflow builds and uploads the HTML
directly, so no branch containing generated HTML is needed.

Pushes to `main` (including merged PRs) build and deploy the book. Pull requests
targeting `main` build and validate the book and upload a downloadable preview
artifact without deploying. Manual runs deploy only when `main` is selected;
manual runs on other branches only build and upload the preview artifact.
If Pages is enabled after the first push, rerun that workflow after setup.

The upstream project URL is https://anton-mel.github.io/pie/.
CI derives the site path and repository button from the current repository,
so the same workflow publishes a fork at
`https://<owner>.github.io/<repository>/` without configuration edits.
For example, `Giotyp/pie-book` publishes at https://giotyp.github.io/pie-book/.
The defaults in `book.toml` target upstream; chapter source links always point
to the pinned tutorial commits in `anton-mel/pie`. The standalone inspection
chapter retains its reference source links to `pie-project/pie`.

The workflow pins mdBook 0.5.4. Update the local and CI versions together.
