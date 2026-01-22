# Pre-submit Bundle (Chapters 00–06)

This folder is a **self-contained review bundle** that can be zipped and sent to external reviewers (e.g., Springer).

## Manuscript (Markdown)

Location: `pre-submit/chapters/`

- **Single combined file (Ch00–Ch06):** `pre-submit/chapters/book_ch00-06_with_labs_and_solutions.md`
- **Per-chapter combined files:**
  - `pre-submit/chapters/ch00_with_labs_and_solutions.md`
  - `pre-submit/chapters/ch01_with_labs_and_solutions.md`
  - `pre-submit/chapters/ch02_with_labs_and_solutions.md`
  - `pre-submit/chapters/ch03_with_labs_and_solutions.md`
  - `pre-submit/chapters/ch04_with_labs_and_solutions.md`
  - `pre-submit/chapters/ch05_with_labs_and_solutions.md`
  - `pre-submit/chapters/ch06_with_labs_and_solutions.md`

Supporting assets (e.g., figures, JSON artifacts) needed by the Markdown sources are included under:
- `pre-submit/chapters/docs/book/ch00/` … `pre-submit/chapters/docs/book/ch06/`

## Code (self-contained)

Location: `pre-submit/code/chapters0-6/`

This is a standalone code snapshot for Chapters 0–6: simulator (`zoosim/`), scripts, and tests.
See `pre-submit/code/chapters0-6/README.md` for setup and commands.

## Zipping

From the repository root:

```bash
zip -r rl_search_presubmit_ch00-06.zip pre-submit
```

