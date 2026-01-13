# Chapter 3 — Actionable Revision Plan (Springer-Quality, Atomic Edits)

This plan captures concrete fixes for `docs/book/ch03/` and the runnable code it references, following `docs/book/authoring_guidelines.md` and `vlad_foundation_mode.md`.

## Status (2026-01-02)

- Completed: cross-reference hazards fixed; `Code ↔ ...` callouts standardized; labs/code outputs reconciled; Chapter 3 prose polished for math rigor and pedagogy (including discrete-time and measurability fine print); Chapter 3 regression test passes.
- Remaining for a publication gate: compile Chapter 3 to PDF with the project Pandoc/LaTeX toolchain and resolve any LaTeX/citation warnings (environment-dependent).

## Scope

- Primary chapter: `docs/book/ch03/ch03_stochastic_processes_bellman_foundations.md`
- Companion materials: `docs/book/ch03/exercises_labs.md`, `docs/book/ch03/ch03_lab_solutions.md`
- Corresponding code: `scripts/ch03/lab_solutions.py`, `scripts/ch03/__init__.py`, `tests/ch03/test_value_iteration.py`

## Audit Findings (What is blocking “Springer-ready”)

### A. Cross-reference correctness (PDF pipeline)

- `docs/book/ch03/ch03_stochastic_processes_bellman_foundations.md` contains a range-style equation reference `[EQ-1.8--1.10]`.
  - **Why it matters:** `docs/book/crossrefs.lua` treats bracket tokens as a single ID; `[EQ-1.8--1.10]` becomes `\eqref{EQ-1.8--1.10}` (a label that does not exist), producing broken references in PDF builds.
  - **Fix:** Expand to discrete references, e.g. `([EQ-1.8], [EQ-1.9], [EQ-1.10])`.

- The RL-bridge section uses code-formatted `` `#EQ-1.2-prime` `` and `` `CH-11` ``.
  - **Why it matters:** code spans are not processed by `crossrefs.lua`, so the reference remains literal; additionally `CH-*` is not a supported prefix, so it cannot resolve to a cross-reference target.
  - **Fix:** Use plain `#EQ-1.2-prime` (or `[EQ-1.2-prime]`) and plain “Chapter 11”.

### B. “Code ↔ X” conventions (consistency + traceability)

- Chapter 3 uses `!!! note "Code <-> ..."` instead of the project convention `Code ↔ ...`.
  - **Why it matters:** other chapters use `Code ↔ ...` consistently; for strict PDF builds we sanitize Unicode titles to ASCII (e.g., `↔` → `<->`) so compilation never relies on Unicode glyph support.
  - **Fix:** Rename titles to `Code ↔ ...` and update module labels to match actual code layout (e.g., `zoosim.dynamics.reward`, `zoosim.multi_episode.session_env`).

### C. Lab outputs are not reproducible as written (docs vs code mismatch)

- `docs/book/ch03/exercises_labs.md` and `docs/book/ch03/ch03_lab_solutions.md` include iteration counts and contraction statistics that do not match `scripts/ch03/lab_solutions.py` when executed with the stated tolerances and defaults.
  - **Why it matters:** the authoring guidelines require runnable blocks with representative outputs, and the lab solutions assert outputs are “actual results”.
  - **Fix options (choose one, but keep it consistent across files):**
    1. **Update docs to match current code** (preferred if code is correct and stable).
    2. **Update code defaults (seeds, max_iters, tolerances)** to reproduce the outputs currently shown in the docs (preferred only if the shown outputs are already vetted).

### D. Script quality issues (small but user-facing)

- `scripts/ch03/lab_solutions.py` prints “MDP Configuration” twice in Lab 3.1.
- Several docstrings/prints are inaccurate or brittle:
  - `extended_perturbation_sensitivity` docstring mentions “Corollary 3.7.3 (if exists)” but the bound is `[PROP-3.7.4]`.
  - `extended_banach_convergence_verification` prints “rate bound violated 0 times” without actually reporting the computed violations.
  - Discount analysis “Avg Ratio” computation includes a degenerate 0-ratio term (indexing bug) and can skew the reported value.
- Default `max_iters=500` is too small for high discounts (e.g. `gamma=0.99`) at tight tolerances; this causes non-convergence and downstream doc mismatches.

## Revision Plan (Atomic, Executable)

### 1) Fix Chapter 3 cross-reference hazards

- [x] Replace `[EQ-1.8--1.10]` with explicit equation references.
  - **How:** edit the Summary section bullet to `([EQ-1.8], [EQ-1.9], [EQ-1.10])`.
  - **Verify:** `rg -n "\\[EQ-[^\\]]*--[^\\]]*\\]" docs/book/ch03/ch03_stochastic_processes_bellman_foundations.md` returns no matches.

- [x] Replace code-formatted `` `#EQ-1.2-prime` `` and `` `CH-11` `` with resolvable references.
  - **How:** use `#EQ-1.2-prime` (or `[EQ-1.2-prime]`) in prose; replace `CH-11` with “Chapter 11”.

### 2) Standardize “Code ↔ X” callouts and module labels

- [x] Rename `Code <-> ...` callouts to `Code ↔ ...` and correct module names.
  - **How:** update the three Chapter 3 admonition titles; keep the existing file:line refs, but correct the `MOD-*` label text to align with:
    - `zoosim/dynamics/reward.py`
    - `zoosim/multi_episode/session_env.py`
    - `zoosim/multi_episode/retention.py`

### 3) Bring Chapter 3 “Production Checklist” into the house style

- [x] Convert the current bold “Production Checklist” block to a numbered section + `!!! tip` admonition (like Chapter 2).
  - **How:** add `## 3.13 Production Checklist` followed by `!!! tip "Production Checklist (Chapter 3)"` and move the checklist bullets inside the admonition.

### 4) Make labs reproducible (code first, then docs)

- [x] Fix `scripts/ch03/lab_solutions.py` output bugs and non-convergence.
  - **How:**
    - Remove duplicated “MDP Configuration” prints in Lab 3.1.
    - Increase `value_iteration` `max_iters` default (or thread `max_iters` through lab entry points) so `gamma=0.99` converges at the tolerances used in the docs.
    - Correct the perturbation lab docstring to reference `[PROP-3.7.4]`.
    - Report Banach rate violations accurately (aggregate actual violations).
    - Fix discount analysis ratio computation (skip the degenerate last-step ratio).
  - **Verify:** run `.venv/bin/python scripts/ch03/lab_solutions.py --all` (should finish quickly; the MDP is tiny).

- [x] Align `docs/book/ch03/exercises_labs.md` outputs with the fixed code.
  - **How:** re-run the code blocks (or call the corresponding functions) with fixed seeds and paste updated “Output:” blocks.
  - **Guideline:** keep outputs short and representative; prefer summary statistics over long tables when possible.

- [x] Align `docs/book/ch03/ch03_lab_solutions.md` outputs and run commands.
  - **How:**
    - Update any numeric outputs that no longer match after the code fixes.
    - Standardize execution commands to `.venv/bin/python ...` throughout.
  - **Verify:** spot-check at least Lab 3.1 and Lab 3.2 by running the import snippets from the markdown.

### 5) Cleanup: chapter polish items (small, safe edits)

- [x] Reorder “What comes next” bullets in §3.11 to follow chapter order (4–5, 6, 7, 9, 10, 11).
- [x] Add Sutton & Barto to the “Key references” list if we cite it in-text (it is currently cited in Remark 3.7.7).

### 6) Math/pedagogy polish (Springer tone)

- [x] Fix minor measure-theory details (define $\mathcal{F}_\infty$, correct stopped-process proof edge case at $t=0$).
- [x] Make the discrete-time convention explicit to avoid statement/proof mismatches in Section 3.2.
  - **Why:** Proposition 3.2.1 is proved by an indicator decomposition that is specific to discrete time; making the discrete-time convention explicit keeps the presentation Bourbaki-clean while staying aligned with RL practice.
  - **How:** add a “Standing convention (Discrete time)” note in §3.2 and make Proposition 3.2.1 assume $T=\mathbb{N}$.
- [x] Remove nonstandard proof tagging (`\\tag{*}`) and tighten operator/notation consistency (bandit case uses $\sup$ and clarifies independence from $V$).
- [x] Align Bellman-operator measurability caveats with Chapter 2’s selection-theorem discussion.
  - **Why:** In general state/action spaces, measurability of the control operator requires topological hypotheses (Chapter 2, §2.8.2); Chapter 3 should not silently overclaim $\mathcal{T}:B_b(\mathcal{S})\to B_b(\mathcal{S})$ without pointing back to that fine print.
  - **How:** in Remark 3.7.0, keep the boundedness argument and reference [PROP-2.8.2] and [THM-2.8.3] for measurability.
- [x] Revise §3.1 motivation and §3.10 bridges to a more formal “we” voice and reduce informal emphasis in lab-solution prose.
- [x] Remove non-numeric `\\tag{Rate}` from Lab 3.2 and cite [EQ-3.18] instead.
  - **Why:** the authoring guidelines require numbered/tagged equations to have stable anchors; a free-form `\\tag{Rate}` is not a stable book reference and can introduce LaTeX/PDF hygiene issues.
  - **How:** rewrite the Lab 3.2 rate discussion in terms of [COR-3.7.3] / [EQ-3.18].

## Verification Checklist (Run in this repo)

All commands assume project root and the existing venv:

- `.venv/bin/pytest -q tests/ch03/test_value_iteration.py`
- `.venv/bin/python scripts/ch03/lab_solutions.py --lab 3.1`
- `.venv/bin/python scripts/ch03/lab_solutions.py --lab 3.2`

If the environment is missing dependencies, we will use `uv` to reconcile `.venv` with `uv.lock` (may require elevated permissions due to restricted network policy):

- `uv sync`
