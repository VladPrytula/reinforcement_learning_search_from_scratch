# Chapter 4 — Actionable Revision Plan (Springer-Quality, Atomic Edits)

This plan captures concrete fixes for `docs/book/ch04/` and the runnable code it references, following `docs/book/authoring_guidelines.md`, `vlad_prytula.md`, and the syllabus expectations in `docs/book/syllabus.md`.

## Status (2026-01-02)

- Current blockers: none known; run the verification checklist to confirm.
- Target outcome: Chapter 4 prose and labs compile to the house voice, numeric outputs match current simulator defaults (with stated seeds), and determinism claims are backed by tests for catalog + (user, query) sampling.

## Scope

- Primary chapter: `docs/book/ch04/ch04_generative_world_design.md`
- Companion labs: `docs/book/ch04/exercises_labs.md`
- Related demo: `scripts/ch04/ch04_demo.py`
- Referenced simulator modules: `zoosim/world/{catalog,users,queries}.py`, `zoosim/core/config.py`
- Tests: `tests/test_catalog_stats.py`

## Audit Findings (What blocks “Springer-ready”)

### A. Voice and tone (Vlad voice, no second person)

- Both Chapter 4 and its labs include second-person passages (“you …”) and informal contractions (“can’t”, “it’s”, “Let’s”).
  - **Why it matters:** violates `docs/book/authoring_guidelines.md` and the unified author voice in `vlad_prytula.md`.

### B. Docs ↔ code drift (correctness)

- Segment mixture weights and segment parameter values in Chapter 4 and labs do not match the simulator defaults in `zoosim/core/config.py`.
  - **Why it matters:** readers cannot reproduce stated tables/plots; later chapters already assume the `0.35/0.25/0.15/0.25` mix introduced in Chapter 2.

- Embedding validation in Chapter 4 asserts “dog-food similarity > toys similarity” using cosine similarity, but this ordering is not guaranteed given random centroid norms.
  - **Why it matters:** the claim can fail even when the implementation is correct.

- The “Complete Catalog Generation” code block is not runnable as written (missing `sample_discount`, mismatched `sample_embedding` signature).

### C. Reference hygiene (local consistency)

- `docs/book/ch04/exercises_labs.md` references “Figure 4.1” which does not exist in Chapter 4.
- Minor Markdown glitch: an extra trailing backtick in a `Code ↔ ...` callout.

### D. Determinism claims not fully tested

- `tests/test_catalog_stats.py` tests catalog determinism, but Chapter 4’s determinism definition also asserts deterministic user/query sampling sequences.

## Revision Plan (Atomic, Executable)

### 1) Fix voice violations and remove informal contractions

- [x] Replace second-person prose with formal “we” voice in:
  - `docs/book/ch04/ch04_generative_world_design.md`
  - `docs/book/ch04/exercises_labs.md`
- [x] Replace informal contractions (`can’t`, `it’s`, `Let’s`, …) with formal equivalents.

### 2) Align Chapter 4 math/text with simulator defaults

- [x] Correct the lognormal equivalence to use $\log X \sim \mathcal{N}(\mu, \sigma^2)$ (std vs variance clarity).
- [x] Align the segment mix + per-segment distributions with `zoosim/core/config.py`.
- [x] Align query-intent coupling expectations with current Dirichlet parameters (notably `litter_heavy`).
- [x] Align the utility preview in Chapter 4 with the production click model in Chapter 2 (Definition 2.5.3 / [EQ-2.4]) rather than introducing a conflicting form.

### 3) Make runnable blocks reproducible (docs match code)

- [x] Replace the broken “Complete Catalog Generation” listing with calls to `zoosim/world/catalog.py::generate_catalog`.
- [x] Replace embedding validation metric with the provable quantity $\mathbb{E}\|e-\mu_c\|_2^2 = d\sigma_c^2$ and update the “Output:” block accordingly.
- [x] Recompute and update representative numeric outputs for the fixed seeds used in the chapter/labs.

### 4) Fix lab sheet inconsistencies

- [x] Update expected tables/numbers in `docs/book/ch04/exercises_labs.md` (segment mix, affinities, query-type counts, LLN example).
- [x] Remove or replace the broken “Figure 4.1” reference.
- [x] Rename headings containing “Your” (e.g., “Testing Your Solutions”) to match the no-second-person rule.

### 5) Code hygiene + determinism tests

- [x] Confirm no stray CR character in `zoosim/world/users.py`.
- [x] Extend `tests/test_catalog_stats.py` with deterministic sampling tests for users and queries (bit-for-bit, including embeddings).

## Verification Checklist (Run in this repo)

All commands assume repo root and the existing venv:

- `uv run python scripts/audit_ch00_ch03_alignment.py --files docs/book/ch04/ch04_generative_world_design.md docs/book/ch04/exercises_labs.md`
- `uv run pytest -q tests/test_catalog_stats.py`
- `uv run python scripts/ch04/ch04_demo.py`
- `uv run python scripts/validate_knowledge_graph.py`
