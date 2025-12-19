# Pre-Press Pipeline: Complete 7-Phase Process

## Overview

This document describes the complete pre-press pipeline for preparing textbook chapters for Springer publication. The pipeline is designed to achieve **100% verification** of all content---not just syntactic correctness, but semantic accuracy.

---

## Pipeline Diagram

```
═══════════════════════════════════════════════════════════════════════════════
                         PRE-PRESS PIPELINE v1.0
═══════════════════════════════════════════════════════════════════════════════

  ┌──────────────┐
  │   PHASE 0    │  INITIALIZATION
  │   Setup      │  • Load personas
  └──────┬───────┘  • Validate Knowledge Graph
         │          • Create tracking structure
         ▼
  ┌──────────────┐
  │   PHASE 1    │  BUILD ANCHOR REGISTRY
  │   Registry   │  • Extract all anchors with semantic context
  └──────┬───────┘  • Reconcile with Knowledge Graph
         │          • Output: registries/registry_chXX.yaml
         ▼
  ┌──────────────┐
  │   PHASE 2    │  INTERNAL REFERENCE AUDIT
  │   Internal   │  • Syntactic validation (existence)
  └──────┬───────┘  • Semantic validation (correctness)
         │          • Cross-chapter validation
         │          • Output: audits/audit_chXX.md, fixes/fixes_chXX.yaml
         ▼
  ┌──────────────┐
  │   PHASE 3    │  EXTERNAL REFERENCE AUDIT
  │   External   │  • URL validity
  └──────┬───────┘  • Paper/DOI verification
         │          • Claim verification
         │          • Output: audits/external_chXX.md
         ▼
  ┌──────────────┐
  │   PHASE 4    │  FIX APPLICATION
  │   Fixes      │  • Pre-flight validation
  └──────┬───────┘  • Apply edits
         │          • Post-flight verification
         │          • Output: fixes/fix_report_chXX.md
         ▼
  ┌──────────────┐
  │   PHASE 5    │  CONTENT REVIEWS (parallel tracks)
  │   Reviews    │  • Track A: Mathematical rigor
  └──────┬───────┘  • Track B: Pedagogical quality
         │          • Track C: RL-practice bridge
         │          • Output: reviews/review_chXX_*.md
         ▼
  ┌──────────────┐
  │   PHASE 6    │  CROSS-VERIFICATION
  │   X-Check    │  • Uncertain claims → /cross-check
  └──────┬───────┘  • Multi-LLM triangulation
         │          • Output: inline in reports
         ▼
  ┌──────────────┐
  │   PHASE 7    │  FINAL VALIDATION
  │   Final      │  • Cross-chapter audit
  └──────────────┘  • Knowledge Graph sync
                    • PDF compilation test
                    • Output: status/final_validation.md

═══════════════════════════════════════════════════════════════════════════════
```

---

## Phase 0: Initialization

### Purpose

Set up the environment and establish a baseline before processing chapters.

### When to Run

- Once at the start of the pre-press process
- Again if Knowledge Graph has been significantly updated

### Steps

1. **Load revision persona**
   ```
   @vlad_prytula.md @vlad_revision_mode.md
   ```

2. **Validate Knowledge Graph baseline**
   ```bash
   python docs/knowledge_graph/validate_kg.py
   ```

   This confirms the Knowledge Graph is internally consistent before we compare chapters against it.

3. **Generate chapter inventory**
   ```
   /audit-refs all --inventory-only
   ```

   Produces a summary of all chapters:
   - File count and locations
   - Anchor counts by type (THM, DEF, EQ, etc.)
   - Reference counts
   - Estimated complexity

4. **Create tracking structure**

   Initialize `pre-press/status/overview.md` with:
   - Chapter list with status
   - Target completion dates
   - Blockers and dependencies

### Outputs

| Output | Location | Purpose |
|--------|----------|---------|
| KG validation report | stdout | Confirm baseline is clean |
| Chapter inventory | `pre-press/status/inventory.md` | Planning reference |
| Tracking spreadsheet | `pre-press/status/overview.md` | Progress tracking |

### Success Criteria

- [ ] Knowledge Graph validates without errors
- [ ] All chapter files are accessible
- [ ] Tracking structure created

---

## Phase 1: Build Anchor Registry

### Purpose

Create the **source of truth** for all anchors in a chapter. This registry includes not just the anchor ID, but semantic context that enables verification.

### When to Run

- First phase for each chapter
- Re-run if chapter structure changes significantly

### Command

```
/audit-refs ch03 --build-registry
```

### What It Extracts

For each anchor `{#ID}`:

```yaml
- id: THM-3.7.1
  file: ch03_stochastic_processes_bellman_foundations.md
  line: 596
  type: theorem

  # Semantic context
  title: "Bellman Operator Contraction"
  snippet: "The Bellman operator T defined by (TV)(s) = max_a {R(s,a) + γ Σ P(s'|s,a)V(s')} is a γ-contraction under the supremum norm..."

  # Auto-extracted topics
  topics:
    - Bellman
    - operator
    - contraction
    - gamma
    - supremum-norm

  # Mathematical relationships
  defines:
    - T-contraction-property
  uses:
    - DEF-3.4.1  # MDP definition
    - EQ-3.8     # Bellman equation
  forward_refs:
    - THM-6.2    # Regret bound uses this
```

### Knowledge Graph Reconciliation

The skill compares the extracted registry with `docs/knowledge_graph/graph.yaml`:

| Issue Type | Description | Action |
|------------|-------------|--------|
| **Missing in KG** | Anchor exists in chapter but not in Knowledge Graph | Add to KG |
| **Stale in KG** | KG entry points to non-existent anchor | Remove from KG |
| **File mismatch** | KG says anchor is in file X, but it's in file Y | Update KG |
| **Metadata drift** | KG title differs from actual title | Update KG |

### Output

```yaml
# pre-press/registries/registry_ch03.yaml

chapter: ch03
generated: 2025-01-15T10:30:00
files_scanned:
  - ch03_stochastic_processes_bellman_foundations.md
  - exercises_labs.md
  - ch03_lab_solutions.md

statistics:
  total_anchors: 45
  by_type:
    theorem: 8
    definition: 12
    equation: 22
    lemma: 2
    corollary: 1

kg_sync:
  in_sync: 42
  missing_in_kg: 2
  stale_in_kg: 1
  file_mismatch: 0

anchors:
  - id: THM-3.5.1-Bellman
    file: ch03_stochastic_processes_bellman_foundations.md
    line: 380
    type: theorem
    title: "Bellman Expectation Equation"
    snippet: "For any policy π, the value function V^π satisfies..."
    topics: [Bellman, expectation, policy, value-function]
    defines: [bellman-expectation-equation]
    uses: [DEF-3.4.3, DEF-3.4.4]

  # ... all other anchors
```

### Success Criteria

- [ ] All anchors extracted with semantic context
- [ ] Knowledge Graph discrepancies documented
- [ ] Registry YAML validates

---

## Phase 2: Internal Reference Audit

### Purpose

Verify all internal references are correct---both syntactically (target exists) and semantically (target matches what the reference claims).

### When to Run

- After Phase 1 (needs registry)
- Re-run after any chapter edits

### Command

```
/audit-refs ch03 --full
```

### Verification Levels

#### Level 1: Syntactic (Existence)

For every reference `[REF-X.Y.Z]`:
- Does `{#REF-X.Y.Z}` exist in the registry?
- Is it in the expected chapter?

**Catches**: Typos, old numbering, deleted anchors

#### Level 2: Topic Alignment

For every reference:
1. Extract surrounding context (±100 characters)
2. Infer claimed topic from context
3. Compare with anchor's actual topics
4. Compute similarity score

**Example**:
```
Reference at ch03_lab_solutions.md:17:
  Context: "Recall from [THM-3.3.3] that the Bellman operator is a contraction"
  Claimed topic: Bellman operator contraction

Anchor THM-3.3.3:
  Status: DOES NOT EXIST

Anchor THM-3.7.1:
  Topics: [Bellman, operator, contraction, gamma, supremum-norm]
  Similarity to claim: 0.95

Verdict: Reference [THM-3.3.3] should be [THM-3.7.1]
```

**Catches**: References to wrong theorems with similar numbering

#### Level 3: Claim Verification

For references with explicit claims ("By [X], property Y holds"):
1. Parse the claim from surrounding text
2. Retrieve the full anchor content
3. LLM analysis: Does the anchor support this claim?

**Example**:
```
Reference at ch03.md:1186:
  Text: "Chapter 10 applies CMDP theory from §3.6"
  Claim: Section 3.6 contains CMDP theory

Section 3.6 actual content:
  Title: "Banach Fixed-Point Theorem"
  Topics: [Banach, fixed-point, contraction, metric-space]

Verdict: ❌ CLAIM MISMATCH
  Section 3.6 is about Banach fixed-point, not CMDP.
  CMDP content is in Remark 3.5.3.
  Suggested fix: "Chapter 10 applies CMDP theory from Remark 3.5.3"
```

**Catches**: Misattributed claims, confused section references

#### Level 4: Cross-Chapter Validation

For references to other chapters:
- **Forward refs**: "Chapter 9 develops X" → Does Ch9 contain X?
- **Backward refs**: "Deferred from Ch1" → Does Ch1 preview this?

### Output

```markdown
# Reference Audit: Chapter 3

Generated: 2025-01-15T11:00:00
Files audited: 3
Total references: 128

## Executive Summary

| Category | Count | Details |
|----------|-------|---------|
| Verified | 116 | All checks passed |
| Critical | 5 | Broken references |
| Semantic | 4 | Wrong target |
| Warnings | 3 | Potential issues |

---

## 🔴 Critical: Dangling References (5)

These references point to non-existent anchors:

| Reference | Location | Context | Suggested Fix |
|-----------|----------|---------|---------------|
| `[THM-3.3.3]` | ch03_lab_solutions.md:17 | "Bellman operator contraction" | → `[THM-3.7.1]` |
| `[THM-3.3.2]` | ch03_lab_solutions.md:122 | "Banach fixed-point" | → `[THM-3.6.2-Banach]` |
| `[EQ-3.3]` | ch03_lab_solutions.md:42 | "contraction inequality" | → `[EQ-3.16]` |
| `Theorem 3.9` | exercises_labs.md:85 | "convergence rate" | → `COR-3.7.3` |
| `[EQ-3.18]` | ch03_main.md:1229 | "rate bound" | Add anchor to COR-3.7.3 |

---

## 🟠 Semantic Issues: Wrong Target (4)

These references exist but point to incorrect content:

| Reference | Location | Claims | Actually | Fix |
|-----------|----------|--------|----------|-----|
| §3.6 | ch00.md:307 | CMDP/Lagrangian | Banach fixed-point | → Remark 3.5.3 |
| Chapter 9 | ch03.md:586 | TD-learning | OPE | → Chapter 7 |
| ... | ... | ... | ... | ... |

---

## 🟡 Warnings (3)

| Issue | Location | Details |
|-------|----------|---------|
| Duplicate anchor | `{#EQ-3.3}` | lab_solutions.md:42 collides with ch03_main.md:145 |
| ... | ... | ... |

---

## ✅ Verified (116)

All other references passed all verification levels.
[Expandable list or link to full verification log]
```

### Fix Specification

Alongside the report, generate machine-readable fix spec:

```yaml
# pre-press/fixes/fixes_ch03.yaml

chapter: ch03
generated: 2025-01-15T11:00:00
generated_from: audit_ch03.md

fixes:
  - type: reference_update
    old: "[THM-3.3.3]"
    new: "[THM-3.7.1]"
    files:
      - docs/book/ch03/ch03_lab_solutions.md
    occurrences: 3
    reason: "THM-3.3.3 was old numbering; THM-3.7.1 is Bellman contraction"
    semantic_verified: true
    confidence: 0.95

  - type: reference_update
    old: "[THM-3.3.2]"
    new: "[THM-3.6.2-Banach]"
    files:
      - docs/book/ch03/ch03_lab_solutions.md
    occurrences: 2
    reason: "THM-3.3.2 was old numbering; THM-3.6.2-Banach is Banach fixed-point"
    semantic_verified: true
    confidence: 0.98

  - type: anchor_collision
    anchor: "{#EQ-3.3}"
    primary_file: docs/book/ch03/ch03_stochastic_processes_bellman_foundations.md
    collision_file: docs/book/ch03/ch03_lab_solutions.md
    resolution: "Remove from lab_solutions.md, use [EQ-3.16] instead"
    reason: "Lab defines contraction inequality as EQ-3.3, but main chapter uses EQ-3.3 for Q-function"

  - type: section_update
    old: "Section 3.6"
    new: "Remark 3.5.3"
    context: "CMDP/Lagrangian theory"
    files:
      - docs/book/ch00/ch00_motivation_first_experiment_revised.md
    reason: "CMDP content moved to Remark 3.5.3; Section 3.6 is now Banach"
    semantic_verified: true
    confidence: 0.92

  - type: add_anchor
    anchor: "{#EQ-3.18}"
    after_pattern: "\\|V_k - V^\\*\\|_\\infty \\leq \\gamma^k"
    file: docs/book/ch03/ch03_stochastic_processes_bellman_foundations.md
    reason: "Rate bound equation referenced but no anchor exists"
```

### Success Criteria

- [ ] All references verified at all 4 levels
- [ ] No critical issues remaining (or documented blockers)
- [ ] Fix specification generated and reviewed

---

## Phase 3: External Reference Audit

### Purpose

Verify all external references (URLs, papers, books) are accessible, accurate, and correctly represent their sources.

### When to Run

- After Phase 2 (internal refs should be clean first)
- Re-run periodically (links can break over time)

### Command

```
/verify-external-refs ch03
```

### Reference Types

#### arXiv Citations

**Pattern**: `[@author:arxiv:XXXX.XXXXX]` or inline `arXiv:XXXX.XXXXX`

**Verification**:
1. Fetch `https://arxiv.org/abs/XXXX.XXXXX`
2. Extract: title, authors, year, abstract
3. Compare with citation metadata
4. Check if newer version exists (v1 → v3)
5. If surrounding text makes claims, verify against abstract

**Example**:
```
Citation: [@mnih:dqn:2015]
URL: arXiv:1312.5602
Fetched title: "Playing Atari with Deep Reinforcement Learning"
Fetched authors: Mnih et al.
Latest version: v3 (2013-12-24)

Status: ✅ Metadata matches, no newer version
```

#### DOI References

**Pattern**: `[@author:doi:10.XXXX/...]` or DOI URLs

**Verification**:
1. Resolve via `https://doi.org/10.XXXX/...`
2. Fetch publisher metadata
3. Verify title, authors, journal, year
4. Check for retractions/corrections (when possible)

#### Direct URLs

**Pattern**: `https://...` or `[text](url)`

**Verification**:
1. HTTP HEAD request
2. If 200: Content accessible
3. If 404/5xx: Flag as broken
4. If redirect: Note final URL
5. For broken links: Check Archive.org

#### Book Citations

**Pattern**: `[@author:book:year]`

**Verification**:
1. Search Google Books / WorldCat
2. Verify ISBN if provided
3. Check edition/year accuracy
4. Note: Often requires manual verification flag

### Claim Verification

For citations with surrounding claims:

```
Text: "Bertsekas [2012] proves that value iteration converges in O(1/(1-γ)) iterations"

Verification:
1. Extract claim: "value iteration converges in O(1/(1-γ))"
2. Fetch source (if available online)
3. LLM analysis: Does Bertsekas prove this?
4. Result: ✅ Confirmed (Theorem 1.3.1 in Bertsekas DP Vol 2)
   or
   Result: 🟡 Uncertain - claim may overstate (Bertsekas proves bound, not tight)
```

For uncertain claims (confidence < 0.8):
- Automatically invoke `/cross-check`
- Require multi-LLM consensus

### Output

```markdown
# External Reference Audit: Chapter 3

Generated: 2025-01-15T12:00:00

## Summary

| Category | Count |
|----------|-------|
| Total external refs | 23 |
| Verified | 19 |
| Broken links | 1 |
| Metadata mismatch | 2 |
| Claim issues | 1 |

---

## 🔴 Broken Links (1)

| Reference | URL | Status | Archive |
|-----------|-----|--------|---------|
| "MDN docs" | https://example.com/old-page | 404 | [Archive link] |

---

## 🟠 Metadata Mismatches (2)

| Citation | Issue | Fix |
|----------|-------|-----|
| [@sutton:rl_book:2018] | Actual publication 2020 (2nd ed) | Update year |
| [@bertsekas:dp:2012] | Title differs: "Dynamic Programming..." vs "Neuro-Dynamic..." | Verify which volume |

---

## 🟡 Claim Issues (1)

| Citation | Claim | Issue | Confidence |
|----------|-------|-------|------------|
| [@puterman:mdps:2014] | "Puterman proves strict contraction" | Puterman proves contraction; strict requires additional assumption | 0.65 |

Cross-check result: Gemini agrees, GPT suggests clarifying "γ < 1" assumption.

---

## ✅ Verified (19)

All other external references verified.
```

### Success Criteria

- [ ] All URLs checked
- [ ] All paper citations verified
- [ ] Claim issues flagged for review
- [ ] No broken links (or documented with alternatives)

---

## Phase 4: Fix Application

### Purpose

Systematically apply fixes from Phases 2-3 with validation before and after.

### When to Run

- After Phases 2 and 3 are complete
- After reviewing and approving fix specifications

### Command

```
/fix-refs pre-press/fixes/fixes_ch03.yaml
```

### Pre-Flight Validation

Before applying any fix:

1. **Verify old pattern exists** in specified files
2. **Verify new target exists** (for reference updates)
3. **Check for collisions** - will the fix create new problems?
4. **Semantic sanity check** - does the fix make sense?

```
Pre-flight report:

Fix 1: [THM-3.3.3] → [THM-3.7.1]
  ✓ Old pattern found: 3 occurrences in ch03_lab_solutions.md
  ✓ New target exists: THM-3.7.1 at ch03_main.md:596
  ✓ No collision: No existing [THM-3.7.1] refs in target file
  ✓ Semantic check: Both refer to Bellman contraction
  → Ready to apply

Fix 2: ...
```

### Fix Application

For each fix:
1. Read the file
2. Find all occurrences of old pattern
3. Show context around each occurrence
4. Apply replacement
5. Log the change

### Post-Flight Validation

After all fixes:
1. Re-extract references from modified files
2. Verify no new dangling references
3. Quick audit check

```
Post-flight report:

Files modified: 3
Total changes: 12

Validation:
  ✓ No new dangling references
  ✓ All fixes applied successfully
  ⚠ 1 warning: New reference [EQ-3.16] - verify this is intentional

Re-audit recommended: /audit-refs ch03 --quick
```

### Output

```markdown
# Fix Report: Chapter 3

Applied: 2025-01-15T13:00:00

## Changes Made

| Fix | File | Occurrences | Status |
|-----|------|-------------|--------|
| [THM-3.3.3] → [THM-3.7.1] | ch03_lab_solutions.md | 3 | ✅ Applied |
| [THM-3.3.2] → [THM-3.6.2-Banach] | ch03_lab_solutions.md | 2 | ✅ Applied |
| [EQ-3.3] → [EQ-3.16] | ch03_lab_solutions.md | 4 | ✅ Applied |
| Remove {#EQ-3.3} collision | ch03_lab_solutions.md | 1 | ✅ Applied |
| "Section 3.6" → "Remark 3.5.3" | ch00.md | 1 | ✅ Applied |
| Add {#EQ-3.18} anchor | ch03_main.md | 1 | ✅ Applied |

## Verification

- [x] No new dangling references created
- [x] All fixes applied at expected locations
- [ ] Knowledge Graph updated (pending)
- [ ] Re-audit confirms clean (run /audit-refs ch03 --quick)

## Files Modified

- docs/book/ch03/ch03_lab_solutions.md (10 changes)
- docs/book/ch03/ch03_stochastic_processes_bellman_foundations.md (1 change)
- docs/book/ch00/ch00_motivation_first_experiment_revised.md (1 change)

## Git Diff Summary

[Expandable diff or link to git diff output]
```

### Iteration

If post-flight finds issues:
```
while issues_remain:
    /audit-refs ch03 --quick
    if clean:
        break
    else:
        review new issues
        update fixes_ch03.yaml
        /fix-refs fixes_ch03.yaml
```

### Success Criteria

- [ ] All fixes applied successfully
- [ ] Post-flight validation passes
- [ ] Re-audit shows no critical issues
- [ ] Knowledge Graph updated

---

## Phase 5: Content Reviews

### Purpose

Deep review of content quality across three dimensions: mathematical rigor, pedagogical quality, and theory-practice integration.

### When to Run

- After Phase 4 (references should be clean)
- Can run all three tracks in parallel

### Track A: Mathematical Rigor

```
/review-math ch03
```

**Checks**:
- Proof correctness (every step valid)
- Notation consistency (within chapter, across book)
- Theorem-hypothesis alignment (proof uses all hypotheses, no extras)
- Edge cases handled
- Counterexamples correct
- Exercise solutions correct

**With persona for depth**:
```
@vlad_prytula.md @vlad_foundation_mode.md
/review-math ch03
```

### Track B: Pedagogical Quality

```
@vlad_revision_mode.md
/review-pedagogy ch03
```

**Checks** (per vlad_revision_mode.md):
- Flow and transitions (5-pass method)
- Prose-bullet balance (~60/40)
- Algorithm explanations ("Why does this work?")
- Theorem interpretations ("What this tells us")
- Design choice justifications
- Section connectivity

### Track C: RL-Practice Bridge

```
/review-rl-bridge ch03
```

**Checks**:
- Theory-practice connections clear
- Code-math alignment (implementations match theory)
- Practical relevance of theoretical results
- Honest treatment of theory-practice gaps
- Code↔Config boxes accurate

### Output

Each track produces a review report:

```
pre-press/reviews/
├── review_ch03_math.md
├── review_ch03_pedagogy.md
└── review_ch03_rl.md
```

### Success Criteria

- [ ] Mathematical review: No proof errors, notation consistent
- [ ] Pedagogical review: Prose-bullet balance achieved, flow smooth
- [ ] RL bridge review: Theory-practice gaps honestly addressed

---

## Phase 6: Cross-Verification

### Purpose

Use multi-LLM triangulation for uncertain or complex items identified in earlier phases.

### When to Run

- Automatically during Phases 2, 3, 5 for uncertain items
- Manually for particularly complex claims

### Command

```
/cross-check "Claim or question to verify"
```

### When It's Invoked

| Trigger | Example |
|---------|---------|
| External claim confidence < 0.8 | "Paper X proves Y" - uncertain if accurate |
| Complex mathematical claim | Proof correctness for subtle argument |
| Architectural decision | Design choice with non-obvious tradeoffs |
| Reviewer disagreement | Math review says X, pedagogy review says Y |

### Process

1. **Reformulate query per model**:
   - Gemini: "Verify step-by-step: [claim]"
   - GPT/Codex: "Challenge this reasoning: [claim]"

2. **Query both models**

3. **Compare responses**:
   - Points of agreement
   - Points of divergence
   - Confidence levels

4. **Synthesize verdict**:
   - If consensus: Accept with high confidence
   - If divergence: Flag for human review with both perspectives

### Output

Inline in the relevant report:

```markdown
## Cross-Verification: THM-3.7.1 Proof Step 3

Query: "The proof uses max-stability: max{a+c, b+c} = max{a,b}+c.
Verify this is applied correctly when going from line 4 to line 5."

Gemini response:
  ✓ Application is correct
  Note: Relies on R(s,a) being bounded (Assumption 3.4.1)

GPT response:
  ✓ Application is correct
  Suggestion: Could add parenthetical noting the bounded reward assumption

Synthesis:
  Verdict: ✅ Proof step is correct
  Recommendation: Consider adding clarifying note about bounded rewards
  Confidence: 0.95
```

### Success Criteria

- [ ] All flagged items cross-verified
- [ ] Divergences resolved or documented
- [ ] Human review completed for remaining uncertainties

---

## Phase 7: Final Validation

### Purpose

Comprehensive cross-chapter validation and publication readiness check.

### When to Run

- After all chapters complete Phases 1-6
- Final gate before declaring "Springer-ready"

### Checks

#### 1. Cross-Chapter Reference Audit

```
/audit-refs all
```

Validates:
- All forward references land correctly
- All backward references exist
- Chapter numbers in references are accurate
- No orphaned anchors (defined but never referenced)

#### 2. Knowledge Graph Validation

```bash
python docs/knowledge_graph/validate_kg.py
```

Confirms:
- All anchors in KG match chapter content
- No stale KG entries
- Dependency graph is acyclic
- All forward_refs are satisfied

#### 3. PDF Compilation Test

```bash
for ch in ch00 ch01 ch02 ch03 ch04 ch05 ch06 ch07 ch08; do
  pandoc docs/book/$ch/*.md \
    --lua-filter=docs/book/callouts.lua \
    --include-in-header=docs/book/preamble.tex \
    --pdf-engine=xelatex \
    -o /tmp/$ch.pdf
done
```

Confirms:
- No LaTeX errors
- No Unicode issues
- All equations render
- Cross-references work

#### 4. Syllabus Alignment

Compare each chapter against `docs/book/syllabus.md`:
- Required topics covered
- Labs implemented
- Acceptance criteria met

### Output

```markdown
# Final Validation Report

Generated: 2025-01-20T16:00:00

## Validation Matrix

| Chapter | Refs | External | Math | Pedagogy | KG | PDF | Syllabus |
|---------|------|----------|------|----------|-----|-----|----------|
| Ch00 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ch01 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ch02 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ch03 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ch04 | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ |
| Ch05 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ch06 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ch07 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ch08 | ✅ | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ |

## Outstanding Items

### Ch04: Pedagogy
- Prose-bullet balance: 55/45 (target 60/40)
- Recommendation: Convert §4.3 motivation bullets to prose

### Ch08: External
- 1 URL returning 503 (temporary?)
- Recommendation: Re-check in 24h, add Archive.org fallback

## Verdict

**Status: READY FOR SUBMISSION** (with minor items noted above)

## Sign-Off

- [ ] Author review complete
- [ ] All critical items resolved
- [ ] Minor items documented for publisher
```

### Success Criteria

- [ ] All chapters pass all validation checks (or have documented exceptions)
- [ ] Knowledge Graph 100% synced
- [ ] PDFs compile without errors
- [ ] Syllabus alignment confirmed
- [ ] Author sign-off obtained

---

## Execution Order

### Recommended Chapter Order

Process chapters in dependency order (downstream chapters reference upstream):

1. **Ch03** - Stochastic Processes & Bellman (mathematical foundation)
2. **Ch02** - Probability & Measure (measure-theoretic foundation)
3. **Ch01** - Search as Optimization (references Ch02, Ch03)
4. **Ch00** - Motivation (references all foundations)
5. **Ch04-Ch08** - After foundations are solid

### Within Each Chapter

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
   ↓         ↓         ↓         ↓         ↓         ↓         ↓
 Setup   Registry  Internal  External   Fixes   Reviews   X-Check
                      ↓         ↓         ↓
                   (may loop back for fixes)
```

Phase 7 runs once after all chapters complete Phases 1-6.

---

## Quick Reference

| Phase | Command | Primary Output |
|-------|---------|----------------|
| 0 | (manual) | `status/overview.md` |
| 1 | `/audit-refs chXX --build-registry` | `registries/registry_chXX.yaml` |
| 2 | `/audit-refs chXX --full` | `audits/audit_chXX.md` |
| 3 | `/verify-external-refs chXX` | `audits/external_chXX.md` |
| 4 | `/fix-refs fixes_chXX.yaml` | `fixes/fix_report_chXX.md` |
| 5 | `/review-*` | `reviews/review_chXX_*.md` |
| 6 | `/cross-check` | (inline) |
| 7 | `/audit-refs all` | `status/final_validation.md` |
