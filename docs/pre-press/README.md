# Pre-Press Pipeline Documentation

## Overview

This documentation describes the systematic process for preparing textbook chapters for Springer publication. The pipeline ensures **100% verification** of all references, mathematical content, and pedagogical quality---meeting publisher standards while maintaining the book's distinctive integration of rigorous theory and production code.

## The Problem We're Solving

### Why This Pipeline Exists

During the writing of chapters 1-8, several factors created reference alignment issues:

1. **Structural evolution**: Chapter organization changed as content developed
2. **Theorem renumbering**: Mathematical results were reordered for pedagogical flow
3. **Cross-chapter dependencies**: Forward/backward references became stale
4. **Lab solutions drift**: Exercise solutions referenced old theorem IDs
5. **External citations**: Paper/URL references weren't systematically verified

**The result**: References like `[THM-3.3.3]` point to non-existent anchors, section references like "Chapter 3, Section 3.6" describe content that moved, and external citations may have broken links or inaccurate metadata.

### Why Existing Tools Were Insufficient

Simple grep-based reference checking fails because:

- **Syntactic matching isn't semantic verification**: Finding that `{#THM-3.7.1}` exists doesn't confirm that a reference claiming "Bellman contraction" actually points to a Bellman contraction theorem
- **Cross-chapter validation is complex**: Verifying "Chapter 9 develops CMDP theory" requires understanding what Chapter 9 actually contains
- **External references need web access**: Checking arXiv links, DOIs, and URLs requires HTTP requests and metadata parsing
- **Claims need verification**: "Paper X proves Y" requires comparing the claim against the actual source

### What This Pipeline Achieves

| Verification Level | What It Catches | Tool |
|-------------------|-----------------|------|
| **Syntactic** | Typos, old numbering, deleted anchors | `/audit-refs` |
| **Semantic** | Wrong targets, misattributed claims | `/audit-refs --full` |
| **Cross-chapter** | Stale forward/backward refs, chapter drift | `/audit-refs all` |
| **External** | Dead links, wrong metadata, misrepresented sources | `/verify-external-refs` |
| **Mathematical** | Proof errors, notation inconsistencies | `/review-math` |
| **Pedagogical** | Flow issues, prose-bullet imbalance | `/review-pedagogy` |

---

## Architecture

### The Three-Layer System

```
Layer 1: PERSONAS (@file)
--------------------------
Define WHO Claude is for a task.
Loaded as context, not executed.

  @vlad_prytula.md           Base identity
  @vlad_foundation_mode.md   Ch01-03 writing mode
  @vlad_application_mode.md  Ch04+ writing mode
  @vlad_revision_mode.md     Polishing mode

Usage: @vlad_prytula.md @vlad_revision_mode.md
       "Revise Chapter 6 for prose-bullet balance"


Layer 2: COMMANDS (/cmd)
------------------------
Define WHAT Claude does (single workflow).
Self-contained markdown files.

  /review-math        Mathematical rigor review
  /review-pedagogy    Pedagogical flow review
  /review-rl-bridge   Theory-practice connections
  /vlad               Quick persona activation

Usage: /review-math
       (operates on current context)


Layer 3: SKILLS (/skill)
------------------------
Define HOW Claude performs complex operations.
Multi-file directories with supporting resources.
May use external tools (web, APIs, other LLMs).

  /audit-refs         Full reference auditing with semantic verification
  /fix-refs           Systematic reference correction with validation
  /verify-external    External reference verification (URLs, papers)
  /cross-check        Multi-LLM epistemic triangulation
  /pre-press          Meta-skill orchestrating the full pipeline

Usage: /audit-refs ch03
       (autonomous operation with structured output)
```

### Why This Separation?

**Personas** are not actions---they're context that shapes *how* Claude approaches any task. You load a persona, then ask Claude to do something. The persona influences voice, standards, and approach.

**Commands** are lightweight workflows for review and analysis. They operate on whatever context is current. They're single markdown files because they don't need supporting resources.

**Skills** are heavyweight autonomous operations. They:
- May span multiple steps
- May call external tools (WebFetch, WebSearch)
- May invoke other skills/commands
- Produce structured outputs (YAML, reports)
- Have supporting files (templates, configs, documentation)

### File Organization

```
Project Root/
├── vlad_prytula.md              # Base persona (root for @access)
├── vlad_foundation_mode.md
├── vlad_application_mode.md
├── vlad_revision_mode.md
│
├── .claude/
│   ├── commands/                # Simple review workflows
│   │   ├── review-math.md
│   │   ├── review-pedagogy.md
│   │   ├── review-rl-bridge.md
│   │   └── vlad.md
│   │
│   └── skills/                  # Complex autonomous operations
│       ├── audit-refs/
│       ├── fix-refs/
│       ├── verify-external/
│       ├── cross-check/
│       └── pre-press/           # Meta-skill (orchestrator)
│
├── docs/
│   ├── pre-press/               # This documentation
│   │   ├── README.md            # You are here
│   │   ├── ARCHITECTURE.md      # Detailed architecture
│   │   ├── PIPELINE.md          # 7-phase process
│   │   └── SEMANTIC_VERIFICATION.md
│   │
│   ├── knowledge_graph/         # Source of truth for anchors
│   │   └── graph.yaml
│   │
│   └── book/                    # Chapter content
│       ├── ch00/ ... ch10/
│       └── syllabus.md
│
└── pre-press/                   # Working directory
    ├── ch01.md ... ch08.md      # Editorial notes (existing)
    ├── registries/              # Anchor registries (generated)
    ├── audits/                  # Audit reports (generated)
    ├── fixes/                   # Fix specifications (generated)
    ├── reviews/                 # Review outputs (generated)
    └── status/                  # Pipeline state tracking
```

---

## Quick Start

### For a Single Chapter

```bash
# 1. Full audit with semantic verification
/audit-refs ch03

# 2. Review the report
# Open: pre-press/audits/audit_ch03.md

# 3. Apply fixes
/fix-refs pre-press/fixes/fixes_ch03.yaml

# 4. Verify external references
/verify-external-refs ch03

# 5. Re-audit to confirm clean
/audit-refs ch03 --quick
```

### For Full Pre-Press Pipeline

```bash
# Use the meta-skill orchestrator
/pre-press ch03

# Or run with full automation (pauses for approval at each phase)
/pre-press ch03 --full
```

### Check Status

```bash
# Single chapter
/pre-press ch03 --status

# All chapters
/pre-press --status
```

---

## Pipeline Phases Overview

| Phase | Purpose | Primary Skill | Output |
|-------|---------|---------------|--------|
| **0** | Initialize environment | (setup) | Knowledge Graph baseline |
| **1** | Build anchor registry | `/audit-refs --registry` | `registries/registry_chXX.yaml` |
| **2** | Internal reference audit | `/audit-refs --full` | `audits/audit_chXX.md` |
| **3** | External reference audit | `/verify-external-refs` | `audits/external_chXX.md` |
| **4** | Apply fixes | `/fix-refs` | `fixes/fix_report_chXX.md` |
| **5** | Content reviews | `/review-*` | `reviews/review_chXX_*.md` |
| **6** | Cross-verification | `/cross-check` | Inline in reports |
| **7** | Final validation | `/audit-refs all` | `status/final_validation.md` |

See [PIPELINE.md](PIPELINE.md) for detailed phase documentation.

---

## Key Concepts

### Semantic Verification

Not just "does the reference exist?" but "does it point to the correct content?"

```
Reference: "By [THM-3.7.1], the Bellman operator is a contraction"
                    │
                    ▼
Syntactic check: Does {#THM-3.7.1} exist? ✓
                    │
                    ▼
Semantic check: Does THM-3.7.1 discuss Bellman contraction? ✓
                    │
                    ▼
Claim check: Does THM-3.7.1 actually prove contraction? ✓
```

See [SEMANTIC_VERIFICATION.md](SEMANTIC_VERIFICATION.md) for details.

### Knowledge Graph as Source of Truth

The file `docs/knowledge_graph/graph.yaml` is the canonical registry of all mathematical objects (theorems, definitions, equations). The pipeline:

1. Extracts anchors from chapter markdown
2. Compares with Knowledge Graph entries
3. Reports discrepancies
4. Can update KG after fixes

### Cross-Chapter Dependencies

References don't exist in isolation. The pipeline validates:

- **Forward references**: "Chapter 9 develops X" --- Does Ch9 actually contain X?
- **Backward references**: "Deferred from Ch1 S1.7" --- Does Ch1 S1.7 preview this?
- **Syllabus alignment**: Does chapter content match `syllabus.md` requirements?

---

## Recommended Execution Order

Because chapters reference each other, fix them in dependency order:

1. **Ch03** (Stochastic Processes & Bellman) --- Mathematical foundation
2. **Ch02** (Probability & Measure) --- Measure-theoretic foundation
3. **Ch01** (Search as Optimization) --- References Ch02/Ch03
4. **Ch00** (Motivation) --- References all foundations
5. **Ch04-Ch08** --- After foundations are solid

---

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) --- Detailed skills/commands architecture
- [PIPELINE.md](PIPELINE.md) --- Complete 7-phase process with examples
- [SEMANTIC_VERIFICATION.md](SEMANTIC_VERIFICATION.md) --- How 100% verification works
- [../book/outline.md](../book/outline.md) --- Chapter-to-code mapping
- [../knowledge_graph/README.md](../knowledge_graph/) --- Knowledge Graph documentation

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-01-15 | 1.0 | Initial pipeline design and implementation |
