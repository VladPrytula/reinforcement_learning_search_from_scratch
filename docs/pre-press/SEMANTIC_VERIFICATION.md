# Semantic Verification: 100% Content Correctness

## The Problem

Traditional reference checking is **syntactic**: it verifies that a reference points to something that exists. But existence doesn't guarantee correctness.

### Example: Syntactic Success, Semantic Failure

```markdown
<!-- In ch03_lab_solutions.md -->
Recall from [THM-3.3.3] that the Bellman operator is a γ-contraction.
```

**Syntactic check**:
- Does `{#THM-3.3.3}` exist? No → FAIL

But even if we "fix" this by finding *some* theorem:

```markdown
<!-- "Fixed" version -->
Recall from [THM-3.5.1] that the Bellman operator is a γ-contraction.
```

**Syntactic check**:
- Does `{#THM-3.5.1}` exist? Yes → PASS ✓

**But wait**: THM-3.5.1 is the **Bellman Expectation Equation**, not the contraction theorem. The contraction theorem is THM-3.7.1. The "fix" is wrong---it passes syntactic checking but fails semantic verification.

### Why This Matters for Springer

Academic publishers like Springer have rigorous standards. A reviewer who follows a reference and finds the wrong theorem will:
1. Question the author's attention to detail
2. Wonder what other errors exist
3. Potentially reject or require major revisions

**Our goal**: Every reference must point to the correct content, not just existing content.

---

## The Four Levels of Verification

### Level 1: Existence (Syntactic)

**Question**: Does the target exist?

**Process**:
```
Reference: [THM-3.7.1]
     ↓
Search for: {#THM-3.7.1}
     ↓
Result: Found at ch03_main.md:596
     ↓
Verdict: EXISTS ✓
```

**Catches**:
- Typos (`[THM-3.7.l]` with letter 'l' instead of digit '1')
- Old numbering (`[THM-3.3.3]` from before renumbering)
- Deleted anchors (theorem removed but references remain)

**Misses**:
- References to wrong theorems
- Section references to wrong content
- Claims that don't match targets

---

### Level 2: Topic Alignment

**Question**: Does the reference context match the target's topic?

**Process**:
```
Reference: [THM-3.5.1] at ch03_lab_solutions.md:17
Context: "Recall from [THM-3.5.1] that the Bellman operator is a γ-contraction"
     ↓
Extract claimed topic from context:
  Keywords: {Bellman, operator, contraction, gamma}
     ↓
Retrieve anchor THM-3.5.1:
  Title: "Bellman Expectation Equation"
  Content: "For any policy π, the value function V^π satisfies V^π = R^π + γP^πV^π"
  Keywords: {Bellman, expectation, equation, policy, value-function}
     ↓
Compute topic similarity:
  Claimed: {Bellman, operator, contraction, gamma}
  Actual: {Bellman, expectation, equation, policy, value-function}
  Overlap: {Bellman}
  Similarity: 0.20
     ↓
Verdict: TOPIC MISMATCH ⚠️
  Expected topic: contraction
  Actual topic: expectation equation
```

**How Topic Extraction Works**:

1. **From reference context**:
   - Extract ±100 characters around reference
   - Identify mathematical terms (theorem names, equation names, properties)
   - Remove stop words
   - Result: Claimed topic keywords

2. **From anchor content**:
   - Extract title (from `**Theorem X.Y.Z (Title)**`)
   - Extract first 300 characters of content
   - Identify mathematical terms
   - Result: Actual topic keywords

3. **Similarity computation**:
   - Jaccard similarity on keyword sets
   - Weight mathematical terms higher than generic words
   - Threshold: 0.7 for "aligned", below for "mismatch"

**Catches**:
- References to wrong theorems with similar numbering
- Section references pointing to moved content
- Generic misattribution

---

### Level 3: Claim Verification

**Question**: Does the target actually support the specific claim made?

**Process**:
```
Reference: [THM-3.7.1] at ch03.md:420
Text: "By [THM-3.7.1], value iteration converges in O(log(1/ε)/(1-γ)) iterations"
     ↓
Parse claim:
  Subject: THM-3.7.1
  Property claimed: "value iteration converges in O(log(1/ε)/(1-γ))"
     ↓
Retrieve full anchor content:
  THM-3.7.1: "Bellman Operator Contraction"
  Statement: "The Bellman operator T is a γ-contraction under ‖·‖∞"
  Proof: [contraction proof]
     ↓
LLM Analysis:
  Question: "Does THM-3.7.1 establish that value iteration converges in O(log(1/ε)/(1-γ))?"

  Analysis: THM-3.7.1 establishes that T is a γ-contraction. Combined with
  Banach fixed-point (THM-3.6.2), this implies geometric convergence. The
  O(log(1/ε)/(1-γ)) rate is a consequence stated in COR-3.7.3, not THM-3.7.1
  directly.
     ↓
Verdict: CLAIM PARTIALLY SUPPORTED ⚠️
  THM-3.7.1 provides the contraction property
  The specific rate bound is COR-3.7.3
  Suggestion: Reference both, or just COR-3.7.3 for the rate
```

**Claim Types We Verify**:

| Claim Pattern | Example | Verification |
|---------------|---------|--------------|
| "X proves Y" | "THM-3.7.1 proves contraction" | Does THM-3.7.1's statement include Y? |
| "By X, property Y holds" | "By DEF-3.4.1, MDPs have bounded rewards" | Does DEF-3.4.1 include the bounded assumption? |
| "X establishes bound Y" | "COR-3.7.3 gives O(1/(1-γ)) rate" | Does COR-3.7.3 state this bound? |
| "From X we have Y" | "From EQ-3.8 we get the Bellman equation" | Is EQ-3.8 actually the Bellman equation? |
| "Chapter N develops X" | "Chapter 9 develops CMDP theory" | Does Chapter 9 contain CMDP content? |
| "Section M.N covers X" | "§3.6 covers Lagrangian methods" | Does §3.6 actually cover Lagrangians? |

**Catches**:
- Misattributed results (citing wrong theorem for a property)
- Overstated claims (claiming more than theorem proves)
- Understated claims (not citing the actual source)
- Confused cross-references

---

### Level 4: Cross-Verification (Multi-LLM)

**Question**: Do multiple independent reasoners agree on the verification?

**When to Use**:
- Claim verification confidence < 0.8
- Complex mathematical claims
- Subtle distinctions between theorems
- Reviewer disagreement

**Process**:
```
Uncertain claim identified at Level 3:
  "Bertsekas proves the contraction is strict"
  Confidence: 0.65
     ↓
Invoke /cross-check:

  Query to Gemini:
    "Verify step-by-step: Bertsekas (Dynamic Programming, 2012) proves that
    the Bellman operator contraction is strict (contraction factor < 1)."

  Query to GPT:
    "Challenge this claim: Bertsekas proves strict contraction of Bellman."
     ↓
Gemini response:
  "Bertsekas proves the Bellman operator is a contraction with factor γ.
  'Strict' contraction (γ < 1) follows when γ < 1 is assumed, which is
  standard. The term 'strict' may be misleading---it's a γ-contraction,
  with γ being the discount factor."

GPT response:
  "The claim needs qualification. Bertsekas establishes γ-contraction.
  Whether this is 'strict' depends on interpretation. In some literature,
  'strict contraction' means factor < 1 (satisfied when γ < 1). In other
  contexts, it implies uniform contraction independent of γ (not satisfied)."
     ↓
Synthesis:
  Agreement: Both confirm γ-contraction is proven
  Divergence: Interpretation of "strict"

  Verdict: CLAIM NEEDS CLARIFICATION
  Suggestion: Replace "strict contraction" with "γ-contraction with γ < 1"
  Confidence: 0.90 (after triangulation)
```

**When Cross-Check Is Automatic**:
- Level 3 confidence < 0.8
- External reference claim verification
- Mathematical proof correctness questions

**When Cross-Check Is Manual**:
- Complex architectural decisions
- Subtle pedagogical choices
- Disputed interpretations

---

## Implementation Details

### Anchor Registry Schema

Each anchor in the registry includes semantic context:

```yaml
- id: THM-3.7.1
  file: ch03_stochastic_processes_bellman_foundations.md
  line: 596
  type: theorem

  # For Level 2: Topic alignment
  title: "Bellman Operator Contraction"
  topics:
    - Bellman
    - operator
    - contraction
    - gamma-contraction
    - supremum-norm

  # For Level 3: Claim verification
  statement: |
    The Bellman operator T defined by (TV)(s) = max_a {R(s,a) + γ Σ P(s'|s,a)V(s')}
    is a γ-contraction under the supremum norm ‖·‖_∞. That is, for any V, W ∈ B(S):
    ‖TV - TW‖_∞ ≤ γ ‖V - W‖_∞

  # Mathematical relationships
  proves:
    - bellman-contraction-property
  uses:
    - DEF-3.4.1  # MDP definition
    - ASM-3.4.1  # Bounded rewards
  implies:
    - COR-3.7.3  # Convergence rate (via Banach)
```

### Reference Context Extraction

For each reference, we capture:

```yaml
- reference: "[THM-3.7.1]"
  file: ch03_lab_solutions.md
  line: 17

  # Raw context
  context_before: "Recall from "
  context_after: " that the Bellman operator is a γ-contraction"
  full_sentence: "Recall from [THM-3.7.1] that the Bellman operator is a γ-contraction."

  # Parsed claim
  claim_type: attribution  # "X states Y"
  subject: THM-3.7.1
  predicate: "Bellman operator is a γ-contraction"

  # Topic keywords (for Level 2)
  claimed_topics:
    - Bellman
    - operator
    - contraction
    - gamma
```

### Similarity Computation

Topic alignment uses weighted Jaccard similarity:

```python
def topic_similarity(claimed: set, actual: set) -> float:
    """
    Compute weighted topic similarity.

    Mathematical terms get weight 2.0
    Generic terms get weight 1.0
    """
    math_terms = {'theorem', 'lemma', 'proposition', 'definition',
                  'contraction', 'convergence', 'bound', 'operator',
                  'Bellman', 'Banach', 'MDP', 'policy', ...}

    def weight(term):
        return 2.0 if term.lower() in math_terms else 1.0

    intersection = claimed & actual
    union = claimed | actual

    weighted_intersection = sum(weight(t) for t in intersection)
    weighted_union = sum(weight(t) for t in union)

    return weighted_intersection / weighted_union if weighted_union > 0 else 0.0
```

### Claim Analysis Prompts

For Level 3 claim verification, we use structured prompts:

```
You are verifying a mathematical claim in a textbook.

REFERENCE: [THM-3.7.1]
CLAIMED: "By [THM-3.7.1], value iteration converges in O(log(1/ε)/(1-γ)) iterations"

ANCHOR CONTENT:
---
**Theorem 3.7.1** (Bellman Operator Contraction)

The Bellman operator T defined by (TV)(s) = max_a {R(s,a) + γ Σ P(s'|s,a)V(s')}
is a γ-contraction under the supremum norm ‖·‖_∞.

*Proof.* [proof content]
---

QUESTION: Does THM-3.7.1 directly establish the claimed property?

Respond with:
1. VERDICT: SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED
2. EXPLANATION: Why?
3. CORRECT_REFERENCE: If not supported, what should be cited?
4. CONFIDENCE: 0.0 to 1.0
```

---

## Verification Report Format

The audit report uses visual indicators for quick scanning:

```markdown
## Reference Verification Summary

| ID | Reference | Location | L1 | L2 | L3 | Status |
|----|-----------|----------|----|----|----| -------|
| 1 | [THM-3.7.1] | main:420 | ✓ | ✓ | ✓ | ✅ |
| 2 | [THM-3.5.1] | lab:17 | ✓ | ✗ | - | 🟠 Topic |
| 3 | [THM-3.3.3] | lab:42 | ✗ | - | - | 🔴 Missing |
| 4 | [EQ-3.8] | main:380 | ✓ | ✓ | ⚠️ | 🟡 Partial |

Legend:
  L1 = Existence, L2 = Topic, L3 = Claim
  ✓ = Pass, ✗ = Fail, ⚠️ = Partial, - = Not checked (prerequisite failed)
  ✅ = All pass, 🟠 = Semantic issue, 🔴 = Critical, 🟡 = Warning
```

---

## Confidence Thresholds

| Level | Threshold | Action if Below |
|-------|-----------|-----------------|
| L2 Topic | 0.7 | Flag as semantic issue |
| L3 Claim | 0.8 | Auto-invoke cross-check |
| L4 Cross-check | 0.9 (consensus) | Flag for human review |

---

## Edge Cases

### Intentional Vague References

Some references are intentionally general:
```
"See Chapter 3 for details on MDPs"
```

This shouldn't be flagged---it's a general pointer, not a specific claim.

**Detection**: References without specific claims (no "proves", "shows", "establishes") are marked as "general" and only checked at L1.

### Multiple Valid Targets

Sometimes a claim could cite multiple theorems:
```
"The contraction property (THM-3.7.1) implies convergence"
```

Both THM-3.7.1 (contraction) and COR-3.7.3 (convergence rate) are relevant. The reference to THM-3.7.1 is correct---it does imply convergence (via Banach). COR-3.7.3 would be more specific for the rate.

**Detection**: L3 analysis notes both as valid, suggests more specific reference if available.

### Evolving Terminology

Sometimes the chapter uses different terms than the anchor:
```
Reference: "the fixed-point theorem"
Anchor: "Banach Fixed-Point Theorem"
```

These should align despite wording differences.

**Detection**: Topic extraction normalizes synonyms:
- "fixed-point" = "fixed point" = "fixpoint"
- "contraction mapping" = "contraction"
- "Bellman equation" = "Bellman optimality equation"

---

## Integration with Pipeline

Semantic verification runs during:

| Phase | Verification Levels | Purpose |
|-------|---------------------|---------|
| Phase 2 | L1, L2, L3 | Internal reference audit |
| Phase 3 | L3 | External claim verification |
| Phase 6 | L4 | Uncertain item triangulation |
| Phase 7 | L1, L2 | Cross-chapter final check |

---

## Summary

**Semantic verification ensures**:

1. **L1 Existence**: The target exists (baseline)
2. **L2 Topic**: The reference context matches the target's topic
3. **L3 Claim**: Specific claims are supported by the target
4. **L4 Cross-check**: Complex/uncertain claims are triangulated

**The result**: Every reference in the textbook points to the correct content, with claims accurately representing what the cited source establishes. This is the standard required for Springer publication.
