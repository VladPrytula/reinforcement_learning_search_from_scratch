# Pre-Press Architecture: Skills, Commands, and Personas

## Design Philosophy

### The Core Question

When building Claude-based tooling, we face a design question: **What is the right abstraction level for different types of operations?**

We identified three distinct usage patterns:

1. **Context shaping**: "Act like Vlad when you write this"
2. **Review workflows**: "Review this chapter for mathematical rigor"
3. **Autonomous operations**: "Audit all references and generate a fix specification"

These patterns have different characteristics and deserve different abstractions.

---

## The Three-Layer Architecture

### Layer 1: Personas

**What they are**: Markdown files that define *who Claude is* for a task.

**Characteristics**:
- Loaded as context with `@filename` syntax
- Shape voice, standards, and approach
- Not executed---they're configuration
- Can be combined: `@vlad_prytula.md @vlad_revision_mode.md`

**Why personas are separate from skills**:

Personas are orthogonal to tasks. You might load `@vlad_revision_mode.md` before:
- Running `/audit-refs`
- Running `/review-pedagogy`
- Just asking a question about the chapter

The persona doesn't change *what* you do, it changes *how* you do it. Mixing these into skills would create a combinatorial explosion (audit-refs-as-vlad, audit-refs-as-reviewer, etc.).

**Current personas**:

| File | Purpose | When to Use |
|------|---------|-------------|
| `vlad_prytula.md` | Base identity | Always load first |
| `vlad_foundation_mode.md` | Ch01-03 writing | New foundation content |
| `vlad_application_mode.md` | Ch04+ writing | New RL/implementation content |
| `vlad_revision_mode.md` | Polishing mode | Editorial refinement |

**Location**: Project root (for easy `@` access)

---

### Layer 2: Commands

**What they are**: Single markdown files that define a review or analysis workflow.

**Characteristics**:
- Self-contained (no supporting files needed)
- Operate on current context
- Shape how Claude approaches a specific task
- Lightweight, quick to invoke

**Why commands stay as commands (not skills)**:

Commands like `/review-math` are contextual workflows. They:
- Don't need external tools
- Don't produce structured artifacts
- Don't have complex state
- Are meant to work with whatever you're currently looking at

Promoting them to skills would add overhead without benefit.

**Current commands**:

| Command | Purpose | Typical Use |
|---------|---------|-------------|
| `/review-math` | Mathematical rigor review | After writing proofs |
| `/review-pedagogy` | Pedagogical flow review | After drafting sections |
| `/review-rl-bridge` | Theory-practice connections | Checking code-math alignment |
| `/enhance-implementation` | Code improvement suggestions | Refining implementations |
| `/vlad` | Quick persona activation | Start of session |

**Location**: `.claude/commands/`

---

### Layer 3: Skills

**What they are**: Multi-file directories that define autonomous, complex operations.

**Characteristics**:
- Have a `skill.md` entry point
- May include supporting files (templates, configs, docs)
- May use external tools (WebFetch, WebSearch, other LLMs)
- Produce structured outputs (YAML, markdown reports)
- Can maintain state across invocations
- Can orchestrate other skills/commands

**Why these operations are skills**:

| Operation | Why It's a Skill |
|-----------|------------------|
| `/audit-refs` | Produces structured YAML registry, runs semantic verification, generates fix specifications |
| `/fix-refs` | Batch operations with pre/post validation, state tracking |
| `/verify-external` | Uses WebFetch/WebSearch, parses citations, calls other LLMs |
| `/cross-check` | Calls external LLM APIs (Gemini, GPT) |
| `/pre-press` | Orchestrates all other skills, maintains pipeline state |

**Current skills**:

| Skill | Purpose | Primary Output |
|-------|---------|----------------|
| `/audit-refs` | Full reference auditing | `audits/audit_chXX.md`, `fixes/fixes_chXX.yaml` |
| `/fix-refs` | Systematic reference fixing | `fixes/fix_report_chXX.md` |
| `/verify-external` | External reference verification | `audits/external_chXX.md` |
| `/cross-check` | Multi-LLM verification | Inline analysis |
| `/pre-press` | Pipeline orchestration | `status/chXX_state.yaml` |

**Location**: `.claude/skills/<skill-name>/`

---

## Skill Directory Structure

Each skill has a consistent structure:

```
.claude/skills/<skill-name>/
├── skill.md           # Entry point (REQUIRED)
│                      # Contains: purpose, usage, parameters, execution steps
│
├── README.md          # Documentation for humans
│                      # Contains: rationale, examples, troubleshooting
│
├── templates/         # Output templates (optional)
│   ├── report.md.template
│   └── spec.yaml.template
│
└── <other files>      # Skill-specific resources
    └── ...
```

### The `skill.md` Entry Point

This file defines what the skill does. Structure:

```markdown
# Skill Name

## Purpose
One paragraph explaining what this skill accomplishes.

## Usage
/skill-name <arguments>
/skill-name --flag

## Parameters
| Parameter | Required | Description |
|-----------|----------|-------------|
| `chapter` | Yes | Chapter to process (e.g., `ch03`) |
| `--full` | No | Enable semantic verification |

## Execution Steps
Detailed steps Claude follows when the skill is invoked.

## Output
Description of what the skill produces.

## Integration
How this skill connects with other skills/commands.
```

---

## Decision Records

### Why Promote `/audit-refs` and `/fix-refs` to Skills?

**Before**: Simple commands that listed references and suggested fixes.

**Problem**:
- No semantic verification (just pattern matching)
- No structured output (couldn't feed to `/fix-refs`)
- No Knowledge Graph integration
- No cross-chapter validation

**After**: Full skills with:
- Anchor registry building
- 4-level semantic verification
- YAML fix specifications
- Knowledge Graph sync
- Cross-chapter validation

**The change enables**: Automated, reliable reference fixing instead of manual grep-and-replace.

---

### Why Create `/verify-external-refs` as a New Skill?

**The gap**: No existing tool verified external references (URLs, papers, DOIs).

**Requirements**:
- HTTP requests to check URLs
- arXiv API to verify paper metadata
- DOI resolution
- LLM analysis for claim verification
- Integration with `/cross-check` for uncertain claims

**This requires**: WebFetch, WebSearch, multi-step workflows, structured output---clearly a skill, not a command.

---

### Why Create `/pre-press` as a Meta-Skill?

**The problem**: The pipeline has 7 phases with dependencies. Running them manually:
- Requires remembering the sequence
- Easy to skip phases or run out of order
- No state tracking (did I already audit ch03?)
- No consistent output locations

**The solution**: A meta-skill that:
- Tracks pipeline state per chapter
- Enforces phase dependencies
- Provides checkpoint/resume capability
- Standardizes output locations
- Offers approval gates between phases

**Why "meta"**: It doesn't do work itself---it orchestrates other skills.

---

### Why Keep Personas in Root (Not `.claude/personas/`)?

**Considered**: Moving personas to `.claude/personas/vlad_prytula.md`

**Problem**: The `@` syntax would become `@.claude/personas/vlad_prytula.md`---verbose and error-prone.

**Decision**: Keep personas in project root for clean `@vlad_prytula.md` access.

**Trade-off accepted**: Slightly messier root directory, but much better UX.

---

## Interaction Patterns

### Pattern 1: Persona + Command

```bash
# Load persona, then invoke command
@vlad_revision_mode.md
/review-pedagogy

# Claude now reviews with Vlad's revision standards
```

### Pattern 2: Skill with Arguments

```bash
# Skill operates autonomously
/audit-refs ch03 --full

# Claude:
# 1. Finds all ch03 files
# 2. Extracts anchors with context
# 3. Extracts references with context
# 4. Runs semantic verification
# 5. Generates report and fix spec
```

### Pattern 3: Skill Orchestration

```bash
# Meta-skill coordinates pipeline
/pre-press ch03 --full

# Claude:
# 1. Runs /audit-refs ch03 --registry
# 2. Reports results, waits for approval
# 3. Runs /audit-refs ch03 --full
# 4. Reports results, waits for approval
# 5. Runs /verify-external ch03
# ... continues through all phases
```

### Pattern 4: Selective Cross-Check

```bash
# During any skill, uncertain items get cross-checked
/verify-external ch03

# If a claim has <80% confidence:
# - Automatically invokes /cross-check
# - Reports triangulated result
```

---

## Extension Points

### Adding a New Command

1. Create `.claude/commands/new-command.md`
2. Define purpose, when to use, workflow steps
3. Document in this file

### Adding a New Skill

1. Create `.claude/skills/new-skill/`
2. Add `skill.md` (required) with full specification
3. Add `README.md` with examples and troubleshooting
4. Add templates if needed
5. Document in this file
6. Consider integration with `/pre-press` if relevant

### Adding a New Persona

1. Create `new_persona.md` in project root
2. Define identity, voice, standards
3. Document when to use vs. other personas
4. Update CLAUDE.md if it's a primary persona

---

## Debugging Skills

### If a skill isn't found

```bash
# Check skill exists
ls .claude/skills/

# Check entry point exists
cat .claude/skills/<name>/skill.md
```

### If skill produces unexpected output

1. Check the skill.md execution steps
2. Run steps manually to identify failure point
3. Check template files for format issues

### If orchestrator state is corrupted

```bash
# View state
cat pre-press/status/chXX_state.yaml

# Reset state (loses progress)
rm pre-press/status/chXX_state.yaml

# Re-run from beginning
/pre-press chXX
```

---

## Summary Table

| Layer | Abstraction | Location | Invocation | State | External Tools |
|-------|-------------|----------|------------|-------|----------------|
| Persona | Context | Root | `@file` | No | No |
| Command | Workflow | `.claude/commands/` | `/cmd` | No | No |
| Skill | Operation | `.claude/skills/` | `/skill args` | Optional | Yes |
