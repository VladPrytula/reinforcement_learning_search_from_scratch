# 🚀 Knowledge Graph — START HERE

Welcome! This is your entry point to the knowledge graph system.

## What Is This?

A powerful yet simple system for managing your textbook's knowledge structure. You edit simple YAML files, and powerful Python tools give you queries, validation, and visualizations.

## The 30-Second Overview

```yaml
# Edit this (graph.yaml)
nodes:
  - id: CH-1
    kind: chapter
    title: Chapter 1
    status: complete

# Use Python to query it
from kg_tools import KnowledgeGraph
kg = KnowledgeGraph("graph.yaml")
untested = kg.find_untested_equations()
```

**That's it.** Edit YAML, run Python for insights.

## Quick Start (5 minutes)

### 1. Run the Examples

```bash
cd docs/knowledge_graph
source ../../.venv/bin/activate
python example_queries.py
```

You'll see your graph's statistics, coverage, blockers, and dependencies.

### 2. Check Validation

```bash
python validate_kg.py
```

Green checkmarks mean everything is consistent. Warnings show improvement areas.

### 3. Generate a Visualization

```bash
python visualize_kg.py --chapter CH-1 --output ch01.png
```

Open `ch01.png` to see Chapter 1's structure visually.

## What Can You Do?

| Need | Command | File |
|------|---------|------|
| **Learn the system** | Read TUTORIAL.md | [TUTORIAL.md](TUTORIAL.md) |
| **Quick reference** | Use INDEX.md | [INDEX.md](INDEX.md) |
| **API documentation** | See README.md | [README.md](README.md) |
| **Implementation details** | Check this summary | [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) |
| **Run commands** | Use `make help` | [Makefile](Makefile) |

## Common Tasks

### "What do I need to finish before Chapter 11?"

```python
from kg_tools import KnowledgeGraph
kg = KnowledgeGraph("graph.yaml")
blockers = kg.find_blockers("CH-11")
for blocker, status, title in blockers:
    print(f"  {blocker} ({status}): {title}")
```

### "Which equations don't have tests yet?"

```python
untested = kg.find_untested_equations()
print(f"Need tests for: {untested}")
```

### "Show me what Chapter 1 depends on"

```bash
python visualize_kg.py --deps CH-1 --output ch1_deps.png
```

### "Validate before committing"

```bash
python validate_kg.py
# Exit code 0 = safe to commit
# Exit code 1 = fix errors first
```

### "Quick stats dashboard"

```bash
python kg_tools.py
```

## File Organization

```
docs/knowledge_graph/
├── START_HERE.md              ← You are here
├── TUTORIAL.md                ← Next step (complete walkthrough)
├── INDEX.md                   ← Quick reference
├── README.md                  ← API reference
├── IMPLEMENTATION_SUMMARY.md  ← System architecture
│
├── graph.yaml                 ← Your knowledge graph (edit this)
├── schema.yaml                ← Schema definition (read-only)
│
├── kg_tools.py                ← Query library (use via Python)
├── validate_kg.py             ← Validator (run before commits)
├── visualize_kg.py            ← Visualizer (generate PNG graphs)
├── example_queries.py         ← Working examples
└── Makefile                   ← Quick commands
```

## The Philosophy

**YAML is your source of truth.** You edit it, Git tracks it, teammates review it.

**Python tools are ephemeral.** They load the YAML, analyze it, and disappear. No separate database to maintain.

**Powerful + Simple.** Graph database capability without the complexity.

## Common Workflow

1. **Write new content** → Update `graph.yaml` with new nodes/edges
2. **Validate** → `python validate_kg.py` (catch errors early)
3. **Check stats** → `python kg_tools.py` (see overall health)
4. **Visualize** → `python visualize_kg.py --chapter CH-X` (see structure)
5. **Commit** → Git tracks `graph.yaml` and visualization PNGs

**Time per commit:** ~30 seconds (validation + visualization).

## Key Concepts in 60 Seconds

**Nodes** = Things in your textbook (chapters, equations, modules, tests)

**Edges** = Relationships (defines, implements, uses, tested_by, depends_on)

**Status** = Where something is (planned, in_progress, complete, archived)

**Validation** = Catches mistakes (broken links, circular dependencies, coverage gaps)

**Queries** = Answer questions ("Which equations lack tests?" "What blocks this chapter?")

**Visualization** = See structure (dependency graphs as PNG images)

## System Health

Your current graph:
- ✅ **29 nodes** (6 chapters, 7 equations, 6 modules, etc.)
- ✅ **62 edges** (defines, implements, uses, etc.)
- ✅ **0 errors** in validation
- ✅ **0 circular dependencies**
- ✅ **0 dangling references**
- ⚠️ **6 untested equations** (expected, tests come later)
- ⚠️ **3 unimplemented equations** (expected, code comes later)

Everything is healthy. Low test coverage is normal at this stage.

## Next Steps

1. **Read [TUTORIAL.md](TUTORIAL.md)** (15 min) — Complete walkthrough with examples
2. **Run `make examples`** (2 min) — See tools in action
3. **Run `python validate_kg.py`** (1 min) — Check your graph
4. **Modify `graph.yaml`** with your content
5. **Query the graph** using Python from the examples

## Still Have Questions?

- **How do I use this?** → [TUTORIAL.md](TUTORIAL.md)
- **What can I do?** → [README.md](README.md)
- **Quick reference?** → [INDEX.md](INDEX.md)
- **How is it built?** → [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **How do I run commands?** → `make help`

---

**Ready to go?** Open [TUTORIAL.md](TUTORIAL.md) and start with "Your First Query"! 🚀
