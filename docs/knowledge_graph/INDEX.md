# Knowledge Graph Documentation Index

Welcome to the knowledge graph system! This directory contains everything you need to maintain, query, and visualize your textbook's knowledge structure.

## Quick Links

**New to the system?** Start here:
- **[TUTORIAL.md](TUTORIAL.md)** — Complete walkthrough covering setup, queries, validation, and visualization

**Need specific documentation?**
- **[README.md](README.md)** — Overview, ID conventions, update workflow, API reference
- **[graph.yaml](graph.yaml)** — The knowledge graph itself (human-editable source of truth)
- **[schema.yaml](schema.yaml)** — Schema definition for nodes and edges

## Tools Reference

### Core Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| **kg_tools.py** | Query library with 20+ methods | `python kg_tools.py` or `from kg_tools import KnowledgeGraph` |
| **validate_kg.py** | Validation and consistency checking | `python validate_kg.py` or `make validate` |
| **visualize_kg.py** | Generate dependency graphs (PNG) | `python visualize_kg.py --chapter CH-1 --output ch01.png` |
| **example_queries.py** | Working examples of common tasks | `python example_queries.py` or `make examples` |
| **Makefile** | Quick command shortcuts | `make stats`, `make validate`, `make visualize` |

### Example Commands

```bash
# Show graph statistics
make stats

# Validate before committing
make validate

# Generate visualizations
make visualize

# Run example queries
make examples

# Quick setup help
make help
```

## Common Tasks

### Planning a New Chapter

```python
from kg_tools import KnowledgeGraph
kg = KnowledgeGraph("graph.yaml")

# What needs to be done first?
blockers = kg.find_blockers("CH-12")

# What does it depend on?
deps = kg.transitive_dependencies("CH-12")

# What's the full scope?
summary = kg.chapter_summary("CH-12")
```

### Checking Test Coverage

```python
# Which equations need tests?
untested = kg.find_untested_equations()

# Overall coverage report
coverage = kg.coverage_report()

# Implementation status
impl_status = kg.implementation_status()
```

### Before Committing Changes to graph.yaml

```bash
# 1. Validate the graph
python validate_kg.py

# 2. Check statistics
python kg_tools.py

# 3. Generate visualizations (optional)
python visualize_kg.py --chapter CH-1 --output ch01.png

# 4. Commit
git add graph.yaml ch01.png  # Include visualization
git commit -m "docs: update knowledge graph"
```

### Generating Visualizations

```bash
# Single chapter with dependencies
python visualize_kg.py --chapter CH-1 --depth 2 --output ch01.png

# Show what blocks a chapter
python visualize_kg.py --deps CH-11 --output ch11_deps.png

# Implementation map (modules → equations)
python visualize_kg.py --impl-map --output impl.png
```

## File Organization

```
docs/knowledge_graph/
├── INDEX.md                 # This file
├── TUTORIAL.md              # Complete tutorial (start here)
├── README.md                # Detailed documentation
├── graph.yaml               # Knowledge graph (source of truth)
├── schema.yaml              # Schema definition
├── kg_tools.py              # Query library (620 lines)
├── validate_kg.py           # Validator (340 lines)
├── visualize_kg.py          # Visualizer (440 lines)
├── example_queries.py       # Usage examples
├── Makefile                 # Command shortcuts
└── INDEX.md                 # This file
```

## Key Concepts

### Node Types

- **chapter** (CH-X) — Major sections of the textbook
- **equation** (EQ-X.Y) — Mathematical formulas
- **theorem** (THM-X.Y.Z) — Mathematical theorems
- **definition** (DEF-X.Y.Z) — Concept definitions
- **concept** (CN-slug) — Abstract ideas
- **algorithm** — Named algorithms
- **module** (MOD-path.to.module) — Code modules
- **test** (TEST-path) — Test files
- **plan** (PLN-slug) — Research plans
- **doc** — Documentation

### Edge Types

- **defines** — Chapter defines an equation/theorem/concept
- **proves** — Chapter proves a theorem
- **uses** — Module uses another module
- **implements** — Module implements an equation/algorithm
- **tested_by** — Test validates an equation/module
- **depends_on** — Node depends on another
- **refers_to_future** — Forward hook to planned content
- **superseded_by** — Archived node replaced by new one

### Status Values

- **planned** — Placeholder, content pending
- **in_progress** — Partial content/code exists
- **complete** — Fully implemented and documented
- **archived** — Superseded by newer content

## API Quick Reference

### Loading the Graph

```python
from kg_tools import KnowledgeGraph
kg = KnowledgeGraph("graph.yaml")
```

### Finding Things

```python
equations = kg.nodes_by_kind('equation')
completed = kg.nodes_by_status('complete')
data = kg.get_node_data('CH-1')
```

### Analysis Queries

```python
untested = kg.find_untested_equations()
unimpl = kg.find_unimplemented_equations()
blockers = kg.find_blockers('CH-11')
deps = kg.transitive_dependencies('CH-11')
dependents = kg.transitive_dependents('CH-1')
```

### Validation

```python
missing = kg.find_missing_refs()
orphans = kg.find_orphan_nodes()
cycles = kg.find_cycles()
coverage = kg.coverage_report()
status = kg.status_summary()
impl_status = kg.implementation_status()
stats = kg.export_stats()
```

### Visualization

```python
from visualize_kg import KGVisualizer
viz = KGVisualizer(kg)
viz.visualize_chapter('CH-1', output_path='ch01.png')
viz.visualize_dependencies('CH-11', output_path='deps.png')
viz.visualize_implementation_map(output_path='impl.png')
```

## Validation Checks

The validator (`validate_kg.py`) automatically checks:

- ✅ **Referential integrity** — No dangling references
- ✅ **File existence** — All referenced files exist
- ✅ **Anchor presence** — Declared anchors in files
- ✅ **Status consistency** — Logical status transitions
- ✅ **Circular dependencies** — No cycles detected
- ✅ **Orphan nodes** — Isolated nodes flagged
- ✅ **Coverage gaps** — Untested equations identified
- ✅ **Schema compliance** — Valid kinds and edge types

Always run `make validate` before committing changes to `graph.yaml`.

## Architecture

The knowledge graph works by:

1. **Loading** — `kg_tools.py` reads `graph.yaml` into a NetworkX MultiDiGraph
2. **Querying** — Methods like `transitive_dependencies()` use graph algorithms
3. **Validating** — `validate_kg.py` checks consistency using NetworkX operations
4. **Visualizing** — `visualize_kg.py` generates PNG graphs using matplotlib

The entire system is ephemeral—the NetworkX graph is created fresh each time you load the YAML. This keeps your source of truth (YAML) simple and Git-friendly.

## Troubleshooting

### "Module not found: yaml"

```bash
source .venv/bin/activate
pip install pyyaml networkx matplotlib
```

### "File not found" in validation

Check that your `graph.yaml` has correct file paths relative to the repo root.

### Visualization produces empty graph

Make sure the node ID exists (e.g., `CH-1` not `ch-1`). Node IDs are case-sensitive.

### "Circular dependency detected"

Run `kg.find_cycles()` to see the exact cycle, then restructure your edges to break it.

## Contributing

When you modify `graph.yaml`:

1. Update nodes/edges with new content
2. Run `python validate_kg.py` (exit 0 = success)
3. Run `python kg_tools.py` to check statistics
4. Generate visualizations if topology changes significantly
5. Commit both `graph.yaml` and visualization PNGs

## Resources

- **[NetworkX Documentation](https://networkx.org/)** — Graph algorithms reference
- **[YAML Syntax](https://yaml.org/)** — YAML format reference
- **[Matplotlib Documentation](https://matplotlib.org/)** — Visualization reference

## Next Steps

1. **Start with TUTORIAL.md** if you're new to the system
2. **Run `make examples`** to see the tools in action
3. **Try `python validate_kg.py`** to check your graph
4. **Modify `graph.yaml`** with your new content
5. **Query the graph** using Python from the examples

Questions? Check the TUTORIAL.md for detailed explanations and practical examples.

Happy graphing! 🎉
