# Knowledge Graph Tutorial — From YAML to Insights

Welcome! This tutorial walks you through the knowledge graph system we've built for the RL Search textbook. By the end, you'll understand how to use YAML as your source of truth while leveraging NetworkX for powerful queries and visualizations.

## What Problem Does This Solve?

Imagine you're writing Chapter 11 and you want to know: *What do I need to finish first?* Or: *Which equations are still untested?* Or: *Show me visually how these concepts connect.*

With just YAML, these questions require manual reading and head-scratching. With NetworkX, you get answers in seconds.

## The Core Idea: YAML + NetworkX

Think of it this way:

**YAML is your source of truth.** You edit `graph.yaml` directly, commit it to Git, and colleagues can review it in pull requests. It's human-readable and version-controlled.

**NetworkX is your query engine.** You load the YAML into a NetworkX graph, ask questions, validate consistency, and generate visualizations. It's ephemeral—regenerated every time you run a query.

This separation is powerful: you get Git-friendly editing without complex tooling, plus on-demand analytics without storing redundant data.

## Setting Up

The tools are already installed in your virtual environment. To verify:

```bash
source .venv/bin/activate
python -c "import networkx; print(f'NetworkX {networkx.__version__}')"
```

All tools live in `docs/knowledge_graph/`:

- `kg_tools.py` — Core query library
- `validate_kg.py` — Validation checker
- `visualize_kg.py` — Graph visualizer
- `example_queries.py` — Working examples
- `Makefile` — Quick command shortcuts

## Your First Query

Let's load the graph and ask a simple question: How many equations do we have?

```python
from pathlib import Path
from kg_tools import KnowledgeGraph

kg = KnowledgeGraph("docs/knowledge_graph/graph.yaml")
equations = kg.nodes_by_kind('equation')
print(f"Found {len(equations)} equations: {equations}")
```

When you run this, NetworkX:
1. Reads `graph.yaml`
2. Creates a directed multigraph (allowing multiple edge types between nodes)
3. Adds all nodes with their attributes (title, status, file, anchor, etc.)
4. Adds all edges with their types (defines, implements, uses, etc.)
5. Returns the result

The loading takes milliseconds. The graph fits easily in memory—even with 500 nodes, it's tiny.

## Understanding the Graph Structure

Every node has metadata:

```yaml
- id: EQ-1.2
  kind: equation
  title: Scalar reward R = α·GMV + β·CM2 + γ·STRAT + δ·CLICKS
  status: complete
  file: docs/book/drafts/ch01_foundations.md
  anchor: "{#EQ-1.2}"
  summary: Single-step reward combining business outcomes and engagement.
```

Every edge has a type:

```yaml
edges:
  - {src: MOD-zoosim.reward, dst: EQ-1.2, rel: implements, status: complete}
```

The relationship types form a vocabulary:

- `defines` — A chapter defines an equation/theorem/concept
- `implements` — A module implements an equation/algorithm
- `uses` — A module uses another module
- `depends_on` — A node depends on another (for ordering)
- `tested_by` — A test validates an equation/module
- `refers_to_future` — A forward hook to planned content
- `proves` — A chapter proves a theorem

Think of these as the *grammar* of your knowledge graph. They tell the story of how concepts build on each other.

## Common Queries

### Finding Blockers

You want to write Chapter 11 but don't know what to finish first:

```python
blockers = kg.find_blockers("CH-11")
for blocker_id, status, title in blockers:
    print(f"  {blocker_id} ({status}): {title}")
```

This does a transitive dependency search: starting from CH-11, it walks backward through all `depends_on` and `uses` edges, finding any node that isn't `complete`. If the output is empty, you're good to start!

### Test Coverage

Which equations need tests?

```python
untested = kg.find_untested_equations()
print(f"Missing tests for: {untested}")
```

This scans all equation nodes and checks: does any node have an incoming `tested_by` edge? If not, it's untested. At your current stage, you expect this list to be long—tests come later.

### Implementation Status

Which equations don't yet have code implementations?

```python
impl_status = kg.implementation_status()
for item in impl_status['unimplemented']:
    print(f"{item['id']}: {item['title']}")
```

This searches for equations/algorithms with no incoming `implements` edges from any module. Useful when you're writing Chapter 3 and wondering: "Have we coded this up yet?"

### Dependency Chains

What must Chapter 11 be built on?

```python
deps = kg.transitive_dependencies("CH-11")
for dep in sorted(deps):
    data = kg.get_node_data(dep)
    print(f"  {dep:15} {data['status']:12} {data['kind']}")
```

This does a depth-first search from CH-11, following all outgoing `depends_on` and `uses` edges, collecting every node you transitively depend on. The result is a set—no duplicates, no cycles (thanks to validation!).

### Global Statistics

Quick health check of the entire graph:

```python
stats = kg.export_stats()
print(f"Total nodes: {stats['total_nodes']}")
print(f"Total edges: {stats['total_edges']}")
print(f"Untested equations: {stats['untested_equations']}")
print(f"Circular dependencies: {stats['cycles']}")
print(f"Dangling references: {stats['missing_refs']}")
```

This gives you a dashboard of graph health in one shot.

## Validation: Catching Mistakes Early

Before committing changes to `graph.yaml`, run the validator:

```bash
python docs/knowledge_graph/validate_kg.py
```

It checks:

**Referential integrity** — Do all edges point to nodes that exist? If you delete a node, this catches dangling edges.

**File references** — Does every node with a `file:` field point to a real file? Does the `anchor:` actually exist in that file?

**Status consistency** — Can a `complete` node depend on a `planned` node? (Usually no.) Can a module implement a planned equation? (Usually no.) The validator catches these contradictions.

**Circular dependencies** — The graph should be a DAG (directed acyclic graph). No cycles allowed. The validator detects loops.

**Coverage gaps** — Which equations lack tests? Which have low coverage? Reports included.

**Schema compliance** — Are all node kinds and edge types valid? Do required fields exist?

The validator exits with code 0 if all checks pass, 1 if errors found. This makes it easy to integrate into CI/CD pipelines.

## Visualization: Seeing the Structure

Sometimes a picture is worth a thousand edge queries. The visualizer generates PNG graphs:

```bash
# Visualize Chapter 1 and what it depends on
python docs/knowledge_graph/visualize_kg.py --chapter CH-1 --depth 2 --output ch01.png

# Show all dependencies for Chapter 11
python docs/knowledge_graph/visualize_kg.py --deps CH-11 --output ch11_deps.png

# Map which modules implement which equations
python docs/knowledge_graph/visualize_kg.py --impl-map --output impl_map.png
```

The visualizations are color-coded by node type (blue for chapters, orange for equations) and shape-coded by status (circle for planned, diamond for complete). Edges are styled by type (solid for implements, dashed for uses).

When you generate a visualization, it appears in the current directory. Commit these PNGs to Git alongside your `graph.yaml` updates—they help reviewers understand structure at a glance.

## The Workflow

Here's how the tools fit into your daily routine:

**When you write a new chapter:**

1. Update `graph.yaml` with new nodes (the chapter, equations it defines, theorems it proves).
2. Add edges connecting to existing content (e.g., `CH-1 defines EQ-1.2`).
3. Run `make validate` to catch typos and consistency issues.
4. Run `make stats` to confirm the graph structure.
5. Generate a visualization: `python docs/knowledge_graph/visualize_kg.py --chapter CH-X --output chX.png`
6. Commit both `graph.yaml` and the PNG to Git.

**When you're planning:**

Use `kg.find_blockers("CH-X")` to understand dependencies before you start writing. Use `kg.transitive_dependencies("CH-X")` to see the full chain of what you're building on.

**When you add code:**

Update the `implements` edges in `graph.yaml` to link modules to the equations/algorithms they implement. Then check: `kg.implementation_status()` to see if anything's been missed.

**When you add tests:**

Add `tested_by` edges from modules/equations to test nodes. Run `kg.coverage_report()` to track progress.

**Before committing:**

Run `make validate` (or `make pre-commit`) to ensure everything's consistent. The exit code tells CI/CD whether to proceed.

## Why YAML + NetworkX?

You might wonder: why not use a real graph database?

**Simplicity.** YAML is human-editable, reviewable in pull requests, and requires zero infrastructure. NetworkX is a single Python import with no server setup.

**Reversibility.** If you decide this isn't useful, delete the Python files and go back to pure YAML. No data lock-in.

**Performance.** For a 50-500 node graph, in-memory NetworkX is faster than querying a database. No network latency, no query optimization needed.

**Coupling.** Everything stays in version control. The graph is part of your repo, not a separate service.

This approach scales to at least 500 nodes (proven by similar projects). Beyond that, you'd want a real database. But you're not there yet, and you won't be for years.

## Common Pitfalls and How to Avoid Them

**Pitfall 1: Forgetting to update edges when you move/rename a node.**

Solution: The validator catches dangling references. Always run validation after major edits.

**Pitfall 2: Creating circular dependencies.**

Solution: Again, the validator catches this. It's rare because you're building a textbook (inherently DAG-like), but possible if you add circular forward-refs.

**Pitfall 3: Nodes with no connections (orphans).**

Solution: The validator reports orphan nodes. Top-level chapters are fine (they have no parents), but mid-level concepts should connect to something.

**Pitfall 4: Equations defined but never implemented or tested.**

Solution: Use `kg.find_unimplemented_equations()` and `kg.find_untested_equations()` regularly. These metrics improve over time.

**Pitfall 5: Forgetting file anchors.**

Solution: When you update a file with new equations/theorems, add the Pandoc-style anchor `{#EQ-X.Y}` right after the equation. The validator checks for these.

## Advanced Usage

Once you're comfortable with the basics, here are power moves:

**Custom queries:**

```python
# Find all nodes by a specific author or tag
completed = kg.nodes_by_status('complete')
in_progress = kg.nodes_by_status('in_progress')

# Get the node data for detailed inspection
eq_data = kg.get_node_data("EQ-1.2")
print(eq_data['title'])
print(eq_data['summary'])
print(eq_data['file'])

# Build complex filters
modules = kg.nodes_by_kind('module')
complete_modules = [m for m in modules
                   if kg.get_node_data(m)['status'] == 'complete']
```

**Detecting cycles:**

```python
cycles = kg.find_cycles()
if cycles:
    for cycle in cycles:
        print(f"Circular dependency: {' → '.join(cycle)}")
```

**Exporting for external tools:**

```python
# Export to JSON for web dashboards
import json
stats = kg.export_stats()
with open('kg_stats.json', 'w') as f:
    json.dump(stats, f)

# Export chapter summaries
for chapter_id in kg.nodes_by_kind('chapter'):
    summary = kg.chapter_summary(chapter_id)
    print(f"{chapter_id}: {summary['defines_count']} definitions")
```

**Integration with CI/CD:**

```bash
#!/bin/bash
# Pre-commit hook
python docs/knowledge_graph/validate_kg.py || exit 1
echo "Graph validation passed ✅"
```

## Practical Example: Planning Chapter 12

Let's say you're about to write Chapter 12 (Slate RL). Here's how you'd use the tools to plan:

**Step 1: Check blockers**

```python
blockers = kg.find_blockers("CH-12")
if blockers:
    print("Can't start CH-12 yet. Finish these first:")
    for b_id, status, title in blockers:
        print(f"  - {b_id} ({status}): {title}")
```

Output tells you exactly what chapters/modules you need to be ready.

**Step 2: Understand the dependency chain**

```python
deps = kg.transitive_dependencies("CH-12")
print(f"CH-12 will build on {len(deps)} other nodes:")
for dep in sorted(deps):
    data = kg.get_node_data(dep)
    if data['kind'] == 'chapter':  # Focus on chapters
        print(f"  - {dep}: {data['title']}")
```

You see exactly which chapters you'll need to reference.

**Step 3: Find similar content**

```python
# What equations and algorithms are defined in related chapters?
equations = kg.nodes_by_kind('equation')
for eq in equations:
    data = kg.get_node_data(eq)
    if 'ranking' in data.get('title', '').lower():
        print(f"  {eq}: {data['title']}")
```

Helps you avoid duplication and understand the landscape.

**Step 4: Generate visualization**

```bash
python docs/knowledge_graph/visualize_kg.py --deps CH-12 --output ch12_context.png
```

You see visually how CH-12 connects to the rest of the material. Attach this to your design doc.

## Conclusion

The knowledge graph tools give you three superpowers:

**1. Queries:** Answer questions about your content in seconds. "What's blocking this chapter?" "Which equations lack tests?" "Show me the full dependency chain."

**2. Validation:** Catch mistakes before they become problems. Dangling references, circular dependencies, status inconsistencies—all caught automatically.

**3. Visualization:** See your knowledge structure. What connects to what? Where are the gaps? What's the critical path?

All of this while keeping YAML as your human-friendly source of truth. No databases, no complex tooling—just Python and the power of graphs.

The investment in learning these tools pays for itself the first time you ask "what do I need to finish before starting this chapter?" and get an instant, correct answer.

Happy graphing! 🎉
