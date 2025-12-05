#!/usr/bin/env python
"""
Example queries demonstrating knowledge graph capabilities.

Run this script to see common use cases for the KG tools.
"""

from pathlib import Path
from kg_tools import KnowledgeGraph


def main():
    """Demonstrate common knowledge graph queries."""
    # Load graph
    kg_path = Path(__file__).parent / "graph.yaml"
    print(f"Loading knowledge graph from {kg_path}...\n")
    kg = KnowledgeGraph(kg_path)

    # Example 1: Find what's blocking a chapter
    print("="*70)
    print("Example 1: What's blocking Chapter 11?")
    print("="*70)
    blockers = kg.find_blockers("CH-11")
    if blockers:
        print(f"Found {len(blockers)} blockers:")
        for blocker_id, status, title in blockers:
            print(f"  [{status:12}] {blocker_id:20} {title}")
    else:
        print("No blockers! Chapter 11 is ready to implement.")
    print()

    # Example 2: Check test coverage
    print("="*70)
    print("Example 2: Test Coverage Report")
    print("="*70)
    coverage = kg.coverage_report()
    for kind, stats in coverage.items():
        coverage_pct = stats['coverage_pct']
        status_icon = "✅" if coverage_pct >= 80 else "⚠️" if coverage_pct >= 50 else "❌"
        print(f"{status_icon} {kind:12} {stats['tested']:2}/{stats['total']:2} ({coverage_pct:5.1f}%)")
    print()

    # Example 3: Find untested equations
    print("="*70)
    print("Example 3: Untested Equations (Need Tests!)")
    print("="*70)
    untested = kg.find_untested_equations()
    if untested:
        print(f"Found {len(untested)} untested equations:")
        for eq_id in untested:
            eq_data = kg.get_node_data(eq_id)
            print(f"  {eq_id:15} {eq_data.get('title', 'No title')[:60]}")
    else:
        print("All equations are tested! ✅")
    print()

    # Example 4: Chapter summary
    print("="*70)
    print("Example 4: Chapter 1 Summary")
    print("="*70)
    summary = kg.chapter_summary("CH-1")
    print(f"Title: {summary['title']}")
    print(f"Status: {summary['status']}")
    print(f"File: {summary['file']}")
    print(f"\nDefines {summary['defines_count']} items:")
    for item in summary['defines'][:5]:
        print(f"  {item['kind']:12} {item['id']:15} {item['title'][:50]}")
    if summary['defines_count'] > 5:
        print(f"  ... and {summary['defines_count'] - 5} more")

    if summary['forward_refs']:
        print(f"\nForward references ({len(summary['forward_refs'])}):")
        for ref in summary['forward_refs']:
            print(f"  [{ref['status']:12}] {ref['id']:10} {ref['title']}")
    print()

    # Example 5: Implementation status
    print("="*70)
    print("Example 5: Implementation Status")
    print("="*70)
    impl_status = kg.implementation_status()
    print(f"Implemented: {len(impl_status['implemented'])}")
    print(f"Unimplemented: {len(impl_status['unimplemented'])}")

    if impl_status['unimplemented']:
        print(f"\nUnimplemented items:")
        for item in impl_status['unimplemented'][:5]:
            print(f"  {item['kind']:12} {item['id']:15} [{item['status']}]")
    print()

    # Example 6: Dependency chains
    print("="*70)
    print("Example 6: What does CH-11 depend on?")
    print("="*70)
    deps = kg.transitive_dependencies("CH-11")
    if deps:
        print(f"CH-11 has {len(deps)} transitive dependencies:")
        for dep_id in sorted(deps):
            dep_data = kg.get_node_data(dep_id)
            kind = dep_data.get('kind', 'unknown')
            status = dep_data.get('status', 'unknown')
            print(f"  {dep_id:20} {kind:12} [{status}]")
    else:
        print("No dependencies.")
    print()

    # Example 7: Find nodes by status
    print("="*70)
    print("Example 7: Nodes by Status")
    print("="*70)
    for status in ['planned', 'in_progress', 'complete']:
        nodes = kg.nodes_by_status(status)
        print(f"{status:15} {len(nodes):3} nodes")
    print()

    # Example 8: Overall statistics
    print("="*70)
    print("Example 8: Overall Statistics")
    print("="*70)
    stats = kg.export_stats()
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    print(f"Untested equations: {stats['untested_equations']}")
    print(f"Unimplemented equations: {stats['unimplemented_equations']}")
    print(f"Orphan nodes: {stats['orphan_nodes']}")
    print(f"Circular dependencies: {stats['cycles']}")
    print(f"Missing references: {stats['missing_refs']}")
    print()

    print("="*70)
    print("✅ Example queries completed!")
    print("="*70)
    print("\nFor more details, see:")
    print("  - kg_tools.py for API reference")
    print("  - validate_kg.py for validation")
    print("  - visualize_kg.py for graph visualization")
    print("  - README.md for complete documentation")


if __name__ == '__main__':
    main()
