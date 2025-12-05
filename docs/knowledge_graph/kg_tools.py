#!/usr/bin/env python
"""
Knowledge Graph Tools — NetworkX-based utilities for querying the YAML knowledge graph.

Usage:
    from kg_tools import KnowledgeGraph

    kg = KnowledgeGraph("docs/knowledge_graph/graph.yaml")

    # Find all untested equations
    untested = kg.find_untested_equations()

    # Get transitive dependencies for a node
    deps = kg.transitive_dependencies("CH-11")

    # Find what's blocking progress
    blockers = kg.find_blockers("CH-11")

    # Get all nodes of a specific kind
    equations = kg.nodes_by_kind("equation")
"""

from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
import networkx as nx
import yaml


class KnowledgeGraph:
    """NetworkX-based knowledge graph loaded from YAML."""

    def __init__(self, yaml_path: str | Path):
        """Load knowledge graph from YAML file.

        Args:
            yaml_path: Path to graph.yaml file
        """
        self.yaml_path = Path(yaml_path)
        self.G = nx.MultiDiGraph()  # Allow multiple edge types between nodes
        self._load_from_yaml()

    def _load_from_yaml(self) -> None:
        """Load nodes and edges from YAML into NetworkX graph."""
        with open(self.yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Add nodes with all their attributes
        for node in data.get('nodes', []):
            node_id = node['id']
            # Store all node attributes
            self.G.add_node(node_id, **node)

        # Add explicit edges from edges section
        for edge in data.get('edges', []):
            src = edge['src']
            dst = edge['dst']
            rel = edge['rel']
            status = edge.get('status', 'complete')
            self.G.add_edge(src, dst, rel=rel, status=status)

        # Also add edges from node adjacency lists (defines, uses, etc.)
        for node in data.get('nodes', []):
            node_id = node['id']

            # Map node list fields to edge types
            edge_mappings = {
                'defines': 'defines',
                'proves': 'proves',
                'uses': 'uses',
                'implements': 'implements',
                'tested_by': 'tested_by',
                'depends_on': 'depends_on',
                'forward_refs': 'refers_to_future',
                'see_also': 'see_also',
            }

            for field, rel_type in edge_mappings.items():
                if field in node and node[field]:
                    for target in node[field]:
                        if not self.G.has_edge(node_id, target, key=rel_type):
                            self.G.add_edge(node_id, target, rel=rel_type,
                                          status=node.get('status', 'complete'))

    def nodes_by_kind(self, kind: str) -> List[str]:
        """Get all node IDs of a specific kind.

        Args:
            kind: Node kind (chapter, equation, theorem, module, etc.)

        Returns:
            List of node IDs matching the kind
        """
        return [n for n, d in self.G.nodes(data=True) if d.get('kind') == kind]

    def nodes_by_status(self, status: str) -> List[str]:
        """Get all node IDs with a specific status.

        Args:
            status: planned, in_progress, complete, archived

        Returns:
            List of node IDs matching the status
        """
        return [n for n, d in self.G.nodes(data=True) if d.get('status') == status]

    def get_node_data(self, node_id: str) -> Dict[str, Any]:
        """Get all data for a node.

        Args:
            node_id: Node identifier

        Returns:
            Dictionary of node attributes
        """
        if node_id not in self.G:
            raise ValueError(f"Node {node_id} not found in graph")
        return dict(self.G.nodes[node_id])

    def find_untested_equations(self) -> List[str]:
        """Find all equations that lack test coverage.

        Returns:
            List of equation IDs without tested_by edges
        """
        equations = self.nodes_by_kind('equation')
        untested = []

        for eq in equations:
            # Check if there's any tested_by edge from this equation
            has_tests = any(
                self.G[eq][target].get(key, {}).get('rel') == 'tested_by'
                for target in self.G.successors(eq)
                for key in self.G[eq][target]
            )
            if not has_tests:
                untested.append(eq)

        return untested

    def find_unimplemented_equations(self) -> List[str]:
        """Find all equations without implementations.

        Returns:
            List of equation IDs without implements edges pointing to them
        """
        equations = self.nodes_by_kind('equation')
        unimplemented = []

        for eq in equations:
            # Check if any module implements this equation
            has_impl = any(
                self.G[pred][eq].get(key, {}).get('rel') == 'implements'
                for pred in self.G.predecessors(eq)
                for key in self.G[pred][eq]
            )
            if not has_impl:
                unimplemented.append(eq)

        return unimplemented

    def transitive_dependencies(self, node_id: str,
                               rel_types: Optional[List[str]] = None) -> Set[str]:
        """Get all nodes that this node transitively depends on.

        Args:
            node_id: Starting node
            rel_types: Edge types to follow (default: depends_on, uses)

        Returns:
            Set of node IDs in the transitive closure
        """
        if rel_types is None:
            rel_types = ['depends_on', 'uses']

        # Build filtered graph with only desired edge types
        filtered = nx.DiGraph()
        for u, v, data in self.G.edges(data=True):
            if data.get('rel') in rel_types:
                filtered.add_edge(u, v)

        if node_id not in filtered:
            return set()

        # Get all descendants (nodes this depends on)
        return nx.descendants(filtered, node_id)

    def transitive_dependents(self, node_id: str,
                             rel_types: Optional[List[str]] = None) -> Set[str]:
        """Get all nodes that transitively depend on this node.

        Args:
            node_id: Starting node
            rel_types: Edge types to follow (default: depends_on, uses)

        Returns:
            Set of node IDs that depend on this node
        """
        if rel_types is None:
            rel_types = ['depends_on', 'uses']

        # Build filtered graph
        filtered = nx.DiGraph()
        for u, v, data in self.G.edges(data=True):
            if data.get('rel') in rel_types:
                filtered.add_edge(u, v)

        if node_id not in filtered:
            return set()

        # Get all ancestors (nodes that depend on this)
        return nx.ancestors(filtered, node_id)

    def find_blockers(self, node_id: str) -> List[Tuple[str, str, str]]:
        """Find all incomplete dependencies blocking this node.

        Args:
            node_id: Node to check

        Returns:
            List of (blocker_id, blocker_status, blocker_title) tuples
        """
        deps = self.transitive_dependencies(node_id)
        blockers = []

        for dep in deps:
            data = self.get_node_data(dep)
            status = data.get('status', 'unknown')
            if status in ['planned', 'in_progress']:
                title = data.get('title', dep)
                blockers.append((dep, status, title))

        return sorted(blockers, key=lambda x: x[1])  # Sort by status

    def find_missing_refs(self) -> List[Tuple[str, str]]:
        """Find edges pointing to non-existent nodes (dangling references).

        Returns:
            List of (source_id, missing_target_id) tuples
        """
        all_nodes = set(self.G.nodes())
        missing = []

        for src, dst in self.G.edges():
            if dst not in all_nodes:
                missing.append((src, dst))

        return missing

    def find_orphan_nodes(self) -> List[str]:
        """Find nodes with no incoming or outgoing edges.

        Returns:
            List of orphaned node IDs
        """
        orphans = []
        for node in self.G.nodes():
            if self.G.in_degree(node) == 0 and self.G.out_degree(node) == 0:
                orphans.append(node)
        return orphans

    def find_cycles(self, rel_types: Optional[List[str]] = None) -> List[List[str]]:
        """Find circular dependencies in the graph.

        Args:
            rel_types: Edge types to consider (default: depends_on, uses)

        Returns:
            List of cycles (each cycle is a list of node IDs)
        """
        if rel_types is None:
            rel_types = ['depends_on', 'uses', 'implements']

        # Build filtered graph
        filtered = nx.DiGraph()
        for u, v, data in self.G.edges(data=True):
            if data.get('rel') in rel_types:
                filtered.add_edge(u, v)

        try:
            cycles = list(nx.simple_cycles(filtered))
            return cycles
        except nx.NetworkXNoCycle:
            return []

    def coverage_report(self) -> Dict[str, Dict[str, int]]:
        """Generate test coverage report by kind.

        Returns:
            Dictionary mapping kind -> {total, tested, coverage_pct}
        """
        report = {}

        for kind in ['equation', 'theorem', 'algorithm', 'module']:
            nodes = self.nodes_by_kind(kind)
            if not nodes:
                continue

            tested_count = 0
            for node in nodes:
                has_tests = any(
                    self.G[node][target].get(key, {}).get('rel') == 'tested_by'
                    for target in self.G.successors(node)
                    for key in self.G[node][target]
                )
                if has_tests:
                    tested_count += 1

            total = len(nodes)
            coverage = (tested_count / total * 100) if total > 0 else 0
            report[kind] = {
                'total': total,
                'tested': tested_count,
                'coverage_pct': round(coverage, 1)
            }

        return report

    def status_summary(self) -> Dict[str, int]:
        """Get count of nodes by status.

        Returns:
            Dictionary mapping status -> count
        """
        summary = {'planned': 0, 'in_progress': 0, 'complete': 0, 'archived': 0}
        for _, data in self.G.nodes(data=True):
            status = data.get('status', 'unknown')
            if status in summary:
                summary[status] += 1
        return summary

    def chapter_summary(self, chapter_id: str) -> Dict[str, Any]:
        """Get summary of a chapter's content and dependencies.

        Args:
            chapter_id: Chapter node ID (e.g., 'CH-1')

        Returns:
            Dictionary with chapter statistics
        """
        if chapter_id not in self.G:
            raise ValueError(f"Chapter {chapter_id} not found")

        chapter_data = self.get_node_data(chapter_id)

        # Find what chapter defines
        defines = []
        for target in self.G.successors(chapter_id):
            for key in self.G[chapter_id][target]:
                if self.G[chapter_id][target][key].get('rel') == 'defines':
                    target_data = self.get_node_data(target)
                    defines.append({
                        'id': target,
                        'kind': target_data.get('kind'),
                        'title': target_data.get('title')
                    })

        # Find what chapter proves
        proves = []
        for target in self.G.successors(chapter_id):
            for key in self.G[chapter_id][target]:
                if self.G[chapter_id][target][key].get('rel') == 'proves':
                    target_data = self.get_node_data(target)
                    proves.append({
                        'id': target,
                        'title': target_data.get('title')
                    })

        # Find forward references
        forward_refs = []
        for target in self.G.successors(chapter_id):
            for key in self.G[chapter_id][target]:
                if self.G[chapter_id][target][key].get('rel') == 'refers_to_future':
                    target_data = self.get_node_data(target)
                    forward_refs.append({
                        'id': target,
                        'title': target_data.get('title'),
                        'status': target_data.get('status')
                    })

        return {
            'chapter_id': chapter_id,
            'title': chapter_data.get('title'),
            'status': chapter_data.get('status'),
            'file': chapter_data.get('file'),
            'defines_count': len(defines),
            'defines': defines,
            'proves_count': len(proves),
            'proves': proves,
            'forward_refs': forward_refs
        }

    def implementation_status(self) -> Dict[str, List[Dict[str, str]]]:
        """Check implementation status of equations and algorithms.

        Returns:
            Dictionary with 'implemented' and 'unimplemented' lists
        """
        equations = self.nodes_by_kind('equation')
        algorithms = self.nodes_by_kind('algorithm')
        all_items = equations + algorithms

        implemented = []
        unimplemented = []

        for item_id in all_items:
            item_data = self.get_node_data(item_id)

            # Check if any module implements this
            impl_modules = []
            for pred in self.G.predecessors(item_id):
                for key in self.G[pred][item_id]:
                    if self.G[pred][item_id][key].get('rel') == 'implements':
                        pred_data = self.get_node_data(pred)
                        impl_modules.append({
                            'module': pred,
                            'file': pred_data.get('file', 'unknown')
                        })

            entry = {
                'id': item_id,
                'kind': item_data.get('kind'),
                'title': item_data.get('title'),
                'status': item_data.get('status')
            }

            if impl_modules:
                entry['implementations'] = impl_modules
                implemented.append(entry)
            else:
                unimplemented.append(entry)

        return {
            'implemented': implemented,
            'unimplemented': unimplemented
        }

    def export_stats(self) -> Dict[str, Any]:
        """Export comprehensive statistics about the knowledge graph.

        Returns:
            Dictionary with various statistics
        """
        return {
            'total_nodes': self.G.number_of_nodes(),
            'total_edges': self.G.number_of_edges(),
            'nodes_by_kind': {
                kind: len(self.nodes_by_kind(kind))
                for kind in ['chapter', 'equation', 'theorem', 'definition',
                           'concept', 'module', 'test', 'plan', 'algorithm']
            },
            'status_summary': self.status_summary(),
            'coverage_report': self.coverage_report(),
            'untested_equations': len(self.find_untested_equations()),
            'unimplemented_equations': len(self.find_unimplemented_equations()),
            'orphan_nodes': len(self.find_orphan_nodes()),
            'cycles': len(self.find_cycles()),
            'missing_refs': len(self.find_missing_refs())
        }


def main():
    """Example usage and testing."""
    import sys

    # Default to graph.yaml in same directory
    yaml_path = Path(__file__).parent / "graph.yaml"
    if len(sys.argv) > 1:
        yaml_path = Path(sys.argv[1])

    print(f"Loading knowledge graph from {yaml_path}...")
    kg = KnowledgeGraph(yaml_path)

    # Print statistics
    stats = kg.export_stats()
    print(f"\n📊 Knowledge Graph Statistics")
    print(f"{'='*50}")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total edges: {stats['total_edges']}")

    print(f"\n📝 Nodes by Kind:")
    for kind, count in stats['nodes_by_kind'].items():
        if count > 0:
            print(f"  {kind:15} {count:3}")

    print(f"\n⏳ Status Summary:")
    for status, count in stats['status_summary'].items():
        print(f"  {status:15} {count:3}")

    print(f"\n🧪 Test Coverage:")
    for kind, data in stats['coverage_report'].items():
        print(f"  {kind:15} {data['tested']}/{data['total']} ({data['coverage_pct']}%)")

    # Find issues
    print(f"\n⚠️  Issues Found:")
    untested_eqs = kg.find_untested_equations()
    if untested_eqs:
        print(f"  Untested equations: {len(untested_eqs)}")
        for eq in untested_eqs[:5]:
            print(f"    - {eq}")
        if len(untested_eqs) > 5:
            print(f"    ... and {len(untested_eqs) - 5} more")

    unimpl = kg.find_unimplemented_equations()
    if unimpl:
        print(f"  Unimplemented equations: {len(unimpl)}")
        for eq in unimpl[:5]:
            print(f"    - {eq}")
        if len(unimpl) > 5:
            print(f"    ... and {len(unimpl) - 5} more")

    missing = kg.find_missing_refs()
    if missing:
        print(f"  Dangling references: {len(missing)}")
        for src, dst in missing[:5]:
            print(f"    - {src} → {dst}")

    cycles = kg.find_cycles()
    if cycles:
        print(f"  Circular dependencies: {len(cycles)}")
        for cycle in cycles[:3]:
            print(f"    - {' → '.join(cycle)}")


if __name__ == '__main__':
    main()
