#!/usr/bin/env python
"""
Knowledge Graph Visualizer — Generate dependency graphs and visual representations.

Features:
- Chapter dependency graphs
- Module implementation diagrams
- Forward reference visualization
- Coverage heatmaps
- Export to PNG, SVG, or DOT format
"""

from pathlib import Path
from typing import Optional, Set, List
import sys

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from kg_tools import KnowledgeGraph


class KGVisualizer:
    """Visualizer for knowledge graph."""

    def __init__(self, kg: KnowledgeGraph):
        """Initialize visualizer.

        Args:
            kg: KnowledgeGraph instance
        """
        self.kg = kg

        # Color scheme by node kind
        self.kind_colors = {
            'chapter': '#4A90E2',      # Blue
            'section': '#7ED321',      # Green
            'equation': '#F5A623',     # Orange
            'theorem': '#D0021B',      # Red
            'definition': '#BD10E0',   # Purple
            'algorithm': '#50E3C2',    # Teal
            'module': '#B8E986',       # Light green
            'test': '#F8E71C',         # Yellow
            'concept': '#8B572A',      # Brown
            'plan': '#9013FE',         # Violet
            'doc': '#417505',          # Dark green
            'remark': '#FF6B6B',       # Light red
        }

        # Status markers
        self.status_markers = {
            'planned': 'o',       # Circle
            'in_progress': 's',   # Square
            'complete': 'D',      # Diamond
            'archived': 'x',      # X
        }

    def visualize_chapter(self,
                         chapter_id: str,
                         output_path: Optional[Path] = None,
                         depth: int = 2,
                         include_tests: bool = False) -> None:
        """Visualize a chapter and its dependencies.

        Args:
            chapter_id: Chapter node ID (e.g., 'CH-1')
            output_path: Path to save PNG (if None, displays interactively)
            depth: How many hops to include in visualization
            include_tests: Whether to include test nodes
        """
        # Build subgraph around chapter
        subgraph_nodes = self._build_subgraph(chapter_id, depth, include_tests)

        # Create filtered graph
        G_viz = self.kg.G.subgraph(subgraph_nodes).copy()

        # Setup plot
        fig, ax = plt.subplots(figsize=(16, 12))

        # Layout
        pos = self._compute_layout(G_viz, chapter_id)

        # Draw edges by type
        self._draw_edges(G_viz, pos, ax)

        # Draw nodes
        self._draw_nodes(G_viz, pos, ax)

        # Add labels
        self._draw_labels(G_viz, pos, ax)

        # Add legend
        self._add_legend(ax)

        # Title
        chapter_data = self.kg.get_node_data(chapter_id)
        title = f"{chapter_id}: {chapter_data.get('title', 'Unknown')}"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()

        plt.close()

    def visualize_dependencies(self,
                               node_id: str,
                               output_path: Optional[Path] = None,
                               direction: str = 'both') -> None:
        """Visualize dependency tree for a node.

        Args:
            node_id: Node to visualize
            output_path: Path to save PNG (if None, displays interactively)
            direction: 'upstream' (dependencies), 'downstream' (dependents), or 'both'
        """
        nodes_to_include = {node_id}

        if direction in ['upstream', 'both']:
            deps = self.kg.transitive_dependencies(node_id)
            nodes_to_include.update(deps)

        if direction in ['downstream', 'both']:
            dependents = self.kg.transitive_dependents(node_id)
            nodes_to_include.update(dependents)

        # Create filtered graph
        G_viz = self.kg.G.subgraph(nodes_to_include).copy()

        # Setup plot
        fig, ax = plt.subplots(figsize=(14, 10))

        # Hierarchical layout
        pos = self._compute_hierarchical_layout(G_viz, node_id)

        # Draw
        self._draw_edges(G_viz, pos, ax)
        self._draw_nodes(G_viz, pos, ax)
        self._draw_labels(G_viz, pos, ax)
        self._add_legend(ax)

        node_data = self.kg.get_node_data(node_id)
        title = f"Dependencies for {node_id}: {node_data.get('title', 'Unknown')}"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved dependency graph to {output_path}")
        else:
            plt.show()

        plt.close()

    def visualize_implementation_map(self,
                                    output_path: Optional[Path] = None) -> None:
        """Visualize which modules implement which equations/algorithms.

        Args:
            output_path: Path to save PNG (if None, displays interactively)
        """
        # Get equations and modules
        equations = self.kg.nodes_by_kind('equation')
        algorithms = self.kg.nodes_by_kind('algorithm')
        modules = self.kg.nodes_by_kind('module')

        # Build bipartite-like graph
        nodes_to_include = set(equations + algorithms + modules)

        # Filter to only implements edges
        G_viz = nx.DiGraph()
        for u, v, data in self.kg.G.edges(data=True):
            if data.get('rel') == 'implements' and u in nodes_to_include and v in nodes_to_include:
                G_viz.add_edge(u, v, **data)

        # Add node attributes
        for node in G_viz.nodes():
            G_viz.nodes[node].update(self.kg.G.nodes[node])

        # Setup plot
        fig, ax = plt.subplots(figsize=(16, 12))

        # Bipartite layout
        left_nodes = [n for n in G_viz.nodes() if self.kg.G.nodes[n].get('kind') == 'module']
        right_nodes = [n for n in G_viz.nodes() if n not in left_nodes]

        pos = {}
        left_y_positions = range(len(left_nodes))
        right_y_positions = range(len(right_nodes))

        for i, node in enumerate(left_nodes):
            pos[node] = (0, left_y_positions[i])
        for i, node in enumerate(right_nodes):
            pos[node] = (1, right_y_positions[i])

        # Draw
        self._draw_edges(G_viz, pos, ax, edge_alpha=0.5)
        self._draw_nodes(G_viz, pos, ax)
        self._draw_labels(G_viz, pos, ax, font_size=8)

        ax.set_title("Implementation Map: Modules → Equations/Algorithms",
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved implementation map to {output_path}")
        else:
            plt.show()

        plt.close()

    def export_dot(self, output_path: Path, node_subset: Optional[Set[str]] = None) -> None:
        """Export graph to Graphviz DOT format.

        Args:
            output_path: Path to save .dot file
            node_subset: Optional set of nodes to include (if None, exports all)
        """
        if node_subset:
            G_export = self.kg.G.subgraph(node_subset).copy()
        else:
            G_export = self.kg.G.copy()

        # Convert to undirected for simpler DOT
        G_dot = G_export.to_undirected()

        # Write DOT file
        nx.drawing.nx_pydot.write_dot(G_dot, output_path)
        print(f"Exported DOT file to {output_path}")
        print(f"Render with: dot -Tpng {output_path} -o {output_path.with_suffix('.png')}")

    def _build_subgraph(self,
                       root_id: str,
                       depth: int,
                       include_tests: bool) -> Set[str]:
        """Build set of nodes to include in subgraph.

        Args:
            root_id: Root node
            depth: How many hops
            include_tests: Whether to include test nodes

        Returns:
            Set of node IDs
        """
        nodes = {root_id}

        # BFS to depth
        current_level = {root_id}
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                # Add successors
                for succ in self.kg.G.successors(node):
                    succ_data = self.kg.get_node_data(succ)
                    if include_tests or succ_data.get('kind') != 'test':
                        next_level.add(succ)
                # Add predecessors
                for pred in self.kg.G.predecessors(node):
                    pred_data = self.kg.get_node_data(pred)
                    if include_tests or pred_data.get('kind') != 'test':
                        next_level.add(pred)

            nodes.update(next_level)
            current_level = next_level

        return nodes

    def _compute_layout(self, G: nx.Graph, center_node: Optional[str] = None) -> dict:
        """Compute node positions using spring layout.

        Args:
            G: Graph to layout
            center_node: Optional node to center

        Returns:
            Dictionary mapping node -> (x, y)
        """
        if center_node and center_node in G:
            # Use spring layout with fixed center
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        else:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        return pos

    def _compute_hierarchical_layout(self, G: nx.DiGraph, root: str) -> dict:
        """Compute hierarchical layout for dependency tree.

        Args:
            G: Directed graph
            root: Root node

        Returns:
            Dictionary mapping node -> (x, y)
        """
        # Try to use graphviz_layout if available, else fall back to spring
        try:
            from networkx.drawing.nx_agraph import graphviz_layout
            pos = graphviz_layout(G, prog='dot')
        except (ImportError, Exception):
            # Fall back to spring layout
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        return pos

    def _draw_edges(self, G: nx.Graph, pos: dict, ax, edge_alpha: float = 0.3) -> None:
        """Draw edges with different styles by relation type.

        Args:
            G: Graph
            pos: Node positions
            ax: Matplotlib axis
            edge_alpha: Edge transparency
        """
        # Group edges by relation type
        edge_styles = {
            'defines': {'color': 'blue', 'style': 'solid', 'width': 2},
            'proves': {'color': 'red', 'style': 'solid', 'width': 2},
            'uses': {'color': 'gray', 'style': 'dashed', 'width': 1},
            'implements': {'color': 'green', 'style': 'solid', 'width': 2.5},
            'tested_by': {'color': 'orange', 'style': 'dotted', 'width': 1.5},
            'depends_on': {'color': 'purple', 'style': 'dashed', 'width': 1.5},
            'refers_to_future': {'color': 'brown', 'style': 'dashdot', 'width': 1},
        }

        for rel, style in edge_styles.items():
            edges = [
                (u, v) for u, v, data in G.edges(data=True)
                if data.get('rel') == rel
            ]
            if edges:
                nx.draw_networkx_edges(
                    G, pos, edgelist=edges,
                    edge_color=style['color'],
                    style=style['style'],
                    width=style['width'],
                    alpha=edge_alpha,
                    arrows=True,
                    arrowsize=10,
                    ax=ax
                )

    def _draw_nodes(self, G: nx.Graph, pos: dict, ax) -> None:
        """Draw nodes with colors by kind and markers by status.

        Args:
            G: Graph
            pos: Node positions
            ax: Matplotlib axis
        """
        # Group nodes by kind
        for kind, color in self.kind_colors.items():
            nodes = [n for n, d in G.nodes(data=True) if d.get('kind') == kind]
            if nodes:
                # Further group by status for marker style
                for status, marker in self.status_markers.items():
                    status_nodes = [
                        n for n in nodes
                        if G.nodes[n].get('status') == status
                    ]
                    if status_nodes:
                        nx.draw_networkx_nodes(
                            G, pos, nodelist=status_nodes,
                            node_color=color,
                            node_shape=marker,
                            node_size=800,
                            alpha=0.9,
                            ax=ax
                        )

    def _draw_labels(self, G: nx.Graph, pos: dict, ax, font_size: int = 10) -> None:
        """Draw node labels.

        Args:
            G: Graph
            pos: Node positions
            ax: Matplotlib axis
            font_size: Label font size
        """
        labels = {n: n for n in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, labels,
            font_size=font_size,
            font_weight='bold',
            ax=ax
        )

    def _add_legend(self, ax) -> None:
        """Add legend for node kinds and statuses.

        Args:
            ax: Matplotlib axis
        """
        # Kind legend (colors)
        kind_patches = [
            mpatches.Patch(color=color, label=kind.capitalize())
            for kind, color in self.kind_colors.items()
        ]

        # Status legend (markers)
        status_handles = [
            ax.scatter([], [], marker=marker, s=100, c='gray', label=status.replace('_', ' ').capitalize())
            for status, marker in self.status_markers.items()
        ]

        # Combine legends
        legend1 = ax.legend(handles=kind_patches, loc='upper left',
                          title='Node Kind', fontsize=8, framealpha=0.9)
        ax.add_artist(legend1)  # Keep first legend when adding second
        ax.legend(handles=status_handles, loc='upper right',
                 title='Status', fontsize=8, framealpha=0.9)


def main():
    """Command-line interface for visualization."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize knowledge graph')
    parser.add_argument('--graph', default='docs/knowledge_graph/graph.yaml',
                       help='Path to graph.yaml')
    parser.add_argument('--chapter', type=str,
                       help='Visualize specific chapter (e.g., CH-1)')
    parser.add_argument('--deps', type=str,
                       help='Visualize dependencies for node')
    parser.add_argument('--impl-map', action='store_true',
                       help='Generate implementation map')
    parser.add_argument('--output', type=Path,
                       help='Output path (PNG/SVG)')
    parser.add_argument('--depth', type=int, default=2,
                       help='Subgraph depth (default: 2)')
    parser.add_argument('--include-tests', action='store_true',
                       help='Include test nodes')

    args = parser.parse_args()

    # Load graph
    print(f"Loading knowledge graph from {args.graph}...")
    kg = KnowledgeGraph(args.graph)

    viz = KGVisualizer(kg)

    if args.chapter:
        print(f"Visualizing chapter: {args.chapter}")
        viz.visualize_chapter(
            args.chapter,
            output_path=args.output,
            depth=args.depth,
            include_tests=args.include_tests
        )
    elif args.deps:
        print(f"Visualizing dependencies for: {args.deps}")
        viz.visualize_dependencies(
            args.deps,
            output_path=args.output
        )
    elif args.impl_map:
        print("Generating implementation map...")
        viz.visualize_implementation_map(output_path=args.output)
    else:
        # Default: visualize CH-1 if it exists
        chapters = kg.nodes_by_kind('chapter')
        if 'CH-1' in chapters:
            print("No visualization specified. Showing CH-1 by default.")
            viz.visualize_chapter('CH-1', output_path=args.output)
        else:
            print("Error: No visualization specified. Use --chapter, --deps, or --impl-map")
            sys.exit(1)


if __name__ == '__main__':
    main()
