#!/usr/bin/env python
"""
Knowledge Graph Validator — Comprehensive validation and consistency checking.

Validates:
1. Referential integrity (no dangling references)
2. File existence and anchor presence
3. Status consistency (can't implement planned nodes, etc.)
4. Schema compliance
5. Circular dependencies detection
6. Orphan nodes
7. Coverage gaps
"""

from pathlib import Path
from typing import List, Dict, Set, Tuple, Any
import sys
import yaml

from kg_tools import KnowledgeGraph


class KGValidator:
    """Validator for knowledge graph consistency."""

    def __init__(self, kg: KnowledgeGraph, repo_root: Path):
        """Initialize validator.

        Args:
            kg: KnowledgeGraph instance
            repo_root: Repository root path for file validation
        """
        self.kg = kg
        self.repo_root = repo_root
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def validate_all(self) -> bool:
        """Run all validation checks.

        Returns:
            True if validation passed (no errors), False otherwise
        """
        print("🔍 Running Knowledge Graph Validation...\n")

        self._validate_referential_integrity()
        self._validate_files_and_anchors()
        self._validate_status_consistency()
        self._detect_cycles()
        self._find_orphans()
        self._check_coverage()
        self._validate_schema()

        self._print_results()

        return len(self.errors) == 0

    def _validate_referential_integrity(self) -> None:
        """Check that all referenced nodes exist."""
        missing = self.kg.find_missing_refs()

        if missing:
            for src, dst in missing:
                self.errors.append(
                    f"Dangling reference: {src} → {dst} (target does not exist)"
                )
        else:
            self.info.append("✓ All node references are valid")

    def _validate_files_and_anchors(self) -> None:
        """Check that files exist and contain declared anchors."""
        checked_files = 0
        missing_files = 0
        missing_anchors = 0

        for node_id, data in self.kg.G.nodes(data=True):
            file_path = data.get('file')
            anchor = data.get('anchor')

            if not file_path:
                continue

            # Check file exists
            full_path = self.repo_root / file_path
            if not full_path.exists():
                self.errors.append(
                    f"Missing file: {node_id} references {file_path} (not found)"
                )
                missing_files += 1
                continue

            checked_files += 1

            # Check anchor exists in file (if specified)
            if anchor:
                try:
                    content = full_path.read_text()
                    # Remove braces for flexible matching: {#EQ-1.2} or #EQ-1.2
                    anchor_clean = anchor.strip('{}')
                    if anchor_clean not in content and anchor not in content:
                        self.warnings.append(
                            f"Missing anchor: {node_id} anchor '{anchor}' not found in {file_path}"
                        )
                        missing_anchors += 1
                except Exception as e:
                    self.warnings.append(
                        f"Could not read file for {node_id}: {file_path} ({e})"
                    )

        if missing_files == 0:
            self.info.append(f"✓ All {checked_files} file references are valid")
        if missing_anchors > 0:
            self.warnings.append(
                f"Found {missing_anchors} missing anchors (may need to add to files)"
            )

    def _validate_status_consistency(self) -> None:
        """Check for status inconsistencies.

        Rules:
        - Can't implement a 'planned' equation (should be in_progress or complete)
        - If a node is 'complete', its dependencies should not be 'planned'
        - Forward refs should point to planned/in_progress nodes
        """
        for node_id, data in self.kg.G.nodes(data=True):
            status = data.get('status', 'unknown')

            # Check 1: Complete nodes depending on planned nodes
            if status == 'complete':
                deps = self.kg.transitive_dependencies(node_id)
                for dep in deps:
                    dep_data = self.kg.get_node_data(dep)
                    if dep_data.get('status') == 'planned':
                        self.warnings.append(
                            f"Status mismatch: {node_id} is 'complete' but depends on "
                            f"planned node {dep}"
                        )

            # Check 2: Modules implementing planned equations
            if data.get('kind') == 'module' and status == 'complete':
                for target in self.kg.G.successors(node_id):
                    for key in self.kg.G[node_id][target]:
                        edge_data = self.kg.G[node_id][target][key]
                        if edge_data.get('rel') == 'implements':
                            target_data = self.kg.get_node_data(target)
                            if target_data.get('status') == 'planned':
                                self.warnings.append(
                                    f"Status mismatch: {node_id} (complete) implements "
                                    f"{target} (planned)"
                                )

            # Check 3: Forward refs should point to planned/in_progress
            for target in self.kg.G.successors(node_id):
                for key in self.kg.G[node_id][target]:
                    edge_data = self.kg.G[node_id][target][key]
                    if edge_data.get('rel') == 'refers_to_future':
                        target_data = self.kg.get_node_data(target)
                        target_status = target_data.get('status')
                        if target_status == 'complete':
                            self.warnings.append(
                                f"Forward ref mismatch: {node_id} has refers_to_future "
                                f"edge to {target} which is already 'complete'"
                            )

    def _detect_cycles(self) -> None:
        """Detect circular dependencies."""
        cycles = self.kg.find_cycles()

        if cycles:
            for cycle in cycles:
                cycle_str = ' → '.join(cycle + [cycle[0]])
                self.errors.append(f"Circular dependency: {cycle_str}")
        else:
            self.info.append("✓ No circular dependencies detected")

    def _find_orphans(self) -> None:
        """Find orphaned nodes (no connections)."""
        orphans = self.kg.find_orphan_nodes()

        if orphans:
            # Orphans are not necessarily errors, especially for top-level chapters
            for orphan in orphans:
                orphan_data = self.kg.get_node_data(orphan)
                kind = orphan_data.get('kind')
                # Only warn for non-chapter orphans
                if kind != 'chapter':
                    self.warnings.append(
                        f"Orphan node: {orphan} ({kind}) has no connections"
                    )
        else:
            self.info.append("✓ No orphan nodes found")

    def _check_coverage(self) -> None:
        """Check test coverage for equations and modules."""
        untested_eqs = self.kg.find_untested_equations()
        unimpl_eqs = self.kg.find_unimplemented_equations()

        # Untested equations are warnings, not errors
        if untested_eqs:
            self.warnings.append(
                f"Found {len(untested_eqs)} untested equations: "
                f"{', '.join(untested_eqs[:5])}"
                + (f" ... and {len(untested_eqs) - 5} more" if len(untested_eqs) > 5 else "")
            )

        # Unimplemented equations might be planned, so just info
        if unimpl_eqs:
            # Filter to only complete/in_progress equations
            unimpl_actionable = [
                eq for eq in unimpl_eqs
                if self.kg.get_node_data(eq).get('status') in ['complete', 'in_progress']
            ]
            if unimpl_actionable:
                self.warnings.append(
                    f"Found {len(unimpl_actionable)} equations without implementations: "
                    f"{', '.join(unimpl_actionable[:5])}"
                    + (f" ... and {len(unimpl_actionable) - 5} more"
                       if len(unimpl_actionable) > 5 else "")
                )

        # Generate coverage report
        coverage = self.kg.coverage_report()
        for kind, stats in coverage.items():
            if stats['total'] > 0:
                if stats['coverage_pct'] < 50:
                    self.warnings.append(
                        f"Low test coverage for {kind}: {stats['coverage_pct']}% "
                        f"({stats['tested']}/{stats['total']})"
                    )
                else:
                    self.info.append(
                        f"✓ {kind} test coverage: {stats['coverage_pct']}% "
                        f"({stats['tested']}/{stats['total']})"
                    )

    def _validate_schema(self) -> None:
        """Validate schema compliance (basic checks)."""
        required_fields = ['id', 'kind', 'title', 'status']
        valid_statuses = {'planned', 'in_progress', 'complete', 'archived'}
        valid_kinds = {
            'chapter', 'section', 'equation', 'definition', 'theorem',
            'remark', 'concept', 'algorithm', 'module', 'test', 'plan', 'doc',
            'assumption', 'lemma', 'proposition'
        }

        for node_id, data in self.kg.G.nodes(data=True):
            # Check required fields
            for field in required_fields:
                if field not in data:
                    self.errors.append(
                        f"Schema violation: {node_id} missing required field '{field}'"
                    )

            # Check valid status
            status = data.get('status')
            if status and status not in valid_statuses:
                self.errors.append(
                    f"Schema violation: {node_id} has invalid status '{status}'"
                )

            # Check valid kind
            kind = data.get('kind')
            if kind and kind not in valid_kinds:
                self.errors.append(
                    f"Schema violation: {node_id} has invalid kind '{kind}'"
                )

        # Check edge relation types
        valid_rels = {
            'defines', 'proves', 'uses', 'implements', 'tested_by',
            'depends_on', 'refers_to_future', 'superseded_by', 'see_also'
        }

        for u, v, data in self.kg.G.edges(data=True):
            rel = data.get('rel')
            if rel and rel not in valid_rels:
                self.errors.append(
                    f"Schema violation: edge {u} → {v} has invalid relation '{rel}'"
                )

    def _print_results(self) -> None:
        """Print validation results."""
        print(f"\n{'='*70}")
        print(f"📋 Validation Results")
        print(f"{'='*70}\n")

        if self.errors:
            print(f"❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   {error}")
            print()

        if self.warnings:
            print(f"⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   {warning}")
            print()

        if self.info:
            print(f"ℹ️  INFO ({len(self.info)}):")
            for info_msg in self.info:
                print(f"   {info_msg}")
            print()

        # Summary
        print(f"{'='*70}")
        if self.errors:
            print(f"❌ Validation FAILED: {len(self.errors)} errors, {len(self.warnings)} warnings")
        elif self.warnings:
            print(f"⚠️  Validation PASSED with warnings: {len(self.warnings)} warnings")
        else:
            print(f"✅ Validation PASSED: No errors or warnings!")
        print(f"{'='*70}\n")


def main():
    """Run validation from command line."""
    # Determine paths
    script_dir = Path(__file__).parent
    yaml_path = script_dir / "graph.yaml"
    repo_root = script_dir.parent.parent  # Up two levels from docs/knowledge_graph/

    if len(sys.argv) > 1:
        yaml_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        repo_root = Path(sys.argv[2])

    print(f"Loading knowledge graph from: {yaml_path}")
    print(f"Repository root: {repo_root}\n")

    # Load graph
    kg = KnowledgeGraph(yaml_path)

    # Run validation
    validator = KGValidator(kg, repo_root)
    success = validator.validate_all()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
