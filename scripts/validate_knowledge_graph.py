#!/usr/bin/env python3
"""Validate docs/knowledge_graph/graph.yaml against schema.yaml.

Checks:
- Node required fields and enums
- ID uniqueness
- File path existence (if provided)
- Anchor existence (best-effort substring search)
- Adjacency references point to existing node IDs
- Edge src/dst existence and enum correctness

Exit code is non-zero on validation errors.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Set, Tuple

try:
    import yaml  # PyYAML
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


Node = Dict[str, Any]
Edge = Dict[str, Any]


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is not None:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    ruby = shutil.which("ruby")
    if not ruby:
        raise ModuleNotFoundError(
            "PyYAML is not installed and no 'ruby' executable was found for a YAML fallback. "
            "Install PyYAML (e.g. `pip install pyyaml`) or run in the project virtualenv."
        )

    try:
        proc = subprocess.run(
            [ruby, "-ryaml", "-rjson", "-e", "puts JSON.generate(YAML.load_file(ARGV[0]))", path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to parse YAML via Ruby fallback: {e.stderr.strip()}") from e
    return json.loads(proc.stdout)


def _schema_enums(schema: Dict[str, Any]) -> Tuple[Set[str], Set[str], Set[str]]:
    # Extract enums for kind/status and edge.rel
    node_fields = schema.get("node", {}).get("fields", {})
    kind_vals = set(node_fields.get("kind", {}).get("values", []))
    status_vals = set(node_fields.get("status", {}).get("values", []))
    edge_rel_vals = set(schema.get("edge", {}).get("fields", {}).get("rel", {}).get("values", []))
    return kind_vals, status_vals, edge_rel_vals


def validate_nodes(nodes: List[Node], kind_vals: Set[str], status_vals: Set[str]) -> Tuple[List[str], List[str], Dict[str, Node]]:
    errors: List[str] = []
    warnings: List[str] = []
    by_id: Dict[str, Node] = {}

    required = {"id", "kind", "title", "status"}
    ref_fields = {
        "depends_on",
        "provides",
        "defines",
        "uses",
        "proves",
        "implements",
        "tested_by",
        "forward_refs",
        "see_also",
    }

    for i, n in enumerate(nodes):
        where = f"nodes[{i}]"
        missing = required - set(n.keys())
        if missing:
            errors.append(f"{where}: missing required fields: {sorted(missing)}")
            continue
        nid = str(n["id"]).strip()
        if not nid:
            errors.append(f"{where}: empty id")
            continue
        if nid in by_id:
            errors.append(f"{where}: duplicate id '{nid}' also used by another node")
        by_id[nid] = n

        kind = str(n["kind"]).strip()
        status = str(n["status"]).strip()
        if kind_vals and kind not in kind_vals:
            errors.append(f"{where}:{nid}: unknown kind '{kind}' not in {sorted(kind_vals)}")
        if status_vals and status not in status_vals:
            errors.append(f"{where}:{nid}: unknown status '{status}' not in {sorted(status_vals)}")

        # File existence
        if "file" in n:
            fpath = n["file"]
            if not isinstance(fpath, str) or not fpath:
                errors.append(f"{where}:{nid}: invalid file field")
            else:
                if not os.path.exists(fpath):
                    errors.append(f"{where}:{nid}: file does not exist: {fpath}")
                else:
                    # Anchor check (best-effort)
                    anchor = n.get("anchor")
                    if anchor:
                        try:
                            with open(fpath, "r", encoding="utf-8") as fh:
                                content = fh.read()
                            if anchor not in content:
                                warnings.append(f"{where}:{nid}: anchor not found in file: {anchor} in {fpath}")
                        except Exception as e:
                            warnings.append(f"{where}:{nid}: could not read file for anchor check: {e}")

        # Collect adjacency reference values for later existence checks
        for rf in ref_fields:
            vals = n.get(rf)
            if vals is None:
                continue
            if not isinstance(vals, list):
                errors.append(f"{where}:{nid}: field '{rf}' must be a list of ids")
            else:
                for v in vals:
                    if not isinstance(v, str):
                        errors.append(f"{where}:{nid}: field '{rf}' contains non-string id: {v!r}")

    return errors, warnings, by_id


def validate_node_refs(nodes: List[Node], by_id: Dict[str, Node]) -> List[str]:
    errors: List[str] = []
    ref_fields = [
        "depends_on",
        "provides",
        "defines",
        "uses",
        "proves",
        "implements",
        "tested_by",
        "forward_refs",
        "see_also",
    ]
    for n in nodes:
        nid = n.get("id")
        for rf in ref_fields:
            vals = n.get(rf) or []
            for v in vals:
                if v not in by_id:
                    errors.append(f"node:{nid}: field '{rf}' references unknown id '{v}'")
    return errors


def validate_edges(edges: List[Edge], by_id: Dict[str, Node], status_vals: Set[str], rel_vals: Set[str]) -> List[str]:
    errors: List[str] = []
    for i, e in enumerate(edges):
        where = f"edges[{i}]"
        for key in ("src", "dst", "rel", "status"):
            if key not in e:
                errors.append(f"{where}: missing field '{key}'")
        src = e.get("src")
        dst = e.get("dst")
        rel = e.get("rel")
        st = e.get("status")
        if src and src not in by_id:
            errors.append(f"{where}: unknown src id '{src}'")
        if dst and dst not in by_id:
            errors.append(f"{where}: unknown dst id '{dst}'")
        if rel_vals and rel not in rel_vals:
            errors.append(f"{where}: unknown rel '{rel}' not in {sorted(rel_vals)}")
        if status_vals and st not in status_vals:
            errors.append(f"{where}: unknown status '{st}' not in {sorted(status_vals)}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate knowledge graph YAML")
    ap.add_argument("--graph", default="docs/knowledge_graph/graph.yaml", help="Path to graph.yaml")
    ap.add_argument("--schema", default="docs/knowledge_graph/schema.yaml", help="Path to schema.yaml")
    args = ap.parse_args()

    graph = load_yaml(args.graph)
    schema = load_yaml(args.schema)

    nodes: List[Node] = graph.get("nodes", [])
    edges: List[Edge] = graph.get("edges", [])

    kind_vals, status_vals, edge_rel_vals = _schema_enums(schema)

    node_errors, node_warnings, by_id = validate_nodes(nodes, kind_vals, status_vals)
    ref_errors = validate_node_refs(nodes, by_id)
    edge_errors = validate_edges(edges, by_id, status_vals, edge_rel_vals)

    errors = node_errors + ref_errors + edge_errors

    print(f"Knowledge Graph Validation")
    print(f"  Nodes: {len(nodes)}  Edges: {len(edges)}")
    print(f"  Errors: {len(errors)}  Warnings: {len(node_warnings)}")
    if node_warnings:
        print("\nWarnings:")
        for w in node_warnings:
            print(f"  - {w}")
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
