# Knowledge Graph Makefile
# Quick commands for common KG operations

# Detect virtual environment
VENV := ../../.venv
PYTHON := $(VENV)/bin/python

.PHONY: help stats validate visualize clean examples

help:
	@echo "Knowledge Graph Utilities"
	@echo "========================="
	@echo ""
	@echo "Available commands:"
	@echo "  make stats       - Show knowledge graph statistics"
	@echo "  make validate    - Run comprehensive validation"
	@echo "  make examples    - Run example queries"
	@echo "  make visualize   - Generate all visualizations"
	@echo "  make clean       - Remove generated files"
	@echo ""

stats:
	@echo "📊 Knowledge Graph Statistics"
	@echo "=============================="
	@$(PYTHON) kg_tools.py

validate:
	@echo "🔍 Validating Knowledge Graph"
	@echo "=============================="
	@$(PYTHON) validate_kg.py

examples:
	@echo "💡 Running Example Queries"
	@echo "=========================="
	@$(PYTHON) example_queries.py

visualize: visualize-ch1 visualize-ch11 visualize-impl

visualize-ch1:
	@echo "📈 Generating Chapter 1 visualization..."
	@$(PYTHON) visualize_kg.py --chapter CH-1 --depth 2 --output viz_ch01.png

visualize-ch11:
	@echo "📈 Generating Chapter 11 dependencies..."
	@$(PYTHON) visualize_kg.py --deps CH-11 --output viz_ch11_deps.png

visualize-impl:
	@echo "📈 Generating implementation map..."
	@$(PYTHON) visualize_kg.py --impl-map --output viz_impl_map.png

clean:
	@echo "🧹 Cleaning generated files..."
	@rm -f viz_*.png
	@rm -rf __pycache__
	@echo "Done!"

# Quick validation before commit
pre-commit: validate
	@echo ""
	@echo "✅ Pre-commit checks passed!"
	@echo "   Safe to commit graph.yaml changes"
