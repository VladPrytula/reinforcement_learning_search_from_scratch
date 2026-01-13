SHELL := /bin/bash

# Render Markdown files to PDF with pandoc.
#
# Requirements:
# - pandoc (https://pandoc.org/)
# - A LaTeX engine (default: xelatex). Install MacTeX/TeXLive or set PDF_ENGINE.
#
# Usage examples:
#   make pdf              # build PDFs for all book markdowns
#   make pdf-drafts       # build PDFs for drafts only
#   make docs/book/drafts/ch01_foundations.pdf  # build one file
#   make clean-pdf        # remove generated PDFs

PANDOC ?= pandoc
PDF_ENGINE ?= xelatex
PANDOC_HEADER ?= docs/book/preamble.tex
CALLOUT_FILTER := docs/book/callouts.lua
ADMONITION_FILTER := docs/book/admonitions.lua
CROSSREF_FILTER := docs/book/crossrefs.lua

# Font overrides (optional). Leave blank to use engine defaults.
# You can override via: make MAIN_FONT="TeX Gyre Pagella" MONO_FONT="DejaVu Sans Mono"
MAIN_FONT ?=
SANS_FONT ?=
MONO_FONT ?=

PANDOC_FONT_FLAGS :=
ifneq ($(strip $(MAIN_FONT)),)
PANDOC_FONT_FLAGS += -V mainfont='$(MAIN_FONT)'
endif
ifneq ($(strip $(SANS_FONT)),)
PANDOC_FONT_FLAGS += -V sansfont='$(SANS_FONT)'
endif
ifneq ($(strip $(MONO_FONT)),)
PANDOC_FONT_FLAGS += -V monofont='$(MONO_FONT)'
endif

# Pandoc flags: Markdown with math extensions; numbered sections and TOC.
PANDOC_COMMON_FLAGS := \
  --from=markdown+tex_math_dollars+tex_math_single_backslash+raw_html+fenced_divs \
  --pdf-engine=$(PDF_ENGINE) \
  --syntax-highlighting=tango \
  --number-sections \
  --toc --toc-depth=2 \
  $(if $(wildcard $(PANDOC_HEADER)),--include-in-header=$(PANDOC_HEADER),) \
  -V geometry:margin=1in \
  $(PANDOC_FONT_FLAGS)

ifneq ($(wildcard $(CALLOUT_FILTER)),)
PANDOC_COMMON_FLAGS += --lua-filter=$(CALLOUT_FILTER)
endif
ifneq ($(wildcard $(ADMONITION_FILTER)),)
PANDOC_COMMON_FLAGS += --lua-filter=$(ADMONITION_FILTER)
endif
ifneq ($(wildcard $(CROSSREF_FILTER)),)
PANDOC_COMMON_FLAGS += --lua-filter=$(CROSSREF_FILTER)
endif

# Book directories to process (can be extended to other doc trees if desired).
BOOK_DIRS := $(wildcard docs/book/ch?? docs/book/drafts docs/book/reviews docs/book/revisions)

# Discover Markdown sources; exclude backups.
MD_FILES := $(shell find $(BOOK_DIRS) -type f -name '*.md' \( -not -name '*.bak.md' \) \( -not -name '*.bak.*' \))
PDFS     := $(MD_FILES:.md=.pdf)

# Draft-only convenience targets
DRAFT_MD_FILES := $(shell find docs/book/drafts -type f -name '*.md' \( -not -name '*.bak.md' \) \( -not -name '*.bak.*' \))
DRAFT_PDFS     := $(DRAFT_MD_FILES:.md=.pdf)

.PHONY: help check-tools pdf pdf-drafts clean-pdf validate-kg
.DEFAULT_GOAL := help

help:
	@echo "Make targets:"
	@echo "  make pdf            Build PDFs for all book markdowns"
	@echo "  make pdf-drafts     Build PDFs for drafts only"
	@echo "  make clean-pdf      Remove generated PDFs in book directories"
	@echo ""
	@echo "  make audit-ch00-ch03  Audit Part I alignment (voice/refs/artifacts/citations)"
	@echo ""
	@echo "  make tex            Generate LaTeX for all chapters"
	@echo "  make tex-ch01       Generate LaTeX for Chapter 1 only"
	@echo "  make clean-tex      Remove generated LaTeX files"
	@echo ""
	@echo "Variables: PANDOC=$(PANDOC) PDF_ENGINE=$(PDF_ENGINE)"

check-tools:
	@command -v $(PANDOC) >/dev/null 2>&1 || { \
	  echo "Error: pandoc not found. Install pandoc to build PDFs." >&2; exit 1; }

# Bulk builds
pdf: check-tools $(PDFS)

pdf-drafts: check-tools $(DRAFT_PDFS)

# Pattern rule: .md → .pdf (next to source)
%.pdf: %.md
	@echo "[pandoc] $< → $@"
	@$(PANDOC) $(PANDOC_COMMON_FLAGS) --resource-path=.:docs/book:$(dir $<) -o "$@" "$<"

clean-pdf:
	@echo "Removing generated PDFs in: $(BOOK_DIRS)"
	@find $(BOOK_DIRS) -type f -name '*.pdf' -print -delete

# -----------------------------------------------------------------------------
# Knowledge Graph validation
# -----------------------------------------------------------------------------

PY ?= python
ifneq ($(wildcard .venv/bin/python),)
ifeq ($(PY),python)
PY := .venv/bin/python
endif
endif

validate-kg:
	@echo "[kg] Validating docs/knowledge_graph/graph.yaml"
	@$(PY) scripts/validate_knowledge_graph.py --graph docs/knowledge_graph/graph.yaml --schema docs/knowledge_graph/schema.yaml

# -----------------------------------------------------------------------------
# Ch00–Ch03 alignment audit (Part I)
# -----------------------------------------------------------------------------

.PHONY: audit-ch00-ch03

audit-ch00-ch03:
	@echo "[audit] Ch00–Ch03 alignment"
	@$(PY) scripts/audit_ch00_ch03_alignment.py

# -----------------------------------------------------------------------------
# LaTeX Generation (Markdown → .tex)
# -----------------------------------------------------------------------------
# Non-breaking: These targets are independent of the existing PDF workflow.
#
# Usage:
#   make tex-ch01         Generate LaTeX for Chapter 1
#   make tex              Generate LaTeX for all chapters
#   make clean-tex        Remove generated LaTeX files
#
# Output: docs/book/latex/ch01.tex, ch02.tex, etc.

TEX_OUTPUT_DIR ?= docs/book/latex
LATEX_TEMPLATE := docs/book/latex_template.tex
PREPROCESS_SCRIPT := scripts/md2tex_preprocess.py

# Chapter source files (main content files only)
CH01_SRC := docs/book/ch01/ch01_foundations.md

# Pandoc flags for LaTeX generation
PANDOC_TEX_FLAGS := \
  --from=markdown-yaml_metadata_block+tex_math_dollars+tex_math_single_backslash+raw_html+fenced_divs \
  --to=latex \
  --standalone \
  --number-sections \
  --toc --toc-depth=2 \
  $(if $(wildcard $(LATEX_TEMPLATE)),--template=$(LATEX_TEMPLATE),) \
  $(if $(wildcard $(CALLOUT_FILTER)),--lua-filter=$(CALLOUT_FILTER),) \
  $(if $(wildcard $(CROSSREF_FILTER)),--lua-filter=$(CROSSREF_FILTER),) \
  $(if $(wildcard $(ADMONITION_FILTER)),--lua-filter=$(ADMONITION_FILTER),)

.PHONY: tex tex-ch01 clean-tex

tex: tex-ch01
	@echo "[tex] All chapters generated in $(TEX_OUTPUT_DIR)/"

tex-ch01: $(TEX_OUTPUT_DIR)/ch01.tex

$(TEX_OUTPUT_DIR)/ch01.tex: $(CH01_SRC) $(LATEX_TEMPLATE) $(CALLOUT_FILTER) $(CROSSREF_FILTER) $(PREPROCESS_SCRIPT)
	@mkdir -p $(TEX_OUTPUT_DIR)
	@echo "[preprocess] $(CH01_SRC)"
	@$(PY) $(PREPROCESS_SCRIPT) "$(CH01_SRC)" -o /tmp/ch01_preprocessed.md
	@echo "[pandoc] /tmp/ch01_preprocessed.md → $@"
	@$(PANDOC) $(PANDOC_TEX_FLAGS) \
	    --metadata title="Chapter 1: Search Ranking as Optimization" \
	    --metadata author="Vlad Prytula" \
	    -o "$@" /tmp/ch01_preprocessed.md
	@rm -f /tmp/ch01_preprocessed.md
	@echo "[done] $@"

clean-tex:
	@echo "Removing generated LaTeX files in: $(TEX_OUTPUT_DIR)"
	@rm -rf $(TEX_OUTPUT_DIR)
