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
PANDOC_HEADER ?= docs/book/pandoc_header.tex
CALLOUT_FILTER := docs/book/callouts.lua

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

# Book directories to process (can be extended to other doc trees if desired).
BOOK_DIRS := docs/book/drafts docs/book/reviews docs/book/revisions docs/book/final

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
	@$(PANDOC) $(PANDOC_COMMON_FLAGS) -o "$@" "$<"

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
