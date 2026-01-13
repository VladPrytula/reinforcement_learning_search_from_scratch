# PDF Compilation Guide

Complete instructions for generating PDFs from markdown chapters.

---

## Prerequisites

### Required Software

1. **Pandoc** (tested with version 3.8.2.1+)
   ```bash
   # macOS
   brew install pandoc

   # Verify installation
   pandoc --version
   ```

2. **LaTeX Distribution** (for XeLaTeX PDF engine)
   ```bash
   # macOS - Full installation (recommended)
   brew install --cask mactex

   # OR - Minimal installation (BasicTeX)
   brew install --cask basictex

   # After BasicTeX, install required packages
   sudo tlmgr update --self
   sudo tlmgr install tcolorbox pgf environ trimspaces

   # Verify XeLaTeX is available
   which xelatex
   ```

3. **Required LaTeX Packages**
   - `tcolorbox` - For callout boxes
   - `amsmath`, `amssymb` - For mathematical typesetting (typically included)
   - Standard packages (geometry, fontspec, etc.) - Usually included

### Required Project Files

These must exist before compilation:

1. **`docs/book/admonitions.lua`** - Pandoc Lua filter for MkDocs `!!!` / `???` admonitions
2. **`docs/book/callouts.lua`** - Pandoc Lua filter for fenced div callouts (`:::`)
3. **`docs/book/crossrefs.lua`** - Pandoc Lua filter for `{#...}` anchors and `#EQ-*` / `[EQ-*]` references
4. **`docs/book/preamble.tex`** - LaTeX preamble with package imports
5. **Source markdown files** - Chapter content

---

## File Locations

!!! note "Folder structure update"
    Older documentation referred to `docs/book/final/`. That folder is an artifact of an older layout. The current canonical sources live directly under `docs/book/chXX/` (for example `docs/book/ch02/`).

```
rl_search_from_scratch/
├── docs/book/
│   ├── callouts.lua          # Pandoc filter
│   ├── preamble.tex          # LaTeX preamble
│   ├── ch00/
│   │   └── ch00_motivation_first_experiment.md
│   └── ch01/
│       └── ch01_foundations.md
└── ch00.pdf                  # Generated (root directory)
    ch01.pdf                  # Generated (root directory)
```

---

## Compilation Commands

### Chapter 0 - Full Command

**From project root directory** (`/Users/vladyslavp/src_local/rl_search_from_scratch/`):

```bash
pandoc docs/book/ch00/ch00_motivation_first_experiment.md \
  --lua-filter=docs/book/admonitions.lua \
  --lua-filter=docs/book/callouts.lua \
  --lua-filter=docs/book/crossrefs.lua \
  --include-in-header=docs/book/preamble.tex \
  --pdf-engine=xelatex \
  --resource-path=.:docs/book:docs/book/ch00 \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  --toc \
  -N \
  -o ch00.pdf
```

**Expected output file**: `ch00.pdf` (109K) in project root

### Chapter 1 - Full Command

**From project root directory**:

```bash
pandoc docs/book/ch01/ch01_foundations.md \
  --lua-filter=docs/book/admonitions.lua \
  --lua-filter=docs/book/callouts.lua \
  --lua-filter=docs/book/crossrefs.lua \
  --include-in-header=docs/book/preamble.tex \
  --pdf-engine=xelatex \
  --resource-path=.:docs/book:docs/book/ch01 \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  --toc \
  -N \
  -o ch01.pdf
```

**Expected output file**: `ch01.pdf` (197K) in project root

### Chapter 2 - Full Command (MkDocs admonitions + crossrefs)

**From project root directory**:

```bash
pandoc docs/book/ch02/ch02_probability_measure_click_models.md \
  --lua-filter=docs/book/admonitions.lua \
  --lua-filter=docs/book/callouts.lua \
  --lua-filter=docs/book/crossrefs.lua \
  --include-in-header=docs/book/preamble.tex \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  --toc \
  -N \
  -o ch02.pdf
```

**Filter order matters:** run `admonitions.lua` (MkDocs `!!!` / `???`) before `crossrefs.lua` so references inside converted blocks are resolved.

---

## Command Breakdown

### Pandoc Flags Explained

| Flag | Purpose | Notes |
|------|---------|-------|
| `--lua-filter=docs/book/admonitions.lua` | Convert MkDocs admonitions | Transforms `!!! tip "..."` / `??? note "..."` to LaTeX tcolorbox |
| `--lua-filter=docs/book/callouts.lua` | Convert fenced div callouts | Transforms `::: {.note title="..."}` to LaTeX tcolorbox |
| `--lua-filter=docs/book/crossrefs.lua` | Resolve anchors and crossrefs | Converts `{#EQ-2.1}` → `\\label{EQ-2.1}` and `#EQ-2.1` → `\\eqref{EQ-2.1}` |
| `--include-in-header=docs/book/preamble.tex` | Include LaTeX preamble | Loads packages, defines environments |
| `--pdf-engine=xelatex` | Use XeLaTeX for PDF generation | Better Unicode support than pdflatex |
| `-V geometry:margin=1in` | Set page margins to 1 inch | Pandoc variable for page layout |
| `-V fontsize=11pt` | Set base font size | Standard academic size |
| `--toc` | Generate table of contents | Automatically from headers |
| `-N` | Number sections automatically | 1, 1.1, 1.2, etc. |
| `-o ch00.pdf` | Output file path | Relative to current directory |

### Input Files

1. **Source markdown**: Chapter content (first argument)
2. **Lua filter**: `docs/book/callouts.lua` - Converts markdown callouts to LaTeX
3. **Preamble**: `docs/book/preamble.tex` - LaTeX package imports and environment definitions

---

## Alternative Compilation Methods

### Method 1: Output to Specific Directory

```bash
# Output to docs/book/pdfs/
mkdir -p docs/book/pdfs

pandoc docs/book/ch00/ch00_motivation_first_experiment.md \
  --lua-filter=docs/book/admonitions.lua \
  --lua-filter=docs/book/callouts.lua \
  --lua-filter=docs/book/crossrefs.lua \
  --include-in-header=docs/book/preamble.tex \
  --pdf-engine=xelatex \
  --resource-path=.:docs/book:docs/book/ch00 \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  --toc \
  -N \
  -o docs/book/pdfs/ch00.pdf
```

### Method 2: Combined Multi-Chapter PDF

```bash
pandoc \
  docs/book/ch00/ch00_motivation_first_experiment.md \
  docs/book/ch01/ch01_foundations.md \
  --lua-filter=docs/book/admonitions.lua \
  --lua-filter=docs/book/callouts.lua \
  --lua-filter=docs/book/crossrefs.lua \
  --include-in-header=docs/book/preamble.tex \
  --pdf-engine=xelatex \
  --resource-path=.:docs/book:docs/book/ch00:docs/book/ch01 \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  --toc \
  -N \
  -o book_ch00-01_combined.pdf
```

### Method 3: With Additional Options

```bash
# Add metadata, custom title page, and enhanced formatting
pandoc docs/book/ch00/ch00_motivation_first_experiment.md \
  --lua-filter=docs/book/admonitions.lua \
  --lua-filter=docs/book/callouts.lua \
  --lua-filter=docs/book/crossrefs.lua \
  --include-in-header=docs/book/preamble.tex \
  --pdf-engine=xelatex \
  --resource-path=.:docs/book:docs/book/ch00 \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V documentclass=report \
  -V papersize=letter \
  -V linkcolor=blue \
  --toc \
  --toc-depth=3 \
  -N \
  --metadata title="Chapter 0: Motivation" \
  --metadata author="Vlad Prytula" \
  -o ch00_enhanced.pdf
```

---

## Verification

### Check for Compilation Warnings

```bash
# Compile and check for Unicode warnings
pandoc docs/book/ch00/ch00_motivation_first_experiment.md \
  --lua-filter=docs/book/admonitions.lua \
  --lua-filter=docs/book/callouts.lua \
  --lua-filter=docs/book/crossrefs.lua \
  --include-in-header=docs/book/preamble.tex \
  --pdf-engine=xelatex \
  --resource-path=.:docs/book:docs/book/ch00 \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  --toc \
  -N \
  -o ch00.pdf 2>&1 | grep -i "missing character"

# Empty output = success (no missing characters)
```

### Check Output File

```bash
# Verify PDF was created
ls -lh ch00.pdf ch01.pdf

# Expected output:
# -rw-r--r--  1 user  staff   109K Nov 25 12:32 ch00.pdf
# -rw-r--r--  1 user  staff   197K Nov 25 12:33 ch01.pdf
```

### Open Generated PDFs

```bash
# macOS
open ch00.pdf ch01.pdf

# Linux
xdg-open ch00.pdf
xdg-open ch01.pdf
```

---

## Troubleshooting

### Error: "callouts.lua: No such file or directory"

**Problem**: Running from wrong directory or file doesn't exist

**Solution**:
```bash
# Check current directory
pwd
# Should be: /Users/vladyslavp/src_local/rl_search_from_scratch/

# Verify filter exists
ls -l docs/book/callouts.lua

# If missing, see docs/book/callouts.lua in repository
```

### Error: "preamble.tex: No such file or directory"

**Problem**: Preamble file missing

**Solution**:
```bash
# Verify preamble exists
ls -l docs/book/preamble.tex

# If missing, see docs/book/preamble.tex in repository
```

### Error: "LaTeX Error: Environment CalloutNote undefined"

**Problem**: preamble.tex not loaded or tcolorbox not installed

**Solution**:
```bash
# Install tcolorbox package
sudo tlmgr install tcolorbox pgf environ trimspaces

# Verify preamble.tex contains CalloutNote definition
grep "CalloutNote" docs/book/preamble.tex
```

### Error: "xelatex not found"

**Problem**: LaTeX distribution not installed

**Solution**:
```bash
# macOS
brew install --cask mactex

# Verify installation
which xelatex
xelatex --version
```

### Warning: "Missing character: There is no → (U+2194)"

**Problem**: Unicode characters in source markdown (should be LaTeX)

**Solution**:
```bash
# Detect Unicode issues
./scripts/detect_unicode_issues.sh docs/book/ch00/file.md

# Fix according to docs/book/UNICODE_CLEANUP_TODO.md or build via:
./scripts/build_book_ch00_06_with_labs_and_solutions.sh
```

---

## Batch Compilation Script

Create `scripts/compile_all_chapters.sh`:

```bash
#!/bin/bash
# Compile chapters to PDF

set -e

CHAPTERS=(
    "ch00:ch00_motivation_first_experiment"
    "ch01:ch01_foundations"
)

OUTPUT_DIR="docs/book/pdfs"
mkdir -p "$OUTPUT_DIR"

for chapter_info in "${CHAPTERS[@]}"; do
    IFS=':' read -r chapter_num filename <<< "$chapter_info"

    echo "Compiling $chapter_num..."

    pandoc "docs/book/$chapter_num/$filename.md" \
        --lua-filter=docs/book/admonitions.lua \
        --lua-filter=docs/book/callouts.lua \
        --lua-filter=docs/book/crossrefs.lua \
        --include-in-header=docs/book/preamble.tex \
        --pdf-engine=xelatex \
        --resource-path=.:docs/book:docs/book/"$chapter_num" \
        -V geometry:margin=1in \
        -V fontsize=11pt \
        --toc \
        -N \
        -o "$OUTPUT_DIR/$chapter_num.pdf"

    echo "OK Generated $OUTPUT_DIR/$chapter_num.pdf"
done

echo ""
echo "All chapters compiled successfully!"
ls -lh "$OUTPUT_DIR"/*.pdf
```

**Usage**:
```bash
chmod +x scripts/compile_all_chapters.sh
./scripts/compile_all_chapters.sh
```

---

## Make-Based Build System (Optional)

Create `Makefile`:

```makefile
# Makefile for PDF generation

PANDOC = pandoc
LUA_FILTER = docs/book/callouts.lua
PREAMBLE = docs/book/preamble.tex
PANDOC_OPTS = --lua-filter=$(LUA_FILTER) \
              --include-in-header=$(PREAMBLE) \
              --pdf-engine=xelatex \
              -V geometry:margin=1in \
              -V fontsize=11pt \
              --toc \
              -N

CH00_SRC = docs/book/ch00/ch00_motivation_first_experiment.md
CH01_SRC = docs/book/ch01/ch01_foundations.md

all: ch00.pdf ch01.pdf

ch00.pdf: $(CH00_SRC) $(LUA_FILTER) $(PREAMBLE)
	$(PANDOC) $(CH00_SRC) $(PANDOC_OPTS) -o $@

ch01.pdf: $(CH01_SRC) $(LUA_FILTER) $(PREAMBLE)
	$(PANDOC) $(CH01_SRC) $(PANDOC_OPTS) -o $@

clean:
	rm -f ch00.pdf ch01.pdf

.PHONY: all clean
```

**Usage**:
```bash
# Compile all chapters
make

# Compile specific chapter
make ch00.pdf

# Clean generated files
make clean
```

---

## Quick Reference Card

### Single Chapter Compilation

```bash
# Template
pandoc docs/book/chXX/<filename>.md \
  --lua-filter=docs/book/admonitions.lua \
  --lua-filter=docs/book/callouts.lua \
  --lua-filter=docs/book/crossrefs.lua \
  --include-in-header=docs/book/preamble.tex \
  --pdf-engine=xelatex \
  --resource-path=.:docs/book:docs/book/chXX \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  --toc \
  -N \
  -o chXX.pdf

# CH00
pandoc docs/book/ch00/ch00_motivation_first_experiment.md \
  --lua-filter=docs/book/admonitions.lua \
  --lua-filter=docs/book/callouts.lua \
  --lua-filter=docs/book/crossrefs.lua \
  --include-in-header=docs/book/preamble.tex \
  --pdf-engine=xelatex \
  --resource-path=.:docs/book:docs/book/ch00 \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  --toc \
  -N \
  -o ch00.pdf

# CH01
pandoc docs/book/ch01/ch01_foundations.md \
  --lua-filter=docs/book/admonitions.lua \
  --lua-filter=docs/book/callouts.lua \
  --lua-filter=docs/book/crossrefs.lua \
  --include-in-header=docs/book/preamble.tex \
  --pdf-engine=xelatex \
  --resource-path=.:docs/book:docs/book/ch01 \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  --toc \
  -N \
  -o ch01.pdf
```

### Verify Compilation

```bash
# Check output files exist
ls -lh ch00.pdf ch01.pdf

# Check for Unicode warnings
pandoc <input> ... -o output.pdf 2>&1 | grep "missing character"
# (empty output = success)

# Open PDFs
open ch00.pdf ch01.pdf  # macOS
```

---

## Files Reference

### Required Files

**`docs/book/callouts.lua`**:
```lua
-- Pandoc Lua filter to render note-style callouts nicely in LaTeX/PDF.
-- See full file in repository
```

**`docs/book/preamble.tex`**:
```latex
% Minimal preamble (callouts/admonitions)
\usepackage{amsmath,amssymb,amsfonts}
\usepackage[most]{tcolorbox}
```

---

## Summary

### Minimal Command (CH00)
```bash
pandoc docs/book/ch00/ch00_motivation_first_experiment.md \
  --lua-filter=docs/book/admonitions.lua \
  --lua-filter=docs/book/callouts.lua \
  --lua-filter=docs/book/crossrefs.lua \
  --include-in-header=docs/book/preamble.tex \
  --pdf-engine=xelatex \
  --resource-path=.:docs/book:docs/book/ch00 \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  --toc \
  -N \
  -o ch00.pdf
```

### Minimal Command (CH01)
```bash
pandoc docs/book/ch01/ch01_foundations.md \
  --lua-filter=docs/book/admonitions.lua \
  --lua-filter=docs/book/callouts.lua \
  --lua-filter=docs/book/crossrefs.lua \
  --include-in-header=docs/book/preamble.tex \
  --pdf-engine=xelatex \
  --resource-path=.:docs/book:docs/book/ch01 \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  --toc \
  -N \
  -o ch01.pdf
```

**Prerequisites**: pandoc, xelatex, tcolorbox package
**Input files**: markdown source, callouts.lua, preamble.tex
**Output**: PDF in current directory
**Verification**: Zero "missing character" warnings
