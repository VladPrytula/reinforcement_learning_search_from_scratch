#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PANDOC_BIN="${PANDOC:-pandoc}"
PDF_ENGINE_BIN="${PDF_ENGINE:-xelatex}"
PY_BIN="${PYTHON:-python}"

OUT_PDF="${1:-book_ch00-06_with_labs_and_solutions.pdf}"
OUT_LOG="${OUT_PDF%.pdf}.log"

SOURCES=(
  docs/book/ch00/ch00_motivation_first_experiment.md
  docs/book/ch00/exercises_labs.md
  docs/book/ch00/ch00_lab_solutions.md
  docs/book/ch01/ch01_foundations.md
  docs/book/ch01/exercises_labs.md
  docs/book/ch01/ch01_lab_solutions.md
  docs/book/ch02/ch02_probability_measure_click_models.md
  docs/book/ch02/exercises_labs.md
  docs/book/ch02/ch02_lab_solutions.md
  docs/book/ch03/ch03_stochastic_processes_bellman_foundations.md
  docs/book/ch03/exercises_labs.md
  docs/book/ch03/ch03_lab_solutions.md
  docs/book/ch04/ch04_generative_world_design.md
  docs/book/ch04/exercises_labs.md
  docs/book/ch05/ch05_relevance_features_reward.md
  docs/book/ch05/exercises_labs.md
  docs/book/ch06/discrete_template_bandits.md
  docs/book/ch06/exercises_labs.md
  docs/book/ch06/ch06_lab_solutions.md
  docs/book/ch06/ch06_advanced_gpu_lab.md
)

for f in "${SOURCES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing source file: $f" >&2
    exit 2
  fi
done

TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

echo "[sanitize] Writing sanitized markdown to: $TMP_DIR" >&2
"$PY_BIN" scripts/sanitize_markdown_for_latex.py --output-dir "$TMP_DIR" "${SOURCES[@]}"

SANITIZED_SOURCES=()
for f in "${SOURCES[@]}"; do
  SANITIZED_SOURCES+=("$TMP_DIR/$f")
done

RESOURCE_PATH=".:docs/book"
for ch in ch00 ch01 ch02 ch03 ch04 ch05 ch06; do
  RESOURCE_PATH="$RESOURCE_PATH:docs/book/$ch"
done

echo "[pandoc] Building: $OUT_PDF" >&2
"$PANDOC_BIN" \
  "${SANITIZED_SOURCES[@]}" \
  --from=markdown+tex_math_dollars+tex_math_single_backslash+raw_html+fenced_divs \
  --lua-filter=docs/book/admonitions.lua \
  --lua-filter=docs/book/callouts.lua \
  --lua-filter=docs/book/crossrefs.lua \
  --include-in-header=docs/book/preamble.tex \
  --pdf-engine="$PDF_ENGINE_BIN" \
  --resource-path="$RESOURCE_PATH" \
  --syntax-highlighting=tango \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  --toc --toc-depth=2 \
  --metadata title="Reinforcement Learning for Search from Scratch (Chapters 0--6)" \
  --metadata author="Vlad Prytula" \
  -o "$OUT_PDF" \
  2>&1 | tee "$OUT_LOG"

if rg -n --no-filename -i "missing character|invalid input character|Unicode" "$OUT_LOG" >/dev/null; then
  echo "" >&2
  echo "[error] LaTeX log contains missing/Unicode character warnings. See: $OUT_LOG" >&2
  exit 1
fi

echo "[done] $OUT_PDF" >&2
echo "[done] $OUT_LOG" >&2

