#!/bin/bash
# Detect Unicode characters that should be LaTeX in markdown files
# Usage: ./scripts/detect_unicode_issues.sh docs/book/final/ch01/file.md

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <markdown-file>"
    echo "Example: $0 docs/book/final/ch01/ch01_foundations.md"
    exit 1
fi

FILE="$1"

if [ ! -f "$FILE" ]; then
    echo "Error: File not found: $FILE"
    exit 1
fi

echo "======================================"
echo "Unicode Detection Report"
echo "======================================"
echo "File: $FILE"
echo ""

# Count arrows and comparisons
ARROWS_COUNT=$(grep -o "[→←↔≥≤≈]" "$FILE" | wc -l | xargs)
echo "Arrows & Comparisons (→,←,↔,≥,≤,≈): $ARROWS_COUNT occurrences"

# Count Greek letters
GREEK_COUNT=$(grep -o "[πεαβγδθλμσ]" "$FILE" | wc -l | xargs)
echo "Greek Letters (π,ε,α,β,γ,δ,θ,λ,μ,σ): $GREEK_COUNT occurrences"

TOTAL=$((ARROWS_COUNT + GREEK_COUNT))
echo ""
echo "Total potentially problematic: $TOTAL occurrences"
echo ""

if [ $TOTAL -eq 0 ]; then
    echo "✅ No Unicode issues detected!"
    exit 0
fi

echo "--------------------------------------"
echo "First 20 examples:"
echo "--------------------------------------"
grep -n "[→←↔≥≤≈πεαβγδθλμσ]" "$FILE" | head -20

echo ""
echo "--------------------------------------"
echo "Context breakdown:"
echo "--------------------------------------"

# Count in callout titles
CALLOUT_COUNT=$(grep -c "title=\".*[→←↔≥≤≈πεαβγδθλμσ].*\"" "$FILE" || true)
echo "In callout titles: $CALLOUT_COUNT"

# Estimate in prose (rough heuristic: not in code blocks)
PROSE_COUNT=$(grep "[→←↔≥≤≈πεαβγδθλμσ]" "$FILE" | grep -v "^    " | grep -v "^\t" | wc -l | xargs)
echo "In prose/headings (estimated): $PROSE_COUNT"

echo ""
echo "--------------------------------------"
echo "Next steps:"
echo "--------------------------------------"
echo "1. Review examples above"
echo "2. Use Read tool to examine context"
echo "3. Apply Edit tool for systematic replacements"
echo "4. See docs/book/UNICODE_LATEX_CLEANUP_GUIDE.md for details"
echo ""
