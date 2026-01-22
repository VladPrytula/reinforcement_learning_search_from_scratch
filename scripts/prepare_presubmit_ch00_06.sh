#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[1/5] Build combined Ch00–Ch06 markdown (docs/book/compiled)..." >&2
./scripts/build_book_ch00_06_with_labs_and_solutions.sh --format md

echo "[2/5] Build per-chapter markdown bundles (docs/book/compiled/chapters_ch00-06)..." >&2
python scripts/build_ch00_06_chapter_bundles.py --out-dir docs/book/compiled/chapters_ch00-06

echo "[3/5] Sync manuscript into pre-submit/chapters/ ..." >&2
mkdir -p pre-submit/chapters/docs/book

cp -f \
  docs/book/compiled/book_ch00-06_with_labs_and_solutions.md \
  pre-submit/chapters/book_ch00-06_with_labs_and_solutions.md

cp -f docs/book/compiled/chapters_ch00-06/*.md pre-submit/chapters/

for ch in ch00 ch01 ch02 ch03 ch04 ch05 ch06; do
  rsync -a "docs/book/$ch/" "pre-submit/chapters/docs/book/$ch/"
done

echo "[4/5] Refresh standalone code snapshot (pre-submit/code/chapters0-6)..." >&2
CODE_DST="pre-submit/code/chapters0-6"

rm -rf "$CODE_DST"
mkdir -p "$CODE_DST"

# Top-level files
rsync -a LICENSE README.md pyproject.toml uv.lock main.py "$CODE_DST/"

# Simulator package
rsync -a --exclude='__pycache__/' zoosim/ "$CODE_DST/zoosim/"

# Scripts (Ch00–Ch06 only + verification utilities)
mkdir -p "$CODE_DST/scripts"
rsync -a --exclude='__pycache__/' \
  scripts/__init__.py \
  scripts/ch00 scripts/ch01 scripts/ch02 scripts/ch03 scripts/ch04 scripts/ch05 scripts/ch06 \
  scripts/validate_ch05.py scripts/verify_ch00_ch06.py \
  "$CODE_DST/scripts/"

# Tests (Ch00–Ch06 only + shared integration tests)
mkdir -p "$CODE_DST/tests"
rsync -a --exclude='__pycache__/' \
  tests/__init__.py \
  tests/ch00 tests/ch01 tests/ch02 tests/ch03 tests/ch05 tests/ch06 \
  tests/test_catalog_stats.py tests/test_env_basic.py tests/test_template_bandits_rich_est.py \
  "$CODE_DST/tests/"

# Verification artifacts (kept small; useful for reviewers)
rsync -a verification_runs/ "$CODE_DST/verification_runs/"

# Strip caches if they exist
find "$CODE_DST" -type d -name '__pycache__' -prune -exec rm -rf {} + || true
rm -rf "$CODE_DST/.pytest_cache" || true

echo "[5/5] Done. pre-submit/ is ready to zip." >&2
