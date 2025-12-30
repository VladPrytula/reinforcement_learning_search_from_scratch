#!/usr/bin/env python3
"""
Markdown to LaTeX Pre-processor

Fixes patterns that break pandoc parsing before conversion to LaTeX.
This script NEVER modifies source files - it reads input and writes to a
separate output file.

Usage:
    python scripts/md2tex_preprocess.py input.md -o output.md
    python scripts/md2tex_preprocess.py input.md  # outputs to stdout

Transforms applied:
    1. Blockquote display math: "> $\n> content\n> $" → proper display math
    2. Unicode em-dash (—) → LaTeX triple hyphen (---) in text
    3. Unicode en-dash (–) → LaTeX double hyphen (--) in text
"""

import argparse
import re
import sys
from pathlib import Path


def fix_blockquote_display_math(content: str) -> str:
    r"""
    Fix display math inside blockquotes by extracting it outside.

    Pandoc doesn't handle display math inside blockquotes well.
    This extracts the math to be outside the blockquote structure.

    Converts:
        > Some text
        > $
        > \max_{\mathbf w} ...
        > $
        > More text

    To:
        > Some text

        $$
        \max_{\mathbf w} ...
        $$

        > More text

    The pattern `> $` on its own line causes pandoc to misparse as YAML.
    """
    lines = content.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check for blockquote line with just $ (start of display math)
        if re.match(r'^>\s*\$\s*$', line):
            # Found start of display math in blockquote
            # Collect math content lines
            math_lines = []
            i += 1

            while i < len(lines):
                next_line = lines[i]

                # Check for closing $ in blockquote
                if re.match(r'^>\s*\$\s*$', next_line):
                    # End of math block
                    i += 1
                    break
                elif next_line.startswith('>'):
                    # Strip the > prefix and add to math content
                    # Remove leading > and optional space
                    stripped = re.sub(r'^>\s?', '', next_line)
                    math_lines.append(stripped)
                    i += 1
                else:
                    # Unexpected - not a blockquote line
                    break

            # Emit the math block outside blockquote
            result.append('')  # blank line before
            result.append('$$')
            result.extend(math_lines)
            result.append('$$')
            result.append('')  # blank line after
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)


def normalize_dashes(content: str) -> str:
    """
    Convert Unicode dashes to LaTeX-friendly ASCII equivalents.

    Only converts dashes that are NOT inside math mode ($...$) or ($$...$$).
    This preserves mathematical minus signs.

    Converts:
        — (U+2014 em-dash) → ---
        – (U+2013 en-dash) → --
    """
    result = []
    in_inline_math = False
    in_display_math = False
    i = 0

    while i < len(content):
        # Check for display math start/end ($$)
        if content[i:i+2] == '$$':
            in_display_math = not in_display_math
            result.append('$$')
            i += 2
            continue

        # Check for inline math start/end ($) - but not $$
        if content[i] == '$' and not in_display_math:
            # Make sure it's not an escaped dollar sign
            if i > 0 and content[i-1] == '\\':
                result.append('$')
                i += 1
                continue
            in_inline_math = not in_inline_math
            result.append('$')
            i += 1
            continue

        # Convert dashes only outside math mode
        if not in_inline_math and not in_display_math:
            if content[i] == '—':  # em-dash
                result.append('---')
                i += 1
                continue
            elif content[i] == '–':  # en-dash
                result.append('--')
                i += 1
                continue

        result.append(content[i])
        i += 1

    return ''.join(result)


def fix_non_breaking_hyphens(content: str) -> str:
    """
    Convert non-breaking hyphens to regular hyphens.

    U+2011 (non-breaking hyphen) → regular hyphen
    """
    return content.replace('‑', '-')


def normalize_typography(content: str) -> str:
    """
    Convert Unicode typography characters to ASCII/LaTeX equivalents.

    This ensures clean LaTeX output without requiring newunicodechar mappings.

    Converts:
        … (U+2026 ellipsis) → ...
        ' (U+2018 left single quote) → '
        ' (U+2019 right single quote) → '
        " (U+201C left double quote) → "
        " (U+201D right double quote) → "
    """
    replacements = [
        ('…', '...'),    # Ellipsis
        (''', "'"),      # Left single quote
        (''', "'"),      # Right single quote
        ('"', '"'),      # Left double quote
        ('"', '"'),      # Right double quote
    ]
    for old, new in replacements:
        content = content.replace(old, new)
    return content


def fix_standalone_anchors(content: str) -> str:
    r"""
    Convert standalone anchors to Pandoc-compatible spans.

    Pandoc only recognizes {#id} when attached to elements (headers, spans, etc.).
    A standalone {#EQ-1.1} is treated as literal text.

    Converts:
        {#EQ-1.1}   → []{#EQ-1.1}
        {#DEF-1.2}  → []{#DEF-1.2}

    The empty brackets create a span that Pandoc recognizes.
    """
    # Match standalone anchors: {#PREFIX-X.Y} or {#PREFIX-X.Y.Z} etc.
    # Prefixes: EQ, DEF, THM, REM, ASM, WDEF
    pattern = r'\{#([A-Z]+-[\w.-]+)\}'

    def replace_anchor(match):
        anchor_id = match.group(1)
        return f'[]{{#{anchor_id}}}'

    return re.sub(pattern, replace_anchor, content)


def convert_admonitions(content: str) -> str:
    """
    Convert MkDocs-style admonitions to Pandoc fenced divs.

    Converts:
        !!! tip "Title"
            - indented content
            - more content

    To:
        ::: {.tip title="Title"}
        - content
        - more content
        :::

    Supported types: tip, note, warning, danger, info, example, important,
                     caution, abstract, summary, question, quote
    """
    lines = content.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Match admonition start: !!! type "title" or !!! type 'title' or !!! type
        admon_match = re.match(
            r'^!!!\s+(\w+)(?:\s+["\']([^"\']*)["\'])?\s*$',
            line
        )

        if admon_match:
            admon_type = admon_match.group(1).lower()
            title = admon_match.group(2) or admon_type.capitalize()

            # Emit opening fenced div
            result.append(f'::: {{.{admon_type} title="{title}"}}')

            # Collect indented content
            i += 1
            while i < len(lines):
                next_line = lines[i]

                # Check if line is indented (part of admonition content)
                if next_line.startswith('    ') or next_line.startswith('\t'):
                    # Remove one level of indentation (4 spaces or 1 tab)
                    if next_line.startswith('    '):
                        result.append(next_line[4:])
                    else:
                        result.append(next_line[1:])
                    i += 1
                elif next_line.strip() == '':
                    # Empty line - could be within admonition or end of it
                    # Look ahead to see if next non-empty line is indented
                    j = i + 1
                    while j < len(lines) and lines[j].strip() == '':
                        j += 1

                    if j < len(lines) and (lines[j].startswith('    ') or lines[j].startswith('\t')):
                        # Still in admonition, keep the empty line
                        result.append('')
                        i += 1
                    else:
                        # End of admonition
                        break
                else:
                    # Non-indented, non-empty line - end of admonition
                    break

            # Emit closing fenced div
            result.append(':::')
            result.append('')  # Add blank line after
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)


def preprocess(content: str) -> str:
    """
    Apply all preprocessing transforms in order.

    Order matters:
    1. Convert admonitions first (structural, before other transforms)
    2. Fix standalone anchors ({#EQ-1.1} → []{#EQ-1.1})
    3. Fix blockquote math (structural fix)
    4. Normalize dashes (character replacement)
    5. Fix non-breaking hyphens
    6. Normalize typography (quotes, ellipsis)
    """
    content = convert_admonitions(content)
    content = fix_standalone_anchors(content)
    content = fix_blockquote_display_math(content)
    content = normalize_dashes(content)
    content = fix_non_breaking_hyphens(content)
    content = normalize_typography(content)
    return content


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess Markdown for LaTeX conversion via Pandoc',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'input',
        type=Path,
        help='Input Markdown file'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Output file (default: stdout)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without writing'
    )

    args = parser.parse_args()

    # Read input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    content = args.input.read_text(encoding='utf-8')

    # Process
    processed = preprocess(content)

    # Output
    if args.dry_run:
        # Show diff-like output
        if content != processed:
            print("Changes would be made:")
            # Simple change detection
            orig_lines = content.splitlines()
            proc_lines = processed.splitlines()
            for i, (orig, proc) in enumerate(zip(orig_lines, proc_lines), 1):
                if orig != proc:
                    print(f"  Line {i}:")
                    print(f"    - {orig[:80]}...")
                    print(f"    + {proc[:80]}...")
        else:
            print("No changes needed.")
    elif args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(processed, encoding='utf-8')
        print(f"Preprocessed: {args.input} → {args.output}", file=sys.stderr)
    else:
        # Output to stdout
        print(processed)


if __name__ == '__main__':
    main()
