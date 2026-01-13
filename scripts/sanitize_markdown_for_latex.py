#!/usr/bin/env python3
"""
Sanitize Markdown for LaTeX/PDF compilation.

Goal: ensure the LaTeX toolchain never sees Unicode characters that are known to
break compilation or fonts, by converting them to ASCII / LaTeX commands.

This script is intentionally conservative: it preserves the original source
files by default and writes sanitized copies to an output directory, preserving
the input relative path.

Usage examples:
  python scripts/sanitize_markdown_for_latex.py --output-dir /tmp/sanitized docs/book/ch01/ch01_foundations.md
  python scripts/sanitize_markdown_for_latex.py --check-only docs/book/ch01/ch01_foundations.md
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_CODE_FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
_INLINE_CODE_RE = re.compile(r"(`+)([^`]*?)\1")
_MKDOCS_ADMON_START_RE = re.compile(r'^(\\s*)(!!!|\\?\\?\\?)\\s+(\\w+)(?:\\s+"([^"]*)")?\\s*$')

# Unicode whitespace that should not reach LaTeX.
_UNICODE_SPACES_TO_SPACE = [
    "\u00A0",  # NO-BREAK SPACE
    "\u202F",  # NARROW NO-BREAK SPACE
    "\u2007",  # FIGURE SPACE
    "\u2009",  # THIN SPACE
    "\u200A",  # HAIR SPACE
]
_UNICODE_SPACES_TO_DELETE = [
    "\u200B",  # ZERO WIDTH SPACE
    "\u2060",  # WORD JOINER
    "\uFEFF",  # ZERO WIDTH NO-BREAK SPACE / BOM
    "\u00AD",  # SOFT HYPHEN
]


def _normalize_common_typography(text: str) -> str:
    # Dashes / hyphens
    text = text.replace("\u2014", "---")  # em dash
    text = text.replace("\u2013", "--")  # en dash
    text = text.replace("\u2011", "-")  # non-breaking hyphen
    # Quotes
    text = text.replace("\u201C", '"').replace("\u201D", '"')  # curly double
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # curly single
    # Ellipsis
    text = text.replace("\u2026", "...")
    return text


def _normalize_unicode_whitespace(text: str) -> str:
    for ch in _UNICODE_SPACES_TO_SPACE:
        text = text.replace(ch, " ")
    for ch in _UNICODE_SPACES_TO_DELETE:
        text = text.replace(ch, "")
    return text


_CODE_CHAR_MAP = {
    # Box drawing
    "─": "-",
    # Arrows
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "↓": "down",
    # Math symbols
    "×": "*",
    "÷": "/",
    "−": "-",
    "·": "*",
    "±": "+/-",
    "≈": "~=",
    "≥": ">=",
    "≤": "<=",
    "≠": "!=",
    "∞": "inf",
    "∈": "in",
    "∩": "cap",
    "∏": "prod",
    "√": "sqrt",
    "‖": "||",
    "⟨": "<",
    "⟩": ">",
    # Greek (code/log output should stay ASCII)
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "θ": "theta",
    "μ": "mu",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "φ": "phi",
    "ω": "omega",
    "Δ": "Delta",
    "Σ": "Sigma",
    "Ω": "Omega",
    "𝔼": "E",
    # Subscripts (common in copied math)
    "₀": "_0",
    "₁": "_1",
    "₂": "_2",
    "ₙ": "_n",
    "ᵢ": "_i",
    "²": "^2",
    # Combining marks (U+0302, U+0303) handled separately in code context.
    # Other text symbols
    "§": "Section",
    "€": "EUR",
    "✅": "OK",
    "❌": "X",
    "∎": "QED",
}


@dataclass
class _MathState:
    in_inline: bool = False
    in_display: bool = False


def _is_ascii(text: str) -> bool:
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _describe_non_ascii(text: str, *, limit: int = 40) -> str:
    counts: dict[str, int] = {}
    for ch in text:
        if ord(ch) > 0x7F:
            counts[ch] = counts.get(ch, 0) + 1
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    parts = []
    for ch, n in items[:limit]:
        parts.append(f"{n}x U+{ord(ch):04X} {repr(ch)}")
    more = "" if len(items) <= limit else f" (+{len(items) - limit} more)"
    return ", ".join(parts) + more


def _sanitize_code_text(text: str) -> str:
    # Handle combining marks by turning x\u0302 into x_hat, x\u0303 into x_tilde.
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\u0302":  # combining circumflex
            out.append("_hat")
            i += 1
            continue
        if ch == "\u0303":  # combining tilde
            out.append("_tilde")
            i += 1
            continue
        if ch in _CODE_CHAR_MAP:
            out.append(_CODE_CHAR_MAP[ch])
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


_GREEK_TO_LATEX = {
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\varepsilon",
    "θ": r"\theta",
    "μ": r"\mu",
    "π": r"\pi",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "φ": r"\phi",
    "ω": r"\omega",
    "Δ": r"\Delta",
    "Σ": r"\Sigma",
    "Ω": r"\Omega",
    "𝔼": r"\mathbb{E}",
}

_MATH_SYMBOL_TO_LATEX = {
    "→": r"\rightarrow",
    "←": r"\leftarrow",
    "↔": r"\leftrightarrow",
    "↓": r"\downarrow",
    "×": r"\times",
    "÷": r"\div",
    "·": r"\cdot",
    "±": r"\pm",
    "≈": r"\approx",
    "≥": r"\geq",
    "≤": r"\leq",
    "≠": r"\neq",
    "∞": r"\infty",
    "∈": r"\in",
    "∩": r"\cap",
    "∏": r"\prod",
    "⟨": r"\langle",
    "⟩": r"\rangle",
    "‖": r"\|",
    # End-of-proof symbol: handled explicitly to avoid math-mode edge cases.
}

_UNICODE_SUBSCRIPTS = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "ₙ": "n",
    "ᵢ": "i",
}


def _latex_for_subscript(token: str) -> str:
    if token.isalpha() and len(token) > 1:
        return r"\text{" + token + "}"
    return token


def _emit_math(latex: str, *, in_math: bool) -> str:
    # Always use \ensuremath so the output is valid in both text and math mode,
    # and so LaTeX commands don't accidentally merge with following letters.
    _ = in_math
    return r"\ensuremath{" + latex + "}"


def _sanitize_text_segment(seg: str, state: _MathState) -> tuple[str, _MathState]:
    out: list[str] = []
    i = 0
    while i < len(seg):
        # Toggle display math on $$ (not escaped)
        if seg.startswith("$$", i):
            state.in_display = not state.in_display
            out.append("$$")
            i += 2
            continue

        ch = seg[i]
        in_math = state.in_inline or state.in_display

        # Toggle inline math on $ (not escaped, and not $$)
        if ch == "$" and not state.in_display:
            if i > 0 and seg[i - 1] == "\\":
                out.append("$")
            else:
                state.in_inline = not state.in_inline
                out.append("$")
            i += 1
            continue

        # Section sign: always textual "Section " (with trailing space).
        if ch == "§":
            out.append("Section ")
            i += 1
            continue

        # Emojis: textual replacements.
        if ch == "✅":
            out.append("OK")
            i += 1
            continue
        if ch == "❌":
            out.append("X")
            i += 1
            continue

        # Greek letters: may need to capture subscripts like π₀ or θ_price.
        if ch in _GREEK_TO_LATEX:
            latex = _GREEK_TO_LATEX[ch]

            # Case A: Unicode subscript immediately after the greek letter.
            if i + 1 < len(seg) and seg[i + 1] in _UNICODE_SUBSCRIPTS:
                sub = _UNICODE_SUBSCRIPTS[seg[i + 1]]
                expr = f"{latex}_{sub}"
                out.append(_emit_math(expr, in_math=in_math))
                i += 2
                continue

            # Case B: ASCII underscore subscript: θ_price, μ_c, etc.
            if i + 1 < len(seg) and seg[i + 1] == "_":
                j = i + 2
                while j < len(seg) and re.match(r"[A-Za-z0-9]", seg[j]):
                    j += 1
                sub_token = seg[i + 2 : j]
                if sub_token:
                    expr = f"{latex}_{{{_latex_for_subscript(sub_token)}}}"
                    out.append(_emit_math(expr, in_math=in_math))
                    i = j
                    continue

            out.append(_emit_math(latex, in_math=in_math))
            i += 1
            continue

        # Math symbols
        if ch == "∎":
            # Robust in both text and math mode; avoids relying on our $-state.
            out.append(r"\ensuremath{\square}")
            i += 1
            continue
        if ch in _MATH_SYMBOL_TO_LATEX:
            out.append(_emit_math(_MATH_SYMBOL_TO_LATEX[ch], in_math=in_math))
            i += 1
            continue

        # Latin-1 accents: prefer plain ASCII.
        if ch == "é":
            out.append("e")
            i += 1
            continue
        if ch == "ï":
            out.append("i")
            i += 1
            continue

        # Unicode minus sign (U+2212) -> ASCII hyphen-minus.
        if ch == "−":
            out.append("-")
            i += 1
            continue

        # Superscript 2 (U+00B2): attach via inline math outside math mode.
        if ch == "²":
            out.append("^2" if in_math else "$^2$")
            i += 1
            continue

        # Combining marks should not appear in prose; drop them.
        if ch in ("\u0302", "\u0303"):
            i += 1
            continue

        out.append(ch)
        i += 1
    return "".join(out), state


def sanitize_markdown(markdown: str) -> str:
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
    markdown = markdown.lstrip("\ufeff")
    markdown = _normalize_unicode_whitespace(markdown)
    markdown = _normalize_common_typography(markdown)

    # Pandoc's tex_math_dollars has a currency heuristic: a closing '$' directly
    # followed by a digit is treated as a literal dollar sign, not math. Fix a
    # common anti-pattern in our sources like: ($\\approx$10% ...) by pulling the
    # trailing numeric payload inside the same math span.
    markdown = re.sub(
        r"\$\\approx\$\s*([+-]?\s*\d+(?:\.\d+)?)\s*%",
        lambda m: fr"$\approx {m.group(1).replace(' ', '')}\%$",
        markdown,
    )
    markdown = re.sub(
        r"\$\\approx\$\s*([+-]?\s*\d+(?:\.\d+)?)",
        lambda m: fr"$\approx {m.group(1).replace(' ', '')}$",
        markdown,
    )

    out_lines: list[str] = []
    fence_char: str | None = None
    fence_len: int = 0
    in_fence = False
    state = _MathState()

    for line in markdown.split("\n"):
        m = _CODE_FENCE_RE.match(line)
        if m:
            _, fence, _ = m.groups()
            if not in_fence:
                in_fence = True
                fence_char = fence[0]
                fence_len = len(fence)
                out_lines.append(_sanitize_code_text(line))
                continue
            if fence_char and fence[0] == fence_char and len(fence) >= fence_len:
                in_fence = False
                fence_char = None
                fence_len = 0
                out_lines.append(_sanitize_code_text(line))
                continue

        if in_fence:
            out_lines.append(_sanitize_code_text(line))
            continue

        # Keep admonition / callout titles plain (no LaTeX math) to avoid escaping
        # issues in filters (titles are "moving arguments" in LaTeX).
        m_admon = _MKDOCS_ADMON_START_RE.match(line)
        if m_admon and m_admon.group(4) is not None:
            indent, marker, kind, title = m_admon.groups()
            title_s = _sanitize_code_text(title)
            # Also normalize common LaTeX arrow macros in titles.
            title_s = (
                title_s.replace(r"\leftrightarrow", "<->")
                .replace(r"\rightarrow", "->")
                .replace(r"\leftarrow", "<-")
                .replace(r"\to", "->")
            )
            title_s = title_s.replace("$", "")
            line = f'{indent}{marker} {kind} "{title_s}"'

        if 'title="' in line:
            def _title_sub(m: re.Match[str]) -> str:
                raw_title = m.group(2)
                title_s = _sanitize_code_text(raw_title)
                title_s = (
                    title_s.replace(r"\leftrightarrow", "<->")
                    .replace(r"\rightarrow", "->")
                    .replace(r"\leftarrow", "<-")
                    .replace(r"\to", "->")
                )
                title_s = title_s.replace("$", "")
                return m.group(1) + title_s + m.group(3)

            line = re.sub(r'(title=")([^"]*)(")', _title_sub, line)

        # Outside fenced code blocks: sanitize inline code spans separately.
        parts: list[str] = []
        last = 0
        for mm in _INLINE_CODE_RE.finditer(line):
            before = line[last : mm.start()]
            before_s, state = _sanitize_text_segment(before, state)
            parts.append(before_s)

            delim = mm.group(1)
            code = mm.group(2)
            parts.append(delim + _sanitize_code_text(code) + delim)
            last = mm.end()
        tail = line[last:]
        tail_s, state = _sanitize_text_segment(tail, state)
        parts.append(tail_s)

        out_lines.append("".join(parts))

    sanitized = "\n".join(out_lines)
    if not _is_ascii(sanitized):
        raise ValueError("Sanitization incomplete: " + _describe_non_ascii(sanitized))
    return sanitized


def _iter_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.md")))
        else:
            files.append(p)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Markdown files or directories")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write sanitized copies (preserves input relative path)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Fail if input contains any non-ASCII characters (does not sanitize)",
    )
    args = parser.parse_args()

    files = _iter_files(args.paths)
    if not files:
        print("No markdown files found.", file=sys.stderr)
        return 2

    if args.check_only:
        bad = []
        for f in files:
            text = f.read_text(encoding="utf-8")
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            text = _normalize_unicode_whitespace(text)
            text = _normalize_common_typography(text)
            if not _is_ascii(text):
                bad.append((f, _describe_non_ascii(text)))
        if bad:
            for f, desc in bad:
                print(f"[non-ascii] {f}: {desc}", file=sys.stderr)
            return 1
        return 0

    if args.output_dir is None:
        print("--output-dir is required unless --check-only is used.", file=sys.stderr)
        return 2

    out_dir: Path = args.output_dir
    repo_root = Path.cwd().resolve()

    for f in files:
        raw = f.read_text(encoding="utf-8")
        sanitized = sanitize_markdown(raw)
        try:
            rel = f.resolve().relative_to(repo_root)
        except ValueError:
            # If invoked outside repo root, fall back to basename-only.
            rel = Path(f.name)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(sanitized, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
