from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MARKDOWN_FILES = [
    # MkDocs nav (Part I)
    Path("docs/book/ch00/ch00_motivation_first_experiment.md"),
    Path("docs/book/ch00/exercises_labs.md"),
    Path("docs/book/ch00/ch00_lab_solutions.md"),
    Path("docs/book/ch01/ch01_foundations.md"),
    Path("docs/book/ch01/exercises_labs.md"),
    Path("docs/book/ch01/ch01_lab_solutions.md"),
    Path("docs/book/ch02/ch02_probability_measure_click_models.md"),
    Path("docs/book/ch02/exercises_labs.md"),
    Path("docs/book/ch02/ch02_lab_solutions.md"),
    Path("docs/book/ch03/ch03_stochastic_processes_bellman_foundations.md"),
    Path("docs/book/ch03/exercises_labs.md"),
    Path("docs/book/ch03/ch03_lab_solutions.md"),
]

BIB_PATH = Path("docs/references.bib")

SECOND_PERSON_RE = re.compile(r"\b(you|your|you're|you'll)\b", flags=re.IGNORECASE)
FIRST_PERSON_CONTRACTIONS_RE = re.compile(r"\b(I'm|I've|I'd|I'll)\b")
CITE_KEY_RE = re.compile(r"\[@([^\];, ]+)")
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
HTML_IMAGE_RE = re.compile(r"<img[^>]*src=[\"']([^\"']+)[\"'][^>]*>", flags=re.IGNORECASE)
CODE_SPAN_RE = re.compile(r"`([^`]+)`")
PATH_WITH_OPTIONAL_LINE_RE = re.compile(
    r"(?P<path>(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.+-]+\.[A-Za-z0-9]+)"
    r"(?:(?::|#L?)(?P<line>\d+))?"
)

ALLOWED_REFERENCE_EXTENSIONS = {
    ".py",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".yml",
    ".yaml",
    ".toml",
    ".json",
    ".tex",
    ".pdf",
}


@dataclass(frozen=True)
class Issue:
    kind: str
    file: Path
    line: int
    message: str


def _is_url(path: str) -> bool:
    return path.startswith(("http://", "https://"))


def _strip_link_extras(link: str) -> str:
    link = link.strip()
    if link.startswith("<") and link.endswith(">"):
        link = link[1:-1].strip()
    link = link.split("#", 1)[0]
    link = link.split("?", 1)[0]
    return link.strip()


def _is_probably_text(path: Path) -> bool:
    return path.suffix.lower() in {".py", ".md", ".yml", ".yaml", ".toml", ".txt", ".json"}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resolve_reference_path(path_str: str, *, base_dir: Path) -> Path | None:
    path_str = path_str.strip()
    if not path_str or _is_url(path_str):
        return None

    if path_str.startswith("/"):
        path_str = path_str[1:]

    ext = Path(path_str).suffix.lower()
    if ext not in ALLOWED_REFERENCE_EXTENSIONS:
        return None

    # 1) Repo-root relative
    candidate = (REPO_ROOT / path_str).resolve()
    if candidate.exists():
        return candidate

    # 2) File-local relative
    local = (base_dir / path_str).resolve()
    if local.exists():
        return local

    # 3) Bare filename: try unique match in repo (best-effort, small scope).
    if "/" not in path_str:
        matches = list(REPO_ROOT.rglob(path_str))
        if len(matches) == 1:
            return matches[0].resolve()

    return None


def audit_files(markdown_files: list[Path], *, bib_path: Path) -> list[Issue]:
    issues: list[Issue] = []

    bib_abs = (REPO_ROOT / bib_path).resolve()
    bib_text = _read_text(bib_abs) if bib_abs.exists() else ""

    all_cite_keys: set[str] = set()

    for rel in markdown_files:
        abs_path = (REPO_ROOT / rel).resolve()
        if not abs_path.exists():
            issues.append(Issue("missing-file", rel, 1, "file not found"))
            continue

        text = _read_text(abs_path)
        lines = text.splitlines()

        # 1) Voice scan
        in_fence = False
        fence_delim: str | None = None
        for i, line in enumerate(lines, start=1):
            stripped = line.lstrip()
            if stripped.startswith(("```", "~~~")):
                delim = stripped[:3]
                if not in_fence:
                    in_fence = True
                    fence_delim = delim
                elif fence_delim == delim:
                    in_fence = False
                    fence_delim = None
                continue
            if in_fence:
                continue

            line_no_code = CODE_SPAN_RE.sub("", line)
            if SECOND_PERSON_RE.search(line_no_code) or FIRST_PERSON_CONTRACTIONS_RE.search(line_no_code):
                issues.append(Issue("voice", rel, i, "second-person or first-person singular detected"))
                continue

            # Catch bare "I " in prose while avoiding Roman numerals ("Part I:", "I.").
            if re.search(r"(^|\s)I\s", line_no_code):
                issues.append(Issue("voice", rel, i, "second-person or first-person singular detected"))

        # 2) Citation keys
        for key in CITE_KEY_RE.findall(text):
            all_cite_keys.add(key)

        # 3) Embedded artifacts (images)
        image_links: list[str] = []
        image_links.extend(MD_IMAGE_RE.findall(text))
        image_links.extend(HTML_IMAGE_RE.findall(text))
        for raw_link in image_links:
            link = _strip_link_extras(raw_link)
            if not link or _is_url(link):
                continue
            if link.startswith("/"):
                link = link[1:]
            artifact_abs = (abs_path.parent / link).resolve()
            if not artifact_abs.exists():
                issues.append(Issue("artifact", rel, 1, f"missing image: {link}"))

        # 4) Doc ↔ code references: validate backticked paths exist; optionally validate :line.
        for i, line in enumerate(lines, start=1):
            for span in CODE_SPAN_RE.findall(line):
                for m in PATH_WITH_OPTIONAL_LINE_RE.finditer(span):
                    path_str = m.group("path")
                    candidate_abs = _resolve_reference_path(path_str, base_dir=abs_path.parent)
                    if candidate_abs is None:
                        continue
                    if not candidate_abs.exists():
                        issues.append(Issue("ref", rel, i, f"missing path: {path_str}"))
                        continue

                    line_str = m.group("line")
                    if line_str is None:
                        continue
                    if not _is_probably_text(candidate_abs):
                        continue

                    try:
                        n_lines = len(_read_text(candidate_abs).splitlines())
                    except UnicodeDecodeError:
                        continue

                    target = int(line_str)
                    if target < 1 or target > n_lines:
                        issues.append(
                            Issue(
                                "ref",
                                rel,
                                i,
                                f"line out of range: {path_str}:{target} (file has {n_lines} lines)",
                            )
                        )

    # Citation-key audit against bibliography
    if not bib_text:
        issues.append(Issue("bib", bib_path, 1, "missing bibliography file"))
    else:
        for key in sorted(all_cite_keys):
            if f"{{{key}," not in bib_text:
                issues.append(Issue("bib", bib_path, 1, f"missing citation key: {key}"))

    return issues


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Audit Ch00–Ch03 alignment (voice, artifacts, refs, citations).")
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Markdown files to audit (defaults to MkDocs Part I list).",
    )
    parser.add_argument("--bib", default=str(BIB_PATH), help="Path to bibliography file.")
    args = parser.parse_args(argv)

    md_files = [Path(p) for p in (args.files if args.files is not None else DEFAULT_MARKDOWN_FILES)]
    bib_path = Path(args.bib)

    issues = audit_files(md_files, bib_path=bib_path)
    if not issues:
        print(f"OK: audited {len(md_files)} markdown files.")
        return 0

    print(f"FAIL: {len(issues)} issue(s) found.")
    for issue in issues:
        print(f"- [{issue.kind}] {issue.file.as_posix()}:{issue.line}: {issue.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
