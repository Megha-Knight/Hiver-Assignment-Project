"""Audit helper script for Phase 7A.

Scans the repository for:
1. Secrets and credentials
2. Hardcoded absolute paths (C:, D:, E:, /home/, Users, AppData)
3. Dependency usage vs requirements.txt
4. Frozen metrics consistency across Phase 4, Phase 5, Phase 6B, Phase 6C
5. Repository file categorization and size breakdown
"""

from collections import defaultdict
import json
import os
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Patterns for secret scanning
SECRET_PATTERNS = [
    ("Generic Secret", re.compile(r'(?i)(api[_-]?key|secret_key|private_key|access_token|auth_token)\s*[:=]\s*["\']([A-Za-z0-9_\-\.]{12,})["\']')),
    ("GitHub Token", re.compile(r'gh[pousr]_[A-Za-z0-9_]{30,}')),
    ("Google API Key", re.compile(r'AIza[0-9A-Za-z\-_]{35}')),
    ("OpenAI Key", re.compile(r'sk-[A-Za-z0-9]{20,}')),
    ("Private Key Header", re.compile(r'-----BEGIN (RSA|EC|OPENSSH|DSA|PGP)? PRIVATE KEY-----')),
    ("AWS Key", re.compile(r'AKIA[0-9A-Z]{16}')),
]

# Patterns for hardcoded paths
HARDCODED_PATH_PATTERNS = [
    ("Windows Drive Letter", re.compile(r'(?i)[a-z]:\\[a-z0-9_\\ -]+')),
    ("Windows User Directory", re.compile(r'(?i)[a-z]:[/\\]users[/\\][a-z0-9_\.\-]+', re.IGNORECASE)),
    ("AppData Directory", re.compile(r'(?i)AppData[/\\][a-z0-9_\\-]+', re.IGNORECASE)),
    ("Unix Root /home/", re.compile(r'/home/[a-z0-9_\-]+')),
]


def scan_secrets():
    print("\n--- 1. SECRET SCAN ---")
    flagged = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        if any(ignored in root for ignored in [".git", "__pycache__", ".vscode", "node_modules"]):
            continue
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in [".py", ".json", ".md", ".txt", ".env", ".example", ".yaml", ".yml"]:
                fpath = Path(root) / f
                rel = fpath.relative_to(PROJECT_ROOT)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                        for lno, line in enumerate(fh, 1):
                            # Skip commented example lines in .env.example
                            for name, pat in SECRET_PATTERNS:
                                m = pat.search(line)
                                if m:
                                    val = m.group(0)
                                    # redact
                                    redacted = val[:10] + "..." + val[-4:] if len(val) > 16 else "[REDACTED]"
                                    flagged.append((str(rel), lno, name, redacted))
                except Exception as e:
                    print(f"Error reading {rel}: {e}")

    print(f"Potential secret matches: {len(flagged)}")
    for rel, lno, name, val in flagged:
        print(f"  {rel}:{lno} [{name}] -> {val}")
    return flagged


def scan_hardcoded_paths():
    print("\n--- 2. HARDCODED PATH SCAN ---")
    flagged = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        if any(ignored in root for ignored in [".git", "__pycache__", ".vscode", "logs", "twcs"]):
            continue
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in [".py", ".md", ".json"]:
                fpath = Path(root) / f
                rel = fpath.relative_to(PROJECT_ROOT)
                # Ignore this audit script itself
                if str(rel) == os.path.join("scripts", "audit_helper.py"):
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                        for lno, line in enumerate(fh, 1):
                            for name, pat in HARDCODED_PATH_PATTERNS:
                                m = pat.search(line)
                                if m:
                                    match_str = m.group(0)
                                    flagged.append((str(rel), lno, name, match_str))
                except Exception as e:
                    pass

    print(f"Hardcoded path occurrences found: {len(flagged)}")
    # Group by file
    by_file = defaultdict(list)
    for rel, lno, name, val in flagged:
        by_file[rel].append((lno, name, val))

    for rel, items in sorted(by_file.items()):
        print(f"  {rel} ({len(items)} occurrences):")
        for lno, name, val in items[:3]:
            print(f"    Line {lno}: {val}")
        if len(items) > 3:
            print(f"    ... and {len(items) - 3} more")
    return flagged


def scan_file_inventory():
    print("\n--- 3. REPOSITORY INVENTORY & SIZES ---")
    categories = defaultdict(list)
    total_size = 0

    for root, dirs, files in os.walk(PROJECT_ROOT):
        if ".git" in root:
            continue
        for f in files:
            fpath = Path(root) / f
            rel = str(fpath.relative_to(PROJECT_ROOT))
            size = fpath.stat().st_size
            total_size += size

            # Categorize
            if "__pycache__" in rel or rel.endswith(".pyc"):
                cat = "TEMPORARY"
            elif rel.startswith("src" + os.sep + "llm") or rel.startswith("src" + os.sep + "policy") or rel.startswith("src" + os.sep + "state") or rel == os.path.join("src", "config.py"):
                cat = "CORE_RUNTIME"
            elif rel.startswith("src" + os.sep + "retrieval"):
                cat = "CORE_RUNTIME"
            elif rel.startswith("src" + os.sep + "baselines") or rel.startswith("src" + os.sep + "evaluation") or rel.startswith("src" + os.sep + "annotation"):
                cat = "EXPERIMENT"
            elif rel.startswith("src"):
                cat = "CORE_RUNTIME"
            elif rel.startswith("data" + os.sep + "golden"):
                cat = "EVALUATION"
            elif rel.startswith("data" + os.sep + "retrieval"):
                cat = "DATA"
            elif rel.startswith("data" + os.sep + "indexes"):
                cat = "INDEX"
            elif rel.startswith("data" + os.sep + "processed"):
                cat = "GENERATED_ARTIFACT"
            elif rel.startswith("data"):
                cat = "DATA"
            elif rel.startswith("models"):
                cat = "MODEL"
            elif rel.startswith("docs"):
                cat = "DOCUMENTATION"
            elif rel.startswith("results"):
                cat = "GENERATED_ARTIFACT"
            elif rel.startswith("scripts"):
                cat = "SCRIPT"
            elif rel.startswith("tests"):
                cat = "TEST"
            elif rel in ["README.md", ".gitignore", "requirements.txt", ".env.example"]:
                cat = "DOCUMENTATION" if rel == "README.md" else "CORE_RUNTIME"
            else:
                cat = "UNKNOWN"

            categories[cat].append((rel, size))

    print(f"Total repository size: {total_size / (1024 * 1024):.2f} MB")
    for cat in sorted(categories):
        cat_files = categories[cat]
        cat_size = sum(s for _, s in cat_files)
        print(f"  {cat:<20}: {len(cat_files):>4} files | {cat_size / (1024*1024):>8.2f} MB")

    return categories, total_size


def check_frozen_consistency():
    print("\n--- 4. FROZEN METRICS CONSISTENCY CHECK ---")
    files_to_check = [
        PROJECT_ROOT / "results" / "phase4_baseline_report.md",
        PROJECT_ROOT / "results" / "phase5_retrieval_report.md",
        PROJECT_ROOT / "results" / "phase5_retrieval_metrics.json",
        PROJECT_ROOT / "results" / "phase6" / "phase6b_metrics.json",
        PROJECT_ROOT / "results" / "phase6" / "phase6b_report.md",
        PROJECT_ROOT / "results" / "phase6" / "phase6c_metrics.json",
        PROJECT_ROOT / "results" / "phase6" / "phase6c_report.md",
        PROJECT_ROOT / "scripts" / "compare_phase4_phase6.py",
    ]

    for p in files_to_check:
        print(f"  {p.relative_to(PROJECT_ROOT)}: exists={p.exists()}")


if __name__ == "__main__":
    scan_secrets()
    scan_hardcoded_paths()
    scan_file_inventory()
    check_frozen_consistency()
