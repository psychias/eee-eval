"""
preflight_check.py — run this BEFORE the batch pipeline to catch all
setup issues in one shot instead of discovering them mid-extraction.

Usage:
    python scripts/preflight_check.py

Exit code 0 = all checks passed, safe to run the pipeline.
Exit code 1 = one or more blocking issues found.
"""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# ── ANSI colours (work on all platforms with modern terminals) ────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg: str)   -> None: print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg: str) -> None: print(f"  {YELLOW}⚠{RESET}  {msg}")
def fail(msg: str) -> None: print(f"  {RED}✗{RESET}  {msg}")


ARXIV_IDS = [
    "2407.21783",
    "2310.06825v1",
    "2505.09388v1",
]

REQUIRED_PACKAGES = [
    # Core pipeline — Docling (PDF + table), no pdfplumber/pypdf
    ("docling",         "docling"),
    # huggingface_hub: Docling needs this to load docling-layout-heron model
    ("huggingface_hub", "huggingface_hub"),
    # LLM calls via OpenRouter (score extraction + protocol detection)
    ("openai",          "openai"),
    # arXiv PDF download + HF Hub author lookup
    ("requests",        "requests"),
    # Docling uses pandas for table.export_to_dataframe()
    ("pandas",          "pandas"),
    # Output schema validation
    ("jsonschema",      "jsonschema"),
    # Figure generation
    ("matplotlib",      "matplotlib"),
    ("seaborn",         "seaborn"),
    ("scipy",           "scipy"),
]

REQUIRED_DIRS = [
    "scripts/scrapers/raw/papers",
    "scripts/scrapers/raw/llm_cache",
    ".cache/llm_protocol",
    "data",
    "data/aggregated",
    "data/figures",
]

REQUIRED_FILES = [
    ("eval.schema.json",                      "blocking",  "Schema validation will be skipped"),
    ("src/extraction/extract_paper.py",       "blocking",  "Core extraction script missing"),
    ("src/figures/generate_all_figures.py",    "blocking",  "Figure generation script missing"),
    ("data/arxiv_ids.txt",                    "blocking",  "Batch input file missing"),
]


# ─────────────────────────────────────────────────────────────────────────────
def check_python_version() -> bool:
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 9):
        fail(f"Python {major}.{minor} — need ≥ 3.9")
        return False
    ok(f"Python {major}.{minor}")
    return True


def check_packages() -> bool:
    all_ok = True
    for import_name, display_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
            ok(f"Package: {display_name}")
        except ImportError:
            fail(f"Package missing: {display_name}  →  pip install {display_name}")
            all_ok = False
    return all_ok


def check_env_vars() -> bool:
    all_ok = True
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key and len(key) > 10:
        ok(f"OPENROUTER_API_KEY  set ({len(key)} chars)")
    else:
        warn("OPENROUTER_API_KEY  not set — LLM extraction will be disabled")
        warn("  Add to shell profile:  export OPENROUTER_API_KEY='sk-...'")
        # Not blocking — pipeline falls back to table+prose
    return all_ok


def check_hf_cache() -> bool:
    hf_home = pathlib.Path(os.environ.get("HF_HOME",
                pathlib.Path.home() / ".cache" / "huggingface"))
    layout_cache = hf_home / "hub" / "models--docling-project--docling-layout-heron"
    if layout_cache.exists():
        ok(f"Docling models cached  ({layout_cache})")
        return True
    warn("Docling models NOT cached — first run will download ~2 GB")
    warn("  After first run, set:  export HF_HUB_OFFLINE=1")
    return True  # not blocking — just slow


def check_directories() -> bool:
    for rel in REQUIRED_DIRS:
        p = ROOT / rel
        if p.exists():
            ok(f"Dir exists:  {rel}/")
        else:
            p.mkdir(parents=True, exist_ok=True)
            ok(f"Dir created: {rel}/")
    return True


def check_files() -> bool:
    all_ok = True
    for rel, severity, note in REQUIRED_FILES:
        p = ROOT / rel
        if p.exists():
            ok(f"File exists: {rel}")
        elif severity == "blocking":
            fail(f"Missing:     {rel}  ({note})")
            all_ok = False
        else:
            warn(f"Missing:     {rel}  ({note})")
    return all_ok


def check_schema() -> bool:
    schema_path = ROOT / "eval.schema.json"
    if not schema_path.exists():
        warn("eval.schema.json not found — validation task will be skipped")
        return True
    try:
        data = json.loads(schema_path.read_text())
        from jsonschema.validators import validator_for
        cls = validator_for(data)
        cls.check_schema(data)
        ok("eval.schema.json  is valid JSON Schema")
        return True
    except Exception as exc:
        fail(f"eval.schema.json  is invalid: {exc}")
        return False


def check_arxiv_reachability() -> bool:
    """Quick HEAD request to arXiv to confirm network access."""
    try:
        req = urllib.request.Request(
            "https://arxiv.org",
            headers={"User-Agent": "EEE-preflight/1.0"},
            method="HEAD",
        )
        with urllib.request.urlopen(req, timeout=5):
            ok("arXiv reachable  (network OK)")
            return True
    except Exception as exc:
        warn(f"arXiv unreachable: {exc}")
        warn("  PDFs must already exist in scripts/scrapers/raw/papers/")
        return True  # not blocking if PDFs are cached


def check_paper_ids() -> bool:
    """Verify arxiv_ids.txt contains exactly the expected test IDs."""
    batch = ROOT / "scripts" / "arxiv_ids.txt"
    if not batch.exists():
        fail("scripts/arxiv_ids.txt not found")
        return False
    ids_in_file = [
        line.strip()
        for line in batch.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    missing = [aid for aid in ARXIV_IDS if aid not in ids_in_file]
    extra   = [aid for aid in ids_in_file if aid not in ARXIV_IDS]
    if not missing and not extra:
        ok(f"arxiv_ids.txt  contains exactly the {len(ARXIV_IDS)} test papers")
    if missing:
        warn(f"IDs missing from batch file: {missing}")
    if extra:
        warn(f"Extra IDs in batch file (will also be processed): {extra}")
    return True


def check_pdf_cache() -> bool:
    """Report which PDFs are already downloaded vs. need fetching."""
    paper_dir = ROOT / "scripts" / "scrapers" / "raw" / "papers"
    for arxiv_id in ARXIV_IDS:
        pdf = paper_dir / f"{arxiv_id}.pdf"
        if pdf.exists():
            size_mb = pdf.stat().st_size / 1_048_576
            ok(f"PDF cached:    {arxiv_id}.pdf  ({size_mb:.1f} MB)")
        else:
            warn(f"PDF missing:   {arxiv_id}.pdf  — will download on first run")
    return True


def check_llm_cache() -> bool:
    """Report which LLM extraction results are already cached."""
    cache_dir = ROOT / "scripts" / "scrapers" / "raw" / "llm_cache"
    protocol_dir = ROOT / ".cache" / "llm_protocol"
    for arxiv_id in ARXIV_IDS:
        base_id = arxiv_id.replace("v1", "").replace("v2", "")
        chunks = list(cache_dir.glob(f"{arxiv_id}_chunk*.json")) + \
                 list(cache_dir.glob(f"{base_id}_chunk*.json"))
        protocol = protocol_dir / f"{arxiv_id}.json"
        protocol_base = protocol_dir / f"{base_id}.json"
        cached_protocol = protocol.exists() or protocol_base.exists()

        if chunks:
            ok(f"LLM cached:    {arxiv_id}  ({len(chunks)} chunk(s))")
        else:
            warn(f"LLM not cached: {arxiv_id}  — will call API on first run")
        if cached_protocol:
            ok(f"Protocol cached: {arxiv_id}")
        else:
            warn(f"Protocol not cached: {arxiv_id}  — will call API on first run")
    return True


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"\n{BOLD}═══ EEE Pipeline Preflight Check ═══════════════════════════════{RESET}")
    print(f"  Workspace: {ROOT}")
    print(f"  Papers:    {', '.join(ARXIV_IDS)}\n")

    sections = [
        ("Python version",        check_python_version),
        ("Required packages",     check_packages),
        ("Environment variables", check_env_vars),
        ("Docling model cache",   check_hf_cache),
        ("Directory structure",   check_directories),
        ("Required files",        check_files),
        ("Schema integrity",      check_schema),
        ("Network (arXiv)",       check_arxiv_reachability),
        ("Batch file contents",   check_paper_ids),
        ("PDF cache status",      check_pdf_cache),
        ("LLM cache status",      check_llm_cache),
    ]

    blocking_failures = 0
    for title, fn in sections:
        print(f"\n{BOLD}── {title}{RESET}")
        try:
            passed = fn()
            if not passed:
                blocking_failures += 1
        except Exception as exc:
            fail(f"Check crashed: {exc}")
            blocking_failures += 1

    print(f"\n{BOLD}═══ Summary ════════════════════════════════════════════════════{RESET}")
    if blocking_failures == 0:
        print(f"  {GREEN}{BOLD}✓ All checks passed — safe to run the batch pipeline{RESET}")
        print(f"\n  Next step in VSCode:")
        print(f"    Cmd+Shift+P  →  Tasks: Run Task  →  pipeline: FULL (batch)")
        print(f"    Batch file:  scripts/arxiv_ids.txt\n")
        sys.exit(0)
    else:
        print(f"  {RED}{BOLD}✗ {blocking_failures} blocking issue(s) found — fix before running{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
