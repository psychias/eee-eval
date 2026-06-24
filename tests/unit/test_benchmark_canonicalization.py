"""Regression test: no variant/casing leaks survive in the canonical benchmark map.

For every canonical benchmark name, asserts that no other casing or known
variant form can reach a *different* canonical name through the normaliser.
This catches both over-collapse (two different benchmarks → one name) and
under-collapse (one benchmark → two different names).
"""
import pytest
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Import both canonical dicts — use importlib to avoid triggering
# eee_eval.extraction.__init__ which pulls in docling and has broken imports
# ---------------------------------------------------------------------------
import importlib.util
import sys

_root = Path(__file__).resolve().parent.parent.parent

def _load_module_from_file(name: str, filepath: str):
    """Load a single .py file as a module without triggering package __init__."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register so inner imports can find it
    spec.loader.exec_module(mod)
    return mod

# constants.py is self-contained (no problematic imports)
_const_mod = _load_module_from_file(
    "extraction_constants",
    str(_root / "src" / "eee_eval" / "extraction" / "constants.py"),
)
CONST_MAP = _const_mod._BENCHMARK_CANONICAL

# _BENCHMARK_CANONICAL was consolidated into constants.py (the extract_paper.py
# copy was removed during SOLID refactoring). Both references now point to the
# same single source of truth.
PAPER_MAP = CONST_MAP


# ---------------------------------------------------------------------------
# 1. No ARC variant should map to anything other than ARC-Challenge / ARC-Easy
# ---------------------------------------------------------------------------
_ARC_CHALLENGE_VARIANTS = [
    "arc-c", "arc-challenge", "arc challenge", "arc_challenge",
    "arc c", "arc (challenge)", "arc- challenge", "arc (c)",
]
_ARC_EASY_VARIANTS = [
    "arc-e", "arc-easy", "arc easy", "arc_easy",
    "arc e", "arc (easy)", "arc- easy", "arc (e)",
]

@pytest.mark.parametrize("variant", _ARC_CHALLENGE_VARIANTS)
def test_arc_challenge_canonical(variant):
    for name, canonical_map in [("constants", CONST_MAP), ("extract_paper", PAPER_MAP)]:
        if variant in canonical_map:
            assert canonical_map[variant] == "ARC-Challenge", (
                f"{name}: '{variant}' maps to '{canonical_map[variant]}' "
                f"instead of 'ARC-Challenge'"
            )

@pytest.mark.parametrize("variant", _ARC_EASY_VARIANTS)
def test_arc_easy_canonical(variant):
    for name, canonical_map in [("constants", CONST_MAP), ("extract_paper", PAPER_MAP)]:
        if variant in canonical_map:
            assert canonical_map[variant] == "ARC-Easy", (
                f"{name}: '{variant}' maps to '{canonical_map[variant]}' "
                f"instead of 'ARC-Easy'"
            )


# ---------------------------------------------------------------------------
# 2. MATH Lvl 5 ≠ MATH-500 — no cross-contamination
# ---------------------------------------------------------------------------
_MATH_LVL5_VARIANTS = ["math lvl 5", "math level 5", "math-hard", "math hard"]
_MATH_500_VARIANTS = ["math500", "math-500"]

@pytest.mark.parametrize("variant", _MATH_LVL5_VARIANTS)
def test_math_lvl5_not_math500(variant):
    for name, canonical_map in [("constants", CONST_MAP), ("extract_paper", PAPER_MAP)]:
        if variant in canonical_map:
            assert canonical_map[variant] == "MATH Lvl 5", (
                f"{name}: '{variant}' maps to '{canonical_map[variant]}'"
            )

@pytest.mark.parametrize("variant", _MATH_500_VARIANTS)
def test_math500_not_lvl5(variant):
    for name, canonical_map in [("constants", CONST_MAP), ("extract_paper", PAPER_MAP)]:
        if variant in canonical_map:
            assert canonical_map[variant] == "MATH-500", (
                f"{name}: '{variant}' maps to '{canonical_map[variant]}'"
            )


# ---------------------------------------------------------------------------
# 3. GPQA-Diamond ≠ GPQA (generic)
# ---------------------------------------------------------------------------
def test_gpqa_diamond_distinct():
    for name, canonical_map in [("constants", CONST_MAP), ("extract_paper", PAPER_MAP)]:
        assert canonical_map.get("gpqa-diamond") == "GPQA-Diamond"
        assert canonical_map.get("gpqa") == "GPQA"
        assert canonical_map.get("gpqa") != "GPQA-Diamond"


# ---------------------------------------------------------------------------
# 4. BBH ≠ BIG-Bench
# ---------------------------------------------------------------------------
def test_bbh_not_bigbench():
    for name, canonical_map in [("constants", CONST_MAP), ("extract_paper", PAPER_MAP)]:
        assert canonical_map.get("bbh") == "BBH"
        assert canonical_map.get("big-bench") == "BIG-Bench", (
            f"{name}: 'big-bench' maps to '{canonical_map.get('big-bench')}'"
        )
        assert canonical_map.get("big-bench hard") == "BBH"


# ---------------------------------------------------------------------------
# 5. IFEval strict-prompt is distinct from generic IFEval
# ---------------------------------------------------------------------------
def test_ifeval_variants_distinct():
    for name, canonical_map in [("constants", CONST_MAP), ("extract_paper", PAPER_MAP)]:
        assert canonical_map.get("ifeval strict prompt") == "IFEval (strict-prompt)", (
            f"{name}: ifeval strict prompt → {canonical_map.get('ifeval strict prompt')}"
        )
        assert canonical_map.get("ifeval") == "IFEval"


# ---------------------------------------------------------------------------
# 6. TruthfulQA MC1/MC2 are distinct from generic TruthfulQA
# ---------------------------------------------------------------------------
def test_truthfulqa_mc_variants():
    for name, canonical_map in [("constants", CONST_MAP), ("extract_paper", PAPER_MAP)]:
        assert canonical_map.get("truthfulqa mc1") == "TruthfulQA-MC1"
        assert canonical_map.get("truthfulqa mc2") == "TruthfulQA-MC2"
        assert canonical_map.get("truthfulqa") == "TruthfulQA"


# ---------------------------------------------------------------------------
# 7. Bare pass@1 should NOT map to HumanEval (ambiguous)
# ---------------------------------------------------------------------------
def test_bare_pass_at_k_not_mapped():
    for name, canonical_map in [("constants", CONST_MAP), ("extract_paper", PAPER_MAP)]:
        assert "pass@1" not in canonical_map, (
            f"{name}: bare 'pass@1' should not be in canonical map"
        )


# ---------------------------------------------------------------------------
# 8. Cross-dict consistency: shared keys must map to the same canonical name
# ---------------------------------------------------------------------------
def test_constants_and_paper_maps_consistent():
    shared_keys = set(CONST_MAP.keys()) & set(PAPER_MAP.keys())
    mismatches = []
    for key in sorted(shared_keys):
        if CONST_MAP[key] != PAPER_MAP[key]:
            mismatches.append(
                f"  '{key}': constants→'{CONST_MAP[key]}' vs extract_paper→'{PAPER_MAP[key]}'"
            )
    assert not mismatches, (
        f"Canonical map divergence on {len(mismatches)} keys:\n"
        + "\n".join(mismatches)
    )


# ---------------------------------------------------------------------------
# 9. No canonical name appears as a value for a conflicting key
#    (e.g., "GPQA-Diamond" should not also be a canonical target for "gpqa")
# ---------------------------------------------------------------------------
_KNOWN_DISTINCT_BENCHMARKS = {
    # (parent, child) — child must not collapse into parent
    ("GPQA", "GPQA-Diamond"),
    ("GPQA", "GPQA-Extended"),
    ("BBH", "BIG-Bench"),
    ("MATH-500", "MATH Lvl 5"),
    ("IFEval", "IFEval (strict-prompt)"),
    ("IFEval", "IFEval (strict-inst)"),
    ("IFEval", "IFEval (loose-prompt)"),
    ("IFEval", "IFEval (loose-inst)"),
    ("TruthfulQA", "TruthfulQA-MC1"),
    ("TruthfulQA", "TruthfulQA-MC2"),
    ("ARC-Challenge", "ARC-Easy"),
}

def test_distinct_benchmarks_not_collapsed():
    """For each (parent, child) pair, no variant key should map child to parent."""
    for canonical_map in [CONST_MAP, PAPER_MAP]:
        # Build reverse map: canonical_name → set of keys
        reverse = {}
        for key, val in canonical_map.items():
            reverse.setdefault(val, set()).add(key)

        for parent, child in _KNOWN_DISTINCT_BENCHMARKS:
            if child in reverse:
                # child has its own canonical entries — should never also appear
                # mapped to parent
                for key in reverse.get(parent, set()):
                    assert canonical_map[key] != child, (
                        f"'{key}' maps to '{child}' but should map to '{parent}'"
                    )
