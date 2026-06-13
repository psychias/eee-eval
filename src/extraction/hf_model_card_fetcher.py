"""
hf_model_card_fetcher.py — extract benchmark results from HF model cards.

Scans HuggingFace model READMEs for:
  1. Structured YAML `model-index` results (official HF format)
  2. ArXiv paper references (extracted from body text)

Results are written as EEE-format JSON records under data/hf_model_card/.

Usage
-----
  python hf_model_card_fetcher.py                 # fetch all models from leaderboard records
  python hf_model_card_fetcher.py --refresh        # bypass cache
  python hf_model_card_fetcher.py --model Qwen/Qwen2.5-72B-Instruct

Requirements
------------
  pip install requests pyyaml
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def _arxiv_id_to_date(arxiv_id: str) -> str:
    """Derive an approximate ISO date from an arxiv ID (YYMM.NNNNN -> YYYY-MM-01)."""
    m = re.match(r"(\d{2})(\d{2})\.", arxiv_id)
    if m:
        yy, mm = int(m.group(1)), int(m.group(2))
        year = 2000 + yy
        if 1 <= mm <= 12:
            return f"{year}-{mm:02d}-01"
    return ""

import requests

_ROOT    = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.extraction.constants import infer_developer

DATA_DIR   = _ROOT / "data"
CACHE_DIR  = pathlib.Path(".cache/hf_model_cards")
CACHE_TTL  = 7 * 24 * 3600  # 7 days

_SOURCE_DIR  = "hf_model_card"
_SOURCE_NAME = "HuggingFace Model Card"
_SOURCE_ORG  = "HuggingFace"
_SOURCE_URL  = "https://huggingface.co"

_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ModelCardResult:
    """A single benchmark result parsed from a model card."""
    benchmark:        str
    metric_name:      str
    score:            float
    dataset:          str = ""
    split:            str = ""
    verified:         bool = False
    shots:            int | None = None
    temperature:      float | None = None
    eval_lib:         str = ""
    chain_of_thought: bool | None = None


@dataclass
class ModelCardData:
    """Parsed model card with results and metadata."""
    model_id:   str
    model_name: str
    developer:  str
    results:    list[ModelCardResult] = field(default_factory=list)
    arxiv_ids:  list[str] = field(default_factory=list)
    paper_url:  str = ""
    tags:       list[str] = field(default_factory=list)
    library:    str = ""


# ---------------------------------------------------------------------------
# Developer inference — delegated to constants.infer_developer()
# ---------------------------------------------------------------------------

def _infer_developer(model_id: str) -> str:
    """Infer developer from model_id (org/model format or model name)."""
    if "/" in model_id:
        return model_id.split("/")[0]
    return infer_developer(model_id)


# ---------------------------------------------------------------------------
# YAML frontmatter parsing
# ---------------------------------------------------------------------------

def _parse_yaml_frontmatter(readme_text: str) -> dict[str, Any]:
    """Extract YAML frontmatter from a README. Returns {} on failure."""
    if not readme_text.startswith("---"):
        return {}
    end = readme_text.find("\n---", 3)
    if end < 0:
        return {}
    yaml_str = readme_text[3:end].strip()
    try:
        import yaml
        return yaml.safe_load(yaml_str) or {}
    except Exception:
        return {}


def _parse_model_index_results(model_index: list[dict]) -> list[ModelCardResult]:
    """Parse HF model-index YAML into ModelCardResult list."""
    results: list[ModelCardResult] = []
    if not isinstance(model_index, list):
        return results

    for entry in model_index:
        if not isinstance(entry, dict):
            continue
        for result_block in entry.get("results", []):
            if not isinstance(result_block, dict):
                continue
            dataset_info = result_block.get("dataset", {})
            dataset_name = ""
            dataset_args = ""
            if isinstance(dataset_info, dict):
                dataset_name = dataset_info.get("name", "") or dataset_info.get("type", "")
                dataset_args = str(dataset_info.get("args", "") or dataset_info.get("config", ""))
            elif isinstance(dataset_info, str):
                dataset_name = dataset_info

            # Extract n-shot from dataset args or name
            shots = _extract_shots_from_text(f"{dataset_name} {dataset_args}")

            for metric in result_block.get("metrics", []):
                if not isinstance(metric, dict):
                    continue
                metric_type = metric.get("type", "")
                metric_name = metric.get("name", metric_type)
                value = metric.get("value")
                verified = metric.get("verified", False)

                if value is None:
                    continue
                try:
                    score = float(value)
                except (ValueError, TypeError):
                    continue

                bench = dataset_name or metric_name or "unknown"
                # Also try extracting shots from metric name
                m_shots = shots
                if m_shots is None:
                    m_shots = _extract_shots_from_text(str(metric_name))

                # Extract CoT and temperature from dataset/metric text
                combined_text = f"{dataset_name} {dataset_args} {metric_name}"
                m_cot = _extract_cot_from_text(combined_text)
                m_temp = _extract_temperature_from_text(combined_text)

                results.append(ModelCardResult(
                    benchmark=bench,
                    metric_name=str(metric_name or metric_type or "accuracy"),
                    score=score,
                    dataset=str(dataset_name),
                    verified=bool(verified),
                    shots=m_shots,
                    temperature=m_temp,
                    chain_of_thought=m_cot,
                ))
    return results


def _extract_shots_from_text(text: str) -> int | None:
    """Try to extract n-shot count from text."""
    m = re.search(r"(\d+)[- ]?shot", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    if re.search(r"\bzero[- ]?shot\b", text, re.IGNORECASE):
        return 0
    if re.search(r"\bfew[- ]?shot\b", text, re.IGNORECASE):
        return -1  # indicates few-shot without exact count
    return None


def _extract_cot_from_text(text: str) -> bool | None:
    """Detect chain-of-thought from text. Returns True/False/None."""
    if re.search(r"\bcot\b|chain[- ]of[- ]thought", text, re.IGNORECASE):
        return True
    return None


def _extract_temperature_from_text(text: str) -> float | None:
    """Extract temperature setting from text."""
    m = re.search(r"temp(?:erature)?\s*=?\s*([0-9.]+)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _extract_eval_lib_from_readme(text: str) -> str:
    """Detect eval framework from README text."""
    lower = text.lower()
    for lib, patterns in [
        ("lm-eval-harness", [r"lm[_-]?eval", r"eleutherai.*harness", r"lm[_-]?harness"]),
        ("helm", [r"\bhelm\b", r"stanford.*helm"]),
        ("inspect", [r"\binspect\b.*eval", r"uk\s*aisi.*inspect"]),
        ("vllm", [r"\bvllm\b"]),
        ("fastchat", [r"\bfastchat\b", r"mt[_-]?bench"]),
        ("alpaca_eval", [r"alpaca[_-]?eval"]),
        ("bigcode-evaluation-harness", [r"bigcode.*harness", r"bigcodebench"]),
    ]:
        for pat in patterns:
            if re.search(pat, lower):
                return lib
    return ""


def _extract_arxiv_ids(text: str) -> list[str]:
    """Extract unique arxiv IDs from text."""
    matches = re.findall(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", text, re.IGNORECASE)
    # Also match bare arxiv IDs in common patterns like "Paper: 2407.10671"
    matches += re.findall(r"(?:paper|preprint|citation)[:\s]+(\d{4}\.\d{4,5})", text, re.IGNORECASE)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


# ---------------------------------------------------------------------------
# HF Model Card Fetcher
# ---------------------------------------------------------------------------

class HFModelCardFetcher:
    """Fetch and parse HF model cards for benchmark results."""

    def __init__(self) -> None:
        self._token = os.environ.get("HF_TOKEN", "")

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _fetch_readme(self, model_id: str, force_refresh: bool = False) -> str | None:
        """Fetch the raw README.md for a model. Returns None on failure."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        safe_id = model_id.replace("/", "__")
        cache_path = CACHE_DIR / f"{safe_id}.md"

        if not force_refresh and cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < CACHE_TTL:
                return cache_path.read_text(encoding="utf-8")

        url = f"https://huggingface.co/{model_id}/raw/main/README.md"
        try:
            r = requests.get(url, headers=self._headers(), timeout=_TIMEOUT)
            if r.status_code == 401:
                # Gated model — skip silently
                return None
            if r.status_code == 404:
                return None
            r.raise_for_status()
        except requests.RequestException:
            return None

        text = r.text
        cache_path.write_text(text, encoding="utf-8")
        return text

    def fetch_and_parse(
        self,
        model_id: str,
        force_refresh: bool = False,
    ) -> ModelCardData | None:
        """Fetch and parse a model card. Returns None if no useful data found."""
        readme = self._fetch_readme(model_id, force_refresh=force_refresh)
        if readme is None:
            return None

        frontmatter = _parse_yaml_frontmatter(readme)
        model_index = frontmatter.get("model-index", [])
        results = _parse_model_index_results(model_index)

        arxiv_ids = _extract_arxiv_ids(readme)
        tags = frontmatter.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        library = frontmatter.get("library_name", "")

        # Detect eval framework from README text
        eval_lib = _extract_eval_lib_from_readme(readme)
        if eval_lib:
            for r in results:
                if not r.eval_lib:
                    r.eval_lib = eval_lib

        # If no structured results and no arxiv IDs, not useful
        if not results and not arxiv_ids:
            return None

        dev = _infer_developer(model_id)
        name = model_id.split("/")[-1] if "/" in model_id else model_id

        paper_url = ""
        if arxiv_ids:
            paper_url = f"https://arxiv.org/abs/{arxiv_ids[0]}"

        return ModelCardData(
            model_id=model_id,
            model_name=name,
            developer=dev,
            results=results,
            arxiv_ids=arxiv_ids,
            paper_url=paper_url,
            tags=tags if isinstance(tags, list) else [],
            library=str(library),
        )


# ---------------------------------------------------------------------------
# Record writing
# ---------------------------------------------------------------------------

def _already_exists(model_id: str) -> bool:
    """Check if we already have a model card record for this model."""
    d = DATA_DIR / _SOURCE_DIR
    if not d.exists():
        return False
    cache = getattr(_already_exists, "_cache", None)
    if cache is None:
        ids: set[str] = set()
        for f in d.rglob("*.json"):
            try:
                mid = json.loads(f.read_text(encoding="utf-8")).get("model_info", {}).get("id")
                if mid:
                    ids.add(mid)
            except Exception:
                pass
        _already_exists._cache = ids  # type: ignore[attr-defined]
        cache = ids
    return model_id in cache


def _guess_scale(benchmark: str, score: float) -> tuple[float, float]:
    """Guess min/max score for a benchmark."""
    if score > 1.0:
        return (0.0, 100.0)
    return (0.0, 1.0)


def write_model_card_record(mcd: ModelCardData) -> int:
    """Write an EEE record from model card data. Returns 1 if written, 0 otherwise."""
    if not mcd.results:
        return 0

    if _already_exists(mcd.model_id):
        return 0

    ts = str(int(time.time()))

    # Derive evaluation date from first arxiv ID if available
    eval_date = ""
    for aid in mcd.arxiv_ids:
        eval_date = _arxiv_id_to_date(aid)
        if eval_date:
            break

    result_dicts = []
    for r in mcd.results:
        min_s, max_s = _guess_scale(r.benchmark, r.score)
        rd: dict[str, Any] = {
            "bench": r.benchmark,
            "score": r.score,
            "min_score": min_s,
            "max_score": max_s,
            "metric_name": r.metric_name,
        }
        if r.shots is not None and r.shots >= 0:
            rd["shots"] = r.shots
        if r.temperature is not None:
            rd["temperature"] = r.temperature
        if r.chain_of_thought is not None:
            rd["chain_of_thought"] = r.chain_of_thought
        if r.eval_lib:
            rd["eval_lib"] = r.eval_lib
        result_dicts.append(rd)

    # Determine the best eval library name
    detected_lib = ""
    for r in mcd.results:
        if r.eval_lib:
            detected_lib = r.eval_lib
            break
    lib_name = detected_lib or mcd.library or "unknown"

    rec = {
        "schema_version": "0.2.1",
        "evaluation_id": f"{_SOURCE_DIR}/{mcd.model_id.replace('/', '_')}/{ts}",
        **({"evaluation_timestamp": eval_date} if eval_date else {}),
        "retrieved_timestamp": ts,
        "source_metadata": {
            "source_name": _SOURCE_NAME,
            "source_type": "documentation",
            "source_organization_name": _SOURCE_ORG,
            "source_organization_url": _SOURCE_URL,
            "evaluator_relationship": "first_party",
            "additional_details": {
                "arxiv_ids": ", ".join(mcd.arxiv_ids) if mcd.arxiv_ids else "",
                "paper_url": mcd.paper_url,
                "tags": ", ".join(mcd.tags[:10]) if mcd.tags else "",
                "library": mcd.library,
            },
        },
        "eval_library": {"name": lib_name, "version": "unknown"},
        "model_info": {
            "name": mcd.model_name,
            "id": mcd.model_id,
            "developer": mcd.developer,
        },
        "evaluation_results": [
            {
                "evaluation_name": rd["bench"],
                **({"evaluation_timestamp": eval_date} if eval_date else {}),
                "source_data": {
                    "dataset_name": rd["bench"],
                    "source_type": "url",
                    "url": [f"{_SOURCE_URL}/{mcd.model_id}"],
                },
                "metric_config": {
                    "metric_name": rd.get("metric_name", "accuracy"),
                    "lower_is_better": False,
                    "score_type": "continuous",
                    "min_score": rd["min_score"],
                    "max_score": rd["max_score"],
                },
                "score_details": {
                    "score": rd["score"],
                },
                "generation_config": {
                    "generation_args": {
                        **({"shots": rd["shots"]} if "shots" in rd else {}),
                        **({"temperature": rd["temperature"]} if "temperature" in rd else {}),
                        **({"chain_of_thought": rd["chain_of_thought"]} if "chain_of_thought" in rd else {}),
                    },
                    "additional_details": {
                        "source": _SOURCE_DIR,
                        **({"harness": rd["eval_lib"]} if rd.get("eval_lib") else {}),
                    },
                },
            }
            for rd in result_dicts
        ],
    }

    # Save
    dev = mcd.developer
    slug = mcd.model_id.split("/")[-1] if "/" in mcd.model_id else mcd.model_id
    out_dir = DATA_DIR / _SOURCE_DIR / dev / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{uuid.uuid4()}.json"
    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    # Update cache
    if hasattr(_already_exists, "_cache"):
        _already_exists._cache.add(mcd.model_id)  # type: ignore[attr-defined]

    return 1


# ---------------------------------------------------------------------------
# Model discovery from leaderboard records
# ---------------------------------------------------------------------------

def models_from_leaderboards() -> list[str]:
    """Collect unique model IDs from all leaderboard records in data/."""
    model_ids: set[str] = set()
    if not DATA_DIR.exists():
        return []

    for json_file in DATA_DIR.rglob("*.json"):
        # Skip our own records
        try:
            rel = json_file.relative_to(DATA_DIR)
        except ValueError:
            continue
        if str(rel).startswith(_SOURCE_DIR):
            continue

        try:
            rec = json.loads(json_file.read_text(encoding="utf-8"))
            mid = rec.get("model_info", {}).get("id", "").strip()
            if mid and "/" in mid and mid != "unknown/unknown":
                model_ids.add(mid)
        except Exception:
            pass

    return sorted(model_ids)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch HF model card benchmark results and write EEE records.",
    )
    parser.add_argument("--refresh", action="store_true", help="Bypass cache.")
    parser.add_argument("--model", type=str, default="",
                        help="Fetch a single model by ID (e.g. Qwen/Qwen2.5-72B-Instruct).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and show results, but don't write records.")
    args = parser.parse_args()

    fetcher = HFModelCardFetcher()

    if args.model:
        model_ids = [args.model]
    else:
        model_ids = models_from_leaderboards()
        if not model_ids:
            print("No leaderboard models found. Run add_leaderboard_records.py first.")
            return

    print(f"Processing {len(model_ids)} models...\n")

    written = 0
    no_data = 0
    skipped = 0
    with_arxiv = 0

    for i, mid in enumerate(model_ids, 1):
        print(f"  [{i:3d}/{len(model_ids)}] {mid}", end=" ")

        if not args.dry_run and _already_exists(mid):
            print("-> already exists")
            skipped += 1
            continue

        mcd = fetcher.fetch_and_parse(mid, force_refresh=args.refresh)
        if mcd is None:
            print("-> no data")
            no_data += 1
            continue

        if mcd.arxiv_ids:
            with_arxiv += 1

        if args.dry_run:
            print(f"-> {len(mcd.results)} results, arxiv: {mcd.arxiv_ids}")
            continue

        n = write_model_card_record(mcd)
        if n:
            written += 1
            benches = ", ".join(r.benchmark for r in mcd.results[:3])
            extra = f" +{len(mcd.results) - 3} more" if len(mcd.results) > 3 else ""
            print(f"-> {len(mcd.results)} results ({benches}{extra})")
        else:
            if not mcd.results:
                print(f"-> arxiv only: {mcd.arxiv_ids}")
            else:
                print("-> write skipped (exists)")

        # Rate limit
        if i % 10 == 0:
            time.sleep(0.5)

    print(f"\n  Written: {written} | No data: {no_data} | Skipped: {skipped}")
    print(f"  Models with arxiv IDs: {with_arxiv}")


if __name__ == "__main__":
    main()
