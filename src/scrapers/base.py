"""
abstract base classes and protocol interfaces for the EEE pipeline.

follows SOLID principles:
  - S: each class has a single, well-defined responsibility
  - O: new scrapers/extractors extend base classes without modifying existing ones
  - L: subclasses are fully substitutable for base classes
  - I: narrow, focused interfaces keep implementors decoupled
  - D: high-level orchestrators depend on abstractions (protocols/ABCs)
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# project root — used to locate eval.schema.json for pre-write validation
_ROOT = Path(__file__).resolve().parent.parent.parent

# lazily initialised once; None means the schema/jsonschema is unavailable
_schema_validator: Any = None
_schema_validator_loaded: bool = False


def _get_schema_validator() -> Any:
    """Return a cached jsonschema validator for eval.schema.json.

    Initialised on the first call.  Returns None when jsonschema is not
    installed or the schema file is absent, so callers can degrade gracefully
    without crashing on optional validation.
    """
    global _schema_validator, _schema_validator_loaded
    if _schema_validator_loaded:
        return _schema_validator
    _schema_validator_loaded = True

    schema_path = _ROOT / "eval.schema.json"
    if not schema_path.exists():
        print(
            f"  [validation] eval.schema.json not found at {schema_path} "
            "— pre-write validation disabled",
            file=sys.stderr,
        )
        return None

    try:
        from jsonschema.validators import validator_for  # type: ignore[import-untyped]

        with schema_path.open(encoding="utf-8") as fh:
            schema = json.load(fh)
        validator_cls = validator_for(schema)
        validator_cls.check_schema(schema)
        _schema_validator = validator_cls(schema)
    except Exception as exc:  # noqa: BLE001
        print(
            f"  [validation] could not load schema validator: {exc} "
            "— pre-write validation disabled",
            file=sys.stderr,
        )
    return _schema_validator

# ---------------------------------------------------------------------------
# narrow protocol interfaces (interface-segregation principle)
# ---------------------------------------------------------------------------


@runtime_checkable
class IFetcher(Protocol):
    """fetch raw data from a remote or local source."""

    def fetch(self, source: str) -> Any:
        """fetch and return raw data from *source* (a URL, file path, etc.)."""
        ...


@runtime_checkable
class IConverter(Protocol):
    """convert raw data to a list of EvaluationLog-compatible dicts."""

    def convert(self, raw: Any) -> list[dict]:
        """convert *raw* data to a list of EEE schema dicts."""
        ...


@runtime_checkable
class IValidator(Protocol):
    """validate a list of EEE schema dicts."""

    def validate(self, records: list[dict]) -> list[dict]:
        """return only the records that pass validation."""
        ...


@runtime_checkable
class IWriter(Protocol):
    """persist a list of EEE schema dicts to disk."""

    def write(self, records: list[dict]) -> list[Path]:
        """write *records* to disk; return paths of written files."""
        ...


# ---------------------------------------------------------------------------
# pipeline orchestrator (dependency-inversion principle)
# ---------------------------------------------------------------------------


class NoopValidator:
    """pass-through validator that accepts every record without modification.

    Use this as the default *validator* argument in EEEPipeline when you want
    to skip inline validation (e.g. you are running a separate validation step
    as a post-processing pass via ``scripts/validate_all.py``).
    """

    def validate(self, records: list[dict]) -> list[dict]:
        """return all *records* unchanged."""
        return records


class EEEPipeline:
    """high-level pipeline that wires fetcher → converter → validator → writer.

    Typical usage with the default pass-through validator::

        pipeline = EEEPipeline(
            fetcher=MyFetcher(),
            converter=MyConverter(),
            validator=NoopValidator(),
            writer=MyWriter(),
        )
        written_paths = pipeline.run("https://example.com/leaderboard")

    The scrapers (``BaseLeaderboardScraper`` subclasses) do not use this class
    directly — they integrate fetch/convert/write in their own ``run()`` method
    for simpler CLI usage.  ``EEEPipeline`` is provided for custom integrations
    that need a composable, dependency-injected alternative.
    """

    def __init__(
        self,
        fetcher: IFetcher,
        converter: IConverter,
        writer: IWriter,
        validator: IValidator | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._converter = converter
        self._validator = validator if validator is not None else NoopValidator()
        self._writer = writer

    def run(self, source: str) -> list[Path]:
        """run the full pipeline for *source* and return written file paths."""
        raw = self._fetcher.fetch(source)
        records = self._converter.convert(raw)
        valid = self._validator.validate(records)
        return self._writer.write(valid)


# ---------------------------------------------------------------------------
# abstract base classes for extension (open/closed principle)
# ---------------------------------------------------------------------------


class BaseLeaderboardScraper(ABC):
    """base class for all leaderboard scrapers.

    new scrapers extend this class; the existing ones are never modified.
    each subclass must implement *fetch_raw* and *convert*.
    """

    # subclasses set these class-level attributes
    eval_name: str = ""
    source_name: str = ""
    source_organization: str = ""
    output_dir: str = ""

    @abstractmethod
    def fetch_raw(self) -> Any:
        """fetch raw data from the leaderboard source."""
        ...

    @abstractmethod
    def convert(self, raw: Any, retrieved_timestamp: str) -> list[dict]:
        """convert raw leaderboard data to EEE schema dicts."""
        ...

    def save_raw(self, raw: Any, raw_dir: Path) -> None:
        """save raw data to *raw_dir* for reproducibility."""
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"{self.eval_name}_raw.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=2, ensure_ascii=False)
        print(f"  raw data saved → {path}")

    def run(self, save_raw: bool = True) -> int:
        """run fetch → convert → write; return count of written files."""
        import time

        raw_dir = Path("scripts/scrapers/raw")
        retrieved_timestamp = str(time.time())

        print(f"[{self.eval_name}] fetching raw data...")
        raw = self.fetch_raw()

        if save_raw:
            self.save_raw(raw, raw_dir)

        print(f"[{self.eval_name}] converting to EEE schema...")
        records = self.convert(raw, retrieved_timestamp)

        print(f"[{self.eval_name}] writing {len(records)} records...")
        count = self._write_records(records)
        self._write_methodology(retrieved_timestamp)
        print(f"[{self.eval_name}] done — {count} files written.")
        return count

    def run_standalone(
        self, limit: int | None = None, dry_run: bool = False
    ) -> int:
        """CLI-friendly run: fetch, optionally limit, skip write if dry_run.

        Provides the common main() boilerplate so scrapers don't need to
        duplicate it. Returns count of records written (0 if dry_run).
        """
        import time

        raw_dir = Path("scripts/scrapers/raw")
        retrieved_timestamp = str(time.time())

        print(f"[{self.eval_name}] fetching raw data...")
        raw = self.fetch_raw()

        self.save_raw(raw, raw_dir)

        if limit and isinstance(raw, list):
            raw = raw[:limit]

        records = self.convert(raw, retrieved_timestamp)
        print(f"[{self.eval_name}] converted {len(records)} records")

        if dry_run:
            print(f"[{self.eval_name}] dry-run mode — no files written")
            return 0

        count = self._write_records(records)
        self._write_methodology(retrieved_timestamp)
        print(f"[{self.eval_name}] done — {count} files written.")
        return count

    def _write_records(self, records: list[dict]) -> int:
        """validate against eval.schema.json then write to data/{eval_name}/.

        Records that fail schema validation are logged to stderr and skipped
        rather than written to disk — this prevents invalid files from
        accumulating silently and failing the CI check at PR time.
        """
        import uuid

        validator = _get_schema_validator()
        if validator is None:
            print(
                f"  [validation] WARNING: schema validator unavailable — "
                f"{len(records)} record(s) will be written without pre-write "
                f"schema validation. Ensure jsonschema is installed and "
                f"eval.schema.json is present before submitting a PR.",
                file=sys.stderr,
            )
        count = 0
        validation_failures = 0
        for rec in records:
            mid = rec.get("model_info", {}).get("id", "?")

            # Guard: leaderboard scrapers must always emit source_type='documentation'.
            # An incorrect value here means the record will likely fail CI schema
            # validation, so warn loudly before the write so it can be fixed fast.
            src_type = rec.get("source_metadata", {}).get("source_type")
            if src_type and src_type != "documentation":
                print(
                    f"  [source_type] WARNING {mid}: source_type='{src_type}' — "
                    f"leaderboard scrapers must use 'documentation'. "
                    f"Record will still be validated; fix before submitting a PR.",
                    file=sys.stderr,
                )

            try:
                # validate before touching the filesystem
                if validator is not None:
                    validator.validate(rec)
            except Exception as exc:  # noqa: BLE001
                # exc from jsonschema has a .message attribute; fall back to str()
                msg = getattr(exc, "message", str(exc))
                print(
                    f"  [validation] SKIP {mid}: {msg}",
                    file=sys.stderr,
                )
                validation_failures += 1
                continue

            try:
                model_id: str = rec["model_info"]["id"]
                # model_id is e.g. "meta-llama/Llama-3.1-8B"
                if "/" in model_id:
                    developer, model_name = model_id.split("/", 1)
                else:
                    developer = rec["model_info"].get("developer", "unknown")
                    model_name = model_id

                developer = _sanitize(developer)
                model_name = _sanitize(model_name)

                out_dir = Path(self.output_dir) / developer / model_name
                out_dir.mkdir(parents=True, exist_ok=True)

                path = out_dir / f"{uuid.uuid4()}.json"
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(rec, fh, indent=2, ensure_ascii=False)

                count += 1
            except Exception as exc:
                print(f"  warning: could not write record for {mid}: {exc}")

        if validation_failures:
            print(
                f"  [validation] {validation_failures}/{len(records)} record(s) "
                f"skipped due to schema validation errors — review stderr above "
                f"and fix source data before submitting a PR.",
                file=sys.stderr,
            )
        return count


    def _write_methodology(self, retrieved_timestamp: str) -> None:
        """Write a methodology.txt deliverable for this scraper.

        The ACL 2026 Shared Task deliverables require documentation of the
        data-extraction methodology for each Track 1 submission.  This method
        auto-generates a file that satisfies that requirement.  Edit the
        generated file to document any known issues or manual decisions made
        during extraction.
        """
        import datetime

        out_dir = Path("scripts/scrapers")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.eval_name}_methodology.txt"
        dt = datetime.datetime.utcfromtimestamp(float(retrieved_timestamp)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        content = (
            f"Methodology: {self.eval_name}\n"
            f"{'=' * (13 + len(self.eval_name))}\n\n"
            f"Source name:         {self.source_name}\n"
            f"Source organisation: {self.source_organization}\n"
            f"Schema version:      0.2.1\n"
            f"Retrieved at:        {dt} (UTC)\n"
            f"Script:              scripts/scrapers/{self.eval_name}_scraper.py\n\n"
            f"Extraction approach\n"
            f"-------------------\n"
            f"Data is fetched programmatically from the {self.source_name} public\n"
            f"API or web interface at the URL(s) documented in the scraper source.\n"
            f"Each entry is converted to the EEE eval.schema.json (v0.2.1) format\n"
            f"and validated against the schema before being written to disk.\n"
            f"source_type is set to 'documentation' for all records, reflecting\n"
            f"that scores originate from a published leaderboard (not a local run).\n\n"
            f"Known issues / manual decisions\n"
            f"--------------------------------\n"
            f"(none documented — update this section with any anomalies encountered\n"
            f" during extraction or conversion before submitting a PR)\n"
        )
        path.write_text(content, encoding="utf-8")
        print(f"  methodology written → {path}")


class BasePaperExtractor(ABC):
    """base class for all paper extractors.

    new extractors extend this class without modifying existing ones.
    """

    @abstractmethod
    def extract(self, source: str) -> list[dict]:
        """extract EEE schema dicts from *source* (PDF path or arXiv ID)."""
        ...


class BaseProprietaryConverter(ABC):
    """base class for Track 2 proprietary evaluation converters.

    Use this when you are the organisation that ran the evaluation
    (``evaluator_relationship = first_party``) and want to convert an
    internal evaluation format into the EEE schema.

    Subclass responsibilities
    -------------------------
    * Set the class-level attributes ``eval_name``, ``source_name``,
      ``source_organization``, and ``output_dir``.
    * Implement ``load_raw`` to read your internal data format (CSV, JSONL,
      internal JSON, etc.) from *source_path*.
    * Implement ``convert`` to turn that raw data into a list of EEE schema
      dicts.  Use ``SourceDataPrivate`` for the ``source_data`` field of each
      ``EvaluationResult`` when the dataset cannot be shared publicly.

    The ``run`` method handles writing and reporting so subclasses stay
    focused on conversion logic.

    Example usage::

        class MyOrgConverter(BaseProprietaryConverter):
            eval_name = "myorg_internal_bench"
            source_name = "MyOrg Internal Benchmark"
            source_organization = "MyOrg"
            output_dir = "data/myorg_internal_bench"

            def load_raw(self, source_path: Path) -> Any:
                import csv
                with open(source_path) as fh:
                    return list(csv.DictReader(fh))

            def convert(self, raw: Any, retrieved_timestamp: str) -> list[dict]:
                ...

        converter = MyOrgConverter()
        converter.run(Path("internal_results.csv"))
    """

    # subclasses must set these
    eval_name: str = ""
    source_name: str = ""
    source_organization: str = ""
    output_dir: str = ""

    # Track 2 converters represent first-party evaluations by default.
    # Override with EvaluatorRelationship.third_party if needed.
    evaluator_relationship: str = "first_party"

    @abstractmethod
    def load_raw(self, source_path: Path) -> Any:
        """load raw proprietary evaluation data from *source_path*.

        *source_path* may be a file (CSV, JSONL, JSON, …) or a directory.
        Return whatever structure is convenient for ``convert``.
        """
        ...

    @abstractmethod
    def convert(self, raw: Any, retrieved_timestamp: str) -> list[dict]:
        """convert raw internal data to a list of EEE schema dicts.

        Each dict must pass validation against ``eval.schema.json``.
        Use ``SourceDataPrivate`` for ``source_data`` fields when the
        underlying dataset cannot be released publicly.
        """
        ...

    def run(self, source_path: Path, *, dry_run: bool = False) -> int:
        """load → convert → write; return the number of files written.

        Parameters
        ----------
        source_path:
            Path to the internal evaluation file or directory.
        dry_run:
            When ``True``, skip disk writes and only report conversion counts.
        """
        import time

        retrieved_timestamp = str(time.time())

        print(f"[{self.eval_name}] loading raw data from {source_path}...")
        raw = self.load_raw(source_path)

        print(f"[{self.eval_name}] converting to EEE schema...")
        records = self.convert(raw, retrieved_timestamp)
        print(f"[{self.eval_name}] converted {len(records)} records")

        if dry_run:
            print(f"[{self.eval_name}] dry-run mode — skipping disk writes")
            return len(records)

        count = self._write_records(records)
        print(f"[{self.eval_name}] done — {count} files written to {self.output_dir}/")
        return count

    def _write_records(self, records: list[dict]) -> int:
        """Validate against eval.schema.json then write to ``output_dir``.

        Path pattern: ``{output_dir}/{developer}/{model_name}/{uuid}.json``

        Records that fail schema validation are logged to stderr and skipped
        so that invalid files never reach disk and don't fail CI at PR time.
        """
        import uuid

        validator = _get_schema_validator()
        count = 0
        for rec in records:
            mid = rec.get("model_info", {}).get("id", "?")
            try:
                if validator is not None:
                    validator.validate(rec)
            except Exception as exc:  # noqa: BLE001
                msg = getattr(exc, "message", str(exc))
                print(
                    f"  [validation] SKIP {mid}: {msg}",
                    file=sys.stderr,
                )
                continue

            try:
                model_id: str = rec["model_info"]["id"]
                if "/" in model_id:
                    developer, model_name = model_id.split("/", 1)
                else:
                    developer = rec["model_info"].get("developer", "unknown")
                    model_name = model_id

                out_dir = Path(self.output_dir) / _sanitize(developer) / _sanitize(model_name)
                out_dir.mkdir(parents=True, exist_ok=True)

                path = out_dir / f"{uuid.uuid4()}.json"
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(rec, fh, indent=2, ensure_ascii=False)

                count += 1
            except Exception as exc:
                print(f"  warning: could not write record for {mid}: {exc}")

        return count


# ---------------------------------------------------------------------------
# helpers used internally
# ---------------------------------------------------------------------------


def _sanitize(name: str) -> str:
    """sanitize *name* for use as a filesystem path component."""
    import re

    return re.sub(r'[<>:"/\\|?*]', "_", name)
