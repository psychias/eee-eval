"""Unit tests for src/validation/ refactored modules.

Tests:
  - schema.py: SchemaValidator, AutoFixer, FileValidator
  - quality_audit.py: Record, checks (SchemaCheck, ShotsCheck, ScoreCheck, etc.)
  - fix_data_quality.py: ParameterLeakFix, BenchmarkCanonFix
"""
import json
import tempfile
from pathlib import Path

import pytest

from eee_eval.validation.schema import SchemaValidator, AutoFixer, FileValidator
from eee_eval.validation.quality_audit import (
    Record, Issue, RecordLoader,
    SchemaCheck, ShotsCheck, HarnessCheck, ScoreCheck,
    DuplicateCheck, RequiredFieldsCheck, BenchmarkNamingCheck,
    ModelIdentityCheck, MetadataCoverageCheck, AuditRunner,
)
from eee_eval.validation.fix_data_quality import (
    ParameterLeakFix, BenchmarkCanonFix, FixStats,
    _is_metadata_metric, _is_score_out_of_range,
)

ROOT = Path(__file__).resolve().parent.parent.parent


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_record() -> dict:
    """A minimal valid EEE schema record."""
    return {
        "schema_version": "0.2.1",
        "evaluation_id": "test__001",
        "retrieved_timestamp": "2025-01-01T00:00:00Z",
        "source_metadata": {
            "source_type": "evaluation_run",
            "source_organization_name": "TestOrg",
            "evaluator_relationship": "first_party",
        },
        "model_info": {
            "name": "Test Model",
            "id": "testorg/test-model",
            "developer": "TestOrg",
        },
        "eval_library": {
            "name": "lm-evaluation-harness",
            "version": "0.4.0",
        },
        "evaluation_results": [
            {
                "evaluation_name": "MMLU",
                "source_data": {"dataset_name": "MMLU", "source_type": "hf_dataset"},
                "metric_config": {
                    "metric_name": "accuracy",
                    "min_score": 0.0,
                    "max_score": 100.0,
                    "lower_is_better": False,
                    "level_names": ["accuracy"],
                    "has_unknown_level": False,
                },
                "score_details": {"score": 75.5},
                "generation_config": {
                    "generation_args": {"shots": 5, "temperature": 0.0}
                },
            }
        ],
    }


@pytest.fixture
def schema_validator() -> SchemaValidator:
    return SchemaValidator()


# ─── SchemaValidator tests ────────────────────────────────────────────────────


class TestSchemaValidator:
    def test_valid_record_passes(self, schema_validator, valid_record):
        schema_validator.validate(valid_record)  # should not raise

    def test_missing_required_field_fails(self, schema_validator, valid_record):
        del valid_record["evaluation_id"]
        with pytest.raises(Exception):
            schema_validator.validate(valid_record)

    def test_iter_errors_returns_list(self, schema_validator, valid_record):
        del valid_record["model_info"]
        errors = schema_validator.collect_errors(valid_record)
        assert len(errors) > 0

    def test_valid_record_no_errors(self, schema_validator, valid_record):
        errors = schema_validator.collect_errors(valid_record)
        assert errors == []


# ─── AutoFixer tests ──────────────────────────────────────────────────────────


class TestAutoFixer:
    def test_fix_score_string_to_float(self, valid_record):
        valid_record["evaluation_results"][0]["score_details"]["score"] = "75.5"
        fixer = AutoFixer()
        fixed, changes = fixer.fix(valid_record)
        assert fixed["evaluation_results"][0]["score_details"]["score"] == 75.5
        assert len(changes) > 0

    def test_fix_missing_schema_version(self, valid_record):
        del valid_record["schema_version"]
        fixer = AutoFixer()
        fixed, changes = fixer.fix(valid_record)
        assert fixed["schema_version"] == "0.2.1"

    def test_fix_evaluator_relationship_unknown(self, valid_record):
        valid_record["source_metadata"]["evaluator_relationship"] = "unknown"
        fixer = AutoFixer()
        fixed, changes = fixer.fix(valid_record)
        assert fixed["source_metadata"]["evaluator_relationship"] == "other"

    def test_no_fix_on_valid_record(self, valid_record):
        fixer = AutoFixer()
        fixed, changes = fixer.fix(valid_record)
        assert changes == []


# ─── FileValidator tests ──────────────────────────────────────────────────────


class TestFileValidator:
    def test_validate_valid_file(self, valid_record, tmp_path):
        p = tmp_path / "test.json"
        p.write_text(json.dumps(valid_record), encoding="utf-8")

        sv = SchemaValidator()
        fixer = AutoFixer()
        fv = FileValidator(sv, fixer, fix_mode=False)
        result = fv.validate_file(p)
        assert result["status"] == "valid"

    def test_validate_invalid_file(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text('{"not": "valid"}', encoding="utf-8")

        sv = SchemaValidator()
        fixer = AutoFixer()
        fv = FileValidator(sv, fixer, fix_mode=False)
        result = fv.validate_file(p)
        assert result["status"] == "invalid"


# ─── Quality Audit Check tests ───────────────────────────────────────────────


def _make_record(data: dict, source: str = "test") -> Record:
    return Record(source=source, path=Path("/fake/path.json"), data=data)


class TestShotsCheck:
    def test_known_shots_mismatch(self, valid_record):
        valid_record["evaluation_results"][0]["evaluation_name"] = "BBH"
        valid_record["evaluation_results"][0]["generation_config"]["generation_args"]["shots"] = 0
        rec = _make_record(valid_record, source="open_llm_leaderboard_v2")
        check = ShotsCheck()
        issues = check.run([rec])
        # BBH should be 3-shot; setting 0 is an ERROR
        errors = [i for i in issues if i.severity == "ERROR"]
        assert len(errors) == 1
        assert "expected 3" in errors[0].message

    def test_correct_shots_no_issue(self, valid_record):
        valid_record["evaluation_results"][0]["evaluation_name"] = "BBH"
        valid_record["evaluation_results"][0]["generation_config"]["generation_args"]["shots"] = 3
        rec = _make_record(valid_record, source="open_llm_leaderboard_v2")
        check = ShotsCheck()
        issues = check.run([rec])
        errors = [i for i in issues if i.severity == "ERROR"]
        assert errors == []


class TestScoreCheck:
    def test_score_above_max(self, valid_record):
        valid_record["evaluation_results"][0]["score_details"]["score"] = 150.0
        valid_record["evaluation_results"][0]["metric_config"]["max_score"] = 100.0
        rec = _make_record(valid_record)
        check = ScoreCheck()
        issues = check.run([rec])
        criticals = [i for i in issues if i.severity == "CRITICAL"]
        assert len(criticals) == 1  # parameter leak detection

    def test_valid_score_no_issue(self, valid_record):
        rec = _make_record(valid_record)
        check = ScoreCheck()
        issues = check.run([rec])
        assert all(i.severity not in ("CRITICAL", "ERROR") for i in issues)

    def test_missing_score(self, valid_record):
        valid_record["evaluation_results"][0]["score_details"]["score"] = None
        rec = _make_record(valid_record)
        check = ScoreCheck()
        issues = check.run([rec])
        errors = [i for i in issues if i.severity == "ERROR"]
        assert len(errors) == 1


class TestDuplicateCheck:
    def test_same_model_bench_different_scores(self, valid_record):
        rec1 = _make_record(valid_record, source="test_src")
        rec2_data = json.loads(json.dumps(valid_record))
        rec2_data["evaluation_results"][0]["score_details"]["score"] = 80.0
        rec2 = _make_record(rec2_data, source="test_src")
        check = DuplicateCheck()
        issues = check.run([rec1, rec2])
        warnings = [i for i in issues if i.severity == "WARNING"]
        assert len(warnings) == 1


class TestRequiredFieldsCheck:
    def test_missing_eval_library(self, valid_record):
        del valid_record["eval_library"]
        rec = _make_record(valid_record)
        check = RequiredFieldsCheck()
        issues = check.run([rec])
        errors = [i for i in issues if i.severity == "ERROR"]
        assert any("eval_library" in e.message for e in errors)


class TestModelIdentityCheck:
    def test_empty_model_name(self, valid_record):
        valid_record["model_info"]["name"] = ""
        rec = _make_record(valid_record)
        check = ModelIdentityCheck()
        issues = check.run([rec])
        errors = [i for i in issues if i.severity == "ERROR"]
        assert len(errors) == 1

    def test_missing_slash_in_id(self, valid_record):
        valid_record["model_info"]["id"] = "testmodel"
        rec = _make_record(valid_record)
        check = ModelIdentityCheck()
        issues = check.run([rec])
        warnings = [i for i in issues if i.severity == "WARNING"]
        assert any("missing developer/" in w.message for w in warnings)


class TestBenchmarkNamingCheck:
    def test_near_duplicates_detected(self, valid_record):
        rec1 = _make_record(valid_record)
        rec1.data["evaluation_results"][0]["evaluation_name"] = "ARC-Challenge"
        rec2_data = json.loads(json.dumps(valid_record))
        rec2_data["evaluation_results"][0]["evaluation_name"] = "arc_challenge"
        rec2 = _make_record(rec2_data, source="other")
        check = BenchmarkNamingCheck()
        issues = check.run([rec1, rec2])
        assert len(issues) == 1
        assert "Near-duplicate" in issues[0].message


class TestAuditRunner:
    def test_run_all_returns_results(self, valid_record):
        rec = _make_record(valid_record)
        runner = AuditRunner()
        results = runner.run_all([rec])
        assert len(results) == 9  # 9 checks
        # Valid record should have no critical/error issues
        for issues in results.values():
            assert all(i.severity not in ("CRITICAL", "ERROR") for i in issues)


# ─── fix_data_quality tests ──────────────────────────────────────────────────


class TestParameterLeakFix:
    def test_removes_metadata_metric(self, valid_record):
        valid_record["evaluation_results"].append({
            "evaluation_name": "Model Parameters",
            "source_data": {"dataset_name": "Params"},
            "metric_config": {"metric_name": "Parameters (Billions)", "max_score": None},
            "score_details": {"score": 7.0},
            "generation_config": {"generation_args": {}},
        })
        fix = ParameterLeakFix()
        stats = FixStats()
        result, modified = fix.apply(valid_record, "test", Path("/fake"), stats)
        assert modified is True
        assert len(result["evaluation_results"]) == 1
        assert stats["param_leaks_removed"] == 1

    def test_removes_score_above_max(self, valid_record):
        valid_record["evaluation_results"][0]["score_details"]["score"] = 500.0
        valid_record["evaluation_results"][0]["metric_config"]["max_score"] = 100.0
        fix = ParameterLeakFix()
        stats = FixStats()
        result, modified = fix.apply(valid_record, "test", Path("/fake"), stats)
        assert modified is True
        assert len(result["evaluation_results"]) == 0


class TestBenchmarkCanonFix:
    def test_canonicalizes_name(self, valid_record):
        valid_record["evaluation_results"][0]["evaluation_name"] = "arc_challenge"
        valid_record["evaluation_results"][0]["source_data"]["dataset_name"] = "arc_challenge"
        fix = BenchmarkCanonFix()
        stats = FixStats()
        result, modified = fix.apply(valid_record, "test", Path("/fake"), stats)
        assert modified is True
        assert result["evaluation_results"][0]["evaluation_name"] == "ARC-Challenge"
        assert result["evaluation_results"][0]["source_data"]["dataset_name"] == "ARC-Challenge"
        assert stats["benchmarks_canonicalized"] == 1

    def test_no_change_for_canonical_name(self, valid_record):
        fix = BenchmarkCanonFix()
        stats = FixStats()
        result, modified = fix.apply(valid_record, "test", Path("/fake"), stats)
        assert modified is False


class TestUtilityFunctions:
    @pytest.mark.parametrize("name,expected", [
        ("Parameters (Billions)", True),
        ("Model Size", True),
        ("accuracy", False),
        ("FLOPs", True),
        ("training tokens", True),
        ("throughput", True),
        ("HumanEval", False),
    ])
    def test_is_metadata_metric(self, name, expected):
        assert _is_metadata_metric(name) == expected

    @pytest.mark.parametrize("score,min_s,max_s,expected", [
        (150.0, 0.0, 100.0, True),
        (75.0, 0.0, 100.0, False),
        (2.0, 0.0, 1.0, True),
        (0.8, 0.0, 1.0, False),
    ])
    def test_is_score_out_of_range(self, score, min_s, max_s, expected):
        assert _is_score_out_of_range(score, min_s, max_s) == expected
