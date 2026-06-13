"""EEE validation package.

Public API:
    SchemaValidator  — validate JSON records against eval.schema.json
    AutoFixer        — heuristic repairs for common schema issues
    FileValidator    — orchestrate validate → fix → re-validate for one file
    BatchValidator   — validate all JSON files in a directory tree
    validate_record  — check extracted values against paper HTML
    AuditRunner      — run comprehensive quality checks
"""
from src.validation.schema import SchemaValidator, AutoFixer, FileValidator, BatchValidator
from src.validation.validate_extractions import validate_record
from src.validation.quality_audit import AuditRunner, RecordLoader

__all__ = [
    "SchemaValidator",
    "AutoFixer",
    "FileValidator",
    "BatchValidator",
    "validate_record",
    "AuditRunner",
    "RecordLoader",
]
