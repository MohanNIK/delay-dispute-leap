# -*- coding: utf-8 -*-
"""Smoke tests for the true LLM Copilot helper functions.

These tests avoid network calls. They verify the audit-critical behavior that
the implementation depends on: robust JSON parsing, exact pre-decision span
validation, responsibility folding, and metric recomputation from predictions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_true_llm_copilot import (  # noqa: E402
    LABELS,
    extract_json_object,
    fold_responsibility,
    normalize_llm_record,
    recompute_outcome_metrics,
    validate_evidence_chain,
)


def test_extract_json_object_accepts_fenced_output() -> None:
    raw = "```json\n{\"outcome_label\":\"support\",\"outcome_confidence\":0.8}\n```"
    parsed = extract_json_object(raw)
    assert parsed["outcome_label"] == "support"
    assert parsed["outcome_confidence"] == 0.8


def test_validate_evidence_chain_requires_exact_pre_decision_span() -> None:
    pre_text = "合同约定工期为60天。承包人未按期施工导致延误。"
    chain = [
        {"role_label": "ENT", "span_text": "合同约定工期为60天。"},
        {"role_label": "CAU", "span_text": "不存在的裁判结论"},
    ]
    validated, metrics = validate_evidence_chain(chain, pre_text)
    assert validated[0]["span_start"] == 0
    assert validated[0]["pre_decision_flag"] == 1
    assert validated[1]["span_start"] == -1
    assert validated[1]["pre_decision_flag"] == 0
    assert metrics["valid_span_rate"] == 0.5
    assert metrics["pre_decision_span_rate"] == 0.5


def test_normalize_llm_record_never_invents_proxy_label() -> None:
    parsed = {
        "outcome_label": "部分支持",
        "primary_responsible_party": "业主",
        "evidence_chain": [{"role_label": "NOT", "span_text": "已提交签证。"}],
    }
    record = normalize_llm_record("case-x", parsed, "已提交签证。")
    assert record["outcome_label"] == "partial"
    assert record["primary_responsible_party"] == "owner"
    assert record["api_status"] == "api_available"
    assert json.loads(record["evidence_chain_json"])[0]["pre_decision_flag"] == 1


def test_fold_responsibility_schema() -> None:
    assert fold_responsibility("subcontractor") == "other_external"
    assert fold_responsibility("designer_supervisor") == "other_external"
    assert fold_responsibility("force_majeure_policy") == "other_external"
    assert fold_responsibility("both") == "shared_or_uncertain"
    assert fold_responsibility("unknown") == "shared_or_uncertain"


def test_recompute_metrics_uses_prediction_rows() -> None:
    rows = [
        {"y_true": "support", "y_pred": "support"},
        {"y_true": "partial", "y_pred": "not_support"},
        {"y_true": "not_support", "y_pred": "not_support"},
    ]
    metrics = recompute_outcome_metrics(rows, LABELS)
    assert metrics["accuracy"] == 2 / 3
    assert 0.0 < metrics["macro_f1"] < 1.0
    assert metrics["confusion_matrix"][0][0] == 1


if __name__ == "__main__":
    test_extract_json_object_accepts_fenced_output()
    test_validate_evidence_chain_requires_exact_pre_decision_span()
    test_normalize_llm_record_never_invents_proxy_label()
    test_fold_responsibility_schema()
    test_recompute_metrics_uses_prediction_rows()
    print("true_llm_copilot helper tests passed")
