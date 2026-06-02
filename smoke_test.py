import json
from pathlib import Path

from src.run_true_llm_copilot import (
    LABELS,
    extract_json_object,
    fold_responsibility,
    normalize_llm_record,
    recompute_outcome_metrics,
    validate_evidence_chain,
)


def main() -> None:
    raw = '```json\n{"outcome_label":"support","outcome_confidence":0.8}\n```'
    parsed = extract_json_object(raw)
    assert parsed["outcome_label"] == "support"

    pre_text = "The contract required 30 days. The contractor missed the deadline."
    chain = [
        {"role_label": "ENT", "span_text": "The contract required 30 days."},
        {"role_label": "CAU", "span_text": "Missing text from the record"},
    ]
    validated, metrics = validate_evidence_chain(chain, pre_text)
    assert validated[0]["pre_decision_flag"] == 1
    assert validated[1]["pre_decision_flag"] == 0
    assert metrics["valid_span_rate"] == 0.5

    record = normalize_llm_record(
        "toy-case",
        {
            "outcome_label": "partial_support",
            "primary_responsible_party": "owner",
            "evidence_chain": [{"role_label": "DOC", "span_text": "The contractor submitted a notice."}],
        },
        "The contractor submitted a notice.",
    )
    assert record["outcome_label"] == "partial"
    assert fold_responsibility("subcontractor") == "other_external"

    recomputed = recompute_outcome_metrics(
        [
            {"y_true": "support", "y_pred": "support"},
            {"y_true": "partial", "y_pred": "not_support"},
            {"y_true": "not_support", "y_pred": "not_support"},
        ],
        LABELS,
    )
    assert recomputed["accuracy"] == 2 / 3

    Path("examples").mkdir(exist_ok=True)
    Path("examples/smoke_test_summary.json").write_text(
        json.dumps(
            {
                "parsed": parsed,
                "record_outcome_label": record["outcome_label"],
                "metrics_accuracy": recomputed["accuracy"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("delay-dispute-leap smoke test passed")


if __name__ == "__main__":
    main()
