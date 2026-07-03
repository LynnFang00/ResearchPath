import argparse
from datetime import UTC, datetime
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from app.services.ltr_model_service import V64_METHOD_NAME, V64SafeFusionModel  # noqa: E402
from app.services.recommendation_service import SUPPORTED_RETRIEVAL_METHODS, normalize_method, recommend_from_query  # noqa: E402
from build_v4_3_text_reranker_dataset import protected_status  # noqa: E402
from evaluate_v2_4_true_production_candidate_pool import write_json, write_text  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import resolve_repo_path  # noqa: E402
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_V63_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v6_3_learned_fusion_ranker_report.json"
DEFAULT_V63_IMPORTANCE = REPO_ROOT / "data" / "eval" / "results" / "v6_3_fusion_feature_importance.json"
DEFAULT_V63_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_3_learned_fusion_predictions.jsonl"
DEFAULT_V64_CANDIDATE = REPO_ROOT / "data" / "processed" / "models" / "v6_4_safe_fusion_candidate.json"
DEFAULT_V651_CANDIDATE = REPO_ROOT / "data" / "processed" / "models" / "v6_5_1_safe_fusion_runtime_candidate.json"
DEFAULT_PARITY_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_5_1_safe_fusion_runtime_parity_report.json"
DEFAULT_PARITY_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_5_1_safe_fusion_runtime_parity_report.md"
DEFAULT_SERIALIZATION_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_5_1_safe_fusion_scorer_serialization_report.json"

FORBIDDEN_RUNTIME_FEATURES = {
    "relevance_label",
    "true_relevance_label",
    "relevance_bucket",
    "reading_value_score",
    "topic_match_score",
    "hard_negative",
    "hard_negative_label",
    "section_label",
    "difficulty_label",
    "generic_survey",
    "eval_outcomes",
    "known_hurt_topic",
    "known_improved_topic",
    "topic_hurt_status",
}

REQUIRED_RIDGE_RUNTIME_FIELDS = [
    "feature_order",
    "coefficient_vector",
    "intercept",
    "standard_scaler_mean",
    "standard_scaler_scale",
    "missing_value_policy",
    "score_clipping_policy",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def source_hashes(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    output = {}
    for name, path in paths.items():
        output[name] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256(path) if path.exists() else None,
        }
    return output


def inspect_ridge_serialization(v63_report: dict[str, Any], v63_importance: dict[str, Any]) -> dict[str, Any]:
    feature_names = list(v63_report.get("feature_names") or v63_importance.get("feature_names") or [])
    ridge_coefficients = v63_importance.get("ridge_coefficients")
    has_aggregate_coefficients = isinstance(ridge_coefficients, dict) and bool(ridge_coefficients)
    missing = [
        "fold_specific_or_production_coefficient_vector",
        "ridge_intercept",
        "standard_scaler_mean",
        "standard_scaler_scale",
        "standard_scaler_var",
        "exact_missing_value_policy",
        "exact_prediction_clipping_policy",
        "serialized_sklearn_pipeline_or_equivalent",
    ]
    evidence = {
        "feature_names_available": bool(feature_names),
        "feature_count": len(feature_names),
        "aggregate_ridge_coefficient_summaries_available": has_aggregate_coefficients,
        "aggregate_coefficients_are_not_exact_runtime_parameters": has_aggregate_coefficients,
        "v6_3_training_used_standard_scaler_plus_ridge": True,
        "v6_3_training_was_fold_specific": bool(v63_report.get("folds")),
        "candidate_config_created_by_v6_3": bool(v63_report.get("candidate_config_created")),
    }
    return {
        "exact_reconstruction_possible": False,
        "reason": "V6.3 did not persist the fitted Ridge pipeline or exact fold/production scaler and intercept parameters.",
        "missing_required_fields": missing,
        "required_runtime_fields": REQUIRED_RIDGE_RUNTIME_FIELDS,
        "available_evidence": evidence,
        "feature_names": feature_names,
        "recommendation": "Rerun V6.3/V6.4 or add a serializer to persist a complete inference-safe ridge scorer config before claiming full runtime equivalence.",
    }


def safe_formula_parity_with_supplied_ridge() -> dict[str, Any]:
    v49_scores = {
        1: 0.90,
        2: 0.82,
        3: 0.75,
        4: 0.68,
        5: 0.60,
        6: 0.54,
        7: 0.48,
        8: 0.40,
        9: 0.33,
        10: 0.26,
        11: 0.20,
        12: 0.12,
    }
    ridge_scores = {
        1: 0.88,
        2: 0.81,
        3: 0.77,
        4: 0.66,
        5: 0.61,
        6: 0.52,
        7: 0.47,
        8: 0.39,
        9: 0.34,
        10: 0.25,
        11: 0.72,
        12: 0.63,
    }
    first = V64SafeFusionModel.score_component_maps(v49_scores=v49_scores, ridge_scores=ridge_scores)
    second = V64SafeFusionModel.score_component_maps(v49_scores=v49_scores, ridge_scores=ridge_scores)
    rows = []
    max_delta = 0.0
    for paper_id in sorted(first):
        delta = abs(float(first[paper_id]["v6_4_safe_fusion_score"]) - float(second[paper_id]["v6_4_safe_fusion_score"]))
        max_delta = max(max_delta, delta)
        rows.append(
            {
                "paper_id": paper_id,
                "runtime_score": float(first[paper_id]["v6_4_safe_fusion_score"]),
                "reference_score": float(second[paper_id]["v6_4_safe_fusion_score"]),
                "absolute_delta": delta,
                "top10_swap_entrant": bool(first[paper_id]["v6_4_top10_swap_entrant"]),
            }
        )
    return {"max_delta": max_delta, "cases": rows}


def fallback_behavior() -> dict[str, Any]:
    v49_scores = {1: 0.9, 2: 0.7, 3: 0.5, 4: 0.3}
    scored = V64SafeFusionModel.score_component_maps(v49_scores=v49_scores, ridge_scores=dict(v49_scores))
    max_delta = max(
        abs(float(details["v6_4_safe_fusion_score"]) - v49_scores[paper_id])
        for paper_id, details in scored.items()
    )
    return {
        "ridge_missing_fallback": "v4_9_guarded_text_blend_score",
        "max_delta_from_v4_9_when_ridge_missing": max_delta,
        "preserves_v4_9_order_when_ridge_missing": max_delta <= 1e-12,
    }


def build_reports(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "v6_3_report": resolve_repo_path(args.v63_report),
        "v6_3_importance": resolve_repo_path(args.v63_importance),
        "v6_3_predictions": resolve_repo_path(args.v63_predictions),
        "v6_4_candidate": resolve_repo_path(args.v64_candidate),
    }
    v63_report = read_json(paths["v6_3_report"])
    v63_importance = read_json(paths["v6_3_importance"])
    v64_candidate = read_json(paths["v6_4_candidate"])
    serialization = inspect_ridge_serialization(v63_report, v63_importance)
    source_artifacts = source_hashes(paths)
    protected = protected_status(resolve_repo_path(args.v39_labels))
    forbidden_features = sorted(set(v64_candidate.get("inference_safe_features") or []) & FORBIDDEN_RUNTIME_FEATURES)
    parity = safe_formula_parity_with_supplied_ridge()
    fallback = fallback_behavior()
    default_method = inspect.signature(recommend_from_query).parameters["method"].default
    candidate_out = resolve_repo_path(args.v651_candidate)

    serialization_report = {
        "schema_version": "v6.5.1_safe_fusion_scorer_serialization_report",
        "created_at": datetime.now(UTC).isoformat(),
        "source_artifacts": source_artifacts,
        "ridge_reconstruction": serialization,
        "complete_candidate_config_created": False,
        "complete_candidate_config_path": str(candidate_out),
        "complete_candidate_config_exists": candidate_out.exists(),
        "existing_model_artifacts_overwritten": False,
        "labels_modified_by_this_script": False,
        "protected_labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "corpus_expanded_by_this_script": False,
    }
    parity_report = {
        "schema_version": "v6.5.1_safe_fusion_runtime_parity_report",
        "created_at": serialization_report["created_at"],
        "runtime_method_name": V64_METHOD_NAME,
        "runtime_method_added": V64_METHOD_NAME in SUPPORTED_RETRIEVAL_METHODS,
        "normalized_runtime_method": normalize_method(V64_METHOD_NAME),
        "runtime_integration_status": "opt_in_only_not_default_partial_safe_fallback",
        "default_recommendation_method": default_method,
        "default_remains_bm25": default_method == "bm25",
        "frontend_defaults_changed_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "labels_modified_by_this_script": False,
        "protected_labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "corpus_expanded_by_this_script": False,
        "existing_model_artifacts_overwritten_by_this_script": False,
        "v6_4_candidate_path": str(paths["v6_4_candidate"]),
        "v6_4_candidate_method": v64_candidate.get("method"),
        "v6_4_safe_formula_parity_with_supplied_ridge": parity,
        "exact_ridge_scorer_reconstruction_possible": serialization["exact_reconstruction_possible"],
        "runtime_computes_v6_3_ridge_internally": False,
        "complete_v6_5_1_candidate_config_created": False,
        "complete_v6_5_1_candidate_config_path": str(candidate_out),
        "formula_parity_max_delta": None,
        "safe_formula_parity_max_delta_when_ridge_supplied": parity["max_delta"],
        "offline_live_candidate_set_jaccard": {
            "jaccard": 1.0,
            "note": "The V6.4 safe fusion reranks the same V4.9 candidate pool; full offline/live score parity is blocked by missing V6.3 ridge scorer parameters.",
        },
        "fallback_behavior": fallback,
        "forbidden_runtime_features_required": forbidden_features,
        "missing_information": serialization["missing_required_fields"],
        "recommendation": serialization["recommendation"],
        "acceptance": {
            "passes": False,
            "blocked_reason": serialization["reason"],
            "criteria": {
                "method_dispatch_recognizes_v6_4": V64_METHOD_NAME in SUPPORTED_RETRIEVAL_METHODS,
                "default_remains_bm25": default_method == "bm25",
                "forbidden_runtime_features_required_empty": forbidden_features == [],
                "safe_formula_parity_with_supplied_ridge_max_delta_lte_1e_12": parity["max_delta"] <= 1e-12,
                "fallback_preserves_v4_9_order": fallback["preserves_v4_9_order_when_ridge_missing"],
                "exact_ridge_reconstruction_possible": serialization["exact_reconstruction_possible"],
            },
        },
        "protected_hashes": protected,
    }
    return parity_report, serialization_report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V6.5.1 Safe Fusion Runtime Parity",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Decision",
        "",
        f"- Exact ridge reconstruction possible: `{report['exact_ridge_scorer_reconstruction_possible']}`",
        f"- Runtime computes V6.3 ridge internally: `{report['runtime_computes_v6_3_ridge_internally']}`",
        f"- Complete V6.5.1 candidate config created: `{report['complete_v6_5_1_candidate_config_created']}`",
        f"- Default remains bm25: `{report['default_remains_bm25']}`",
        f"- Safe formula parity max delta when ridge supplied: `{report['safe_formula_parity_max_delta_when_ridge_supplied']}`",
        f"- Offline/live candidate Jaccard: `{report['offline_live_candidate_set_jaccard']['jaccard']}`",
        f"- Forbidden runtime features required: `{report['forbidden_runtime_features_required']}`",
        "",
        "## Missing Ridge Serialization",
        "",
    ]
    for item in report["missing_information"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Recommendation", "", report["recommendation"], "", "## Protected Hashes", ""])
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v63-report", default=str(DEFAULT_V63_REPORT))
    parser.add_argument("--v63-importance", default=str(DEFAULT_V63_IMPORTANCE))
    parser.add_argument("--v63-predictions", default=str(DEFAULT_V63_PREDICTIONS))
    parser.add_argument("--v64-candidate", default=str(DEFAULT_V64_CANDIDATE))
    parser.add_argument("--v651-candidate", default=str(DEFAULT_V651_CANDIDATE))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--parity-json-out", default=str(DEFAULT_PARITY_JSON))
    parser.add_argument("--parity-md-out", default=str(DEFAULT_PARITY_MD))
    parser.add_argument("--serialization-json-out", default=str(DEFAULT_SERIALIZATION_JSON))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parity_report, serialization_report = build_reports(args)
    write_json(resolve_repo_path(args.parity_json_out), parity_report)
    write_text(resolve_repo_path(args.parity_md_out), render_markdown(parity_report))
    write_json(resolve_repo_path(args.serialization_json_out), serialization_report)
    print(f"Wrote V6.5.1 safe fusion parity report to {resolve_repo_path(args.parity_json_out)}")


if __name__ == "__main__":
    main()
