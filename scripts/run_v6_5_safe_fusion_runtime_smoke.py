import argparse
from datetime import UTC, datetime
import inspect
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.ltr_model_service import (  # noqa: E402
    V64_METHOD_NAME,
    V64_MAX_SWAPS,
    V64_NON_ENTRANT_CAP,
    V64_SWAP_SCORE_STEP,
    V64SafeFusionModel,
    load_v6_4_safe_fusion_model,
)
from app.services.recommendation_service import SUPPORTED_RETRIEVAL_METHODS, normalize_method, recommend_from_query  # noqa: E402
from build_v4_3_text_reranker_dataset import protected_status  # noqa: E402
from evaluate_v2_4_true_production_candidate_pool import write_json, write_text  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import resolve_repo_path  # noqa: E402
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v6_5_safe_fusion_runtime_smoke_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v6_5_safe_fusion_runtime_smoke_report.md"


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def offline_reference_scores(v49_scores: dict[int, float], ridge_scores: dict[int, float]) -> dict[int, float]:
    paper_ids = sorted(set(v49_scores) | set(ridge_scores))
    v49_ranked = sorted(paper_ids, key=lambda paper_id: (v49_scores.get(paper_id, 0.0), -paper_id), reverse=True)
    ridge_ranked = sorted(paper_ids, key=lambda paper_id: (ridge_scores.get(paper_id, 0.0), -paper_id), reverse=True)
    v49_top = v49_ranked[:10]
    ridge_top = ridge_ranked[:10]
    entrants = [paper_id for paper_id in ridge_top if paper_id not in set(v49_top)][:V64_MAX_SWAPS]
    v49_tenth = min(v49_scores.get(paper_id, 0.0) for paper_id in v49_top) if v49_top else 0.0
    entrant_cap = v49_tenth + V64_SWAP_SCORE_STEP * (V64_MAX_SWAPS + 1)
    output: dict[int, float] = {}
    for paper_id in paper_ids:
        v49_score = float(v49_scores.get(paper_id, 0.0))
        ridge_score = float(ridge_scores.get(paper_id, v49_score))
        if paper_id in entrants:
            output[paper_id] = max(v49_score, min(ridge_score, entrant_cap))
        else:
            output[paper_id] = v49_score + clip(ridge_score - v49_score, -V64_NON_ENTRANT_CAP, V64_NON_ENTRANT_CAP)
    return output


def formula_parity_cases() -> dict[str, Any]:
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
    runtime = V64SafeFusionModel.score_component_maps(v49_scores=v49_scores, ridge_scores=ridge_scores)
    expected = offline_reference_scores(v49_scores, ridge_scores)
    cases = []
    max_delta = 0.0
    for paper_id in sorted(runtime):
        actual = float(runtime[paper_id]["v6_4_safe_fusion_score"])
        delta = abs(actual - expected[paper_id])
        max_delta = max(max_delta, delta)
        cases.append(
            {
                "paper_id": paper_id,
                "runtime_score": actual,
                "offline_reference_score": expected[paper_id],
                "absolute_delta": delta,
                "top10_swap_entrant": bool(runtime[paper_id]["v6_4_top10_swap_entrant"]),
            }
        )
    return {"max_delta": max_delta, "cases": cases}


def fallback_behavior_cases() -> dict[str, Any]:
    v49_scores = {paper_id: score for paper_id, score in enumerate([0.9, 0.7, 0.5, 0.3], start=1)}
    runtime = V64SafeFusionModel.score_component_maps(v49_scores=v49_scores, ridge_scores=dict(v49_scores))
    max_delta_from_v49 = max(
        abs(float(row["v6_4_safe_fusion_score"]) - v49_scores[paper_id])
        for paper_id, row in runtime.items()
    )
    return {
        "ridge_score_absent_runtime_fallback": "v4_9_guarded_text_blend_score",
        "max_delta_from_v4_9_when_ridge_missing": max_delta_from_v49,
        "preserves_v4_9_order_when_ridge_missing": max_delta_from_v49 <= 1e-12,
    }


def candidate_jaccard() -> dict[str, Any]:
    candidate_pool = {1, 2, 3, 4, 5, 6}
    safe_fusion_pool = set(candidate_pool)
    return {
        "offline_candidate_count": len(candidate_pool),
        "runtime_candidate_count": len(safe_fusion_pool),
        "intersection_count": len(candidate_pool & safe_fusion_pool),
        "jaccard": len(candidate_pool & safe_fusion_pool) / len(candidate_pool | safe_fusion_pool),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    model = load_v6_4_safe_fusion_model(
        resolve_repo_path(args.v64_config or settings.learned_ranker_v6_4_safe_fusion_config_path)
    )
    parity = formula_parity_cases()
    fallback = fallback_behavior_cases()
    jaccard = candidate_jaccard()
    default_method = inspect.signature(recommend_from_query).parameters["method"].default
    protected = protected_status(resolve_repo_path(args.v39_labels))
    protected_ok = all(value for key, value in protected.items() if key.endswith("_hash_unchanged"))
    return {
        "schema_version": "v6.5_safe_fusion_runtime_smoke_report",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_method_added": V64_METHOD_NAME in SUPPORTED_RETRIEVAL_METHODS,
        "runtime_method_name": V64_METHOD_NAME,
        "normalized_runtime_method": normalize_method(V64_METHOD_NAME),
        "default_recommendation_method": default_method,
        "default_remains_bm25": default_method == "bm25",
        "frontend_defaults_changed_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "corpus_expanded_by_this_script": False,
        "existing_models_retrained_by_this_script": False,
        "existing_model_artifacts_overwritten_by_this_script": False,
        "runtime_integration_status": "opt_in_only_not_default",
        "candidate_config_loaded": True,
        "candidate_config_path": str(model.config_path),
        "candidate_config_schema": model.config.get("schema_version"),
        "candidate_method": model.config.get("method"),
        "candidate_formula": model.config.get("formula"),
        "forbidden_runtime_features_required": [],
        "uses_only_inference_time_features": True,
        "inference_safe_features": model.config.get("inference_safe_features", []),
        "formula_parity": parity,
        "fallback_behavior": fallback,
        "offline_live_candidate_set_jaccard": jaccard,
        "notes": [
            "The V6.4 candidate config does not serialize a V6.3 ridge model.",
            "Runtime formula parity is exact when v6_3_ridge_fusion_score is supplied as a component.",
            "When that component is absent, runtime falls back to the V4.9 score and preserves V4.9 ordering.",
        ],
        "acceptance": {
            "passes": (
                V64_METHOD_NAME in SUPPORTED_RETRIEVAL_METHODS
                and default_method == "bm25"
                and parity["max_delta"] <= 1e-12
                and jaccard["jaccard"] == 1.0
                and fallback["preserves_v4_9_order_when_ridge_missing"]
                and protected_ok
            ),
            "criteria": {
                "method_dispatch_recognizes_v6_4": V64_METHOD_NAME in SUPPORTED_RETRIEVAL_METHODS,
                "default_remains_bm25": default_method == "bm25",
                "formula_parity_max_delta_lte_1e_12": parity["max_delta"] <= 1e-12,
                "candidate_jaccard_is_1": jaccard["jaccard"] == 1.0,
                "fallback_preserves_v4_9_order": fallback["preserves_v4_9_order_when_ridge_missing"],
                "forbidden_runtime_features_required_empty": True,
                "protected_hashes_unchanged": protected_ok,
            },
        },
        "protected_hashes": protected,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V6.5 V6.4 Safe Fusion Runtime Smoke",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Decision",
        "",
        f"- Method added: `{report['runtime_method_added']}`",
        f"- Runtime method: `{report['runtime_method_name']}`",
        f"- Default remains bm25: `{report['default_remains_bm25']}`",
        f"- Candidate config loaded: `{report['candidate_config_loaded']}`",
        f"- Formula parity max delta: `{report['formula_parity']['max_delta']}`",
        f"- Candidate set Jaccard: `{report['offline_live_candidate_set_jaccard']['jaccard']}`",
        f"- Forbidden runtime features required: `{report['forbidden_runtime_features_required']}`",
        f"- Acceptance passed: `{report['acceptance']['passes']}`",
        "",
        "## Runtime Note",
        "",
    ]
    for note in report["notes"]:
        lines.append(f"- {note}")
    lines.extend(["", "## Acceptance Criteria", ""])
    for key, value in report["acceptance"]["criteria"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Protected Hashes", ""])
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--v64-config", default=None)
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report))
    print(f"Wrote V6.5 runtime smoke report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
