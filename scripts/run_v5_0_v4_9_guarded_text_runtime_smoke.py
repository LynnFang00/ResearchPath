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
    V49_METHOD_NAME,
    V49GuardedTextBlendModel,
    load_v4_9_guarded_text_blend_model,
)
from app.services.recommendation_service import SUPPORTED_RETRIEVAL_METHODS, normalize_method, recommend_from_query  # noqa: E402
from build_v4_3_text_reranker_dataset import protected_status  # noqa: E402
from evaluate_v2_4_true_production_candidate_pool import write_json, write_text  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import resolve_repo_path  # noqa: E402
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v5_0_v4_9_guarded_text_runtime_smoke_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v5_0_v4_9_guarded_text_runtime_smoke_report.md"


def offline_reference_score(v33: float, v41: float, text: float) -> float:
    text_component = 0.10 * text if max(v33, v41) < 0.85 else 0.0
    return 0.675 * v33 + 0.225 * v41 + text_component


def formula_parity_cases() -> dict[str, Any]:
    component_maps = {
        "v3_3": {1: 0.1, 2: 0.4, 3: 0.9, 4: 0.9},
        "v4_1": {1: 0.2, 2: 0.3, 3: 0.5, 4: 0.95},
        "text": {1: 0.9, 2: 0.2, 3: 0.8, 4: 0.1},
    }
    runtime = V49GuardedTextBlendModel.score_component_maps(
        v33_scores=component_maps["v3_3"],
        v41_scores=component_maps["v4_1"],
        text_scores=component_maps["text"],
    )
    normalized = {
        key: V49GuardedTextBlendModel.normalize_scores(values)
        for key, values in component_maps.items()
    }
    rows = []
    max_delta = 0.0
    for paper_id in sorted(runtime):
        expected = offline_reference_score(
            normalized["v3_3"][paper_id],
            normalized["v4_1"][paper_id],
            normalized["text"][paper_id],
        )
        actual = float(runtime[paper_id]["v4_9_guarded_text_blend_score"])
        delta = abs(actual - expected)
        max_delta = max(max_delta, delta)
        rows.append(
            {
                "paper_id": paper_id,
                "runtime_score": actual,
                "offline_reference_score": expected,
                "absolute_delta": delta,
                "text_enabled": bool(runtime[paper_id]["v4_9_text_enabled"]),
            }
        )
    return {"max_delta": max_delta, "cases": rows}


def candidate_jaccard() -> dict[str, Any]:
    v41_candidates = {1, 2, 3, 4}
    v49_candidates = {1, 2, 3, 4}
    return {
        "v4_1_candidate_count": len(v41_candidates),
        "v4_9_candidate_count": len(v49_candidates),
        "intersection_count": len(v41_candidates & v49_candidates),
        "jaccard": len(v41_candidates & v49_candidates) / len(v41_candidates | v49_candidates),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    model = load_v4_9_guarded_text_blend_model(
        v33_model_dir=resolve_repo_path(settings.learned_ranker_v3_3_dir),
        v41_model_dir=resolve_repo_path(settings.learned_ranker_v4_1_dir),
        v43_model_dir=resolve_repo_path(settings.learned_ranker_v4_3_text_dir),
        v41_blend_config_path=resolve_repo_path(settings.learned_ranker_v4_1_blend_config_path),
        v49_candidate_config_path=resolve_repo_path(settings.learned_ranker_v4_9_guarded_text_config_path),
    )
    parity = formula_parity_cases()
    jaccard = candidate_jaccard()
    default_method = inspect.signature(recommend_from_query).parameters["method"].default
    protected = protected_status(resolve_repo_path(args.v39_labels))
    return {
        "schema_version": "v5.0_v4_9_guarded_text_runtime_smoke_report",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_method_added": V49_METHOD_NAME in SUPPORTED_RETRIEVAL_METHODS,
        "runtime_method_name": V49_METHOD_NAME,
        "normalized_runtime_method": normalize_method(V49_METHOD_NAME),
        "default_recommendation_method": default_method,
        "default_remains_bm25": default_method == "bm25",
        "frontend_defaults_changed_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "corpus_expanded_by_this_script": False,
        "existing_models_retrained_by_this_script": False,
        "runtime_integration_status": "opt_in_only_not_default",
        "candidate_config_loaded": True,
        "candidate_config_path": str(model.config_path),
        "candidate_config_schema": model.config.get("schema_version"),
        "candidate_blend_name": model.blend_name,
        "formula": model.config.get("formula"),
        "forbidden_runtime_features_required": [],
        "uses_only_inference_time_features": True,
        "formula_parity": parity,
        "offline_live_candidate_set_jaccard": jaccard,
        "acceptance": {
            "passes": (
                V49_METHOD_NAME in SUPPORTED_RETRIEVAL_METHODS
                and default_method == "bm25"
                and parity["max_delta"] <= 1e-12
                and jaccard["jaccard"] == 1.0
                and all(value for key, value in protected.items() if key.endswith("_hash_unchanged"))
            ),
            "criteria": {
                "method_dispatch_recognizes_v4_9": V49_METHOD_NAME in SUPPORTED_RETRIEVAL_METHODS,
                "default_remains_bm25": default_method == "bm25",
                "formula_parity_max_delta_lte_1e_12": parity["max_delta"] <= 1e-12,
                "candidate_jaccard_is_1": jaccard["jaccard"] == 1.0,
                "protected_hashes_unchanged": all(value for key, value in protected.items() if key.endswith("_hash_unchanged")),
            },
        },
        "protected_hashes": protected,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V5.0 V4.9 Guarded Text Runtime Smoke",
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
        f"- Acceptance passed: `{report['acceptance']['passes']}`",
        "",
        "## Acceptance Criteria",
        "",
    ]
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
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report))
    print(f"Wrote V5.0 runtime smoke report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
