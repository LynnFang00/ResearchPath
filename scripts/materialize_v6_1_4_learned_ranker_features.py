import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from build_v4_3_text_reranker_dataset import protected_status  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import resolve_repo_path, write_json, write_text  # noqa: E402
from evaluate_v6_1_3_exact_learned_baselines import (  # noqa: E402
    LEARNED_METHODS,
    METHODS,
    NEURAL_METHOD,
    load_artifacts,
    load_jsonl,
    score_field,
    write_jsonl,
)
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_EXAMPLES = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_neural_reranker_examples.jsonl"
DEFAULT_V612_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_1_2_neural_cv_predictions.jsonl"
DEFAULT_V33_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v3_3_ltr_800"
DEFAULT_V41_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v4_1_weighted_ltr_2400"
DEFAULT_V43_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v4_3_text_reranker"
DEFAULT_V41_BLEND_CONFIG = REPO_ROOT / "data" / "processed" / "models" / "v4_1_calibrated_blend.json"
DEFAULT_V49_CONFIG = REPO_ROOT / "data" / "processed" / "models" / "v4_9_guarded_text_blend_candidate.json"
DEFAULT_RUNTIME_DB = REPO_ROOT / "backend" / "data" / "researchpath.db"
DEFAULT_RAW_CORPUS = REPO_ROOT / "data" / "raw" / "arxiv_ml_ai_50k_incremental.jsonl"

DEFAULT_FEATURE_MATRIX = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_1_4_learned_ranker_feature_matrix.jsonl"
DEFAULT_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_1_4_learned_baseline_predictions.jsonl"
DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_1_4_learned_ranker_feature_materialization_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_1_4_learned_ranker_feature_materialization_report.md"
DEFAULT_PARITY = REPO_ROOT / "data" / "eval" / "results" / "v6_1_4_feature_parity_report.json"

FORBIDDEN_FEATURES = {
    "relevance_bucket",
    "classification_target",
    "reading_value_score",
    "topic_match_score",
    "regression_target_reading",
    "regression_target_topic",
    "section_label",
    "section_scores",
    "difficulty_label",
    "difficulty_scores",
    "hard_negative",
    "generic_survey",
    "eval_outcomes",
    "topic_hurt_status",
    "topic_improved_status",
}

FEATURE_CATEGORIES = {
    "bm25_score": "bm25",
    "bm25_rank": "bm25",
    "bm25_reciprocal_rank": "bm25",
    "tfidf_score": "tfidf",
    "tfidf_rank": "tfidf",
    "tfidf_reciprocal_rank": "tfidf",
    "embedding_score": "dense",
    "embedding_rank": "dense",
    "embedding_reciprocal_rank": "dense",
    "faiss_embedding_score": "faiss",
    "faiss_embedding_rank": "faiss",
    "faiss_embedding_reciprocal_rank": "faiss",
    "hybrid_score": "hybrid",
    "hybrid_rank": "hybrid",
    "hybrid_reciprocal_rank": "hybrid",
    "old_v2_2b_score": "v2_2b",
    "v2_6_score": "v2_6",
    "year": "metadata",
    "paper_age_years": "metadata",
    "citation_count_log1p": "citation",
    "full_text_available_bool": "full_text",
}


def feature_hash(feature_values: dict[str, float], missing: list[str]) -> str:
    payload = {"feature_values": feature_values, "missing_feature_names": missing}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def nested_baseline_value(row: dict[str, Any], method: str, key: str) -> float | None:
    value = ((row.get("baseline_rankings") or {}).get(method) or {}).get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(examples_path: Path, predictions_path: Path) -> list[dict[str, Any]]:
    examples = {
        (row["query_id"], int(row["paper_id"])): row
        for row in load_jsonl(examples_path)
    }
    output = []
    for pred in load_jsonl(predictions_path):
        key = (pred["query_id"], int(pred["paper_id"]))
        row = dict(examples[key])
        row["fold_id"] = pred["fold_id"]
        row["split"] = pred["split"]
        row[f"{NEURAL_METHOD}_score_for_eval"] = float(pred["scores"][NEURAL_METHOD])
        output.append(row)
    return sorted(output, key=lambda row: (row["fold_id"], row["split"], row["query_id"], int(row["paper_id"])))


def required_features(artifacts: dict[str, Any], artifact_status: dict[str, Any]) -> dict[str, list[str]]:
    required: dict[str, list[str]] = {}
    if artifact_status.get("v3_3_ltr", {}).get("artifact_loaded"):
        required["v3_3_ltr"] = list(artifacts["v3_3_ltr"].feature_columns)
    else:
        required["v3_3_ltr"] = []
    if artifact_status.get("v4_1_blend", {}).get("artifact_loaded"):
        required["v4_1_blend"] = list(artifacts["v4_1_blend"].v41_model.feature_columns)
    else:
        required["v4_1_blend"] = []
    if artifact_status.get("v4_9_guarded_text_blend", {}).get("artifact_loaded"):
        model = artifacts["v4_9_guarded_text_blend"]
        required["v4_9_guarded_text_blend"] = sorted(
            set(model.v41_blend_model.v41_model.feature_columns) | set(model.text_model.numeric_ranker_features)
        )
    else:
        required["v4_9_guarded_text_blend"] = []
    return required


def runtime_db_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "paper_count": 0, "usable_for_exact_runtime_features": False}
    try:
        with sqlite3.connect(path) as conn:
            count = int(conn.execute("select count(*) from papers").fetchone()[0])
            columns = [row[1] for row in conn.execute("pragma table_info(papers)").fetchall()]
    except Exception as exc:
        return {"path": str(path), "exists": True, "paper_count": 0, "usable_for_exact_runtime_features": False, "error": str(exc)}
    return {
        "path": str(path),
        "exists": True,
        "paper_count": count,
        "columns": columns,
        "usable_for_exact_runtime_features": count >= 50000,
        "blocker": None if count >= 50000 else f"Runtime SQLite corpus has {count} rows, not the current ~50K paper corpus.",
    }


def raw_corpus_alignment_status(path: Path, examples: list[dict[str, Any]]) -> dict[str, Any]:
    sample_ids = [1, 2, 5053, 30046]
    example_title_by_id = {int(row["paper_id"]): row["title"] for row in examples if int(row["paper_id"]) in sample_ids}
    raw_title_by_id: dict[int, str] = {}
    if not path.exists():
        return {"path": str(path), "exists": False, "usable_for_exact_runtime_features": False}
    needed = set(example_title_by_id)
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if index in needed:
                raw_title_by_id[index] = str(json.loads(line).get("title") or "")
            if len(raw_title_by_id) == len(needed):
                break
    mismatches = [
        {"paper_id": paper_id, "example_title": example_title_by_id[paper_id], "raw_row_title": raw_title_by_id.get(paper_id)}
        for paper_id in sorted(example_title_by_id)
        if raw_title_by_id.get(paper_id) != example_title_by_id[paper_id]
    ]
    return {
        "path": str(path),
        "exists": True,
        "sample_ids_checked": sorted(example_title_by_id),
        "mismatches": mismatches,
        "usable_for_exact_runtime_features": not mismatches,
        "blocker": None if not mismatches else "Raw JSONL row number does not align with ResearchPath paper_id.",
    }


def materialize_available_features(row: dict[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[str, float] = {}
    sources: dict[str, str] = {}
    bm25_score = nested_baseline_value(row, "bm25", "score")
    bm25_rank = nested_baseline_value(row, "bm25", "rank")
    if bm25_score is not None:
        values["bm25_score"] = bm25_score
        sources["bm25_score"] = "v6_0_2_baseline_rankings"
    if bm25_rank is not None and bm25_rank > 0:
        values["bm25_rank"] = bm25_rank
        values["bm25_reciprocal_rank"] = 1.0 / bm25_rank
        sources["bm25_rank"] = "v6_0_2_baseline_rankings"
        sources["bm25_reciprocal_rank"] = "derived_from_v6_0_2_bm25_rank"
    if row.get("year") is not None:
        values["year"] = float(row["year"])
        values["paper_age_years"] = float(max(0, 2026 - int(row["year"])))
        sources["year"] = "v6_0_2_example_metadata"
        sources["paper_age_years"] = "derived_from_year_current_year_2026"
    citation_count = row.get("citation_count")
    if citation_count is not None:
        values["citation_count_log1p"] = math.log1p(max(float(citation_count), 0.0))
        sources["citation_count_log1p"] = "v6_0_2_example_metadata"
    full_text_available = row.get("full_text_available")
    if full_text_available is not None:
        values["full_text_available_bool"] = 1.0 if full_text_available else 0.0
        sources["full_text_available_bool"] = "v6_0_2_example_metadata"
    return values, sources


def build_feature_matrix(rows: list[dict[str, Any]], required_by_method: dict[str, list[str]]) -> list[dict[str, Any]]:
    required_union = sorted({feature for features in required_by_method.values() for feature in features})
    output = []
    for row in rows:
        values, sources = materialize_available_features(row)
        missing = [feature for feature in required_union if feature not in values]
        row_missing_by_method = {
            method: [feature for feature in features if feature not in values]
            for method, features in required_by_method.items()
        }
        output.append(
            {
                "schema_version": "v6.1.4_learned_ranker_feature_row",
                "query_id": row["query_id"],
                "query": row["query"],
                "paper_id": int(row["paper_id"]),
                "title": row["title"],
                "fold_id": row["fold_id"],
                "split": row["split"],
                "feature_values": values,
                "missing_feature_names": missing,
                "missing_features_by_method": row_missing_by_method,
                "feature_source": sources,
                "feature_hash": feature_hash(values, missing),
                "forbidden_features_used": [],
                "exact_runtime_feature_complete": not missing,
            }
        )
    return output


def score_rows(
    matrix_rows: list[dict[str, Any]],
    artifacts: dict[str, Any],
    artifact_status: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = []
    coverage: dict[str, Any] = {}
    by_method_scorable: dict[str, list[dict[str, Any]]] = {}
    for method in LEARNED_METHODS:
        method_rows = [
            row for row in matrix_rows
            if not row["missing_features_by_method"].get(method)
            and artifact_status.get(method, {}).get("artifact_loaded")
        ]
        by_method_scorable[method] = method_rows
        missing_counts = Counter(
            feature
            for row in matrix_rows
            for feature in row["missing_features_by_method"].get(method, [])
        )
        coverage[method] = {
            "rows_scored": len(method_rows),
            "rows_missing": len(matrix_rows) - len(method_rows),
            "coverage": len(method_rows) / len(matrix_rows) if matrix_rows else 0.0,
            "exact_scoring_available": len(method_rows) == len(matrix_rows) and bool(matrix_rows),
            "missing_feature_counts": dict(sorted(missing_counts.items())),
            "missing_feature_categories": category_counts(missing_counts),
            "rows_silently_dropped": 0,
        }

    score_maps: dict[str, dict[tuple[str, str, str, int], float]] = {method: {} for method in LEARNED_METHODS}
    if by_method_scorable["v3_3_ltr"]:
        model = artifacts["v3_3_ltr"]
        scores = model.score_feature_rows([row["feature_values"] for row in by_method_scorable["v3_3_ltr"]])
        for row, score in zip(by_method_scorable["v3_3_ltr"], scores, strict=True):
            score_maps["v3_3_ltr"][row_key(row)] = float(score)
    if by_method_scorable["v4_1_blend"]:
        model = artifacts["v4_1_blend"]
        scores = model.score_feature_rows([row["feature_values"] for row in by_method_scorable["v4_1_blend"]])
        for row, score_row in zip(by_method_scorable["v4_1_blend"], scores, strict=True):
            score_maps["v4_1_blend"][row_key(row)] = float(score_row["v4_1_blend_score"])
    if by_method_scorable["v4_9_guarded_text_blend"]:
        coverage["v4_9_guarded_text_blend"]["blocker"] = None
    else:
        coverage["v4_9_guarded_text_blend"]["blocker"] = "V4.9 requires complete V3.3/V4.1/V4.3 ranker feature inputs before guarded text scores can be normalized per query."

    for row in matrix_rows:
        key = row_key(row)
        missing_score_reasons = {}
        for method in LEARNED_METHODS:
            missing = row["missing_features_by_method"].get(method, [])
            missing_score_reasons[method] = None if not missing else f"Missing exact runtime features: {', '.join(missing[:12])}"
        predictions.append(
            {
                "schema_version": "v6.1.4_learned_baseline_prediction",
                "query_id": row["query_id"],
                "paper_id": row["paper_id"],
                "fold_id": row["fold_id"],
                "split": row["split"],
                "v3_3_score": score_maps["v3_3_ltr"].get(key),
                "v4_1_score": score_maps["v4_1_blend"].get(key),
                "v4_9_score": score_maps["v4_9_guarded_text_blend"].get(key),
                "missing_score_reasons": missing_score_reasons,
            }
        )
    return predictions, coverage


def row_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (str(row["fold_id"]), str(row["split"]), str(row["query_id"]), int(row["paper_id"]))


def category_counts(counter: Counter[str]) -> dict[str, int]:
    categories: Counter[str] = Counter()
    for feature, count in counter.items():
        categories[FEATURE_CATEGORIES.get(feature, "other")] += int(count)
    return dict(sorted(categories.items()))


def overall_feature_coverage(matrix_rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_features = sorted({feature for row in matrix_rows for feature in set(row["feature_values"]) | set(row["missing_feature_names"])})
    return {
        feature: {
            "present_rows": sum(1 for row in matrix_rows if feature in row["feature_values"]),
            "missing_rows": sum(1 for row in matrix_rows if feature in row["missing_feature_names"]),
            "category": FEATURE_CATEGORIES.get(feature, "other"),
        }
        for feature in all_features
    }


def parity_report(
    *,
    matrix_rows: list[dict[str, Any]],
    coverage: dict[str, Any],
    runtime_db: dict[str, Any],
    raw_alignment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "v6.1.4_feature_parity_report",
        "created_at": datetime.now(UTC).isoformat(),
        "offline_only": True,
        "runtime_scoring_invoked": False,
        "runtime_corpus_database": runtime_db,
        "raw_corpus_alignment": raw_alignment,
        "max_score_delta": {
            "v3_3_ltr": None,
            "v4_1_blend": None,
            "v4_9_guarded_text_blend": None,
        },
        "offline_live_candidate_set_jaccard": None,
        "parity_comparable": False,
        "parity_blocker": "Exact runtime corpus metadata is unavailable locally, and V6 rows do not contain complete runtime feature inputs.",
        "forbidden_features_required": [],
        "forbidden_features_used": [],
        "coverage": coverage,
        "deterministic_feature_hash_sample": [row["feature_hash"] for row in matrix_rows[:10]],
    }


def build_report(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
    coverage: dict[str, Any],
    artifact_status: dict[str, Any],
    runtime_db: dict[str, Any],
    raw_alignment: dict[str, Any],
    parity: dict[str, Any],
) -> dict[str, Any]:
    feature_coverage = overall_feature_coverage(matrix_rows)
    exact_success = all(coverage[method]["exact_scoring_available"] for method in LEARNED_METHODS)
    blockers = {
        "runtime_corpus_database": runtime_db.get("blocker"),
        "raw_corpus_alignment": raw_alignment.get("blocker"),
        "missing_feature_categories": {
            method: coverage[method]["missing_feature_categories"]
            for method in LEARNED_METHODS
        },
    }
    return {
        "schema_version": "v6.1.4_learned_ranker_feature_materialization_report",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_integration_status": "offline_only_not_integrated",
        "default_ranker_status": "unchanged_bm25",
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "protected_labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "models_retrained_by_this_script": False,
        "existing_model_artifacts_overwritten": False,
        "production_candidate_created": False,
        "dataset": {
            "row_count": len(rows),
            "topic_count": len({row["query_id"] for row in rows}),
            "fold_count": len({row["fold_id"] for row in rows}),
        },
        "artifact_status": artifact_status,
        "runtime_corpus_database": runtime_db,
        "raw_corpus_alignment": raw_alignment,
        "feature_coverage": feature_coverage,
        "learned_baseline_coverage": coverage,
        "exact_feature_materialization_succeeded": exact_success,
        "exact_neural_vs_v4_9_comparison_possible": coverage["v4_9_guarded_text_blend"]["exact_scoring_available"],
        "rows_silently_dropped": 0,
        "forbidden_features_used": [],
        "blockers": blockers,
        "smallest_safe_fix": "Create an offline full-corpus feature snapshot keyed by ResearchPath paper_id using the same runtime corpus table and retrieval services, then join it to V6 rows before scoring learned baselines.",
        "parity": parity,
        "protected_hashes": protected_status(resolve_repo_path(args.v39_labels)),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V6.1.4 Learned-Ranker Feature Materialization Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Rows: `{report['dataset']['row_count']}`",
        f"- Exact feature materialization succeeded: `{report['exact_feature_materialization_succeeded']}`",
        f"- Exact neural-vs-V4.9 comparison possible: `{report['exact_neural_vs_v4_9_comparison_possible']}`",
        f"- Rows silently dropped: `{report['rows_silently_dropped']}`",
        "",
        "## Learned Baseline Coverage",
        "",
    ]
    for method, row in report["learned_baseline_coverage"].items():
        lines.append(
            f"- `{method}`: scored `{row['rows_scored']}` / `{report['dataset']['row_count']}`, "
            f"exact `{row['exact_scoring_available']}`"
        )
    lines.extend(
        [
            "",
            "## Blockers",
            "",
            f"- Runtime DB: `{report['blockers']['runtime_corpus_database']}`",
            f"- Raw corpus alignment: `{report['blockers']['raw_corpus_alignment']}`",
            f"- Smallest safe fix: {report['smallest_safe_fix']}",
            "",
            "## Protected Hashes",
            "",
        ]
    )
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--v6-1-2-predictions", default=str(DEFAULT_V612_PREDICTIONS))
    parser.add_argument("--v33-model-dir", default=str(DEFAULT_V33_MODEL_DIR))
    parser.add_argument("--v41-model-dir", default=str(DEFAULT_V41_MODEL_DIR))
    parser.add_argument("--v43-model-dir", default=str(DEFAULT_V43_MODEL_DIR))
    parser.add_argument("--v41-blend-config", default=str(DEFAULT_V41_BLEND_CONFIG))
    parser.add_argument("--v49-config", default=str(DEFAULT_V49_CONFIG))
    parser.add_argument("--runtime-db", default=str(DEFAULT_RUNTIME_DB))
    parser.add_argument("--raw-corpus", default=str(DEFAULT_RAW_CORPUS))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--feature-matrix-out", default=str(DEFAULT_FEATURE_MATRIX))
    parser.add_argument("--predictions-out", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--parity-out", default=str(DEFAULT_PARITY))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_jsonl(resolve_repo_path(args.examples))
    rows = load_rows(resolve_repo_path(args.examples), resolve_repo_path(args.v6_1_2_predictions))
    artifacts, artifact_status = load_artifacts(args)
    required_by_method = required_features(artifacts, artifact_status)
    matrix_rows = build_feature_matrix(rows, required_by_method)
    predictions, coverage = score_rows(matrix_rows, artifacts, artifact_status)
    runtime_db = runtime_db_status(resolve_repo_path(args.runtime_db))
    raw_alignment = raw_corpus_alignment_status(resolve_repo_path(args.raw_corpus), examples)
    parity = parity_report(matrix_rows=matrix_rows, coverage=coverage, runtime_db=runtime_db, raw_alignment=raw_alignment)
    report = build_report(
        args=args,
        rows=rows,
        matrix_rows=matrix_rows,
        coverage=coverage,
        artifact_status=artifact_status,
        runtime_db=runtime_db,
        raw_alignment=raw_alignment,
        parity=parity,
    )
    write_jsonl(resolve_repo_path(args.feature_matrix_out), matrix_rows)
    write_jsonl(resolve_repo_path(args.predictions_out), predictions)
    write_json(resolve_repo_path(args.parity_out), parity)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report))
    print(f"Wrote V6.1.4 materialization report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
