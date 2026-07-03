import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import re
from statistics import mean
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_manual_benchmark import (
    DEFAULT_LABELS,
    DEFAULT_POOL,
    REPO_ROOT,
    label_summary,
    labels_by_query,
    load_jsonl,
    mean_metric,
    mrr,
    ndcg_at_k,
    pool_by_query,
    precision_at_k,
    recall_at_k,
    ranked_pool_rows,
)


DEFAULT_BASELINE = REPO_ROOT / "data" / "eval" / "results" / "manual_benchmark_method_comparison.json"
DEFAULT_JSON_OUTPUT = REPO_ROOT / "data" / "eval" / "results" / "learned_reranker_cv_comparison.json"
DEFAULT_MD_OUTPUT = REPO_ROOT / "data" / "eval" / "results" / "learned_reranker_cv_comparison.md"
RANK_METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
METRIC_FIELDS = [
    "recall_at_5",
    "recall_at_10",
    "precision_at_5",
    "precision_at_10",
    "ndcg_at_5",
    "ndcg_at_10",
    "mrr",
    "would_recommend_count_at_10",
    "duplicate_count_at_10",
    "judged_retrieved_at_10",
]


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def title_duplicate_flags(pool_rows: list[dict[str, Any]]) -> dict[tuple[str, int], bool]:
    titles_by_query: dict[str, Counter[str]] = defaultdict(Counter)
    normalized_by_key: dict[tuple[str, int], str] = {}
    for row in pool_rows:
        key = (str(row["query_id"]), int(row["paper_id"]))
        normalized = normalize_title(str(row.get("title") or ""))
        normalized_by_key[key] = normalized
        if normalized:
            titles_by_query[key[0]][normalized] += 1
    return {
        key: bool(normalized and titles_by_query[key[0]][normalized] > 1)
        for key, normalized in normalized_by_key.items()
    }


def safe_year(value: Any) -> float:
    if isinstance(value, (int, float)) and value:
        return float(value)
    return 0.0


def log1p_number(value: Any) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    return math.log1p(max(0.0, float(value)))


def feature_names() -> list[str]:
    names: list[str] = []
    for method in RANK_METHODS:
        names.extend([f"{method}_present", f"{method}_rank", f"{method}_reciprocal_rank"])
    names.extend(
        [
            "appears_in_n_methods",
            "best_rank",
            "mean_rank",
            "rank_std",
            "year",
            "paper_age",
            "log_citation_count",
            "source_methods_count",
            "title_char_length",
            "abstract_char_length",
            "title_word_length",
            "abstract_word_length",
            "metadata_title_duplicate",
        ]
    )
    return names


def build_feature_vector(
    row: dict[str, Any],
    *,
    metadata_duplicate_flags: dict[tuple[str, int], bool],
    names: list[str],
) -> list[float]:
    ranks = {str(method): int(rank) for method, rank in (row.get("retrieval_ranks_by_method") or {}).items()}
    features: dict[str, float] = {}
    observed_ranks: list[int] = []
    for method in RANK_METHODS:
        rank = ranks.get(method)
        present = rank is not None
        if present:
            observed_ranks.append(int(rank))
        features[f"{method}_present"] = 1.0 if present else 0.0
        features[f"{method}_rank"] = float(rank if present else 999)
        features[f"{method}_reciprocal_rank"] = (1.0 / float(rank)) if present and rank else 0.0

    title = str(row.get("title") or "")
    abstract = str(row.get("abstract") or "")
    year = safe_year(row.get("year"))
    key = (str(row["query_id"]), int(row["paper_id"]))
    features.update(
        {
            "appears_in_n_methods": float(len(observed_ranks)),
            "best_rank": float(min(observed_ranks)) if observed_ranks else 999.0,
            "mean_rank": float(mean(observed_ranks)) if observed_ranks else 999.0,
            "rank_std": float(np.std(observed_ranks)) if observed_ranks else 0.0,
            "year": year,
            "paper_age": max(0.0, 2026.0 - year) if year else 0.0,
            "log_citation_count": log1p_number(row.get("citation_count")),
            "source_methods_count": float(len(row.get("source_methods") or [])),
            "title_char_length": float(len(title)),
            "abstract_char_length": float(len(abstract)),
            "title_word_length": float(len(title.split())),
            "abstract_word_length": float(len(abstract.split())),
            "metadata_title_duplicate": 1.0 if metadata_duplicate_flags.get(key, False) else 0.0,
        }
    )
    return [features[name] for name in names]


def training_rows_for_query_ids(
    *,
    query_ids: set[str],
    labels_by_q: dict[str, dict[int, dict[str, Any]]],
    pool_by_q: dict[str, list[dict[str, Any]]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pool_by_key = {
        (str(row["query_id"]), int(row["paper_id"])): row
        for query_id in query_ids
        for row in pool_by_q.get(query_id, [])
    }
    for query_id in sorted(query_ids):
        for paper_id, label in labels_by_q[query_id].items():
            pool_row = pool_by_key.get((query_id, paper_id))
            if pool_row is not None:
                rows.append((pool_row, label))
    return rows


def model_specs() -> dict[str, dict[str, Any]]:
    return {
        "learned_relevance_gbr": {
            "target": "relevance",
            "model": GradientBoostingRegressor(random_state=17, n_estimators=40, max_depth=2, learning_rate=0.05),
        },
        "learned_relevance_rf": {
            "target": "relevance",
            "model": RandomForestRegressor(random_state=17, n_estimators=120, min_samples_leaf=2),
        },
        "learned_recommend_logreg": {
            "target": "would_recommend",
            "model": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        },
        "learned_recommend_gbc": {
            "target": "would_recommend",
            "model": GradientBoostingClassifier(random_state=17, n_estimators=40, max_depth=2, learning_rate=0.05),
        },
    }


def fit_model(spec: dict[str, Any], x_train: np.ndarray, labels: list[dict[str, Any]]) -> Any:
    target_name = spec["target"]
    if target_name == "relevance":
        y_train = np.array([float(label["relevance"]) for label in labels], dtype=float)
    else:
        y_train = np.array([1 if bool(label["would_recommend"]) else 0 for label in labels], dtype=int)
        if len(set(y_train.tolist())) < 2:
            y_train = np.zeros(len(labels), dtype=int)
    model = spec["model"]
    model.fit(x_train, y_train)
    return model


def score_model(model: Any, x_test: np.ndarray, *, target_name: str) -> np.ndarray:
    if target_name == "would_recommend" and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        if probabilities.shape[1] == 1:
            return probabilities[:, 0]
        return probabilities[:, 1]
    return np.asarray(model.predict(x_test), dtype=float)


def evaluate_ranked_ids(
    *,
    method: str,
    query_id: str,
    query_labels: dict[int, dict[str, Any]],
    retrieved_ids: list[int],
) -> dict[str, Any]:
    relevant_ids = {paper_id for paper_id, label in query_labels.items() if int(label["relevance"]) >= 2}
    recommend_ids = {paper_id for paper_id, label in query_labels.items() if bool(label["would_recommend"])}
    duplicate_ids = {paper_id for paper_id, label in query_labels.items() if bool(label["duplicate"])}
    labels_relevance = {paper_id: int(label["relevance"]) for paper_id, label in query_labels.items()}
    return {
        "query_id": query_id,
        "query": str(next(iter(query_labels.values()))["query"]) if query_labels else "",
        "method": method,
        "retrieved_ids_at_10": retrieved_ids[:10],
        "judged_retrieved_at_10": sum(1 for paper_id in retrieved_ids[:10] if paper_id in query_labels),
        "relevant_label_count": len(relevant_ids),
        "would_recommend_label_count": len(recommend_ids),
        "recall_at_5": recall_at_k(retrieved_ids, relevant_ids, 5),
        "recall_at_10": recall_at_k(retrieved_ids, relevant_ids, 10),
        "precision_at_5": precision_at_k(retrieved_ids, recommend_ids, 5),
        "precision_at_10": precision_at_k(retrieved_ids, recommend_ids, 10),
        "ndcg_at_5": ndcg_at_k(retrieved_ids, labels_relevance, 5),
        "ndcg_at_10": ndcg_at_k(retrieved_ids, labels_relevance, 10),
        "mrr": mrr(retrieved_ids, relevant_ids),
        "would_recommend_count_at_10": sum(1 for paper_id in retrieved_ids[:10] if paper_id in recommend_ids),
        "duplicate_count_at_10": sum(1 for paper_id in retrieved_ids[:10] if paper_id in duplicate_ids),
    }


def evaluate_baseline_for_query(
    *,
    method: str,
    query_id: str,
    query_rows: list[dict[str, Any]],
    query_labels: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    retrieved_ids = [int(row["paper_id"]) for row in ranked_pool_rows(query_rows, method)]
    return evaluate_ranked_ids(method=method, query_id=query_id, query_labels=query_labels, retrieved_ids=retrieved_ids)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {field: mean_metric(rows, field) for field in METRIC_FIELDS}


def baseline_table_from_report(baseline_report: dict[str, Any]) -> dict[str, Any]:
    method_order = ["hybrid", "bm25", "tfidf", "embedding", "faiss_embedding"]
    return {
        method: baseline_report["methods"][method]["averages"]
        for method in method_order
        if method in baseline_report.get("methods", {})
    }


def build_cv_report(
    *,
    labels: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
    baseline_report: dict[str, Any],
) -> dict[str, Any]:
    labels_by_q = labels_by_query(labels)
    pool_by_q = pool_by_query(pool_rows)
    metadata_duplicate_flags = title_duplicate_flags(pool_rows)
    names = feature_names()
    specs = model_specs()
    query_ids = sorted(labels_by_q)

    learned_rows_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    baseline_rows_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_fold: dict[str, dict[str, Any]] = {}

    for held_out_query_id in query_ids:
        train_query_ids = set(query_ids) - {held_out_query_id}
        train_pairs = training_rows_for_query_ids(
            query_ids=train_query_ids,
            labels_by_q=labels_by_q,
            pool_by_q=pool_by_q,
        )
        x_train = np.array(
            [
                build_feature_vector(row, metadata_duplicate_flags=metadata_duplicate_flags, names=names)
                for row, _ in train_pairs
            ],
            dtype=float,
        )
        train_labels = [label for _, label in train_pairs]
        test_rows = pool_by_q.get(held_out_query_id, [])
        x_test = np.array(
            [
                build_feature_vector(row, metadata_duplicate_flags=metadata_duplicate_flags, names=names)
                for row in test_rows
            ],
            dtype=float,
        )
        fold_payload: dict[str, Any] = {
            "train_query_ids": sorted(train_query_ids),
            "train_label_count": len(train_pairs),
            "test_query_id": held_out_query_id,
            "test_candidate_count": len(test_rows),
            "learned": {},
            "baselines": {},
        }

        for method in RANK_METHODS:
            baseline_row = evaluate_baseline_for_query(
                method=method,
                query_id=held_out_query_id,
                query_rows=test_rows,
                query_labels=labels_by_q[held_out_query_id],
            )
            baseline_rows_by_method[method].append(baseline_row)
            fold_payload["baselines"][method] = baseline_row

        for method_name, spec in specs.items():
            model = fit_model(spec, x_train, train_labels)
            scores = score_model(model, x_test, target_name=spec["target"])
            scored_rows = sorted(
                zip(test_rows, scores, range(len(test_rows)), strict=False),
                key=lambda item: (-float(item[1]), int((item[0].get("retrieval_ranks_by_method") or {}).get("hybrid", 999)), item[2]),
            )
            retrieved_ids = [int(row["paper_id"]) for row, _, _ in scored_rows]
            learned_row = evaluate_ranked_ids(
                method=method_name,
                query_id=held_out_query_id,
                query_labels=labels_by_q[held_out_query_id],
                retrieved_ids=retrieved_ids,
            )
            learned_row["target"] = spec["target"]
            learned_row["top_10_scores"] = [float(score) for _, score, _ in scored_rows[:10]]
            learned_rows_by_method[method_name].append(learned_row)
            fold_payload["learned"][method_name] = learned_row

        per_fold[held_out_query_id] = fold_payload

    learned_payload = {
        method: {"averages": aggregate_rows(rows), "queries": rows}
        for method, rows in learned_rows_by_method.items()
    }
    baseline_payload = {
        method: {"averages": aggregate_rows(rows), "queries": rows}
        for method, rows in baseline_rows_by_method.items()
    }
    best_learned = max(
        learned_payload,
        key=lambda method: learned_payload[method]["averages"]["ndcg_at_10"],
    )
    hybrid_avg = baseline_payload["hybrid"]["averages"]
    best_avg = learned_payload[best_learned]["averages"]

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "label_source": str(DEFAULT_LABELS.relative_to(REPO_ROOT)),
        "candidate_pool": str(DEFAULT_POOL.relative_to(REPO_ROOT)),
        "baseline_report": str(DEFAULT_BASELINE.relative_to(REPO_ROOT)),
        "evaluation_scope": "leave_one_topic_out_cv_candidate_pool_reranking",
        "label_summary": label_summary(labels),
        "feature_names": names,
        "model_specs": {name: {"target": spec["target"], "estimator": type(spec["model"]).__name__} for name, spec in specs.items()},
        "baseline_method_table_from_manual_benchmark": baseline_table_from_report(baseline_report),
        "cv_baselines": baseline_payload,
        "learned_methods": learned_payload,
        "best_learned_method_by_ndcg_at_10": best_learned,
        "best_learned_vs_hybrid": {
            "learned_method": best_learned,
            "learned_ndcg_at_10": best_avg["ndcg_at_10"],
            "hybrid_ndcg_at_10": hybrid_avg["ndcg_at_10"],
            "delta_ndcg_at_10": best_avg["ndcg_at_10"] - hybrid_avg["ndcg_at_10"],
            "beats_hybrid_on_ndcg_at_10": best_avg["ndcg_at_10"] > hybrid_avg["ndcg_at_10"],
            "delta_precision_at_10": best_avg["precision_at_10"] - hybrid_avg["precision_at_10"],
            "delta_would_recommend_count_at_10": best_avg["would_recommend_count_at_10"]
            - hybrid_avg["would_recommend_count_at_10"],
        },
        "per_fold": per_fold,
        "caveats": [
            "Only 80 judged labels are available across 8 query topics; each fold trains on about 70 judged rows.",
            "The CV protocol holds out an entire query_id to avoid topic label leakage.",
            "Unjudged candidate-pool rows are eligible to be ranked but count as zero relevance and not recommended.",
            "Raw retrieval scores are not present in the manual label pool, so retrieval features use ranks and agreement signals.",
            "The metadata title-duplicate feature is computed from normalized candidate titles, not from manual duplicate labels.",
            "No persistent model is written; models are trained in memory per fold.",
        ],
    }


def metric_table(method_payload: dict[str, Any]) -> list[str]:
    lines = [
        "| method | R@5 | R@10 | P@5 | P@10 | NDCG@5 | NDCG@10 | MRR | rec@10 | dup@10 | judged@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, payload in sorted(
        method_payload.items(),
        key=lambda item: item[1]["averages"]["ndcg_at_10"],
        reverse=True,
    ):
        avg = payload["averages"]
        lines.append(
            f"| `{method}` | {avg['recall_at_5']:.3f} | {avg['recall_at_10']:.3f} | "
            f"{avg['precision_at_5']:.3f} | {avg['precision_at_10']:.3f} | "
            f"{avg['ndcg_at_5']:.3f} | {avg['ndcg_at_10']:.3f} | {avg['mrr']:.3f} | "
            f"{avg['would_recommend_count_at_10']:.2f} | {avg['duplicate_count_at_10']:.2f} | "
            f"{avg['judged_retrieved_at_10']:.2f} |"
        )
    return lines


def copied_baseline_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "| method | R@5 | R@10 | P@5 | P@10 | NDCG@5 | NDCG@10 | MRR | rec@10 | dup@10 | judged@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, avg in sorted(
        report["baseline_method_table_from_manual_benchmark"].items(),
        key=lambda item: item[1]["ndcg_at_10"],
        reverse=True,
    ):
        lines.append(
            f"| `{method}` | {avg['recall_at_5']:.3f} | {avg['recall_at_10']:.3f} | "
            f"{avg['precision_at_5']:.3f} | {avg['precision_at_10']:.3f} | "
            f"{avg['ndcg_at_5']:.3f} | {avg['ndcg_at_10']:.3f} | {avg['mrr']:.3f} | "
            f"{avg['would_recommend_count_at_10']:.2f} | {avg['duplicate_count_at_10']:.2f} | "
            f"{avg['judged_retrieved_at_10']:.2f} |"
        )
    return lines


def per_fold_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "| held_out_query | best_learned | learned NDCG@10 | hybrid NDCG@10 | delta | learned P@10 | hybrid P@10 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for query_id, fold in sorted(report["per_fold"].items()):
        learned_rows = fold["learned"]
        best_method = max(learned_rows, key=lambda method: learned_rows[method]["ndcg_at_10"])
        learned = learned_rows[best_method]
        hybrid = fold["baselines"]["hybrid"]
        lines.append(
            f"| `{query_id}` | `{best_method}` | {learned['ndcg_at_10']:.3f} | "
            f"{hybrid['ndcg_at_10']:.3f} | {learned['ndcg_at_10'] - hybrid['ndcg_at_10']:+.3f} | "
            f"{learned['precision_at_10']:.3f} | {hybrid['precision_at_10']:.3f} |"
        )
    return lines


def markdown_report(report: dict[str, Any]) -> str:
    comparison = report["best_learned_vs_hybrid"]
    lines = [
        "# Learned Reranker Leave-One-Topic-Out CV",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Label Summary",
        "",
        f"- Total labels: `{report['label_summary']['total_labels']}`",
        f"- Labels per query: `{report['label_summary']['labels_per_query']}`",
        f"- Duplicate count: `{report['label_summary']['duplicate_count']}`",
        "",
        "## Baseline Method Table",
    ]
    lines.extend(copied_baseline_table(report))
    lines.extend(["", "## Cross-Validated Learned Rerankers"])
    lines.extend(metric_table(report["learned_methods"]))
    lines.extend(["", "## Cross-Validated Baselines"])
    lines.extend(metric_table(report["cv_baselines"]))
    lines.extend(["", "## Per-Topic Fold Results"])
    lines.extend(per_fold_table(report))
    lines.extend(
        [
            "",
            "## Hybrid Comparison",
            f"Best learned method by NDCG@10: `{comparison['learned_method']}`.",
            f"Mean NDCG@10 delta vs hybrid: `{comparison['delta_ndcg_at_10']:.3f}`.",
            f"Beats hybrid on mean NDCG@10: `{comparison['beats_hybrid_on_ndcg_at_10']}`.",
            f"Mean P@10 delta vs hybrid: `{comparison['delta_precision_at_10']:.3f}`.",
            f"Mean would_recommend_count@10 delta vs hybrid: `{comparison['delta_would_recommend_count_at_10']:.3f}`.",
            "",
            "## Caveats",
        ]
    )
    for caveat in report["caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], json_output: Path, md_output: Path) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_output.write_text(markdown_report(report), encoding="utf-8")


def print_tables(report: dict[str, Any]) -> None:
    print("\nBaseline Method Table")
    for line in copied_baseline_table(report):
        print(line)
    print("\nCross-Validated Learned Rerankers")
    for line in metric_table(report["learned_methods"]):
        print(line)
    print("\nCross-Validated Baselines")
    for line in metric_table(report["cv_baselines"]):
        print(line)
    print("\nPer-Topic Fold Results")
    for line in per_fold_table(report):
        print(line)
    comparison = report["best_learned_vs_hybrid"]
    print("\nHybrid Comparison")
    print(f"Best learned method: {comparison['learned_method']}")
    print(f"Mean NDCG@10 delta vs hybrid: {comparison['delta_ndcg_at_10']:.3f}")
    print(f"Beats hybrid: {comparison['beats_hybrid_on_ndcg_at_10']}")
    print(f"Mean P@10 delta vs hybrid: {comparison['delta_precision_at_10']:.3f}")
    print(f"Mean rec@10 delta vs hybrid: {comparison['delta_would_recommend_count_at_10']:.3f}")


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-topic-out CV for lightweight learned rerankers.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--pool", default=str(DEFAULT_POOL))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUTPUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_jsonl(resolve_repo_path(args.labels))
    pool_rows = load_jsonl(resolve_repo_path(args.pool))
    baseline_report = json.loads(resolve_repo_path(args.baseline).read_text(encoding="utf-8"))
    report = build_cv_report(labels=labels, pool_rows=pool_rows, baseline_report=baseline_report)
    json_output = resolve_repo_path(args.json_out)
    md_output = resolve_repo_path(args.md_out)
    write_report(report, json_output, md_output)
    print_tables(report)
    print(f"\nJSON: {json_output}")
    print(f"Markdown: {md_output}")


if __name__ == "__main__":
    main()
