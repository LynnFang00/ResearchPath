import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from build_v4_3_text_reranker_dataset import protected_status  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import resolve_repo_path, write_json, write_text  # noqa: E402
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_TRAIN = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_train.jsonl"
DEFAULT_VAL = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_val.jsonl"
DEFAULT_TEST = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_test.jsonl"
DEFAULT_PAIRWISE = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_pairwise_train.jsonl"
DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_1_cross_encoder_reranker_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_1_cross_encoder_reranker_report.md"
DEFAULT_VAL_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_1_cross_encoder_predictions_val.jsonl"
DEFAULT_TEST_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_1_cross_encoder_predictions_test.jsonl"
DEFAULT_FAILURE_EXAMPLES = REPO_ROOT / "data" / "eval" / "results" / "v6_1_cross_encoder_failure_examples.jsonl"
DEFAULT_CONFIG = REPO_ROOT / "data" / "processed" / "models" / "v6_1_cross_encoder_reranker_config.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 613
HASH_FEATURES = 4096
MAX_ABSTRACT_WORDS = 220
EPOCHS = 8
PATIENCE = 3
LEARNING_RATE = 2e-3
TRANSFORMER_EPOCHS = 3
TRANSFORMER_PATIENCE = 2
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_LEARNING_RATE = 2e-5
TRANSFORMER_MAX_LENGTH = 192

FORBIDDEN_INPUT_FIELDS = {
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
    "baseline_rankings",
    "label_source",
    "query_grouped_split",
    "v6_0_2_grouped_split",
}
ALLOWED_INPUT_FIELDS = {"query", "title", "abstract"}
BASELINE_METHODS = ["bm25", "v3_3_ltr", "v4_1_blend", "v4_9_guarded_text_blend"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def set_deterministic_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def truncate_words(value: str | None, max_words: int = MAX_ABSTRACT_WORDS) -> str:
    return " ".join(str(value or "").split()[:max_words])


def construct_model_input(row: dict[str, Any]) -> str:
    return f"{row.get('query', '')} [SEP] {row.get('title', '')} [SEP] {truncate_words(row.get('abstract'))}"


def assert_no_forbidden_input_fields(input_fields: set[str]) -> dict[str, Any]:
    forbidden_used = sorted(input_fields & FORBIDDEN_INPUT_FIELDS)
    return {
        "input_fields": sorted(input_fields),
        "allowed_input_fields": sorted(ALLOWED_INPUT_FIELDS),
        "forbidden_input_fields": sorted(FORBIDDEN_INPUT_FIELDS),
        "forbidden_fields_used": forbidden_used,
        "input_is_inference_safe": not forbidden_used and input_fields <= ALLOWED_INPUT_FIELDS,
    }


def transformer_weights_available(model_name: str) -> tuple[bool, str]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, local_files_only=True)
        return True, "pretrained tokenizer and model weights found in local cache"
    except Exception as exc:
        return False, f"pretrained transformer weights unavailable locally: {type(exc).__name__}: {str(exc)[:180]}"


def text_features(rows: list[dict[str, Any]], vectorizer: Any | None = None) -> tuple[np.ndarray, Any]:
    from sklearn.feature_extraction.text import HashingVectorizer

    texts = [construct_model_input(row) for row in rows]
    if vectorizer is None:
        vectorizer = HashingVectorizer(
            n_features=HASH_FEATURES,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            lowercase=True,
        )
    matrix = vectorizer.transform(texts)
    return matrix.astype(np.float32).toarray(), vectorizer


def train_fallback_mlp(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    from torch import nn

    set_deterministic_seed()
    x_train, vectorizer = text_features(train_rows)
    x_val, _ = text_features(val_rows, vectorizer)
    y_train = np.array([float(row["reading_value_score"]) for row in train_rows], dtype=np.float32).reshape(-1, 1)
    y_val = np.array([float(row["reading_value_score"]) for row in val_rows], dtype=np.float32).reshape(-1, 1)
    weights = np.array([1.5 if row["hard_negative"] else 1.0 for row in train_rows], dtype=np.float32).reshape(-1, 1)

    model = nn.Sequential(
        nn.Linear(HASH_FEATURES, 128),
        nn.ReLU(),
        nn.Dropout(0.05),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    weights_t = torch.from_numpy(weights)
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)

    history = []
    best_state = None
    best_epoch = 0
    best_val_mse = float("inf")
    stale_epochs = 0
    for epoch in range(1, EPOCHS + 1):
        set_deterministic_seed(SEED + epoch)
        model.train()
        optimizer.zero_grad()
        predictions = model(x_train_t)
        loss = (((predictions - y_train_t) ** 2) * weights_t).mean()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(x_val_t)
            val_mse = torch.mean((val_predictions - y_val_t) ** 2).item()
        history.append({"epoch": epoch, "train_weighted_mse": float(loss.item()), "val_mse": float(val_mse)})
        if val_mse < best_val_mse - 1e-7:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "model": model,
        "vectorizer": vectorizer,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
    }


def encode_transformer_rows(tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return tokenizer(
        [construct_model_input(row) for row in rows],
        padding=True,
        truncation=True,
        max_length=TRANSFORMER_MAX_LENGTH,
        return_tensors="pt",
    )


def train_transformer_cross_encoder(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_deterministic_seed()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1, local_files_only=True)
    model.train()

    train_encoded = encode_transformer_rows(tokenizer, train_rows)
    val_encoded = encode_transformer_rows(tokenizer, val_rows)
    y_train = torch.tensor([float(row["reading_value_score"]) for row in train_rows], dtype=torch.float32).view(-1, 1)
    y_val = torch.tensor([float(row["reading_value_score"]) for row in val_rows], dtype=torch.float32).view(-1, 1)
    train_weights = torch.tensor([1.5 if row["hard_negative"] else 1.0 for row in train_rows], dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(
        train_encoded["input_ids"],
        train_encoded["attention_mask"],
        y_train,
        train_weights,
    )
    train_loader = DataLoader(train_dataset, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRANSFORMER_LEARNING_RATE, weight_decay=0.01)

    history = []
    best_state = None
    best_epoch = 0
    best_val_mse = float("inf")
    stale_epochs = 0
    for epoch in range(1, TRANSFORMER_EPOCHS + 1):
        set_deterministic_seed(SEED + epoch)
        model.train()
        train_losses = []
        for input_ids, attention_mask, labels, weights in train_loader:
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions = torch.sigmoid(logits)
            loss = (((predictions - labels) ** 2) * weights).mean()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_logits = model(input_ids=val_encoded["input_ids"], attention_mask=val_encoded["attention_mask"]).logits
            val_predictions = torch.sigmoid(val_logits)
            val_mse = torch.mean((val_predictions - y_val) ** 2).item()
        history.append(
            {
                "epoch": epoch,
                "train_weighted_mse": float(sum(train_losses) / len(train_losses)),
                "val_mse": float(val_mse),
            }
        )
        if val_mse < best_val_mse - 1e-7:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= TRANSFORMER_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
    }


def predict_fallback_mlp(model: Any, vectorizer: Any, rows: list[dict[str, Any]]) -> list[float]:
    import torch

    x, _ = text_features(rows, vectorizer)
    model.eval()
    with torch.no_grad():
        values = model(torch.from_numpy(x)).numpy().reshape(-1)
    return [float(value) for value in values]


def predict_transformer_cross_encoder(model: Any, tokenizer: Any, rows: list[dict[str, Any]]) -> list[float]:
    import torch

    encoded = encode_transformer_rows(tokenizer, rows)
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(rows), TRANSFORMER_BATCH_SIZE):
            end = start + TRANSFORMER_BATCH_SIZE
            logits = model(
                input_ids=encoded["input_ids"][start:end],
                attention_mask=encoded["attention_mask"][start:end],
            ).logits
            scores.extend(torch.sigmoid(logits).cpu().numpy().reshape(-1).tolist())
    return [float(value) for value in scores]


def dcg(values: list[float]) -> float:
    return sum((2.0**value - 1.0) / math.log2(index + 2) for index, value in enumerate(values))


def ndcg_at_k(rows: list[dict[str, Any]], score_field: str, label_field: str, k: int = 10) -> float:
    ranked = sorted(rows, key=lambda row: row[score_field], reverse=True)[:k]
    ideal = sorted(rows, key=lambda row: float(row[label_field]), reverse=True)[:k]
    ideal_dcg = dcg([float(row[label_field]) for row in ideal])
    if ideal_dcg <= 0:
        return 0.0
    return dcg([float(row[label_field]) for row in ranked]) / ideal_dcg


def mrr_at_k(rows: list[dict[str, Any]], score_field: str, k: int = 10) -> float:
    ranked = sorted(rows, key=lambda row: row[score_field], reverse=True)[:k]
    for index, row in enumerate(ranked, start=1):
        if int(row["relevance_bucket"]) == 2:
            return 1.0 / index
    return 0.0


def recall_at_k(rows: list[dict[str, Any]], score_field: str, k: int = 10) -> float:
    positives = sum(1 for row in rows if int(row["relevance_bucket"]) == 2)
    if positives == 0:
        return 0.0
    ranked = sorted(rows, key=lambda row: row[score_field], reverse=True)[:k]
    return sum(1 for row in ranked if int(row["relevance_bucket"]) == 2) / positives


def ranking_metrics(rows: list[dict[str, Any]], score_field: str) -> dict[str, float]:
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_topic[row["query_id"]].append(row)
    per_topic = {}
    for topic, topic_rows in sorted(by_topic.items()):
        top10 = sorted(topic_rows, key=lambda row: row[score_field], reverse=True)[:10]
        per_topic[topic] = {
            "reading_ndcg_at_10": ndcg_at_k(topic_rows, score_field, "reading_value_score"),
            "topic_ndcg_at_10": ndcg_at_k(topic_rows, score_field, "topic_match_score"),
            "mrr_at_10": mrr_at_k(topic_rows, score_field),
            "recall_at_10": recall_at_k(topic_rows, score_field),
            "hard_neg_at_10": sum(1 for row in top10 if row["hard_negative"]) / 10.0,
            "judged_at_10": float(len(top10)),
            "unjudged_at_10": 0.0,
        }
    keys = ["reading_ndcg_at_10", "topic_ndcg_at_10", "mrr_at_10", "recall_at_10", "hard_neg_at_10", "judged_at_10", "unjudged_at_10"]
    aggregate = {
        key: sum(metrics[key] for metrics in per_topic.values()) / len(per_topic) if per_topic else 0.0
        for key in keys
    }
    return {"aggregate": aggregate, "per_topic": per_topic}


def pearson(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) < 2:
        return None
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman(values_a: list[float], values_b: list[float]) -> float | None:
    try:
        from scipy.stats import spearmanr

        result = spearmanr(values_a, values_b)
        return None if math.isnan(float(result.correlation)) else float(result.correlation)
    except Exception:
        return None


def confusion_summary(rows: list[dict[str, Any]], score_field: str) -> dict[str, Any]:
    matrix: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        score = float(row[score_field])
        predicted = 2 if score >= 0.66 else 1 if score >= 0.33 else 0
        matrix[str(row["relevance_bucket"])][str(predicted)] += 1
    return {actual: dict(sorted(counts.items())) for actual, counts in sorted(matrix.items())}


def score_bin_summary(rows: list[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0000001)]
    summary = []
    for low, high in bins:
        bucket = [row for row in rows if low <= float(row[score_field]) < high]
        summary.append(
            {
                "score_min": low,
                "score_max": high if high <= 1.0 else 1.0,
                "count": len(bucket),
                "average_reading_value_score": sum(float(row["reading_value_score"]) for row in bucket) / len(bucket) if bucket else 0.0,
                "average_topic_match_score": sum(float(row["topic_match_score"]) for row in bucket) / len(bucket) if bucket else 0.0,
            }
        )
    return summary


def add_predictions(rows: list[dict[str, Any]], scores: list[float], split: str) -> list[dict[str, Any]]:
    output = []
    for row, score in zip(rows, scores, strict=True):
        output.append(
            {
                "schema_version": "v6.1_cross_encoder_prediction",
                "split": split,
                "query_id": row["query_id"],
                "query": row["query"],
                "paper_id": int(row["paper_id"]),
                "title": row["title"],
                "neural_score": float(score),
                "reading_value_score": float(row["reading_value_score"]),
                "topic_match_score": float(row["topic_match_score"]),
                "relevance_bucket": int(row["relevance_bucket"]),
                "hard_negative": bool(row["hard_negative"]),
                "label_source": row.get("label_source"),
                "baseline_rankings": row.get("baseline_rankings") or {},
            }
        )
    return output


def baseline_score(row: dict[str, Any], method: str) -> float | None:
    metadata = row.get("baseline_rankings") or {}
    method_row = metadata.get(method) or {}
    score = method_row.get("score")
    rank = method_row.get("rank")
    if score is not None:
        try:
            return float(score)
        except (TypeError, ValueError):
            pass
    if rank is not None:
        try:
            return 1.0 / (1.0 + float(rank))
        except (TypeError, ValueError):
            return None
    return None


def baseline_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparison = {}
    for method in BASELINE_METHODS:
        scored = []
        coverage = 0
        for row in rows:
            copied = dict(row)
            score = baseline_score(row, method)
            if score is not None:
                coverage += 1
            copied[f"{method}_score_for_eval"] = score if score is not None else -1.0
            scored.append(copied)
        comparison[method] = {
            "available_rows": coverage,
            "total_rows": len(rows),
            "coverage": coverage / len(rows) if rows else 0.0,
            "fully_comparable": coverage == len(rows),
            "metrics": ranking_metrics(scored, f"{method}_score_for_eval")["aggregate"] if coverage else None,
        }
    return comparison


def hard_negative_rank_diagnostics(rows: list[dict[str, Any]], score_field: str) -> dict[str, Any]:
    promoted = []
    for topic in sorted({row["query_id"] for row in rows}):
        ranked = sorted([row for row in rows if row["query_id"] == topic], key=lambda row: row[score_field], reverse=True)
        for rank, row in enumerate(ranked, start=1):
            if row["hard_negative"] and rank <= 10:
                promoted.append(
                    {
                        "query_id": topic,
                        "paper_id": int(row["paper_id"]),
                        "title": row["title"],
                        "rank": rank,
                        "neural_score": float(row[score_field]),
                        "reading_value_score": float(row["reading_value_score"]),
                    }
                )
    return {
        "hard_negatives_in_top10": len(promoted),
        "promoted_hard_negatives": promoted[:25],
    }


def failure_examples(rows: list[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
    failures = []
    for topic in sorted({row["query_id"] for row in rows}):
        ranked = sorted([row for row in rows if row["query_id"] == topic], key=lambda row: row[score_field], reverse=True)
        for rank, row in enumerate(ranked[:20], start=1):
            if row["hard_negative"] or int(row["relevance_bucket"]) == 0:
                failures.append(
                    {
                        "schema_version": "v6.1_cross_encoder_failure_example",
                        "failure_type": "negative_or_hard_negative_promoted",
                        "query_id": topic,
                        "paper_id": int(row["paper_id"]),
                        "title": row["title"],
                        "rank": rank,
                        "neural_score": float(row[score_field]),
                        "reading_value_score": float(row["reading_value_score"]),
                        "topic_match_score": float(row["topic_match_score"]),
                        "hard_negative": bool(row["hard_negative"]),
                    }
                )
        for rank, row in enumerate(ranked, start=1):
            if int(row["relevance_bucket"]) == 2 and rank > 10:
                failures.append(
                    {
                        "schema_version": "v6.1_cross_encoder_failure_example",
                        "failure_type": "strong_positive_below_top10",
                        "query_id": topic,
                        "paper_id": int(row["paper_id"]),
                        "title": row["title"],
                        "rank": rank,
                        "neural_score": float(row[score_field]),
                        "reading_value_score": float(row["reading_value_score"]),
                        "topic_match_score": float(row["topic_match_score"]),
                        "hard_negative": bool(row["hard_negative"]),
                    }
                )
                break
    return failures[:100]


def pairwise_audit(path: Path, train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = load_jsonl(path)
    source_by_key = {(row["query_id"], int(row["paper_id"])): row.get("label_source") for row in train_rows}
    represented = 0
    for pair in pairs:
        if source_by_key.get((pair["query_id"], int(pair["preferred_paper_id"]))) == "v6_0_2" or source_by_key.get((pair["query_id"], int(pair["rejected_paper_id"]))) == "v6_0_2":
            represented += 1
    return {
        "pairwise_example_count": len(pairs),
        "v6_0_2_labels_represented_pair_count": represented,
        "count_capped_at_7200": len(pairs) == 7200,
        "used_in_v6_1_training": False,
        "usage_note": "Reserved for a later V6.2 pairwise reranker; V6.1 uses pointwise reading_value_score regression.",
    }


def evaluate_predictions(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    labels = [float(row["reading_value_score"]) for row in rows]
    scores = [float(row["neural_score"]) for row in rows]
    topics = [float(row["topic_match_score"]) for row in rows]
    return {
        "split": split,
        "row_count": len(rows),
        "ranking": ranking_metrics(rows, "neural_score"),
        "reading_score_pearson": pearson(scores, labels),
        "reading_score_spearman": spearman(scores, labels),
        "topic_score_pearson": pearson(scores, topics),
        "topic_score_spearman": spearman(scores, topics),
        "bucket_confusion": confusion_summary(rows, "neural_score"),
        "score_bin_summary": score_bin_summary(rows, "neural_score"),
        "hard_negative_rank_diagnostics": hard_negative_rank_diagnostics(rows, "neural_score"),
    }


def build_report(
    *,
    train_rows: list[dict[str, Any]],
    val_predictions: list[dict[str, Any]],
    test_predictions: list[dict[str, Any]],
    train_result: dict[str, Any],
    transformer_status: dict[str, Any],
    model_type: str,
    transformer_training_ran: bool,
    leakage: dict[str, Any],
    pairwise: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    val_eval = evaluate_predictions(val_predictions, "val")
    test_eval = evaluate_predictions(test_predictions, "test")
    return {
        "schema_version": "v6.1_cross_encoder_reranker_report",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_integration_status": "offline_only_not_integrated",
        "default_ranker_status": "unchanged_bm25",
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "protected_labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "existing_model_artifacts_overwritten": False,
        "model_type": model_type,
        "transformer_training_ran": transformer_training_ran,
        "transformer_status": transformer_status,
        "training_config": {
            "seed": SEED,
            "input_format": "query [SEP] title [SEP] abstract",
            "input_fields": sorted(ALLOWED_INPUT_FIELDS),
            "max_abstract_words": MAX_ABSTRACT_WORDS,
            "target": "reading_value_score",
            "secondary_topic_target_reporting": "topic_match_score correlation only",
            "feature_extractor": "transformer_tokenizer" if transformer_training_ran else "sklearn HashingVectorizer",
            "hash_features": None if transformer_training_ran else HASH_FEATURES,
            "model": f"{MODEL_NAME} sequence regression head" if transformer_training_ran else "PyTorch MLP 4096->128->32->1 sigmoid",
            "epochs_requested": TRANSFORMER_EPOCHS if transformer_training_ran else EPOCHS,
            "early_stopping_patience": TRANSFORMER_PATIENCE if transformer_training_ran else PATIENCE,
            "learning_rate": TRANSFORMER_LEARNING_RATE if transformer_training_ran else LEARNING_RATE,
            "transformer_max_length": TRANSFORMER_MAX_LENGTH if transformer_training_ran else None,
            "batch_size": TRANSFORMER_BATCH_SIZE if transformer_training_ran else None,
            "hard_negative_loss_weight": 1.5,
            "pairwise_training_used": False,
        },
        "dataset": {
            "train_rows": len(train_rows),
            "val_rows": len(val_predictions),
            "test_rows": len(test_predictions),
            "train_v6_0_2_rows": sum(1 for row in train_rows if row.get("label_source") == "v6_0_2"),
            "val_v6_0_2_rows": sum(1 for row in val_predictions if row.get("label_source") == "v6_0_2"),
            "test_v6_0_2_rows": sum(1 for row in test_predictions if row.get("label_source") == "v6_0_2"),
        },
        "training_history": train_result["history"],
        "best_validation_epoch": train_result["best_epoch"],
        "best_validation_mse": train_result["best_val_mse"],
        "validation": val_eval,
        "test": test_eval,
        "baseline_comparison": {
            "val": baseline_comparison(val_predictions),
            "test": baseline_comparison(test_predictions),
            "note": "Only bm25 has substantial V6.0.2 baseline metadata; v3_3_ltr, v4_1_blend, and v4_9_guarded_text_blend are unavailable in the V6.0.2 examples.",
        },
        "leakage_check": leakage,
        "pairwise_data_audit": pairwise,
        "promising_for_v6_2": val_eval["ranking"]["aggregate"]["reading_ndcg_at_10"] >= 0.70 and test_eval["ranking"]["aggregate"]["hard_neg_at_10"] <= 0.25,
        "protected_hashes": protected_status(resolve_repo_path(args.v39_labels)),
    }


def render_markdown(report: dict[str, Any]) -> str:
    val = report["validation"]["ranking"]["aggregate"]
    test = report["test"]["ranking"]["aggregate"]
    lines = [
        "# V6.1 Cross-Encoder Reranker Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Model type: `{report['model_type']}`",
        f"- Transformer training ran: `{report['transformer_training_ran']}`",
        f"- Transformer status: {report['transformer_status']['reason']}",
        f"- Train/val/test rows: `{report['dataset']['train_rows']}` / `{report['dataset']['val_rows']}` / `{report['dataset']['test_rows']}`",
        f"- Best validation epoch: `{report['best_validation_epoch']}`",
        "",
        "## Validation Metrics",
        "",
        f"- Reading NDCG@10: `{val['reading_ndcg_at_10']:.4f}`",
        f"- Topic NDCG@10: `{val['topic_ndcg_at_10']:.4f}`",
        f"- MRR@10: `{val['mrr_at_10']:.4f}`",
        f"- Recall@10: `{val['recall_at_10']:.4f}`",
        f"- Hard-neg@10: `{val['hard_neg_at_10']:.4f}`",
        "",
        "## Test Metrics",
        "",
        f"- Reading NDCG@10: `{test['reading_ndcg_at_10']:.4f}`",
        f"- Topic NDCG@10: `{test['topic_ndcg_at_10']:.4f}`",
        f"- MRR@10: `{test['mrr_at_10']:.4f}`",
        f"- Recall@10: `{test['recall_at_10']:.4f}`",
        f"- Hard-neg@10: `{test['hard_neg_at_10']:.4f}`",
        "",
        "## Leakage And Pairwise Audit",
        "",
        f"- Input is inference-safe: `{report['leakage_check']['input_is_inference_safe']}`",
        f"- Pairwise examples: `{report['pairwise_data_audit']['pairwise_example_count']}`",
        f"- Pairwise used in V6.1 training: `{report['pairwise_data_audit']['used_in_v6_1_training']}`",
        f"- Promising for V6.2: `{report['promising_for_v6_2']}`",
        "",
        "## Protected Hashes",
        "",
    ]
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def save_config(path: Path, report: dict[str, Any]) -> None:
    config = {
        "schema_version": "v6.1_cross_encoder_reranker_config",
        "created_at": report["created_at"],
        "model_type": report["model_type"],
        "runtime_integration_status": "offline_only_not_integrated",
        "transformer_training_ran": report["transformer_training_ran"],
        "training_config": report["training_config"],
        "best_validation_epoch": report["best_validation_epoch"],
        "best_validation_mse": report["best_validation_mse"],
        "checkpoint_saved": False,
        "weights_committed": False,
        "input_fields": sorted(ALLOWED_INPUT_FIELDS),
        "forbidden_fields_used": report["leakage_check"]["forbidden_fields_used"],
    }
    write_json(path, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--val", default=str(DEFAULT_VAL))
    parser.add_argument("--test", default=str(DEFAULT_TEST))
    parser.add_argument("--pairwise", default=str(DEFAULT_PAIRWISE))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--val-predictions", default=str(DEFAULT_VAL_PREDICTIONS))
    parser.add_argument("--test-predictions", default=str(DEFAULT_TEST_PREDICTIONS))
    parser.add_argument("--failure-examples", default=str(DEFAULT_FAILURE_EXAMPLES))
    parser.add_argument("--config-out", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic_seed()
    train_rows = load_jsonl(resolve_repo_path(args.train))
    val_rows = load_jsonl(resolve_repo_path(args.val))
    test_rows = load_jsonl(resolve_repo_path(args.test))
    transformer_available, transformer_reason = transformer_weights_available(MODEL_NAME)
    transformer_status = {
        "preferred_model_name": MODEL_NAME,
        "dependencies_available": True,
        "local_pretrained_weights_available": transformer_available,
        "reason": transformer_reason,
    }
    leakage = assert_no_forbidden_input_fields({"query", "title", "abstract"})
    if not leakage["input_is_inference_safe"]:
        raise SystemExit(f"Leakage check failed: {json.dumps(leakage, sort_keys=True)}")
    if transformer_available:
        model_type = "cross_encoder_transformer"
        transformer_training_ran = True
        train_result = train_transformer_cross_encoder(train_rows, val_rows)
        val_scores = predict_transformer_cross_encoder(train_result["model"], train_result["tokenizer"], val_rows)
        test_scores = predict_transformer_cross_encoder(train_result["model"], train_result["tokenizer"], test_rows)
    else:
        model_type = "fallback_neural_mlp"
        transformer_training_ran = False
        train_result = train_fallback_mlp(train_rows, val_rows)
        val_scores = predict_fallback_mlp(train_result["model"], train_result["vectorizer"], val_rows)
        test_scores = predict_fallback_mlp(train_result["model"], train_result["vectorizer"], test_rows)
    val_predictions = add_predictions(val_rows, val_scores, "val")
    test_predictions = add_predictions(test_rows, test_scores, "test")
    pairwise = pairwise_audit(resolve_repo_path(args.pairwise), train_rows)
    report = build_report(
        train_rows=train_rows,
        val_predictions=val_predictions,
        test_predictions=test_predictions,
        train_result=train_result,
        transformer_status=transformer_status,
        model_type=model_type,
        transformer_training_ran=transformer_training_ran,
        leakage=leakage,
        pairwise=pairwise,
        args=args,
    )
    write_jsonl(resolve_repo_path(args.val_predictions), val_predictions)
    write_jsonl(resolve_repo_path(args.test_predictions), test_predictions)
    write_jsonl(resolve_repo_path(args.failure_examples), failure_examples(val_predictions + test_predictions, "neural_score"))
    write_json(resolve_repo_path(args.report_json), report)
    write_text(resolve_repo_path(args.report_md), render_markdown(report))
    save_config(resolve_repo_path(args.config_out), report)
    print(f"Wrote V6.1 cross-encoder reranker report to {resolve_repo_path(args.report_json)}")


if __name__ == "__main__":
    main()
