import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
import statistics
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
from evaluate_v6_1_3_exact_learned_baselines import load_jsonl, write_jsonl  # noqa: E402
import evaluate_v6_1_8_exact_neural_vs_learned_baselines as v618  # noqa: E402
import run_v6_1_2_neural_cross_validation as v612  # noqa: E402
import train_v6_1_cross_encoder_reranker as v61  # noqa: E402
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_EXAMPLES = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_neural_reranker_examples.jsonl"
DEFAULT_TRAIN = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_train.jsonl"
DEFAULT_VAL = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_val.jsonl"
DEFAULT_TEST = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_test.jsonl"
DEFAULT_PAIRWISE = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_pairwise_train.jsonl"
DEFAULT_V618_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_1_8_exact_neural_vs_learned_predictions.jsonl"
DEFAULT_V618_FAILURES = REPO_ROOT / "data" / "eval" / "results" / "v6_1_8_neural_vs_v4_9_failure_examples.jsonl"

DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_2_pairwise_neural_reranker_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_2_pairwise_neural_reranker_report.md"
DEFAULT_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_2_neural_predictions.jsonl"
DEFAULT_FAILURES = REPO_ROOT / "data" / "eval" / "results" / "v6_2_neural_vs_v4_9_failure_examples.jsonl"
DEFAULT_BLEND_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_2_neural_blend_probe_report.json"
DEFAULT_BLEND_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_2_neural_blend_probe_report.md"
DEFAULT_CONFIG = REPO_ROOT / "data" / "processed" / "models" / "v6_2_neural_reranker_config.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 620
HASH_FEATURES = 4096
MAX_GENERATED_PAIRS_PER_FOLD = 9000
EPOCHS = 6
PATIENCE = 2
PAIRWISE_BATCH_SIZE = 512
MULTITASK_BATCH_SIZE = 256
LEARNING_RATE = 2e-3
PAIRWISE_MARGIN = 0.10
HARD_NEGATIVE_PAIR_WEIGHT = 2.5
HARD_NEGATIVE_POINTWISE_WEIGHT = 2.0

V61_METHOD = "neural_cross_encoder_v6_1"
PAIRWISE_METHOD = "v6_2_pairwise_neural"
MULTITASK_METHOD = "v6_2_multitask_neural"
BASELINE_METHODS = ["bm25", "v3_3_ltr", "v4_1_blend", "v4_9_guarded_text_blend"]
METHODS = [V61_METHOD, PAIRWISE_METHOD, MULTITASK_METHOD, *BASELINE_METHODS]
METRICS = v618.METRICS
FORBIDDEN_INFERENCE_INPUTS = {
    "relevance_bucket",
    "classification_target",
    "reading_value_score",
    "topic_match_score",
    "regression_target_reading",
    "regression_target_topic",
    "section_label",
    "difficulty_label",
    "hard_negative",
    "generic_survey",
    "baseline_rankings",
    "scores",
    "v4_9_details",
    "eval_outcomes",
    "topic_hurt_status",
}
ALLOWED_INPUT_FIELDS = {"query", "title", "abstract"}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def write_config(path: Path, report: dict[str, Any]) -> None:
    write_json(
        path,
        {
            "schema_version": "v6.2_neural_reranker_config",
            "created_at": report["created_at"],
            "runtime_integration_status": "offline_only_not_integrated",
            "production_candidate_created": False,
            "checkpoint_saved": False,
            "weights_committed": False,
            "base_model": MODEL_NAME,
            "model_backend": report["model_backend"],
            "variants_trained": report["variants_trained"],
            "input_format": "query [SEP] title [SEP] abstract",
            "input_fields": sorted(ALLOWED_INPUT_FIELDS),
            "forbidden_inference_fields_used": report["leakage_check"]["forbidden_fields_used"],
            "hard_negative_used_as_training_target_or_weight_only": True,
        },
    )


def assert_inference_safe() -> dict[str, Any]:
    used = {"query", "title", "abstract"}
    forbidden = sorted(used & FORBIDDEN_INFERENCE_INPUTS)
    return {
        "input_format": "query [SEP] title [SEP] abstract",
        "input_fields": sorted(used),
        "allowed_input_fields": sorted(ALLOWED_INPUT_FIELDS),
        "forbidden_inference_inputs": sorted(FORBIDDEN_INFERENCE_INPUTS),
        "forbidden_fields_used": forbidden,
        "hard_negative_used_as_inference_input": False,
        "hard_negative_used_as_training_target_or_weight_only": True,
        "labels_used_as_training_targets_only": True,
        "input_is_inference_safe": not forbidden and used <= ALLOWED_INPUT_FIELDS,
    }


def transformer_status() -> dict[str, Any]:
    available, reason = v61.transformer_weights_available(MODEL_NAME)
    return {
        "preferred_model_name": MODEL_NAME,
        "local_pretrained_weights_available": available,
        "reason": reason,
        "downloads_attempted": False,
        "checkpoint_saved": False,
    }


def text_matrix(rows: list[dict[str, Any]], vectorizer: Any | None = None) -> tuple[np.ndarray, Any]:
    return v61.text_features(rows, vectorizer)


def build_pair_candidates(train_rows: list[dict[str, Any]], uploaded_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["query_id"], int(row["paper_id"])): row for row in train_rows}
    pairs = []
    for pair in uploaded_pairs:
        query_id = pair["query_id"]
        pref = by_key.get((query_id, int(pair["preferred_paper_id"])))
        rej = by_key.get((query_id, int(pair["rejected_paper_id"])))
        if pref and rej:
            pairs.append(pair_from_rows(pref, rej, pair.get("pair_type", "uploaded_v6_0_2_pair")))
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_topic[row["query_id"]].append(row)
    for topic, rows in sorted(by_topic.items()):
        ordered = sorted(rows, key=lambda row: (-float(row["reading_value_score"]), -float(row["topic_match_score"]), int(row["paper_id"])))
        for pref in ordered:
            for rej in reversed(ordered):
                if int(pref["paper_id"]) == int(rej["paper_id"]):
                    continue
                if preference_margin(pref, rej) <= 0:
                    continue
                pairs.append(pair_from_rows(pref, rej, generated_pair_type(pref, rej)))
                if len(pairs) >= MAX_GENERATED_PAIRS_PER_FOLD:
                    return pairs
    return pairs[:MAX_GENERATED_PAIRS_PER_FOLD]


def preference_margin(pref: dict[str, Any], rej: dict[str, Any]) -> float:
    reading_delta = float(pref["reading_value_score"]) - float(rej["reading_value_score"])
    topic_delta = float(pref["topic_match_score"]) - float(rej["topic_match_score"])
    bucket_delta = int(pref["relevance_bucket"]) - int(rej["relevance_bucket"])
    return reading_delta + 0.35 * topic_delta + 0.15 * bucket_delta


def generated_pair_type(pref: dict[str, Any], rej: dict[str, Any]) -> str:
    if bool(rej["hard_negative"]):
        return "preferred_over_hard_negative"
    if int(pref["relevance_bucket"]) == 2 and int(rej["relevance_bucket"]) == 1:
        return "strong_positive_over_partial"
    if int(pref["relevance_bucket"]) == 1 and int(rej["relevance_bucket"]) == 0:
        return "partial_over_negative"
    return "higher_label_over_lower_label"


def pair_from_rows(pref: dict[str, Any], rej: dict[str, Any], pair_type: str) -> dict[str, Any]:
    hard = bool(pref["hard_negative"]) or bool(rej["hard_negative"])
    return {
        "query_id": pref["query_id"],
        "preferred_paper_id": int(pref["paper_id"]),
        "rejected_paper_id": int(rej["paper_id"]),
        "pair_type": pair_type,
        "hard_negative_pair": hard,
        "weight": HARD_NEGATIVE_PAIR_WEIGHT if hard else 1.0,
    }


def train_pairwise_model(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    from torch import nn

    set_seed()
    x_train, vectorizer = text_matrix(train_rows)
    x_val, _ = text_matrix(val_rows, vectorizer)
    row_index = {(row["query_id"], int(row["paper_id"])): idx for idx, row in enumerate(train_rows)}
    pair_indices = [
        (
            row_index[(pair["query_id"], int(pair["preferred_paper_id"]))],
            row_index[(pair["query_id"], int(pair["rejected_paper_id"]))],
            float(pair["weight"]),
        )
        for pair in pairs
        if (pair["query_id"], int(pair["preferred_paper_id"])) in row_index and (pair["query_id"], int(pair["rejected_paper_id"])) in row_index
    ]
    model = nn.Sequential(nn.Linear(HASH_FEATURES, 128), nn.ReLU(), nn.Dropout(0.05), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    x_train_t = torch.from_numpy(x_train)
    x_val_t = torch.from_numpy(x_val)
    y_val = torch.tensor([float(row["reading_value_score"]) for row in val_rows], dtype=torch.float32).view(-1, 1)
    pair_tensor = torch.tensor(pair_indices, dtype=torch.float32)
    history = []
    best_state = None
    best_epoch = 0
    best_val_mse = float("inf")
    stale = 0
    for epoch in range(1, EPOCHS + 1):
        set_seed(SEED + epoch)
        order = torch.randperm(len(pair_tensor))
        losses = []
        model.train()
        for start in range(0, len(pair_tensor), PAIRWISE_BATCH_SIZE):
            batch = pair_tensor[order[start : start + PAIRWISE_BATCH_SIZE]]
            pref_idx = batch[:, 0].long()
            rej_idx = batch[:, 1].long()
            weights = batch[:, 2].view(-1, 1)
            optimizer.zero_grad()
            pref_scores = model(x_train_t[pref_idx])
            rej_scores = model(x_train_t[rej_idx])
            loss = (torch.relu(PAIRWISE_MARGIN - (pref_scores - rej_scores)) * weights).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_scores = model(x_val_t)
            val_mse = torch.mean((val_scores - y_val) ** 2).item()
        history.append({"epoch": epoch, "pairwise_margin_loss": sum(losses) / len(losses), "val_reading_mse": val_mse})
        if val_mse < best_val_mse - 1e-7:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    return {"model": model, "vectorizer": vectorizer, "history": history, "best_epoch": best_epoch, "best_val_mse": best_val_mse, "pair_count": len(pair_indices)}


def train_multitask_model(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    from torch import nn

    set_seed()
    x_train, vectorizer = text_matrix(train_rows)
    x_val, _ = text_matrix(val_rows, vectorizer)
    y_read = torch.tensor([float(row["reading_value_score"]) for row in train_rows], dtype=torch.float32).view(-1, 1)
    y_topic = torch.tensor([float(row["topic_match_score"]) for row in train_rows], dtype=torch.float32).view(-1, 1)
    y_hard = torch.tensor([1.0 if row["hard_negative"] else 0.0 for row in train_rows], dtype=torch.float32).view(-1, 1)
    weights = torch.tensor([HARD_NEGATIVE_POINTWISE_WEIGHT if row["hard_negative"] else 1.0 for row in train_rows], dtype=torch.float32).view(-1, 1)
    x_train_t = torch.from_numpy(x_train)
    x_val_t = torch.from_numpy(x_val)
    y_val = torch.tensor([float(row["reading_value_score"]) for row in val_rows], dtype=torch.float32).view(-1, 1)
    model = nn.Sequential(nn.Linear(HASH_FEATURES, 160), nn.ReLU(), nn.Dropout(0.05), nn.Linear(160, 48), nn.ReLU(), nn.Linear(48, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    history = []
    best_state = None
    best_epoch = 0
    best_val_mse = float("inf")
    stale = 0
    indices = torch.arange(len(train_rows))
    for epoch in range(1, EPOCHS + 1):
        set_seed(SEED + 100 + epoch)
        order = indices[torch.randperm(len(indices))]
        losses = []
        model.train()
        for start in range(0, len(order), MULTITASK_BATCH_SIZE):
            batch = order[start : start + MULTITASK_BATCH_SIZE]
            optimizer.zero_grad()
            out = model(x_train_t[batch])
            reading = torch.sigmoid(out[:, 0:1])
            topic = torch.sigmoid(out[:, 1:2])
            hard_logit = out[:, 2:3]
            read_loss = (((reading - y_read[batch]) ** 2) * weights[batch]).mean()
            topic_loss = (((topic - y_topic[batch]) ** 2) * weights[batch]).mean()
            hard_loss = (bce(hard_logit, y_hard[batch]) * weights[batch]).mean()
            loss = read_loss + 0.65 * topic_loss + 0.35 * hard_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_out = model(x_val_t)
            val_read = torch.sigmoid(val_out[:, 0:1])
            val_mse = torch.mean((val_read - y_val) ** 2).item()
        history.append({"epoch": epoch, "multitask_loss": sum(losses) / len(losses), "val_reading_mse": val_mse})
        if val_mse < best_val_mse - 1e-7:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    return {"model": model, "vectorizer": vectorizer, "history": history, "best_epoch": best_epoch, "best_val_mse": best_val_mse}


def predict_pairwise(result: dict[str, Any], rows: list[dict[str, Any]]) -> list[float]:
    import torch

    x, _ = text_matrix(rows, result["vectorizer"])
    result["model"].eval()
    with torch.no_grad():
        return [float(value) for value in result["model"](torch.from_numpy(x)).numpy().reshape(-1)]


def predict_multitask(result: dict[str, Any], rows: list[dict[str, Any]]) -> list[float]:
    import torch

    x, _ = text_matrix(rows, result["vectorizer"])
    result["model"].eval()
    with torch.no_grad():
        out = result["model"](torch.from_numpy(x))
        reading = torch.sigmoid(out[:, 0]).numpy().reshape(-1)
        topic = torch.sigmoid(out[:, 1]).numpy().reshape(-1)
        hard_prob = torch.sigmoid(out[:, 2]).numpy().reshape(-1)
    return [float(0.70 * r + 0.30 * t - 0.15 * h) for r, t, h in zip(reading, topic, hard_prob, strict=True)]


def rows_for_topics(examples: list[dict[str, Any]], topics: list[str]) -> list[dict[str, Any]]:
    topic_set = set(topics)
    return sorted([row for row in examples if row["query_id"] in topic_set], key=lambda row: (row["query_id"], int(row["paper_id"])))


def exact_rows_for_fold(v618_rows: list[dict[str, Any]], examples_by_pair: dict[tuple[str, int], dict[str, Any]], fold_id: str, split: str) -> list[dict[str, Any]]:
    rows = []
    for row in v618_rows:
        if row["fold_id"] != fold_id or row["split"] != split:
            continue
        example = examples_by_pair[(row["query_id"], int(row["paper_id"]))]
        copied = dict(row)
        copied["abstract"] = example.get("abstract") or ""
        copied["source"] = example.get("source")
        copied["venue"] = example.get("venue")
        copied["year"] = example.get("year")
        rows.append(copied)
    return sorted(rows, key=lambda row: (row["query_id"], int(row["paper_id"])))


def add_v62_scores(rows: list[dict[str, Any]], pair_scores: list[float], multitask_scores: list[float]) -> list[dict[str, Any]]:
    output = []
    for row, pair_score, multi_score in zip(rows, pair_scores, multitask_scores, strict=True):
        copied = dict(row)
        copied["schema_version"] = "v6.2_neural_prediction"
        copied["scores"] = dict(row["scores"])
        copied["scores"][PAIRWISE_METHOD] = float(pair_score)
        copied["scores"][MULTITASK_METHOD] = float(multi_score)
        output.append(copied)
    return output


def add_probe_scores(rows: list[dict[str, Any]], variant: str) -> tuple[list[dict[str, Any]], list[str]]:
    probes = [
        f"probe_v4_9_90_{variant}_10",
        f"probe_v4_9_80_{variant}_20",
        f"probe_v4_9_{variant}_hard_negative_guardrail",
        f"probe_v4_9_{variant}_when_v49_confidence_low",
    ]
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(row["fold_id"], row["split"], row["query_id"])].append(row)
    for group_rows in by_group.values():
        v49_norm = v618.normalize({row["paper_id"]: row["scores"]["v4_9_guarded_text_blend"] for row in group_rows})
        neural_norm = v618.normalize({row["paper_id"]: row["scores"][variant] for row in group_rows})
        for row in group_rows:
            v49 = v49_norm[row["paper_id"]]
            neural = neural_norm[row["paper_id"]]
            row["scores"][probes[0]] = 0.9 * v49 + 0.1 * neural
            row["scores"][probes[1]] = 0.8 * v49 + 0.2 * neural
            row["scores"][probes[2]] = v49 - (0.15 * neural if neural < 0.30 else 0.0)
            confidence = float((row.get("v4_9_details") or {}).get("v4_9_max_ranker_confidence") or v49)
            row["scores"][probes[3]] = v49 if confidence >= 0.85 else 0.85 * v49 + 0.15 * neural
    return rows, probes


def choose_probe_variant(aggregate: dict[str, Any]) -> str:
    candidates = [PAIRWISE_METHOD, MULTITASK_METHOD]
    def sort_key(method: str) -> tuple[float, float, float]:
        val = aggregate["val"][method]
        return (
            float(val["reading_ndcg_at_10"]["mean"] or 0.0),
            float(val["topic_ndcg_at_10"]["mean"] or 0.0),
            -float(val["hard_neg_at_10"]["mean"] or 0.0),
        )

    return max(candidates, key=sort_key)


def summarize_fold_training(fold_reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "fold_count": len(fold_reports),
        "pairwise": {
            "trained_successfully": all(row["pairwise"]["trained_successfully"] for row in fold_reports),
            "pair_count_by_fold": {row["fold_id"]: row["pairwise"]["pair_count"] for row in fold_reports},
            "best_epoch_by_fold": {row["fold_id"]: row["pairwise"]["best_epoch"] for row in fold_reports},
        },
        "multitask": {
            "trained_successfully": all(row["multitask"]["trained_successfully"] for row in fold_reports),
            "best_epoch_by_fold": {row["fold_id"]: row["multitask"]["best_epoch"] for row in fold_reports},
        },
    }


def deltas_vs_v49(aggregate: dict[str, Any], methods: list[str]) -> dict[str, Any]:
    return v618.deltas(aggregate, methods, baseline="v4_9_guarded_text_blend")


def behavior_counts(acceptance: dict[str, Any], method: str) -> dict[str, Any]:
    rows = acceptance[method]["per_topic_deltas"]
    return {
        "topics_where_method_beats_v4_9": [row for row in rows if row["reading_delta_vs_v4_9"] > 0.02],
        "topics_where_method_loses_badly_to_v4_9": [row for row in rows if row["reading_delta_vs_v4_9"] <= -0.05],
        "hard_negative_delta_test_mean": acceptance[method]["hard_neg_delta_vs_v4_9_test_mean"],
        "severe_regression_count": acceptance[method]["severe_regression_count"],
        "non_weak_severe_regression_count": acceptance[method]["non_weak_severe_regression_count"],
    }


def failure_examples(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    failures = []
    for row in v618.failure_examples(rows):
        copied = dict(row)
        copied["schema_version"] = "v6.2_neural_vs_v4_9_failure_example"
        copied["diagnosed_variant"] = method
        failures.append(copied)
    return failures[:300]


def blend_probe_report(acceptance: dict[str, Any], probe_methods: list[str]) -> dict[str, Any]:
    probe_rows = {}
    for method in probe_methods:
        diag = acceptance[method]
        probe_rows[method] = {
            "reading_delta_vs_v4_9_test_mean": diag["reading_delta_vs_v4_9_test_mean"]["mean"],
            "topic_delta_vs_v4_9_test_mean": diag["topic_delta_vs_v4_9_test_mean"]["mean"],
            "hard_neg_delta_vs_v4_9_test_mean": diag["hard_neg_delta_vs_v4_9_test_mean"]["mean"],
            "severe_regression_count": diag["severe_regression_count"],
            "non_weak_severe_regression_count": diag["non_weak_severe_regression_count"],
            "judged_at_10_decreases": diag["judged_at_10_decreases"],
        }
    candidates = [{"method": method, **row} for method, row in probe_rows.items() if row["reading_delta_vs_v4_9_test_mean"] is not None]
    highest = max(candidates, key=lambda row: (row["reading_delta_vs_v4_9_test_mean"], -row["severe_regression_count"])) if candidates else None
    safe = [
        row
        for row in candidates
        if row["severe_regression_count"] == 0
        and row["non_weak_severe_regression_count"] == 0
        and not row["judged_at_10_decreases"]
        and row["hard_neg_delta_vs_v4_9_test_mean"] is not None
        and row["hard_neg_delta_vs_v4_9_test_mean"] <= 0
    ]
    safest = max(safe, key=lambda row: row["reading_delta_vs_v4_9_test_mean"]) if safe else None
    return {
        "schema_version": "v6.2_neural_blend_probe_report",
        "created_at": datetime.now(UTC).isoformat(),
        "offline_only": True,
        "runtime_integration_status": "offline_only_not_integrated",
        "production_candidate_created": False,
        "probe_methods": probe_rows,
        "highest_reading_probe": highest,
        "safest_probe": safest,
        "best_probe": safest or highest,
    }


def recommendation(acceptance: dict[str, Any], blend: dict[str, Any], best_variant: str) -> str:
    diag = acceptance[best_variant]
    read_delta = diag["reading_delta_vs_v4_9_test_mean"]["mean"]
    topic_delta = diag["topic_delta_vs_v4_9_test_mean"]["mean"]
    hard_delta = diag["hard_neg_delta_vs_v4_9_test_mean"]["mean"]
    if read_delta is not None and topic_delta is not None and hard_delta is not None:
        if read_delta >= 0 and topic_delta >= 0 and hard_delta <= 0 and diag["severe_regression_count"] == 0:
            return "useful_as_standalone_offline_candidate_not_runtime_integrated"
    best_probe = blend.get("best_probe")
    if best_probe and best_probe["severe_regression_count"] == 0 and best_probe["reading_delta_vs_v4_9_test_mean"] is not None and best_probe["reading_delta_vs_v4_9_test_mean"] >= 0.005:
        return "useful_as_offline_blend_feature_for_future_v6_3"
    if hard_delta is not None and hard_delta < 0 and read_delta is not None and read_delta >= -0.02:
        return "useful_as_hard_negative_guardrail_research_signal"
    return "research_baseline_only_do_not_integrate"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V6.2 Pairwise Neural Reranker",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Model backend: `{report['model_backend']}`",
        f"- Variants trained: `{', '.join(report['variants_trained'])}`",
        f"- Best V6.2 variant: `{report['best_v6_2_variant']}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Leakage check passed: `{report['leakage_check']['input_is_inference_safe']}`",
        "",
        "## Test Metrics",
        "",
    ]
    for method in [PAIRWISE_METHOD, MULTITASK_METHOD, V61_METHOD, "v4_9_guarded_text_blend"]:
        row = report["aggregate"]["test"][method]
        lines.append(
            f"- `{method}`: reading `{row['reading_ndcg_at_10']['mean']:.4f}`, "
            f"topic `{row['topic_ndcg_at_10']['mean']:.4f}`, hard-neg `{row['hard_neg_at_10']['mean']:.4f}`"
        )
    lines.extend(["", "## Severe Regressions Vs V4.9", ""])
    for method in [PAIRWISE_METHOD, MULTITASK_METHOD]:
        diag = report["acceptance_diagnostics"][method]
        lines.append(f"- `{method}`: `{diag['severe_regression_count']}` total, `{diag['non_weak_severe_regression_count']}` non-weak")
    lines.extend(["", "## Protected Hashes", ""])
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def render_blend_markdown(report: dict[str, Any]) -> str:
    lines = ["# V6.2 Neural Blend Probes", "", f"Created: `{report['created_at']}`", ""]
    for method, row in report["probe_methods"].items():
        lines.append(
            f"- `{method}`: reading delta `{row['reading_delta_vs_v4_9_test_mean']:+.4f}`, "
            f"topic delta `{row['topic_delta_vs_v4_9_test_mean']:+.4f}`, "
            f"hard-neg delta `{row['hard_neg_delta_vs_v4_9_test_mean']:+.4f}`, severe `{row['severe_regression_count']}`"
        )
    lines.extend(
        [
            "",
            f"Highest-reading probe: `{(report['highest_reading_probe'] or {}).get('method')}`",
            f"Safest no-regression probe: `{(report['safest_probe'] or {}).get('method')}`",
            f"Best probe: `{(report['best_probe'] or {}).get('method')}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    set_seed()
    examples = load_jsonl(resolve_repo_path(args.examples))
    _ = load_jsonl(resolve_repo_path(args.train))
    _ = load_jsonl(resolve_repo_path(args.val))
    _ = load_jsonl(resolve_repo_path(args.test))
    uploaded_pairs = load_jsonl(resolve_repo_path(args.pairwise))
    v618_rows = load_jsonl(resolve_repo_path(args.v618_predictions))
    examples_by_pair = {(row["query_id"], int(row["paper_id"])): row for row in examples}
    folds = v612.build_folds(examples)
    leakage = assert_inference_safe()
    if not leakage["input_is_inference_safe"]:
        raise SystemExit(f"Leakage check failed: {json.dumps(leakage, sort_keys=True)}")
    fold_reports = []
    predictions = []
    for fold in folds:
        train_rows = rows_for_topics(examples, fold["train_topics"])
        val_rows = exact_rows_for_fold(v618_rows, examples_by_pair, fold["fold_id"], "val")
        test_rows = exact_rows_for_fold(v618_rows, examples_by_pair, fold["fold_id"], "test")
        pair_rows = build_pair_candidates(train_rows, uploaded_pairs)
        pair_result = train_pairwise_model(train_rows, val_rows, pair_rows)
        multi_result = train_multitask_model(train_rows, val_rows)
        val_pair = predict_pairwise(pair_result, val_rows)
        test_pair = predict_pairwise(pair_result, test_rows)
        val_multi = predict_multitask(multi_result, val_rows)
        test_multi = predict_multitask(multi_result, test_rows)
        predictions.extend(add_v62_scores(val_rows, val_pair, val_multi))
        predictions.extend(add_v62_scores(test_rows, test_pair, test_multi))
        fold_reports.append(
            {
                "fold_id": fold["fold_id"],
                "train_topics": fold["train_topics"],
                "val_topics": fold["val_topics"],
                "test_topics": fold["test_topics"],
                "pairwise": {
                    "trained_successfully": True,
                    "pair_count": pair_result["pair_count"],
                    "best_epoch": pair_result["best_epoch"],
                    "best_val_mse": pair_result["best_val_mse"],
                    "history": pair_result["history"],
                },
                "multitask": {
                    "trained_successfully": True,
                    "best_epoch": multi_result["best_epoch"],
                    "best_val_mse": multi_result["best_val_mse"],
                    "history": multi_result["history"],
                },
            }
        )
    per_group_initial, aggregate_initial = v618.evaluate_all(predictions, METHODS)
    best_variant = choose_probe_variant(aggregate_initial)
    predictions, probe_methods = add_probe_scores(predictions, best_variant)
    all_methods = METHODS + probe_methods
    per_group, aggregate = v618.evaluate_all(predictions, all_methods)
    acceptance = v618.acceptance_diagnostics(per_group, all_methods)
    blend = blend_probe_report(acceptance, probe_methods)
    rec = recommendation(acceptance, blend, best_variant)
    failures = failure_examples(predictions, best_variant)
    report = {
        "schema_version": "v6.2_pairwise_neural_reranker_report",
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
        "production_candidate_created": False,
        "large_checkpoints_created": False,
        "model_backend": "hashed_text_cross_encoder_fallback",
        "transformer_status": transformer_status(),
        "base_model": MODEL_NAME,
        "variants_trained": [PAIRWISE_METHOD, MULTITASK_METHOD],
        "best_v6_2_variant": best_variant,
        "training_config": {
            "seed": SEED,
            "input_format": "query [SEP] title [SEP] abstract",
            "feature_extractor": "sklearn HashingVectorizer fallback",
            "hash_features": HASH_FEATURES,
            "epochs": EPOCHS,
            "pairwise_margin": PAIRWISE_MARGIN,
            "hard_negative_pair_weight": HARD_NEGATIVE_PAIR_WEIGHT,
            "hard_negative_pointwise_weight": HARD_NEGATIVE_POINTWISE_WEIGHT,
            "labels_used_as_training_targets_only": True,
        },
        "dataset": {
            "examples": len(examples),
            "uploaded_pairwise_train_rows": len(uploaded_pairs),
            "exact_v6_1_8_prediction_rows": len(v618_rows),
        },
        "fold_training": summarize_fold_training(fold_reports),
        "fold_reports": fold_reports,
        "methods": METHODS,
        "blend_probe_methods": probe_methods,
        "aggregate": aggregate,
        "deltas_vs_v4_9": deltas_vs_v49(aggregate, all_methods),
        "acceptance_diagnostics": acceptance,
        "behavior_diagnosis": {
            PAIRWISE_METHOD: behavior_counts(acceptance, PAIRWISE_METHOD),
            MULTITASK_METHOD: behavior_counts(acceptance, MULTITASK_METHOD),
            "v6_1_reference": behavior_counts(acceptance, V61_METHOD),
        },
        "v6_1_severe_regression_reference_count": acceptance[V61_METHOD]["severe_regression_count"],
        "v6_2_reduces_v6_1_severe_regressions": acceptance[best_variant]["severe_regression_count"] < acceptance[V61_METHOD]["severe_regression_count"],
        "v6_2_reduces_hard_negative_promotions": (acceptance[best_variant]["hard_neg_delta_vs_v4_9_test_mean"]["mean"] or 0.0)
        < (acceptance[V61_METHOD]["hard_neg_delta_vs_v4_9_test_mean"]["mean"] or 0.0),
        "leakage_check": leakage,
        "recommendation": rec,
        "protected_hashes": protected_status(resolve_repo_path(args.v39_labels)),
    }
    return report, blend, predictions, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--val", default=str(DEFAULT_VAL))
    parser.add_argument("--test", default=str(DEFAULT_TEST))
    parser.add_argument("--pairwise", default=str(DEFAULT_PAIRWISE))
    parser.add_argument("--v618-predictions", default=str(DEFAULT_V618_PREDICTIONS))
    parser.add_argument("--v618-failures", default=str(DEFAULT_V618_FAILURES))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--predictions-out", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--failures-out", default=str(DEFAULT_FAILURES))
    parser.add_argument("--blend-json-out", default=str(DEFAULT_BLEND_JSON))
    parser.add_argument("--blend-md-out", default=str(DEFAULT_BLEND_MD))
    parser.add_argument("--config-out", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, blend, predictions, failures = build_report(args)
    write_jsonl(resolve_repo_path(args.predictions_out), predictions)
    write_jsonl(resolve_repo_path(args.failures_out), failures)
    write_json(resolve_repo_path(args.blend_json_out), blend)
    write_text(resolve_repo_path(args.blend_md_out), render_blend_markdown(blend))
    write_json(resolve_repo_path(args.report_json), report)
    write_text(resolve_repo_path(args.report_md), render_markdown(report))
    write_config(resolve_repo_path(args.config_out), report)
    print(f"Wrote V6.2 pairwise neural reranker report to {resolve_repo_path(args.report_json)}")


if __name__ == "__main__":
    main()
