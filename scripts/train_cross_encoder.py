import argparse
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = REPO_ROOT / "data" / "processed" / "training" / "cross_encoder_train.jsonl"
DEFAULT_VAL = REPO_ROOT / "data" / "processed" / "training" / "cross_encoder_val.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "models" / "cross_encoder_minilm_smoke"
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Reranker dataset file was not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ("query_text", "candidate_text", "label"):
            if field not in row:
                raise ValueError(f"Line {line_number}: missing required field '{field}'.")
        if row["label"] not in (0, 1, 0.0, 1.0):
            raise ValueError(f"Line {line_number}: label must be 0 or 1.")
        rows.append(row)
    return rows


def sample_balanced_rows(
    rows: list[dict[str, Any]],
    *,
    max_examples: int | None = None,
    positive_fraction: float = 0.5,
    seed: int = 13,
) -> list[dict[str, Any]]:
    if max_examples is None or len(rows) <= max_examples:
        sampled = list(rows)
        random.Random(seed).shuffle(sampled)
        return sampled

    if not 0.0 < positive_fraction < 1.0:
        raise ValueError("positive_fraction must be between 0 and 1.")

    rng = random.Random(seed)
    positives = [row for row in rows if int(row["label"]) == 1]
    negatives = [row for row in rows if int(row["label"]) == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    target_positives = min(len(positives), int(max_examples * positive_fraction))
    target_negatives = min(len(negatives), max_examples - target_positives)

    # Fill unused capacity if one side is smaller than requested.
    remaining = max_examples - target_positives - target_negatives
    if remaining > 0:
        extra_positives = min(len(positives) - target_positives, remaining)
        target_positives += extra_positives
        remaining -= extra_positives
    if remaining > 0:
        target_negatives += min(len(negatives) - target_negatives, remaining)

    sampled = positives[:target_positives] + negatives[:target_negatives]
    rng.shuffle(sampled)
    return sampled


def rows_to_cross_encoder_examples(rows: list[dict[str, Any]]) -> list[tuple[str, str, float]]:
    return [
        (row["query_text"], row["candidate_text"], float(row["label"]))
        for row in rows
    ]


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(row["label"]) for row in rows)
    return {"positive": counts.get(1, 0), "negative": counts.get(0, 0)}


def write_metadata(path: Path, metadata: dict[str, Any]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    metadata_path = path / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResearchPath cross-encoder reranker.")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--val", default=str(DEFAULT_VAL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-examples", type=int, default=2000)
    parser.add_argument("--max-val-examples", type=int, default=500)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-evaluator", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    val_path = Path(args.val)
    output_dir = Path(args.output_dir)
    if not train_path.is_absolute():
        train_path = REPO_ROOT / train_path
    if not val_path.is_absolute():
        val_path = REPO_ROOT / val_path
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    from sentence_transformers import CrossEncoder, InputExample
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    from torch.utils.data import DataLoader

    train_rows = load_reranker_rows(train_path)
    val_rows = load_reranker_rows(val_path)
    sampled_train = sample_balanced_rows(
        train_rows,
        max_examples=args.max_train_examples,
        positive_fraction=args.positive_fraction,
        seed=args.seed,
    )
    sampled_val = sample_balanced_rows(
        val_rows,
        max_examples=args.max_val_examples,
        positive_fraction=args.positive_fraction,
        seed=args.seed,
    )
    if not sampled_train:
        raise ValueError("No cross-encoder training examples were generated.")

    train_examples = [
        InputExample(texts=[query, candidate], label=label)
        for query, candidate, label in rows_to_cross_encoder_examples(sampled_train)
    ]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    evaluator = None
    if not args.no_evaluator and sampled_val:
        evaluator = CEBinaryClassificationEvaluator(
            sentence_pairs=[(row["query_text"], row["candidate_text"]) for row in sampled_val],
            labels=[int(row["label"]) for row in sampled_val],
            name="researchpath_cross_encoder_val",
            show_progress_bar=True,
        )

    model = CrossEncoder(args.model_name, num_labels=1)
    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(output_dir),
        save_best_model=True,
        show_progress_bar=True,
    )
    model.save(str(output_dir))

    metadata_path = write_metadata(
        output_dir,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "base_model": args.model_name,
            "output_dir": str(output_dir),
            "train_path": str(train_path),
            "val_path": str(val_path),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": warmup_steps,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_examples_used": len(sampled_train),
            "val_examples_used": len(sampled_val),
            "train_label_counts": label_counts(sampled_train),
            "val_label_counts": label_counts(sampled_val),
            "notes": (
                "ResearchPath cross-encoder reranker training run. Use after a first-stage "
                "retriever such as BM25, FAISS, or the MNRL bi-encoder."
            ),
        },
    )

    print(f"Model output: {output_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Train examples used: {len(sampled_train)}")
    print(f"Val examples used: {len(sampled_val)}")
    print(f"Train label counts: {label_counts(sampled_train)}")


if __name__ == "__main__":
    main()
