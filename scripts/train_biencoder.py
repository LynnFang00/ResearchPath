import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))


DEFAULT_TRAIN = REPO_ROOT / "data" / "processed" / "training" / "bi_encoder_train.jsonl"
DEFAULT_VAL = REPO_ROOT / "data" / "processed" / "training" / "bi_encoder_val.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "models" / "biencoder_all_minilm_smoke"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_training_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Training file was not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ("query_text", "positive_text", "negative_texts"):
            if field not in row:
                raise ValueError(f"Line {line_number}: missing required field '{field}'.")
        if not isinstance(row["negative_texts"], list) or not row["negative_texts"]:
            raise ValueError(f"Line {line_number}: negative_texts must be a non-empty list.")
        rows.append(row)
    return rows


def rows_to_triplets(
    rows: list[dict[str, Any]],
    *,
    max_triplets: int | None = None,
    seed: int = 13,
) -> list[tuple[str, str, str]]:
    triplets: list[tuple[str, str, str]] = []
    for row in rows:
        query_text = row["query_text"]
        positive_text = row["positive_text"]
        for negative_text in row["negative_texts"]:
            triplets.append((query_text, positive_text, negative_text))

    rng = random.Random(seed)
    rng.shuffle(triplets)
    if max_triplets is not None:
        triplets = triplets[:max_triplets]
    return triplets


def rows_to_pairs(
    rows: list[dict[str, Any]],
    *,
    max_pairs: int | None = None,
    seed: int = 13,
) -> list[tuple[str, str]]:
    pairs = [(row["query_text"], row["positive_text"]) for row in rows]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs


def write_metadata(path: Path, metadata: dict[str, Any]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    metadata_path = path / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a ResearchPath bi-encoder with triplet loss.")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--val", default=str(DEFAULT_VAL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--loss", choices=["triplet", "mnrl"], default="triplet")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-triplets", type=int, default=2000)
    parser.add_argument("--max-val-triplets", type=int, default=500)
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

    # Heavy imports stay inside main so tests can cover parsing helpers without loading torch.
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from sentence_transformers.evaluation import TripletEvaluator
    from torch.utils.data import DataLoader

    train_rows = load_training_rows(train_path)
    val_rows = load_training_rows(val_path)

    if args.loss == "triplet":
        train_items = rows_to_triplets(train_rows, max_triplets=args.max_train_triplets, seed=args.seed)
        val_triplets = rows_to_triplets(val_rows, max_triplets=args.max_val_triplets, seed=args.seed)
        if not train_items:
            raise ValueError("No training triplets were generated.")
    else:
        train_items = rows_to_pairs(train_rows, max_pairs=args.max_train_triplets, seed=args.seed)
        val_triplets = rows_to_triplets(val_rows, max_triplets=args.max_val_triplets, seed=args.seed)
        if not train_items:
            raise ValueError("No training pairs were generated.")

    model = SentenceTransformer(args.model_name)
    train_examples = [InputExample(texts=list(item)) for item in train_items]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = (
        losses.TripletLoss(model=model)
        if args.loss == "triplet"
        else losses.MultipleNegativesRankingLoss(model=model)
    )

    evaluator = None
    if not args.no_evaluator and val_triplets:
        evaluator = TripletEvaluator(
            anchors=[triplet[0] for triplet in val_triplets],
            positives=[triplet[1] for triplet in val_triplets],
            negatives=[triplet[2] for triplet in val_triplets],
            name="researchpath_val_triplets",
            batch_size=args.batch_size,
            show_progress_bar=True,
        )

    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(output_dir),
        save_best_model=True,
        show_progress_bar=True,
    )

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
            "loss": args.loss,
            "warmup_steps": warmup_steps,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_examples_used": len(train_items),
            "train_triplets_used": len(train_items) if args.loss == "triplet" else 0,
            "train_pairs_used": len(train_items) if args.loss == "mnrl" else 0,
            "val_triplets_used": len(val_triplets),
            "loss_name": "TripletLoss" if args.loss == "triplet" else "MultipleNegativesRankingLoss",
            "notes": (
                "ResearchPath bi-encoder training run. Compare against frozen embeddings and "
                "previous trained checkpoints before increasing scale."
            ),
        },
    )

    print(f"Model output: {output_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Loss: {args.loss}")
    print(f"Train examples used: {len(train_items)}")
    print(f"Val triplets used: {len(val_triplets)}")


if __name__ == "__main__":
    main()
