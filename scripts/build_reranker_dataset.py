import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "processed" / "training"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "training"
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class RerankerExample:
    query_paper_id: int
    candidate_paper_id: int
    split: str
    label: int
    label_source: str
    candidate_source: str
    query_text: str
    candidate_text: str

    def to_json(self) -> dict[str, Any]:
        return {
            "query_paper_id": self.query_paper_id,
            "candidate_paper_id": self.candidate_paper_id,
            "split": self.split,
            "label": self.label,
            "label_source": self.label_source,
            "candidate_source": self.candidate_source,
            "query_text": self.query_text,
            "candidate_text": self.candidate_text,
        }


def load_biencoder_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Bi-encoder dataset file was not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = (
            "query_paper_id",
            "positive_paper_id",
            "negative_paper_ids",
            "split",
            "query_text",
            "positive_text",
            "negative_texts",
        )
        for field in required:
            if field not in row:
                raise ValueError(f"Line {line_number}: missing required field '{field}'.")
        if not isinstance(row["negative_paper_ids"], list):
            raise ValueError(f"Line {line_number}: negative_paper_ids must be a list.")
        if not isinstance(row["negative_texts"], list):
            raise ValueError(f"Line {line_number}: negative_texts must be a list.")
        if len(row["negative_paper_ids"]) != len(row["negative_texts"]):
            raise ValueError(f"Line {line_number}: negative ids/texts length mismatch.")
        rows.append(row)
    return rows


def rows_to_reranker_examples(
    rows: list[dict[str, Any]],
    *,
    max_negatives_per_positive: int | None = None,
    seed: int = 13,
) -> list[RerankerExample]:
    rng = random.Random(seed)
    positive_keys = {
        (int(row["query_paper_id"]), int(row["positive_paper_id"]), row["split"])
        for row in rows
    }
    examples_by_key: dict[tuple[int, int, str], RerankerExample] = {}

    for row in rows:
        query_id = int(row["query_paper_id"])
        positive_id = int(row["positive_paper_id"])
        split = str(row["split"])
        key = (query_id, positive_id, split)
        examples_by_key[key] = RerankerExample(
            query_paper_id=query_id,
            candidate_paper_id=positive_id,
            split=split,
            label=1,
            label_source=row.get("label_source", "citation_graph"),
            candidate_source="positive",
            query_text=row["query_text"],
            candidate_text=row["positive_text"],
        )

    for row in rows:
        query_id = int(row["query_paper_id"])
        split = str(row["split"])
        negatives = list(zip(row["negative_paper_ids"], row["negative_texts"], strict=True))
        rng.shuffle(negatives)
        if max_negatives_per_positive is not None:
            negatives = negatives[:max_negatives_per_positive]
        for negative_id_raw, negative_text in negatives:
            negative_id = int(negative_id_raw)
            key = (query_id, negative_id, split)
            if key in positive_keys or key in examples_by_key:
                continue
            examples_by_key[key] = RerankerExample(
                query_paper_id=query_id,
                candidate_paper_id=negative_id,
                split=split,
                label=0,
                label_source="hard_negative",
                candidate_source=row.get("negative_source", "unknown"),
                query_text=row["query_text"],
                candidate_text=negative_text,
            )

    return sorted(
        examples_by_key.values(),
        key=lambda item: (item.split, item.query_paper_id, -item.label, item.candidate_paper_id),
    )


def split_examples(examples: list[RerankerExample]) -> dict[str, list[RerankerExample]]:
    splits: dict[str, list[RerankerExample]] = {split: [] for split in SPLITS}
    for example in examples:
        splits.setdefault(example.split, []).append(example)
    return splits


def write_jsonl(examples: list[RerankerExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_json(), ensure_ascii=True) + "\n")


def build_report(
    *,
    examples: list[RerankerExample],
    output_paths: dict[str, str],
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    split_counts = Counter(example.split for example in examples)
    label_counts = Counter(example.label for example in examples)
    query_counts = Counter(example.query_paper_id for example in examples)
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "example_count": len(examples),
        "query_count": len(query_counts),
        "positive_count": label_counts.get(1, 0),
        "negative_count": label_counts.get(0, 0),
        "split_counts": {split: split_counts.get(split, 0) for split in SPLITS},
        "output_paths": output_paths,
        "config": vars(args) if args is not None else {},
        "notes": (
            "Cross-encoder reranker data generated from bi-encoder weak labels. "
            "Positive pairs come from citation graph positives; negatives come from mined hard negatives."
        ),
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weakly supervised cross-encoder reranker data.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-negatives-per-positive", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_absolute():
        input_dir = REPO_ROOT / input_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    all_examples: list[RerankerExample] = []
    for split in SPLITS:
        rows = load_biencoder_rows(input_dir / f"bi_encoder_{split}.jsonl")
        all_examples.extend(
            rows_to_reranker_examples(
                rows,
                max_negatives_per_positive=args.max_negatives_per_positive,
                seed=args.seed,
            )
        )

    splits = split_examples(all_examples)
    output_paths = {
        "train": str(output_dir / "cross_encoder_train.jsonl"),
        "val": str(output_dir / "cross_encoder_val.jsonl"),
        "test": str(output_dir / "cross_encoder_test.jsonl"),
        "report": str(output_dir / "cross_encoder_dataset_report.json"),
    }
    for split, split_examples_ in splits.items():
        if split in output_paths:
            write_jsonl(split_examples_, Path(output_paths[split]))

    report = build_report(examples=all_examples, output_paths=output_paths, args=args)
    write_report(report, Path(output_paths["report"]))

    print(f"Examples: {len(all_examples)}")
    print(f"Train/val/test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"Positives/negatives: {report['positive_count']}/{report['negative_count']}")
    print(f"Report: {output_paths['report']}")


if __name__ == "__main__":
    main()
