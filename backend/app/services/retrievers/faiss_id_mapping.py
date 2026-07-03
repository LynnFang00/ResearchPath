from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FaissIdMapping:
    """Position-to-paper-id mapping stored beside a FAISS index."""

    id_map_path: Path
    document_ids: np.ndarray
    model_name: str | None = None

    @property
    def count(self) -> int:
        return int(self.document_ids.shape[0])

    @property
    def min_paper_id(self) -> int | None:
        return int(self.document_ids.min()) if self.count else None

    @property
    def max_paper_id(self) -> int | None:
        return int(self.document_ids.max()) if self.count else None

    def paper_id_for_position(self, position: int) -> int:
        if position < 0 or position >= self.count:
            raise IndexError(f"FAISS position {position} is outside id map with {self.count} entries.")
        return int(self.document_ids[position])

    def position_by_paper_id(self) -> dict[int, int]:
        return {int(paper_id): int(position) for position, paper_id in enumerate(self.document_ids)}

    def diagnostics(self) -> dict[str, Any]:
        return {
            "id_map_path": str(self.id_map_path),
            "document_id_count": self.count,
            "unique_document_id_count": len(set(int(value) for value in self.document_ids.tolist())),
            "min_paper_id": self.min_paper_id,
            "max_paper_id": self.max_paper_id,
            "model_name": self.model_name,
        }


def load_faiss_id_mapping(id_map_path: Path) -> FaissIdMapping:
    payload = np.load(id_map_path, allow_pickle=False)
    if "document_ids" not in payload.files:
        raise ValueError(f"FAISS id map does not contain document_ids: {id_map_path}")
    model_name = str(payload["model_name"]) if "model_name" in payload.files else None
    return FaissIdMapping(
        id_map_path=id_map_path,
        document_ids=payload["document_ids"].astype(np.int64),
        model_name=model_name,
    )
