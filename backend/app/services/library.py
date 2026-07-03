from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.paper import Paper
from app.models.saved_library_item import SavedLibraryItem
from app.schemas.paper import LibraryItemResponse, LibraryItemUpsert, LibraryResponse
from app.services.formatting import paper_to_recommendation
from app.services.profile import (
    DEFAULT_USER_KEY,
    add_to_profile_list,
    add_to_profile_topics,
    decode_int_list,
    decode_string_list,
    encode_int_list,
    encode_string_list,
    get_or_create_profile,
)


def upsert_library_item(db: Session, payload: LibraryItemUpsert) -> LibraryItemResponse:
    paper = db.get(Paper, payload.paper_id)
    if paper is None:
        raise ValueError(f"Paper with id={payload.paper_id} was not found.")

    item = db.scalar(
        select(SavedLibraryItem).where(
            SavedLibraryItem.user_key == DEFAULT_USER_KEY,
            SavedLibraryItem.paper_id == payload.paper_id,
        )
    )
    if item is None:
        item = SavedLibraryItem(user_key=DEFAULT_USER_KEY, paper_id=payload.paper_id)
        db.add(item)

    existing_tags = decode_string_list(item.tags)
    item.tags = encode_string_list([*existing_tags, *payload.tags])
    item.notes = payload.notes
    item.updated_at = datetime.now(UTC)

    profile = get_or_create_profile(db)
    add_to_profile_list(profile, "saved_paper_ids", payload.paper_id)
    add_to_profile_topics(profile, payload.tags)
    profile.updated_at = datetime.now(UTC)
    db.add(profile)
    db.commit()
    db.refresh(item)
    return library_item_to_response(item, paper)


def list_library_items(db: Session, *, tag: str | None = None) -> LibraryResponse:
    papers = {paper.id: paper for paper in db.scalars(select(Paper)).all()}
    items = list(
        db.scalars(
            select(SavedLibraryItem)
            .where(SavedLibraryItem.user_key == DEFAULT_USER_KEY)
            .order_by(SavedLibraryItem.updated_at.desc(), SavedLibraryItem.id.desc())
        ).all()
    )
    responses: list[LibraryItemResponse] = []
    all_tags: set[str] = set()
    normalized_tag = tag.strip().lower() if tag else None
    for item in items:
        paper = papers.get(item.paper_id)
        if paper is None:
            continue
        tags = decode_string_list(item.tags)
        all_tags.update(tags)
        if normalized_tag and normalized_tag not in tags:
            continue
        responses.append(library_item_to_response(item, paper))
    return LibraryResponse(items=responses, tags=sorted(all_tags))


def delete_library_item(db: Session, paper_id: int) -> None:
    item = db.scalar(
        select(SavedLibraryItem).where(
            SavedLibraryItem.user_key == DEFAULT_USER_KEY,
            SavedLibraryItem.paper_id == paper_id,
        )
    )
    if item is not None:
        db.delete(item)
    profile = get_or_create_profile(db)
    saved_ids = [item_id for item_id in decode_int_list(profile.saved_paper_ids) if item_id != paper_id]
    profile.saved_paper_ids = encode_int_list(saved_ids)
    profile.updated_at = datetime.now(UTC)
    db.add(profile)
    db.commit()


def library_item_to_response(item: SavedLibraryItem, paper: Paper) -> LibraryItemResponse:
    return LibraryItemResponse(
        id=item.id,
        paper_id=item.paper_id,
        tags=decode_string_list(item.tags),
        notes=item.notes,
        created_at=item.created_at,
        updated_at=item.updated_at,
        paper=paper_to_recommendation(
            paper=paper,
            score=0.0,
            method="library",
            explanation="saved to your research library",
        ),
    )
