from collections import Counter
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.feedback import FeedbackEvent
from app.schemas.paper import FeedbackCreate, FeedbackSummaryResponse
from app.schemas.paper import LibraryItemUpsert
from app.services.library import upsert_library_item
from app.services.profile import add_to_profile_list, add_to_profile_topics, decode_int_list, get_or_create_profile


SKIP_ACTIONS = {"already_read", "not_relevant", "less_like_this"}


def create_feedback_event(db: Session, payload: FeedbackCreate) -> FeedbackEvent:
    event = FeedbackEvent(**payload.model_dump(exclude={"tags"}))
    db.add(event)
    profile = get_or_create_profile(db)
    if payload.action in {"save", "more_like_this"}:
        add_to_profile_list(profile, "saved_paper_ids", payload.paper_id)
        if payload.action == "save":
            add_to_profile_topics(profile, payload.tags)
            upsert_library_item(
                db,
                LibraryItemUpsert(paper_id=payload.paper_id, tags=payload.tags, notes=""),
            )
    if payload.action in SKIP_ACTIONS:
        add_to_profile_list(profile, "skipped_paper_ids", payload.paper_id)
    if payload.action == "too_easy":
        add_to_profile_list(profile, "too_easy_paper_ids", payload.paper_id)
    if payload.action == "too_hard":
        add_to_profile_list(profile, "too_hard_paper_ids", payload.paper_id)
    profile.background_level = payload.background_level or profile.background_level
    profile.updated_at = datetime.now(UTC)
    db.add(profile)
    db.commit()
    db.refresh(event)
    return event


def feedback_summary(db: Session) -> FeedbackSummaryResponse:
    events = list(db.scalars(select(FeedbackEvent)).all())
    action_counts = Counter(event.action for event in events)
    profile = get_or_create_profile(db)
    return FeedbackSummaryResponse(
        total_events=len(events),
        action_counts=dict(sorted(action_counts.items())),
        saved_paper_ids=decode_int_list(profile.saved_paper_ids),
        skipped_paper_ids=decode_int_list(profile.skipped_paper_ids),
        too_easy_paper_ids=decode_int_list(profile.too_easy_paper_ids),
        too_hard_paper_ids=decode_int_list(profile.too_hard_paper_ids),
    )
