from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.user_profile import UserProfile
from app.schemas.paper import ProfileRead, ProfileUpdate


DEFAULT_USER_KEY = "default"


def decode_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return []
    return sorted({int(item) for item in raw})


def encode_int_list(values: list[int]) -> str:
    return json.dumps(sorted({int(item) for item in values}))


def decode_string_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return []
    return sorted({str(item).strip() for item in raw if str(item).strip()})


def encode_string_list(values: list[str]) -> str:
    return json.dumps(sorted({str(item).strip() for item in values if str(item).strip()}))


def get_or_create_profile(db: Session) -> UserProfile:
    profile = db.scalar(select(UserProfile).where(UserProfile.user_key == DEFAULT_USER_KEY))
    if profile is not None:
        return profile
    profile = UserProfile(user_key=DEFAULT_USER_KEY)
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


def profile_to_read(profile: UserProfile) -> ProfileRead:
    return ProfileRead(
        background_level=profile.background_level,
        saved_paper_ids=decode_int_list(profile.saved_paper_ids),
        skipped_paper_ids=decode_int_list(profile.skipped_paper_ids),
        too_easy_paper_ids=decode_int_list(profile.too_easy_paper_ids),
        too_hard_paper_ids=decode_int_list(profile.too_hard_paper_ids),
        preferred_topics=decode_string_list(profile.preferred_topics),
        avoid_topics=decode_string_list(profile.avoid_topics),
        current_status=profile.current_status,
        research_goal=profile.research_goal,
        paper_taste=profile.paper_taste,
        updated_at=profile.updated_at,
    )


def update_profile(db: Session, payload: ProfileUpdate) -> ProfileRead:
    profile = get_or_create_profile(db)
    updates = payload.model_dump(exclude_unset=True)
    if "background_level" in updates and updates["background_level"] is not None:
        profile.background_level = updates["background_level"]
    for field_name in ("saved_paper_ids", "skipped_paper_ids", "too_easy_paper_ids", "too_hard_paper_ids"):
        if field_name in updates and updates[field_name] is not None:
            setattr(profile, field_name, encode_int_list(updates[field_name]))
    if "preferred_topics" in updates and updates["preferred_topics"] is not None:
        profile.preferred_topics = encode_string_list(updates["preferred_topics"])
    if "avoid_topics" in updates and updates["avoid_topics"] is not None:
        profile.avoid_topics = encode_string_list(updates["avoid_topics"])
    for field_name in ("current_status", "research_goal", "paper_taste"):
        if field_name in updates and updates[field_name] is not None:
            setattr(profile, field_name, str(updates[field_name]).strip())
    profile.updated_at = datetime.now(UTC)
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile_to_read(profile)


def profile_as_dict(db: Session) -> dict[str, Any]:
    return profile_to_read(get_or_create_profile(db)).model_dump()


def add_to_profile_list(profile: UserProfile, field_name: str, paper_id: int) -> None:
    values = decode_int_list(getattr(profile, field_name))
    if paper_id not in values:
        values.append(paper_id)
    setattr(profile, field_name, encode_int_list(values))


def add_to_profile_topics(profile: UserProfile, topics: list[str]) -> None:
    values = decode_string_list(profile.preferred_topics)
    for topic in topics:
        clean = str(topic).strip().lower()
        if clean and clean not in values:
            values.append(clean)
    profile.preferred_topics = encode_string_list(values)
