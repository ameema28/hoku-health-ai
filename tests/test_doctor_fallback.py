"""Doctor suggestion falls back to a General Physician when the mapped specialty has none."""

from unittest.mock import AsyncMock, patch

import pytest

from app.ai.chatbot import GENERAL_PHYSICIAN, HokuChatbot
from app.ai.specialist_mapper import SpecialistMapper
from app.models.models_doctor import Doctor
from app.schemas.schemas_chat import ChatMessageResponse
from app.schemas.schemas_doctor import DoctorSuggestion
from app.scripts.seed_doctors import SAMPLE_DOCTORS, seed_doctors

SEEDED = {specialty for specialty, _, _ in SAMPLE_DOCTORS}


async def _suggest(db, specialist):
    with (
        patch(
            "app.ai.chatbot.extract_symptoms_from_text",
            new=AsyncMock(return_value=["some symptom"]),
        ),
        patch.object(
            SpecialistMapper, "map_symptoms_to_specialist", return_value=specialist
        ),
    ):
        return await HokuChatbot()._lookup_doctor_suggestion("some symptom", db)


@pytest.mark.parametrize(
    "specialist", sorted(set(SpecialistMapper.SPECIALIST_MAP.values()))
)
async def test_every_mapped_specialty_gets_a_doctor(db, specialist):
    seed_doctors(db)
    suggestion = await _suggest(db, specialist)
    assert suggestion is not None
    if specialist in SEEDED:
        assert suggestion.specialist == specialist
        assert suggestion.requested_specialist is None
    else:
        assert suggestion.specialist == GENERAL_PHYSICIAN
        assert suggestion.requested_specialist == specialist
    doctor = db.get(Doctor, suggestion.doctor_id)
    assert doctor.specialty == suggestion.specialist


def test_some_mapped_specialties_have_no_seeded_doctor():
    """Guards the premise of the fallback (F8: 8 unseeded specialties)."""
    assert set(SpecialistMapper.SPECIALIST_MAP.values()) - SEEDED


async def test_no_general_physician_returns_none(db):
    db.add(Doctor(specialty="Cardiologist", experience_years=5, license_number="X-1"))
    db.commit()
    assert await _suggest(db, "Dermatologist") is None


async def test_unavailable_doctor_triggers_fallback(db):
    seed_doctors(db)
    for doctor in db.query(Doctor).filter(Doctor.specialty == "Cardiologist"):
        doctor.is_available = False
    db.commit()
    suggestion = await _suggest(db, "Cardiologist")
    assert suggestion.specialist == GENERAL_PHYSICIAN
    assert suggestion.requested_specialist == "Cardiologist"


def test_requested_specialist_is_null_by_default_in_json():
    dumped = DoctorSuggestion(specialist="Cardiologist").model_dump(mode="json")
    assert "requested_specialist" in dumped and dumped["requested_specialist"] is None
    schema = ChatMessageResponse.model_json_schema()
    assert "requested_specialist" in schema["$defs"]["DoctorSuggestion"]["properties"]
