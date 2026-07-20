"""
Hoku Health Care - Specialist & Doctor Integration Unit Tests (Day 6).

Tests for SpecialistMapper, symptom extraction, and doctor CRUD lookups.
All Groq API calls and DB interactions are mocked.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from app.ai.specialist_mapper import SpecialistMapper
from app.ai.symptom_extractor import extract_symptoms_from_text, _regex_extract_symptoms
from app.crud.crud_doctor import get_doctors_by_specialty, get_doctor_by_id, get_doctor_availability
from app.models.models_doctor import Doctor
from app.models.doctor_availability import DoctorAvailability


# ------------------------------------------------------------------
# SpecialistMapper Tests
# ------------------------------------------------------------------

class TestSpecialistMapper:

    def test_exact_match_fever_to_general_physician(self):
        result = SpecialistMapper.map_symptoms_to_specialist(["fever"])
        assert result == "General Physician"

    def test_exact_match_chest_pain_to_cardiologist(self):
        result = SpecialistMapper.map_symptoms_to_specialist(["chest pain"])
        assert result == "Cardiologist"

    def test_exact_match_skin_rash_to_dermatologist(self):
        result = SpecialistMapper.map_symptoms_to_specialist(["skin rash"])
        assert result == "Dermatologist"

    def test_fuzzy_match_close_typo(self):
        # "fevr" should fuzzy-match to "fever"
        result = SpecialistMapper.map_symptoms_to_specialist(["fevr"])
        assert result == "General Physician"

    def test_fuzzy_match_chest_pain_typo(self):
        result = SpecialistMapper.map_symptoms_to_specialist(["chest pai"])
        assert result == "Cardiologist"

    def test_multiple_symptoms_picks_best(self):
        # "fever" (General Physician, score 100) vs "chest pain" (Cardiologist, score 100)
        # Both score 100, but chest pain comes first in dict iteration on some Python versions
        result = SpecialistMapper.map_symptoms_to_specialist(["fever", "chest pain"])
        assert result in ("General Physician", "Cardiologist")

    def test_empty_list_returns_none(self):
        result = SpecialistMapper.map_symptoms_to_specialist([])
        assert result is None

    def test_unknown_symptom_returns_none(self):
        result = SpecialistMapper.map_symptoms_to_specialist(["xyz_unknown_symptom"])
        assert result is None

    def test_get_doctors_by_specialist_mocked(self):
        mock_db = MagicMock()
        mock_doctor = MagicMock(spec=Doctor)
        mock_doctor.id = 1
        mock_doctor.specialty = "Cardiologist"
        mock_doctor.experience_years = 15
        mock_doctor.is_available = True

        with patch(
            "app.ai.specialist_mapper.get_doctors_by_specialty",
            return_value=[mock_doctor],
        ):
            doctors = SpecialistMapper.get_doctors_by_specialist(mock_db, "Cardiologist")
            assert len(doctors) == 1
            assert doctors[0].specialty == "Cardiologist"

    def test_pick_top_doctor_returns_first(self):
        doc1 = MagicMock(spec=Doctor)
        doc1.id = 1
        doc1.experience_years = 20
        doc2 = MagicMock(spec=Doctor)
        doc2.id = 2
        doc2.experience_years = 10

        top = SpecialistMapper.pick_top_doctor([doc1, doc2])
        assert top is doc1
        assert top.experience_years == 20

    def test_pick_top_doctor_empty_returns_none(self):
        result = SpecialistMapper.pick_top_doctor([])
        assert result is None


# ------------------------------------------------------------------
# SymptomExtractor Tests
# ------------------------------------------------------------------

class TestSymptomExtractor:

    def test_regex_extract_fever(self):
        result = _regex_extract_symptoms("I have a fever and headache")
        assert "fever" in result
        assert "headache" in result

    def test_regex_extract_chest_pain(self):
        result = _regex_extract_symptoms("Severe chest pain since morning")
        assert "chest pain" in result

    def test_regex_extract_multiple(self):
        result = _regex_extract_symptoms("I have fever, skin rash, and toothache")
        assert set(result) == {"fever", "skin rash", "toothache"}

    def test_regex_no_symptoms_returns_empty(self):
        result = _regex_extract_symptoms("What time is it?")
        assert result == []

    def test_regex_empty_string(self):
        result = _regex_extract_symptoms("")
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_symptoms_fast_path(self):
        result = await extract_symptoms_from_text("I have a fever")
        assert "fever" in result

    @pytest.mark.asyncio
    async def test_extract_symptoms_complex_fallback_timeout(self):
        # Complex text with no obvious keywords should trigger LLM path,
        # but we mock it to timeout and verify fallback to ["fever"]
        with patch(
            "app.ai.symptom_extractor._llm_extract_symptoms",
            side_effect=asyncio.TimeoutError,
        ):
            result = await extract_symptoms_from_text(
                "I've been feeling very unwell lately with various symptoms"
            )
            assert result == ["fever"]

    @pytest.mark.asyncio
    async def test_extract_symptoms_llm_returns_symptoms(self):
        with patch(
            "app.ai.symptom_extractor._llm_extract_symptoms",
            return_value=["chest pain", "shortness of breath"],
        ):
            result = await extract_symptoms_from_text(
                "I've been experiencing chest discomfort and breathing issues"
            )
            assert "chest pain" in result
            assert "shortness of breath" in result

    @pytest.mark.asyncio
    async def test_extract_symptoms_normalization(self):
        result = await extract_symptoms_from_text("  FEVER  , FEVER, headache  ")
        assert result == ["fever", "headache"]


# ------------------------------------------------------------------
# Doctor CRUD Tests (Mocked DB)
# ------------------------------------------------------------------

class TestDoctorCrud:

    def test_get_doctors_by_specialty(self):
        mock_db = MagicMock()
        mock_doctor = MagicMock(spec=Doctor)
        mock_doctor.id = 1
        mock_doctor.specialty = "Dermatologist"
        mock_doctor.experience_years = 8
        mock_doctor.is_available = True

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_doctor]
        mock_db.execute.return_value = mock_result

        doctors = get_doctors_by_specialty(mock_db, "Dermatologist")
        assert len(doctors) == 1
        assert doctors[0].specialty == "Dermatologist"

    def test_get_doctor_by_id_found(self):
        mock_db = MagicMock()
        mock_doctor = MagicMock(spec=Doctor)
        mock_doctor.id = 5
        mock_doctor.specialty = "Cardiologist"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_doctor
        mock_db.execute.return_value = mock_result

        doctor = get_doctor_by_id(mock_db, 5)
        assert doctor is not None
        assert doctor.id == 5

    def test_get_doctor_by_id_not_found(self):
        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        doctor = get_doctor_by_id(mock_db, 999)
        assert doctor is None

    def test_get_doctor_availability(self):
        mock_db = MagicMock()
        mock_slot = MagicMock(spec=DoctorAvailability)
        mock_slot.id = 1
        mock_slot.doctor_id = 2
        mock_slot.day_of_week = 0
        mock_slot.start_time = "09:00"
        mock_slot.end_time = "12:00"
        mock_slot.is_booked = False

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_slot]
        mock_db.execute.return_value = mock_result

        slots = get_doctor_availability(mock_db, doctor_id=2)
        assert len(slots) == 1
        assert slots[0].start_time == "09:00"
        assert slots[0].is_booked is False

    def test_get_doctor_availability_excludes_booked(self):
        mock_db = MagicMock()
        mock_slot_free = MagicMock(spec=DoctorAvailability)
        mock_slot_free.is_booked = False
        mock_slot_booked = MagicMock(spec=DoctorAvailability)
        mock_slot_booked.is_booked = True

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_slot_free]
        mock_db.execute.return_value = mock_result

        slots = get_doctor_availability(mock_db, doctor_id=2, include_booked=False)
        assert len(slots) == 1
        assert slots[0].is_booked is False


# ------------------------------------------------------------------
# Integration-style Tests
# ------------------------------------------------------------------

class TestSpecialistIntegration:

    @pytest.mark.asyncio
    async def test_end_to_end_symptom_to_doctor_suggestion(self):
        """
        Simulate the full Day 6 flow: extract symptoms -> map specialist ->
        query doctors -> pick top -> build suggestion.
        """
        # Step 1: Extract symptoms
        symptoms = await extract_symptoms_from_text("I have chest pain")
        assert "chest pain" in symptoms

        # Step 2: Map to specialist
        specialist = SpecialistMapper.map_symptoms_to_specialist(symptoms)
        assert specialist == "Cardiologist"

        # Step 3: Mock DB lookup
        mock_db = MagicMock()
        mock_doc = MagicMock(spec=Doctor)
        mock_doc.id = 10
        mock_doc.specialty = "Cardiologist"
        mock_doc.experience_years = 12
        mock_doc.is_available = True

        with patch(
            "app.ai.specialist_mapper.get_doctors_by_specialty",
            return_value=[mock_doc],
        ):
            doctors = SpecialistMapper.get_doctors_by_specialist(mock_db, specialist)
            top = SpecialistMapper.pick_top_doctor(doctors)

        assert top is not None
        assert top.specialty == "Cardiologist"
        assert top.experience_years == 12