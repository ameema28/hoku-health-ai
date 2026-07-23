"""
Hoku Health Care - Specialist Mapper (Day 6).

Maps extracted symptoms to medical specialties using fuzzy matching,
then queries the database for available doctors. Designed for
sub-second latency to stay within the NFR-02 < 4s total budget.
"""

import logging
from typing import List, Optional

from rapidfuzz import process, fuzz
from sqlalchemy.orm import Session

from app.crud.crud_doctor import get_doctors_by_specialty
from app.models.models_doctor import Doctor

logger = logging.getLogger(__name__)


class SpecialistMapper:
    """
    Maps patient symptoms to medical specialists and retrieves
    the most suitable available doctor from the database.

    Uses rapidfuzz for pure-Python fuzzy string matching (no C
    extensions required, MIT license, ~3-5ms per lookup).
    """

    SPECIALIST_MAP: dict[str, str] = {
        "fever": "General Physician",
        "chest pain": "Cardiologist",
        "skin rash": "Dermatologist",
        "rash": "Dermatologist",
        "itchy": "Dermatologist",
        "skin": "Dermatologist",
        "pregnancy": "Gynecologist",
        "child fever": "Child Specialist",
        "toothache": "Dental Specialist",
        "diabetes": "Endocrinologist",
        "depression": "Psychiatrist",
        "fracture": "Orthopedic Surgeon",
        "headache": "General Physician",
        "back pain": "Orthopedic Surgeon",
        "cough": "General Physician",
        "sore throat": "General Physician",
        "stomach pain": "Gastroenterologist",
        "abdominal pain": "Gastroenterologist",
        "digestive": "Gastroenterologist",
        "eye pain": "Ophthalmologist",
        "ear pain": "ENT Specialist",
        "anxiety": "Psychiatrist",
        "joint pain": "Rheumatologist",
        "high blood pressure": "Cardiologist",
        "shortness of breath": "Pulmonologist",
    }

    @classmethod
    def map_symptoms_to_specialist(cls, symptoms: List[str]) -> Optional[str]:
        """
        Map a list of extracted symptom keywords to a medical specialty.

        Uses rapidfuzz fuzzy matching against SPECIALIST_MAP keys.
        Returns the specialty with the highest aggregate score, or None
        if no reasonable match is found.

        Args:
            symptoms: List of normalized symptom strings (lowercase).

        Returns:
            str | None: Matching specialty name or None.
        """
        if not symptoms:
            logger.debug("Empty symptom list, no specialist mapping")
            return None

        best_specialty: Optional[str] = None
        best_score = 0.0

        for symptom in symptoms:
            if not symptom or not isinstance(symptom, str):
                continue

            # rapidfuzz.extractOne returns (match, score, index)
            result = process.extractOne(
                symptom,
                list(cls.SPECIALIST_MAP.keys()),
                scorer=fuzz.partial_ratio,
                score_cutoff=60,
            )

            if result:
                match, score, _ = result  # type: ignore
                specialty = cls.SPECIALIST_MAP.get(match)
                logger.debug(
                    "Fuzzy match: symptom='%s' -> match='%s' (score=%.1f) -> specialty='%s'",
                    symptom,
                    match,
                    score,
                    specialty,
                )
                if score > best_score:
                    best_score = score
                    best_specialty = specialty

        if best_specialty:
            logger.info(
                "Mapped symptoms to specialist '%s' (best_score=%.1f)",
                best_specialty,
                best_score,
            )
        else:
            logger.info("No specialist match found for symptoms: %s", symptoms)

        return best_specialty

    @staticmethod
    def get_doctors_by_specialist(db: Session, specialist: str) -> List[Doctor]:
        """
        Query the database for available doctors matching a specialty.

        Results are ordered by experience_years DESC (most experienced first).

        Args:
            db: SQLAlchemy database session.
            specialist: Medical specialty name (e.g., "Cardiologist").

        Returns:
            List[Doctor]: Available doctors for the specialty.
        """
        logger.info("Looking up doctors for specialty='%s'", specialist)
        doctors = get_doctors_by_specialty(db, specialty=specialist)
        logger.info("Found %d available doctors for '%s'", len(doctors), specialist)
        return doctors

    @staticmethod
    def pick_top_doctor(doctors: List[Doctor]) -> Optional[Doctor]:
        """
        Select the top doctor from a pre-sorted list.

        The list is expected to be ordered by experience descending.
        Returns the first available doctor, or None if the list is empty.

        Args:
            doctors: List of Doctor objects ordered by experience DESC.

        Returns:
            Doctor | None: The top-ranked available doctor.
        """
        if not doctors:
            logger.debug("No doctors in list to pick from")
            return None

        top = doctors[0]
        logger.info(
            "Picked top doctor: id=%d, name='%s', experience=%d years",
            top.id,
            getattr(top, "name", f"Doctor #{top.id}"),
            top.experience_years,
        )
        return top