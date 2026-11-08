"""
Hoku Health Care - Doctor seed script.

Inserts the sample specialists and their weekly availability slots.
Idempotent: a doctor whose ``license_number`` already exists is skipped,
so running it twice still leaves 7 doctors.

Usage:
    python -m app.scripts.seed_doctors
"""

from typing import Dict, List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.doctor_availability import DoctorAvailability
from app.models.models_doctor import Doctor

#: (specialty, experience_years, license_number)
SAMPLE_DOCTORS: List[Tuple[str, int, str]] = [
    ("General Physician", 10, "GP-PAK-2024-001"),
    ("Cardiologist", 15, "CD-PAK-2024-002"),
    ("Dermatologist", 8, "DER-PAK-2024-003"),
    ("Child Specialist", 12, "PED-PAK-2024-004"),
    ("Psychiatrist", 20, "PSY-PAK-2024-005"),
    ("Neurologist", 14, "NEU-PAK-2024-006"),
    ("Pulmonologist", 11, "PUL-PAK-2024-007"),
]

#: (day_of_week, start_time, end_time); 0 = Monday
WEEKLY_SLOTS: List[Tuple[int, str, str]] = [
    (0, "09:00", "12:00"),
    (0, "14:00", "17:00"),
    (2, "10:00", "13:00"),
    (4, "09:00", "15:00"),
]


def seed_doctors(db: Session) -> Dict[str, int]:
    """
    Insert missing sample doctors and their slots.

    Returns:
        dict: ``{"doctors_added": n, "doctors_skipped": m, "slots_added": s}``.
    """
    existing = set(
        db.execute(
            select(Doctor.license_number).where(
                Doctor.license_number.in_([lic for _, _, lic in SAMPLE_DOCTORS])
            )
        ).scalars()
    )

    new_doctors = [
        Doctor(
            user_id=None,
            specialty=specialty,
            experience_years=years,
            is_available=True,
            license_number=license_number,
        )
        for specialty, years, license_number in SAMPLE_DOCTORS
        if license_number not in existing
    ]
    db.add_all(new_doctors)
    db.flush()  # assigns ids

    slots = [
        DoctorAvailability(
            doctor_id=doctor.id,
            day_of_week=day,
            start_time=start,
            end_time=end,
            is_booked=False,
        )
        for doctor in new_doctors
        for day, start, end in WEEKLY_SLOTS
    ]
    db.add_all(slots)
    db.commit()

    return {
        "doctors_added": len(new_doctors),
        "doctors_skipped": len(existing),
        "slots_added": len(slots),
    }


def main() -> None:
    """Seed doctors into the configured database."""
    from app.core.database import SessionLocal

    db = SessionLocal()
    try:
        result = seed_doctors(db)
    finally:
        db.close()
    print(
        f"Seeded {result['doctors_added']} doctors "
        f"({result['doctors_skipped']} already present), "
        f"{result['slots_added']} availability slots."
    )


if __name__ == "__main__":
    main()
