"""
Hoku Health Care - FAQ Seed Script (Day 5).

Runnable standalone: connects to the configured database, checks for
the pgvector extension, creates the "hoku_health_faqs" collection, and
seeds it with 20 realistic FAQ entries covering Hoku's Pakistan/UAE/UK
home healthcare services.

Usage:
    python -m app.scripts.seed_faqs
"""

import logging
import sys

from app.ai.rag import HokuRAG
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FAQS = [
    {
        "question": "What home healthcare services does Hoku Health Care offer?",
        "answer": (
            "Hoku Health Care offers home nursing, physiotherapy, elderly care, "
            "post-surgical care, chronic disease management, and doctor "
            "teleconsultations across Pakistan, the UAE, and the UK."
        ),
        "category": "services",
    },
    {
        "question": "Do you provide home nursing care in Lahore?",
        "answer": (
            "Yes, Hoku Health Care provides licensed home nursing in Lahore, "
            "including wound care, injections, vital sign monitoring, and "
            "post-operative recovery support."
        ),
        "category": "services",
    },
    {
        "question": "Is Hoku Health Care available in Dubai?",
        "answer": (
            "Yes, Hoku operates in Dubai and across the UAE, offering home "
            "nursing, physiotherapy, and elderly companion care licensed under "
            "local healthcare regulations."
        ),
        "category": "services",
    },
    {
        "question": "Does Hoku Health Care operate in the UK?",
        "answer": (
            "Yes, Hoku Health Care serves patients across the UK, offering "
            "domiciliary nursing care, medication management, and NHS-aligned "
            "referral support."
        ),
        "category": "services",
    },
    {
        "question": "How do I book a home nurse visit?",
        "answer": (
            "You can book a home nurse visit through the Hoku Health Care app "
            "or patient dashboard by selecting 'Book a Service', choosing your "
            "preferred date and time, and confirming your address."
        ),
        "category": "booking",
    },
    {
        "question": "Can I reschedule or cancel an appointment?",
        "answer": (
            "Yes, appointments can be rescheduled or cancelled up to 4 hours "
            "before the scheduled visit at no charge through the app's "
            "'My Appointments' section."
        ),
        "category": "booking",
    },
    {
        "question": "How do I book a doctor teleconsultation?",
        "answer": (
            "Open the Hoku app, select 'Teleconsultation', choose a specialist "
            "and time slot, and join the video call at your scheduled time."
        ),
        "category": "booking",
    },
    {
        "question": "What is the typical wait time for a home visit booking?",
        "answer": (
            "Standard home visits are typically confirmed within 2-4 hours in "
            "major cities (Karachi, Lahore, Islamabad, Dubai, London). Urgent "
            "same-day requests are prioritized when available."
        ),
        "category": "booking",
    },
    {
        "question": "Can Hoku nurses administer prescribed injections at home?",
        "answer": (
            "Yes, our licensed nurses can administer prescribed injections, "
            "including insulin and vitamin B12, at home, provided a valid "
            "prescription is uploaded to your patient profile."
        ),
        "category": "medication",
    },
    {
        "question": "Do you offer medication reminder services?",
        "answer": (
            "Yes, patients can set up medication reminders in the Hoku app, "
            "and home care nurses can also assist with medication schedules "
            "during scheduled visits."
        ),
        "category": "medication",
    },
    {
        "question": "Can I ask Hoku AI about drug interactions?",
        "answer": (
            "Hoku AI can share general information about common medications, "
            "but for specific drug interaction concerns, always confirm with "
            "your prescribing doctor or a licensed pharmacist."
        ),
        "category": "medication",
    },
    {
        "question": "Does Hoku Health Care help refill prescriptions?",
        "answer": (
            "Hoku partners with local pharmacies in Pakistan and the UAE for "
            "prescription refill and home delivery; UK availability depends on "
            "your registered pharmacy's delivery options."
        ),
        "category": "medication",
    },
    {
        "question": "What should I do if I have a fever and body aches?",
        "answer": (
            "For mild fever and body aches, rest, stay hydrated, and monitor "
            "your temperature. If fever persists beyond 3 days or exceeds "
            "103°F/39.4°C, book a teleconsultation or home nurse visit."
        ),
        "category": "general",
    },
    {
        "question": "What specialists are available through Hoku?",
        "answer": (
            "Hoku's network includes general physicians, cardiologists, "
            "pediatricians, physiotherapists, dermatologists, and "
            "psychologists available via teleconsultation or home visit."
        ),
        "category": "general",
    },
    {
        "question": "Is Hoku Health Care suitable for elderly care?",
        "answer": (
            "Yes, Hoku offers dedicated elderly care packages including "
            "companionship, mobility assistance, medication supervision, and "
            "regular health monitoring visits."
        ),
        "category": "general",
    },
    {
        "question": "How is my health data protected on the Hoku platform?",
        "answer": (
            "Hoku Health Care encrypts patient data in transit and at rest, "
            "restricts access to authorized care staff, and complies with "
            "applicable data protection regulations in each operating region."
        ),
        "category": "general",
    },
    {
        "question": "What are the signs I should seek emergency care instead of home care?",
        "answer": (
            "Seek immediate emergency care for chest pain, difficulty "
            "breathing, severe bleeding, loss of consciousness, or signs of "
            "stroke -- home care visits are not a substitute for emergency "
            "services in these situations."
        ),
        "category": "emergency",
    },
    {
        "question": "Does Hoku Health Care handle medical emergencies directly?",
        "answer": (
            "No, Hoku Health Care is not an emergency response service. In an "
            "emergency, call your local emergency number immediately: 1122 in "
            "Pakistan, 998/999 in the UAE, or 999 in the UK."
        ),
        "category": "emergency",
    },
    {
        "question": "What post-surgical care does Hoku provide at home?",
        "answer": (
            "Hoku offers post-surgical wound dressing, pain management "
            "support, mobility assistance, and recovery monitoring by "
            "licensed nurses, coordinated with your surgeon's care plan."
        ),
        "category": "services",
    },
    {
        "question": "What are Hoku Health Care's operating hours?",
        "answer": (
            "Teleconsultations are available 24/7. Home visit bookings can be "
            "scheduled anytime through the app, with visits typically carried "
            "out between 7 AM and 10 PM local time; urgent overnight visits "
            "are available on request in select cities."
        ),
        "category": "general",
    },
]


def main() -> None:
    """Seed the Hoku FAQ vector store."""
    logger.info("Connecting to database: %s", settings.DATABASE_URL.split("@")[-1])

    rag = HokuRAG()

    logger.info("Creating/verifying 'hoku_health_faqs' vector store collection...")
    rag.create_vector_store()

    logger.info("Seeding %d FAQ entries...", len(FAQS))
    added = rag.add_faq_documents(FAQS)

    logger.info("Done. %d FAQ documents added to the Hoku knowledge base.", added)
    print(f"Seeded {added} FAQ entries into '{rag.collection_name}'.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        logger.exception("Seed script failed: %s", exc)
        sys.exit(1)
