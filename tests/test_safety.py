"""
Hoku Health Care - Safety Guardrails & Emergency Escalation Unit Tests (Day 7).

Comprehensive tests for:
- EmergencyDetector (Tier 1 regex + Tier 2 LLM fallback)
- SafetyGuardrails (diagnosis/prescription detection, disclaimer enforcement)
- 3-strike safety retry mechanism
- Monitoring metrics counters
- Zero regression: all Day 0-6 functionality preserved.

All Groq API calls are mocked — no real API usage.
"""

import asyncio
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from app.ai.config import ai_settings
from app.ai.emergency_detector import EmergencyDetector
from app.ai.safety_guardrails import SafetyGuardrails
from app.core.monitoring import HokuMetrics, get_metrics
from app.utils.constants import SAFETY_DISCLAIMER


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def mock_settings():
    """Mock AI settings for safety tests."""
    settings = MagicMock()
    settings.groq_api_key = "test-api-key"
    settings.EMERGENCY_CHECK_TIMEOUT = 0.3
    settings.SAFETY_MAX_RETRIES = 3
    settings.SAFETY_FALLBACK_RESPONSE = (
        "I am unable to provide a medical opinion for this query. "
        "Please consult a qualified doctor immediately."
    )
    return settings


@pytest.fixture
def detector(mock_settings):
    """Create EmergencyDetector with mocked settings."""
    with patch("app.ai.emergency_detector.ai_settings", mock_settings):
        yield EmergencyDetector()


@pytest.fixture
def metrics():
    """Get a fresh metrics instance and reset counters."""
    m = get_metrics()
    m.reset_counters()
    yield m
    m.reset_counters()


# ------------------------------------------------------------------
# EmergencyDetector Tests — Tier 1 Regex
# ------------------------------------------------------------------

class TestEmergencyDetectorTier1:

    def test_detect_high_urgency_chest_pain(self, detector):
        """Tier 1: High urgency detection of chest pain."""
        is_emergency, urgency, reason = detector.detect_emergency("I have severe chest pain")
        assert is_emergency is True
        assert urgency == "high"
        assert "chest pain" in reason.lower()

    def test_detect_high_urgency_cant_breathe(self, detector):
        """Tier 1: High urgency detection of breathing difficulty."""
        is_emergency, urgency, reason = detector.detect_emergency("I can't breathe properly")
        assert is_emergency is True
        assert urgency == "high"
        assert "breathe" in reason.lower()

    def test_detect_high_urgency_unconscious(self, detector):
        """Tier 1: High urgency detection of unconsciousness."""
        is_emergency, urgency, reason = detector.detect_emergency("My father is unconscious")
        assert is_emergency is True
        assert urgency == "high"

    def test_detect_high_urgency_heart_attack(self, detector):
        """Tier 1: High urgency detection of heart attack."""
        is_emergency, urgency, reason = detector.detect_emergency("I think I'm having a heart attack")
        assert is_emergency is True
        assert urgency == "high"

    def test_detect_high_urgency_stroke(self, detector):
        """Tier 1: High urgency detection of stroke."""
        is_emergency, urgency, reason = detector.detect_emergency("Signs of a stroke, slurred speech")
        assert is_emergency is True
        assert urgency == "high"

    def test_detect_high_urgency_suicide(self, detector):
        """Tier 1: High urgency detection of suicide."""
        is_emergency, urgency, reason = detector.detect_emergency("I want to suicide")
        assert is_emergency is True
        assert urgency == "high"

    def test_detect_high_urgency_seizure(self, detector):
        """Tier 1: High urgency detection of seizure."""
        is_emergency, urgency, reason = detector.detect_emergency("She is having a seizure")
        assert is_emergency is True
        assert urgency == "high"

    def test_detect_high_urgency_severe_bleeding(self, detector):
        """Tier 1: High urgency detection of severe bleeding."""
        is_emergency, urgency, reason = detector.detect_emergency("Severe bleeding from my leg")
        assert is_emergency is True
        assert urgency == "high"

    def test_detect_high_urgency_severe_allergic_reaction(self, detector):
        """Tier 1: High urgency detection of severe allergic reaction."""
        is_emergency, urgency, reason = detector.detect_emergency("Severe allergic reaction, swelling")
        assert is_emergency is True
        assert urgency == "high"

    def test_detect_moderate_urgency_high_fever(self, detector):
        """Tier 1: Moderate urgency detection of high fever."""
        is_emergency, urgency, reason = detector.detect_emergency("I have a high fever of 104")
        assert is_emergency is True
        assert urgency == "moderate"
        assert "fever" in reason.lower()

    def test_detect_moderate_urgency_dehydration(self, detector):
        """Tier 1: Moderate urgency detection of dehydration."""
        is_emergency, urgency, reason = detector.detect_emergency("Signs of dehydration, very thirsty")
        assert is_emergency is True
        assert urgency == "moderate"

    def test_no_emergency_normal_message(self, detector):
        """Tier 1: Normal messages should not trigger emergency."""
        is_emergency, urgency, reason = detector.detect_emergency("I have a mild headache")
        assert is_emergency is False
        assert urgency == "none"
        assert reason == ""

    def test_no_emergency_booking_query(self, detector):
        """Tier 1: Booking queries should not trigger emergency."""
        is_emergency, urgency, reason = detector.detect_emergency("How do I book an appointment?")
        assert is_emergency is False
        assert urgency == "none"

    def test_no_emergency_empty_message(self, detector):
        """Tier 1: Empty messages should not trigger emergency."""
        is_emergency, urgency, reason = detector.detect_emergency("")
        assert is_emergency is False
        assert urgency == "none"

    def test_no_emergency_none_message(self, detector):
        """Tier 1: None input should not trigger emergency."""
        is_emergency, urgency, reason = detector.detect_emergency(None)
        assert is_emergency is False
        assert urgency == "none"

    def test_detect_emergency_case_insensitive(self, detector):
        """Tier 1: Detection should be case-insensitive."""
        is_emergency, urgency, _ = detector.detect_emergency("CHEST PAIN")
        assert is_emergency is True
        assert urgency == "high"

        is_emergency, urgency, _ = detector.detect_emergency("Can't Breathe")
        assert is_emergency is True
        assert urgency == "high"

    def test_detect_emergency_substring_boundary(self, detector):
        """Tier 1: Partial word matches should not trigger false positives."""
        is_emergency, _, _ = detector.detect_emergency("Something seizuresque")
        assert is_emergency is False

    def test_detect_emergency_performance_under_50ms(self, detector):
        """Tier 1: Emergency detection must complete in < 50ms."""
        start = time.perf_counter()
        for _ in range(100):
            detector.detect_emergency("I have chest pain and can't breathe")
        elapsed = time.perf_counter() - start
        # 100 calls should take < 100ms total (1ms per call average)
        assert elapsed < 0.1


# ------------------------------------------------------------------
# EmergencyDetector Tests — Tier 2 LLM Fallback
# ------------------------------------------------------------------

class TestEmergencyDetectorTier2:

    @pytest.mark.asyncio
    async def test_tier2_ambiguous_edge_case(self, detector):
        """Tier 2: LLM fallback for ambiguous edge cases."""
        # Mock the LLM chain to return emergency for ambiguous text
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "text": json.dumps({"is_emergency": True, "urgency": "high", "reason": "ambiguous chest discomfort"})
        }

        with patch.object(EmergencyDetector, "_build_tier2_chain", return_value=mock_chain):
            is_emergency, urgency, reason = await detector.detect_emergency_async(
                "My chest feels a bit funny and I'm worried"
            )
            assert is_emergency is True
            assert urgency == "high"
            assert "ambiguous" in reason.lower()

    @pytest.mark.asyncio
    async def test_tier2_timeout_falls_back_to_tier1(self, detector):
        """Tier 2: On timeout, fall back to Tier 1 result."""
        slow_chain = MagicMock()
        slow_chain.invoke = lambda **kwargs: asyncio.sleep(10) or {}

        with patch.object(EmergencyDetector, "_build_tier2_chain", return_value=slow_chain):
            is_emergency, urgency, reason = await detector.detect_emergency_async(
                "I have chest pain"
            )
            # Should still detect via Tier 1 even if Tier 2 times out
            assert is_emergency is True
            assert urgency == "high"

    @pytest.mark.asyncio
    async def test_tier2_llm_unavailable_uses_tier1(self, detector):
        """Tier 2: When LLM is unavailable, Tier 1 still works."""
        with patch.object(EmergencyDetector, "_build_tier2_chain", return_value=None):
            is_emergency, urgency, reason = await detector.detect_emergency_async(
                "I can't breathe"
            )
            assert is_emergency is True
            assert urgency == "high"


# ------------------------------------------------------------------
# EmergencyDetector Tests — Urgency Responses
# ------------------------------------------------------------------

class TestEmergencyUrgencyResponses:

    def test_high_urgency_response_structure(self, detector):
        """High urgency response must contain all required fields."""
        response = detector.get_urgency_response("high")

        assert "reply" in response
        assert "suggestedSpecialist" in response
        assert "severity" in response
        assert "shouldSeeDoctor" in response
        assert "intent" in response
        assert "confidence" in response

        assert response["intent"] == "emergency"
        assert response["confidence"] == 1.0
        assert response["severity"] == "severe"
        assert response["shouldSeeDoctor"] is True
        assert "🚨" in response["reply"]
        assert "999" in response["reply"] or "1122" in response["reply"]

    def test_moderate_urgency_response_structure(self, detector):
        """Moderate urgency response must contain all required fields."""
        response = detector.get_urgency_response("moderate")

        assert "reply" in response
        assert "suggestedSpecialist" in response
        assert "severity" in response
        assert "shouldSeeDoctor" in response
        assert "intent" in response
        assert "confidence" in response

        assert response["intent"] == "emergency"
        assert response["confidence"] == 1.0
        assert response["severity"] == "moderate"
        assert response["shouldSeeDoctor"] is True
        assert SAFETY_DISCLAIMER in response["reply"]

    def test_unknown_urgency_defaults_to_high(self, detector):
        """Unknown urgency level should default to high response."""
        response = detector.get_urgency_response("unknown")
        assert response["severity"] == "severe"


# ------------------------------------------------------------------
# SafetyGuardrails Tests — Diagnosis Detection
# ------------------------------------------------------------------

class TestSafetyGuardrailsDiagnosis:

    def test_detect_diagnosis_you_have(self):
        """Detect 'you have [condition]' pattern."""
        text = "You have pneumonia and should rest."
        is_safe, violations = SafetyGuardrails.validate_response(text)
        assert is_safe is False
        assert SafetyGuardrails.VIOLATION_DIAGNOSIS in violations

    def test_detect_diagnosis_suffering_from(self):
        """Detect 'you are suffering from' pattern."""
        text = "You are suffering from diabetes."
        is_safe, violations = SafetyGuardrails.validate_response(text)
        assert is_safe is False
        assert SafetyGuardrails.VIOLATION_DIAGNOSIS in violations

    def test_detect_diagnosis_diagnosis_is(self):
        """Detect 'diagnosis is' pattern."""
        text = "Your diagnosis is bronchitis."
        is_safe, violations = SafetyGuardrails.validate_response(text)
        assert is_safe is False
        assert SafetyGuardrails.VIOLATION_DIAGNOSIS in violations

    def test_detect_diagnosis_your_condition_is(self):
        """Detect 'your condition is' pattern."""
        text = "Your condition is asthma."
        is_safe, violations = SafetyGuardrails.validate_response(text)
        assert is_safe is False
        assert SafetyGuardrails.VIOLATION_DIAGNOSIS in violations

    def test_detect_diagnosis_diagnosed_with(self):
        """Detect 'diagnosed with' pattern."""
        text = "You have been diagnosed with hypertension."
        is_safe, violations = SafetyGuardrails.validate_response(text)
        assert is_safe is False
        assert SafetyGuardrails.VIOLATION_DIAGNOSIS in violations

    def test_safe_response_no_diagnosis(self):
        """Safe response without diagnosis language should pass."""
        text = f"You mentioned some symptoms. Please consult a doctor for proper diagnosis. {SAFETY_DISCLAIMER}"
        is_safe, violations = SafetyGuardrails.validate_response(text)
        assert is_safe is True
        assert len(violations) == 0


# ------------------------------------------------------------------
# SafetyGuardrails Tests — Prescription Detection
# ------------------------------------------------------------------

class TestSafetyGuardrailsPrescription:

    def test_detect_prescription_take_mg(self):
        """Detect 'take [X]mg' pattern."""
        text = "Take 500mg of paracetamol every 6 hours."
        is_safe, violations = SafetyGuardrails.validate_response(text)
        assert is_safe is False
        assert SafetyGuardrails.VIOLATION_PRESCRIPTION in violations

    def test_detect_prescription_prescribe(self):
        """Detect 'prescribe' pattern."""
        text = "I prescribe amoxicillin for your infection."
        is_safe, violations = SafetyGuardrails.validate_response(text)
        assert is_safe is False
        assert SafetyGuardrails.VIOLATION_PRESCRIPTION in violations

    def test_detect_prescription_dosage(self):
        """Detect 'dosage' pattern."""
        text = "The dosage of 1000mg should be sufficient."
        is_safe, violations = SafetyGuardrails.validate_response(text)
        assert is_safe is False
        assert SafetyGuardrails.VIOLATION_PRESCRIPTION in violations

    def test_detect_prescription_take_twice_daily(self):
        """Detect 'take [med] twice daily' pattern."""
        text = "Take this medication twice daily."
        is_safe, violations = SafetyGuardrails.validate_response(text)
        assert is_safe is False
        assert SafetyGuardrails.VIOLATION_PRESCRIPTION in violations

    def test_detect_prescription_start_taking(self):
        """Detect 'start taking' pattern."""
        text = "Start taking insulin immediately."
        is_safe, violations = SafetyGuardrails.validate_response(text)
        assert is_safe is False
        assert SafetyGuardrails.VIOLATION_PRESCRIPTION in violations

    def test_detect_prescription_stop_taking(self):
        """Detect 'stop taking' pattern."""
        text = "Stop taking your blood pressure medicine."
        is_safe, violations = SafetyGuardrails.validate_response(text)
        assert is_safe is False
        assert SafetyGuardrails.VIOLATION_PRESCRIPTION in violations

    def test_safe_response_no_prescription(self):
        """Safe response without prescription language should pass."""
        text = f"Please follow your doctor's instructions for any medication. {SAFETY_DISCLAIMER}"
        is_safe, violations = SafetyGuardrails.validate_response(text)
        assert is_safe is True
        assert len(violations) == 0


# ------------------------------------------------------------------
# SafetyGuardrails Tests — Disclaimer Enforcement
# ------------------------------------------------------------------

class TestSafetyGuardrailsDisclaimer:

    def test_missing_disclaimer_detected(self):
        """Response without disclaimer should be flagged."""
        text = "You should rest and drink water."
        is_safe, violations = SafetyGuardrails.validate_response(text)
        assert is_safe is False
        assert SafetyGuardrails.VIOLATION_MISSING_DISCLAIMER in violations

    def test_disclaimer_present_passes(self):
        """Response with disclaimer should pass."""
        text = f"Please rest and drink water. {SAFETY_DISCLAIMER}"
        is_safe, violations = SafetyGuardrails.validate_response(text)
        # Should still fail if no other violations, but disclaimer is present
        # This text has no diagnosis or prescription, so it should be safe
        assert is_safe is True
        assert len(violations) == 0

    def test_add_disclaimer_appends_when_missing(self):
        """add_disclaimer should append disclaimer if missing."""
        text = "Please rest and drink water."
        result = SafetyGuardrails.add_disclaimer(text)
        assert SAFETY_DISCLAIMER in result

    def test_add_disclaimer_no_duplicate(self):
        """add_disclaimer should not duplicate existing disclaimer."""
        text = f"Please rest. {SAFETY_DISCLAIMER}"
        result = SafetyGuardrails.add_disclaimer(text)
        assert result.count(SAFETY_DISCLAIMER) == 1


# ------------------------------------------------------------------
# SafetyGuardrails Tests — Sanitization
# ------------------------------------------------------------------

class TestSafetyGuardrailsSanitization:

    def test_sanitize_diagnosis_language(self):
        """Sanitize should replace diagnosis language."""
        text = "You have pneumonia. Take 500mg of antibiotics."
        sanitized = SafetyGuardrails.sanitize_response(text)
        assert "You have pneumonia" not in sanitized
        assert "only a doctor can confirm" in sanitized or "possible considerations" in sanitized
        assert SAFETY_DISCLAIMER in sanitized

    def test_sanitize_prescription_language(self):
        """Sanitize should replace prescription language."""
        text = "Take 500mg of paracetamol every 6 hours."
        sanitized = SafetyGuardrails.sanitize_response(text)
        assert "Take 500mg" not in sanitized
        assert "consult your doctor" in sanitized.lower()
        assert SAFETY_DISCLAIMER in sanitized

    def test_sanitize_combined_violations(self):
        """Sanitize should handle both diagnosis and prescription violations."""
        text = "You have diabetes. Take 10 units of insulin daily."
        sanitized = SafetyGuardrails.sanitize_response(text)
        assert SAFETY_DISCLAIMER in sanitized
        # Should not contain the original violations
        assert "You have diabetes" not in sanitized or "only a doctor can confirm" in sanitized

    def test_sanitize_empty_input(self):
        """Sanitize empty input should return safe fallback."""
        sanitized = SafetyGuardrails.sanitize_response("")
        assert ai_settings.SAFETY_FALLBACK_RESPONSE in sanitized
        assert SAFETY_DISCLAIMER in sanitized

    def test_sanitize_none_input(self):
        """Sanitize None input should return safe fallback."""
        sanitized = SafetyGuardrails.sanitize_response(None)
        assert ai_settings.SAFETY_FALLBACK_RESPONSE in sanitized
        assert SAFETY_DISCLAIMER in sanitized


# ------------------------------------------------------------------
# SafetyGuardrails Tests — 3-Strike Retry
# ------------------------------------------------------------------

class TestSafetyGuardrails3Strike:

    def test_3_strike_passes_on_first_attempt(self):
        """Safe response should pass on first attempt."""
        text = f"Please rest and stay hydrated. {SAFETY_DISCLAIMER}"
        result, violations, severity = SafetyGuardrails.apply_3_strike_safety(
            text, user_id=1, db=None
        )
        assert result == text
        assert len(violations) == 0
        assert severity == "low"

    def test_3_strike_sanitizes_and_passes(self):
        """Unsafe response should be sanitized and eventually pass."""
        text = "You have pneumonia. Take 500mg of medicine."
        result, violations, severity = SafetyGuardrails.apply_3_strike_safety(
            text, user_id=1, db=None
        )
        # After sanitization, should be safe
        assert SAFETY_DISCLAIMER in result
        # Should have detected violations initially
        assert len(violations) > 0
        # Severity should be high due to diagnosis + prescription
        assert severity == "high"

    def test_3_strike_fallback_after_max_retries(self):
        """After max retries, should return safe fallback."""
        # Create a text that will always fail validation (impossible to sanitize)
        # by using a pattern not covered by sanitization
        text = "DIAGNOSIS_IS_HARDCODED_UNSANITIZABLE"
        # Mock validate_response to always return unsafe
        with patch.object(SafetyGuardrails, "validate_response", return_value=(False, ["diagnosis_attempt"])):
            with patch.object(SafetyGuardrails, "sanitize_response", return_value="STILL_UNSAFE"):
                result, violations, severity = SafetyGuardrails.apply_3_strike_safety(
                    text, user_id=1, db=None
                )
                assert ai_settings.SAFETY_FALLBACK_RESPONSE in result
                assert severity == "high"

    def test_3_strike_logs_violations_with_db(self):
        """3-strike should log violations when DB session is provided."""
        mock_db = MagicMock()
        text = "You have pneumonia."
        with patch("app.ai.safety_guardrails.log_safety_violation") as mock_log:
            result, violations, severity = SafetyGuardrails.apply_3_strike_safety(
                text, user_id=1, db=mock_db
            )
            # Should have logged at least once
            assert mock_log.called


# ------------------------------------------------------------------
# Monitoring Metrics Tests
# ------------------------------------------------------------------

class TestMonitoringMetrics:

    def test_singleton_instance(self):
        """Metrics should be a singleton."""
        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2

    def test_increment_emergency_detection(self, metrics):
        """Emergency detection counter should increment."""
        metrics.increment_emergency_detection()
        assert metrics.get_emergency_detections_total() == 1

    def test_increment_safety_violation(self, metrics):
        """Safety violation counter should increment."""
        metrics.increment_safety_violation("diagnosis_attempt")
        assert metrics.get_safety_violations_total() == 1

    def test_increment_3_strike_fallback(self, metrics):
        """3-strike fallback counter should increment."""
        metrics.increment_3_strike_fallback()
        assert metrics.get_3_strike_fallbacks_total() == 1

    def test_increment_nfr02_breach(self, metrics):
        """NFR-02 breach counter should increment."""
        metrics.increment_nfr02_breach("/api/ai/chat")
        assert metrics.get_nfr02_breaches_total() == 1

    def test_increment_request(self, metrics):
        """Request counter should increment."""
        metrics.increment_request("/api/ai/chat")
        assert metrics.get_requests_total() == 1

    def test_record_latency(self, metrics):
        """Latency recording should work correctly."""
        metrics.record_latency("/api/ai/chat", elapsed_seconds=2.5)
        assert metrics.get_average_latency_ms("/api/ai/chat") > 0

    def test_record_latency_breach(self, metrics):
        """Latency breach should increment breach counter."""
        metrics.record_latency("/api/ai/chat", elapsed_seconds=5.0)
        assert metrics.get_nfr02_breaches_total() == 1

    def test_record_latency_no_breach(self, metrics):
        """Latency under limit should not increment breach counter."""
        metrics.record_latency("/api/ai/chat", elapsed_seconds=2.0)
        assert metrics.get_nfr02_breaches_total() == 0

    def test_get_summary(self, metrics):
        """Summary should contain all expected keys."""
        metrics.increment_emergency_detection()
        metrics.increment_safety_violation()
        metrics.record_latency("/api/ai/chat", elapsed_seconds=2.0)
        summary = metrics.get_summary()
        assert "emergency_detections_total" in summary
        assert "safety_violations_total" in summary
        assert "nfr02_breaches_total" in summary
        assert "requests_total" in summary
        assert "average_latency_ms" in summary
        assert "p99_latency_ms" in summary
        assert "breach_rate_percent" in summary

    def test_reset_counters(self, metrics):
        """Reset should zero all counters."""
        metrics.increment_emergency_detection()
        metrics.increment_safety_violation()
        metrics.record_latency("/api/ai/chat", elapsed_seconds=2.0)
        metrics.reset_counters()
        assert metrics.get_emergency_detections_total() == 0
        assert metrics.get_safety_violations_total() == 0
        assert metrics.get_nfr02_breaches_total() == 0
        assert metrics.get_requests_total() == 0

    def test_thread_safety(self, metrics):
        """Metrics should be thread-safe under concurrent access."""
        import threading

        def worker():
            for _ in range(100):
                metrics.increment_request()
                metrics.increment_emergency_detection()

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert metrics.get_requests_total() == 500
        assert metrics.get_emergency_detections_total() == 500


# ------------------------------------------------------------------
# Integration Tests — End-to-End Safety Flow
# ------------------------------------------------------------------

class TestSafetyIntegration:

    @pytest.mark.asyncio
    async def test_emergency_bypasses_all_safety_checks(self, detector):
        """Emergency detection should bypass LLM and safety checks entirely."""
        is_emergency, urgency, reason = detector.detect_emergency("I can't breathe and my chest hurts")
        assert is_emergency is True
        assert urgency == "high"

        response = detector.get_urgency_response(urgency)
        assert response["intent"] == "emergency"
        assert response["severity"] == "severe"
        assert SAFETY_DISCLAIMER in response["reply"]

    def test_safe_response_with_disclaimer_passes_all_checks(self):
        """A properly formatted safe response should pass all checks."""
        text = (
            "I understand you're experiencing some discomfort. "
            "It's important to monitor your symptoms and seek medical attention "
            "if they worsen. Please consult a doctor for proper diagnosis."
        )
        is_safe, violations = SafetyGuardrails.validate_response(text)
        assert is_safe is True
        assert len(violations) == 0

    def test_multiple_violations_detected(self):
        """Response with multiple violations should detect all of them."""
        text = "You have pneumonia. Take 500mg of antibiotics."
        is_safe, violations = SafetyGuardrails.validate_response(text)
        assert is_safe is False
        assert SafetyGuardrails.VIOLATION_DIAGNOSIS in violations
        assert SafetyGuardrails.VIOLATION_PRESCRIPTION in violations
        assert SafetyGuardrails.VIOLATION_MISSING_DISCLAIMER in violations