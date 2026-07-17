"""
Hoku Health Care - Intent Classification Unit Tests (Day 4).

Tests for IntentClassifier and EmergencyDetector.
All Groq API calls are mocked — no real API usage.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from app.ai.emergency_detector import detect_emergency, get_emergency_response
from app.ai.intent_classifier import IntentClassifier, IntentEnum


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def mock_settings():
    """Mock AI settings for intent classification."""
    settings = MagicMock()
    settings.groq_api_key = "test-api-key"
    settings.INTENT_MODEL = "llama-3.1-8b-instant"
    settings.INTENT_CONFIDENCE_THRESHOLD = 0.7
    settings.INTENT_CLASSIFICATION_TIMEOUT = 0.5
    return settings


@pytest.fixture
def classifier(mock_settings):
    """Create IntentClassifier with mocked settings."""
    with patch("app.ai.intent_classifier.ai_settings", mock_settings):
        yield IntentClassifier()


@pytest.fixture
def mock_chain():
    """Mock LLMChain that returns predictable intent responses."""
    chain = MagicMock()

    def mock_invoke(inputs):
        message = inputs.get("message", "").lower()
        # More specific keyword matching — order matters!
        # Emergency first (safety), then specific categories
        if any(kw in message for kw in ["emergency", "chest pain", "can't breathe", "unconscious"]):
            return {"text": json.dumps({"intent": "emergency", "confidence": 0.99})}
        elif any(kw in message for kw in ["book", "appointment", "schedule"]):
            return {"text": json.dumps({"intent": "booking", "confidence": 0.92})}
        elif any(kw in message for kw in ["medicine", "paracetamol", "insulin", "dosage", "prescription", "take my"]):
            return {"text": json.dumps({"intent": "medication", "confidence": 0.93})}
        elif any(kw in message for kw in ["headache", "fever", "pain", "symptom", "hurt", "ache"]):
            return {"text": json.dumps({"intent": "symptom", "confidence": 0.95})}
        else:
            return {"text": json.dumps({"intent": "general", "confidence": 0.88})}

    chain.invoke = mock_invoke
    return chain


# ------------------------------------------------------------------
# IntentClassifier Tests
# ------------------------------------------------------------------

class TestIntentClassifier:

    @pytest.mark.asyncio
    async def test_classify_symptom_intent(self, classifier, mock_chain):
        """Test classification of symptom-related messages."""
        classifier._chain = mock_chain

        intent, confidence = await classifier.classify_intent(
            "I have a severe headache and fever"
        )

        assert intent == IntentEnum.SYMPTOM
        assert 0.0 <= confidence <= 1.0
        assert confidence >= 0.7

    @pytest.mark.asyncio
    async def test_classify_booking_intent(self, classifier, mock_chain):
        """Test classification of booking-related messages."""
        classifier._chain = mock_chain

        intent, confidence = await classifier.classify_intent(
            "How do I book an appointment with a cardiologist?"
        )

        assert intent == IntentEnum.BOOKING
        assert 0.0 <= confidence <= 1.0
        assert confidence >= 0.7

    @pytest.mark.asyncio
    async def test_classify_medication_intent(self, classifier, mock_chain):
        """Test classification of medication-related messages."""
        classifier._chain = mock_chain

        intent, confidence = await classifier.classify_intent(
            "Can I take paracetamol for my fever?"
        )

        assert intent == IntentEnum.MEDICATION
        assert 0.0 <= confidence <= 1.0
        assert confidence >= 0.7

    @pytest.mark.asyncio
    async def test_classify_general_intent(self, classifier, mock_chain):
        """Test classification of general health questions."""
        classifier._chain = mock_chain

        intent, confidence = await classifier.classify_intent(
            "What are your services?"
        )

        assert intent == IntentEnum.GENERAL
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_classify_emergency_intent(self, classifier, mock_chain):
        """Test classification of emergency messages."""
        classifier._chain = mock_chain

        intent, confidence = await classifier.classify_intent(
            "I have chest pain and can't breathe"
        )

        assert intent == IntentEnum.EMERGENCY
        assert confidence >= 0.7

    @pytest.mark.asyncio
    async def test_low_confidence_fallback_to_general(self, classifier):
        """Test that low confidence scores fall back to GENERAL."""
        low_conf_chain = MagicMock()
        low_conf_chain.invoke.return_value = {
            "text": json.dumps({"intent": "symptom", "confidence": 0.45})
        }
        classifier._chain = low_conf_chain

        intent, confidence = await classifier.classify_intent(
            "Something vague and unclear"
        )

        assert intent == IntentEnum.GENERAL
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_timeout_fallback_to_general(self, classifier):
        """Test that timeout falls back to GENERAL gracefully."""
        slow_chain = MagicMock()
        slow_chain.invoke = lambda **kwargs: asyncio.sleep(10) or {"text": "{}"}
        classifier._chain = slow_chain

        intent, confidence = await classifier.classify_intent("test message")

        assert intent == IntentEnum.GENERAL
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_invalid_intent_string_fallback(self, classifier):
        """Test fallback when LLM returns unknown intent string."""
        bad_chain = MagicMock()
        bad_chain.invoke.return_value = {
            "text": json.dumps({"intent": "unknown_category", "confidence": 0.9})
        }
        classifier._chain = bad_chain

        intent, confidence = await classifier.classify_intent("test")

        assert intent == IntentEnum.GENERAL
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_llm_unavailable_fallback(self, classifier):
        """Test fallback when LLM/chain is not available."""
        classifier._chain = None

        intent, confidence = await classifier.classify_intent("test")

        assert intent == IntentEnum.GENERAL
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_malformed_json_regex_fallback(self, classifier):
        """Test regex fallback for malformed LLM output."""
        malformed_chain = MagicMock()
        malformed_chain.invoke.return_value = {
            "text": 'Here is my answer: "intent": "symptom", "confidence": 0.85'
        }
        classifier._chain = malformed_chain

        intent, confidence = await classifier.classify_intent("test")

        assert intent == IntentEnum.SYMPTOM
        assert confidence == 0.85

    def test_parse_intent_output_valid_json(self, classifier):
        """Test parsing of valid intent JSON output."""
        result = classifier._parse_intent_output(
            '{"intent": "symptom", "confidence": 0.92}'
        )
        assert result["intent"] == "symptom"
        assert result["confidence"] == 0.92

    def test_parse_intent_output_markdown_json(self, classifier):
        """Test parsing of JSON inside markdown fences."""
        result = classifier._parse_intent_output(
            '```json\n{"intent": "booking", "confidence": 0.88}\n```'
        )
        assert result["intent"] == "booking"
        assert result["confidence"] == 0.88

    def test_parse_intent_output_clamps_confidence(self, classifier):
        """Test that confidence values are clamped to [0, 1]."""
        result = classifier._parse_intent_output(
            '{"intent": "symptom", "confidence": 1.5}'
        )
        assert result["confidence"] == 1.0

        result = classifier._parse_intent_output(
            '{"intent": "symptom", "confidence": -0.5}'
        )
        assert result["confidence"] == 0.0


# ------------------------------------------------------------------
# EmergencyDetector Tests
# ------------------------------------------------------------------

class TestEmergencyDetector:

    def test_detect_emergency_chest_pain(self):
        """Test detection of chest pain emergency."""
        assert detect_emergency("I have severe chest pain") is True

    def test_detect_emergency_cant_breathe(self):
        """Test detection of breathing difficulty."""
        assert detect_emergency("I can't breathe properly") is True

    def test_detect_emergency_unconscious(self):
        """Test detection of unconsciousness."""
        assert detect_emergency("My father is unconscious") is True

    def test_detect_emergency_seizure(self):
        """Test detection of seizure."""
        assert detect_emergency("She is having a seizure") is True

    def test_detect_emergency_suicide(self):
        """Test detection of suicide-related emergency."""
        assert detect_emergency("I want to suicide") is True

    def test_detect_emergency_stroke(self):
        """Test detection of stroke symptoms."""
        assert detect_emergency("Signs of a stroke, slurred speech") is True

    def test_detect_emergency_heart_attack(self):
        """Test detection of heart attack."""
        assert detect_emergency("I think I'm having a heart attack") is True

    def test_detect_emergency_case_insensitive(self):
        """Test that detection is case-insensitive."""
        assert detect_emergency("CHEST PAIN") is True
        assert detect_emergency("Can't Breathe") is True

    def test_detect_emergency_no_emergency(self):
        """Test that normal messages are not flagged."""
        assert detect_emergency("I have a mild headache") is False
        assert detect_emergency("How do I book an appointment?") is False
        assert detect_emergency("What time should I take my medicine?") is False

    def test_detect_emergency_empty_message(self):
        """Test handling of empty messages."""
        assert detect_emergency("") is False
        assert detect_emergency(None) is False

    def test_detect_emergency_substring_boundary(self):
        """Test that partial word matches don't trigger false positives."""
        # "seizure" should not match "seizuresque" (word boundary check)
        assert detect_emergency("Something seizuresque") is False

    def test_detect_emergency_performance(self):
        """Test that emergency detection completes in < 50ms."""
        import time
        start = time.perf_counter()
        for _ in range(100):
            detect_emergency("I have chest pain and can't breathe")
        elapsed = time.perf_counter() - start
        # 100 calls should take < 100ms total (1ms per call)
        assert elapsed < 0.1

    def test_get_emergency_response_structure(self):
        """Test emergency response dict has all required fields."""
        response = get_emergency_response()

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
        assert "🚨" in response["reply"]  # Contains emergency indicator
        assert "999" in response["reply"] or "1122" in response["reply"]


# ------------------------------------------------------------------
# Integration Tests
# ------------------------------------------------------------------

class TestIntentIntegration:

    @pytest.mark.asyncio
    async def test_all_intent_categories(self, classifier, mock_chain):
        """Test that all 5 intent categories are correctly classified."""
        classifier._chain = mock_chain

        test_cases = [
            ("I have fever and body aches", IntentEnum.SYMPTOM),
            ("Book me a doctor appointment", IntentEnum.BOOKING),
            ("When should I take my insulin dosage?", IntentEnum.MEDICATION),
            ("What is Hoku Health Care?", IntentEnum.GENERAL),
            ("Emergency chest pain can't breathe", IntentEnum.EMERGENCY),
        ]

        for message, expected_intent in test_cases:
            intent, confidence = await classifier.classify_intent(message)
            assert intent == expected_intent, f"Failed for: {message}"
            assert confidence >= 0.7, f"Low confidence for: {message}"