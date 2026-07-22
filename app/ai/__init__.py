"""
Hoku Health Care - AI Package Initialization.

Exports the core AI components for clean imports across the application.
Day 8 additions: performance, caching, fallback_responses, llm_optimizer, connection_pool.
"""

from app.ai.chatbot import HokuChatbot
from app.ai.config import ai_settings, get_ai_settings
from app.ai.embeddings import EmbeddingManager, batch_embed, get_embedding
from app.ai.emergency_detector import EmergencyDetector, detect_emergency, get_emergency_response
from app.ai.intent_classifier import IntentClassifier, IntentEnum
from app.ai.memory import HokuConversationMemory
from app.ai.rag import HokuRAG
from app.ai.safety_guardrails import SafetyGuardrails
from app.ai.specialist_mapper import SpecialistMapper
from app.ai.symptom_extractor import extract_symptoms_from_text

# Day 8: Performance optimization layer exports
from app.ai.ai_performance import ResponseOptimizer, generate_with_timeout
from app.ai.caching import ResponseCache
from app.ai.fallback_responses import FALLBACK_GENERAL, FALLBACK_EMERGENCY, FALLBACK_BOOKING
from app.ai.llm_optimizer import LLMFactory, compress_prompt

__all__ = [
    "HokuChatbot",
    "ai_settings",
    "get_ai_settings",
    "EmbeddingManager",
    "get_embedding",
    "batch_embed",
    "EmergencyDetector",
    "detect_emergency",
    "get_emergency_response",
    "IntentClassifier",
    "IntentEnum",
    "HokuConversationMemory",
    "HokuRAG",
    "SafetyGuardrails",
    "SpecialistMapper",
    "extract_symptoms_from_text",
    # Day 8 exports
    "ResponseOptimizer",
    "generate_with_timeout",
    "ResponseCache",
    "FALLBACK_GENERAL",
    "FALLBACK_EMERGENCY",
    "FALLBACK_BOOKING",
    "LLMFactory",
    "compress_prompt",
]