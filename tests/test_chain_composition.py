"""The real runnable chains, composed with fake chat models (no network, no key)."""

import json
from unittest.mock import MagicMock, patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from app.ai.chatbot import HokuChatbot
from app.ai.emergency_detector import EmergencyDetector
from app.ai.intent_classifier import IntentClassifier, IntentEnum
from app.ai.prompts import chat_prompt_template, rag_chat_prompt_template
from app.utils.constants import SAFETY_DISCLAIMER

REPLY = {
    "reply": "Rest and drink fluids. " + SAFETY_DISCLAIMER,
    "suggestedSpecialist": "General Physician",
    "severity": "mild",
    "shouldSeeDoctor": False,
}


def _recording_model(reply: str, seen: list):
    """A chat-model stand-in that records the prompt messages it receives."""

    def _call(prompt_value):
        seen.append(prompt_value.to_messages())
        return AIMessage(content=reply)

    return RunnableLambda(_call)


def test_generation_chain_returns_text_from_fake_model():
    bot = HokuChatbot()
    bot._main_llm = FakeListChatModel(responses=[json.dumps(REPLY)])
    out = bot._generation_chain(chat_prompt_template).invoke(
        {"message": "headache", "context": "general", "history": []}
    )
    assert isinstance(out, str)
    assert json.loads(out) == REPLY


def test_history_and_faq_context_reach_the_prompt():
    seen: list = []
    bot = HokuChatbot()
    bot._main_llm = _recording_model(json.dumps(REPLY), seen)
    history = [
        HumanMessage(content="I had a fever yesterday"),
        AIMessage(content="Noted."),
    ]

    bot._generation_chain(rag_chat_prompt_template).invoke(
        {
            "message": "Is it still a concern?",
            "context": "symptom",
            "faq_context": "- Q: Opening hours?\n  A: 24/7 teleconsultations",
            "history": history,
        }
    )

    messages = seen[0]
    assert "24/7 teleconsultations" in messages[0].content  # system prompt has FAQ
    assert [m.content for m in messages[1:3]] == ["I had a fever yesterday", "Noted."]
    assert "Is it still a concern?" in messages[-1].content


async def test_get_response_sends_db_history_to_the_model():
    """End to end through get_response: stored turns arrive as prompt messages."""
    seen: list = []
    bot = HokuChatbot()
    bot._main_llm = _recording_model(json.dumps(REPLY), seen)
    history = [
        HumanMessage(content="earlier question"),
        AIMessage(content="earlier answer"),
    ]

    with (
        patch("app.ai.chatbot.HokuConversationMemory") as memory_cls,
        patch.object(
            IntentClassifier,
            "classify_intent",
            return_value=(IntentEnum.GENERAL, 0.9),
        ),
    ):
        memory_cls.return_value.load_memory.return_value = history
        result = await bot.get_response("new question", user_id=1, db=MagicMock())

    assert result["severity"] == "mild"
    assert SAFETY_DISCLAIMER in result["reply"]
    contents = [m.content for m in seen[0]]
    assert "earlier question" in contents and "earlier answer" in contents


async def test_intent_chain_composes_with_fake_model(monkeypatch):
    classifier = IntentClassifier()
    classifier._llm = FakeListChatModel(
        responses=[json.dumps({"intent": "booking", "confidence": 0.95})]
    )
    intent, confidence = await classifier.classify_intent("book a nurse visit")
    assert intent == IntentEnum.BOOKING
    assert confidence == 0.95


def test_tier2_emergency_chain_is_a_runnable(monkeypatch):
    fake = FakeListChatModel(responses=['{"is_emergency": false}'])
    with patch("langchain_groq.ChatGroq", return_value=fake):
        chain = EmergencyDetector._build_tier2_chain()
    assert chain.invoke({"text": "hello"}) == '{"is_emergency": false}'
