"""
The model's reply text reaches the visitor without JSON scaffolding or
Markdown, whatever wrapping the model put around it. No LLM, no network.
"""

import json

import pytest

from app.ai.chatbot import HokuChatbot, _strip_markdown

GOOD = {
    "reply": "Rest and drink fluids.",
    "suggestedSpecialist": "General Physician",
    "severity": "mild",
    "shouldSeeDoctor": False,
}


@pytest.fixture
def bot():
    return HokuChatbot()


@pytest.mark.parametrize(
    "text",
    [
        json.dumps(GOOD),
        "```json\n" + json.dumps(GOOD) + "\n```",
        "Here is my response: " + json.dumps(GOOD),
        json.dumps(GOOD) + "\nI hope this helps!",
        "Sure!\n```\n" + json.dumps(GOOD, indent=2) + "\n```\nAnything else?",
    ],
)
def test_wrapped_json_yields_only_the_reply(bot, text):
    parsed = bot._parse_llm_output(text)
    assert parsed["reply"] == "Rest and drink fluids."
    assert parsed["suggestedSpecialist"] == "General Physician"
    assert parsed["severity"] == "mild"
    assert parsed["shouldSeeDoctor"] is False


def test_malformed_json_still_extracts_reply_string(bot):
    text = '{"reply": "Rest well.\\nDrink water.", "severity": "mild", "shouldSeeDoctor": true'
    parsed = bot._parse_llm_output(text)
    assert parsed["reply"] == "Rest well.\nDrink water."
    assert "{" not in parsed["reply"]
    assert parsed["severity"] == "mild"


def test_plain_text_is_kept(bot):
    parsed = bot._parse_llm_output("Please rest and drink fluids.")
    assert parsed["reply"] == "Please rest and drink fluids."
    assert parsed["severity"] == "unknown"


def test_markdown_is_stripped_from_reply(bot):
    payload = dict(GOOD, reply="## Tips\n**Rest** well.\n- Drink water\n1. Sleep early")
    parsed = bot._parse_llm_output(json.dumps(payload))
    assert parsed["reply"] == "Tips\nRest well.\nDrink water\nSleep early"


def test_strip_markdown_leaves_plain_text_alone():
    text = "Call 1122 in Pakistan. Rates vary: ask the clinic."
    assert _strip_markdown(text) == text
