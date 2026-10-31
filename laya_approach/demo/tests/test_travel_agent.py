"""Unit tests for the Onda Travel demo's deterministic half.

The GPU stack is exercised by ``python -m laya_approach.demo.server`` and the
websocket smoke test in ``scripts/smoke_demo.py``.  These tests keep the filter
contract and the bilingual fallback honest without loading a model.
"""

from __future__ import annotations

from laya_approach.demo.travel_agent import (
    QwenTravelLLM,
    TravelAgent,
    _clean_reply,
    _parse_money,
    catalogue_payload,
    detect_language,
    empty_filters,
    fallback_plan,
    normalize_filters,
    requested_language,
    search,
)
from laya_approach.demo.travel_data import TAGS, TRAVEL_ITEMS


def test_catalog_is_consistent() -> None:
    assert len(TRAVEL_ITEMS) == 12
    ids = [item["id"] for item in TRAVEL_ITEMS]
    assert len(ids) == len(set(ids))
    for item in TRAVEL_ITEMS:
        assert item["price"] > 0
        assert item["nights"] > 0
        assert 0 < item["rating"] <= 5
        assert set(item["tags"]) <= set(TAGS)
        assert len(item["colors"]) == 2
        assert item["country"]["en"] and item["country"]["pt-BR"]
        assert item["summary"]["en"] and item["summary"]["pt-BR"]


def test_filters_under_budget_with_trailing_punctuation() -> None:
    filters, reset = fallback_plan(
        "I would like a beach trip with good food under $1,500.", empty_filters()
    )
    assert reset is False
    assert filters["tags"] == ["beach", "food"]
    assert filters["max_price"] == 1500
    names = [item["name"] for item in search(filters).items]
    assert names == ["Maragogi", "Salvador", "Tulum"]


def test_portuguese_spoken_budget() -> None:
    assert _parse_money("até mil e quinhentos dólares") == 1500
    filters, _ = fallback_plan(
        "Olá, quero uma viagem para uma praia com boa comida, até mil e quinhentos dólares.",
        empty_filters(),
    )
    assert filters["tags"] == ["beach", "food"]
    assert filters["max_price"] == 1500


def test_all_tags_are_required_then_relaxed() -> None:
    filters = normalize_filters({"tags": ["mountains", "romantic"]})
    names = [item["name"] for item in search(filters).items]
    assert names == ["Gramado"]
    relaxed = normalize_filters({"tags": ["beach", "skiing"]})
    result = search(relaxed)
    assert result.relaxed is True
    assert result.items


def test_language_detection_and_explicit_switch() -> None:
    assert detect_language("Quero uma praia com boa comida", "en") == "pt-BR"
    assert detect_language("I want a beach with good food", "pt-BR") == "en"
    assert detect_language("Who is Kenny Wallace?", "pt-BR") == "en"
    assert detect_language("Quem é Kenny Wallace?", "en") == "pt-BR"
    assert requested_language("Switch to Portuguese") == "pt-BR"
    assert requested_language("Quero trocar para inglês") == "en"
    assert requested_language("Tell me about the beach") is None


def test_fallback_agent_is_bilingual() -> None:
    agent = TravelAgent(None)
    turn = agent.respond(
        "Quero uma praia com boa comida até US$ 1.500",
        filters=empty_filters(),
        language="pt-BR",
    )
    assert turn.language == "pt-BR"
    assert turn.fallback_used is True
    assert turn.filters["max_price"] == 1500
    assert [item["name"] for item in turn.items] == ["Maragogi", "Salvador", "Tulum"]
    assert "Maragogi" in turn.reply

    switch = agent.respond(
        "Switch to Portuguese", filters=turn.filters, language="en"
    )
    assert switch.language == "pt-BR"
    assert switch.filters == turn.filters
    assert "português" in switch.reply


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0
        self.last_messages = None

    def generate(self, messages):  # noqa: ANN001
        self.calls += 1
        self.last_messages = messages
        assert messages and messages[0]["role"] == "system"
        return self.response


def test_llm_gets_only_the_filtered_shortlist() -> None:
    llm = _FakeLLM("Maragogi is the beach pick, with seven nights from US$ 1,280.")
    agent = TravelAgent(llm)  # type: ignore[arg-type]
    turn = agent.respond(
        "I want a beach with good food under $1,500",
        filters=empty_filters(),
        language="en",
    )
    assert llm.calls == 1
    assert turn.fallback_used is False
    assert turn.reply.startswith("Maragogi")
    assert [item["name"] for item in turn.items] == ["Maragogi", "Salvador", "Tulum"]


def test_smalltalk_uses_general_prompt_and_keeps_filters() -> None:
    llm = _FakeLLM("Kenny Wallace is a former NASCAR driver and broadcaster.")
    agent = TravelAgent(llm)  # type: ignore[arg-type]
    filters = normalize_filters({"tags": ["beach"], "max_price": 1500})
    turn = agent.respond("Who is Kenny Wallace?", filters=filters, language="en")
    assert turn.intent == "smalltalk"
    assert turn.catalog_changed is False
    assert turn.filters == filters
    assert turn.reply.startswith("Kenny Wallace")
    assert llm.calls == 1
    assert "MATCHING TRIPS" not in llm.last_messages[1]["content"]
    assert "everyday questions" in llm.last_messages[0]["content"]


def test_joke_about_beach_does_not_filter_the_page() -> None:
    llm = _FakeLLM("Why did the beach go to therapy? It had too many unresolved tides.")
    agent = TravelAgent(llm)  # type: ignore[arg-type]
    filters = normalize_filters({"tags": ["mountains"]})
    turn = agent.respond("Tell me a joke about the beach", filters=filters, language="en")
    assert turn.intent == "smalltalk"
    assert turn.filters == filters
    assert turn.catalog_changed is False


def test_plural_travel_question_reaches_catalog() -> None:
    llm = _FakeLLM("We have twelve mock escapes, including Maragogi and Kyoto.")
    turn = TravelAgent(llm).respond(  # type: ignore[arg-type]
        "What trips do you have?", filters=empty_filters(), language="en"
    )
    assert turn.intent == "travel"
    assert turn.catalog_changed is True


def test_general_topic_does_not_trigger_catalog() -> None:
    llm = _FakeLLM("Try a cozy mystery, or a travel memoir if you want inspiration.")
    turn = TravelAgent(llm).respond(  # type: ignore[arg-type]
        "Recommend a movie", filters=empty_filters(), language="en"
    )
    assert turn.intent == "smalltalk"
    assert turn.catalog_changed is False
    assert turn.filters == empty_filters()


def test_mixed_trivia_and_travel_updates_catalog() -> None:
    llm = _FakeLLM("Kenny Wallace is a NASCAR broadcaster. For a beach, try Maragogi.")
    agent = TravelAgent(llm)  # type: ignore[arg-type]
    turn = agent.respond(
        "Who is Kenny Wallace? Also find me a beach trip",
        filters=empty_filters(),
        language="en",
    )
    assert turn.intent == "mixed"
    assert turn.catalog_changed is True
    assert turn.filters["tags"] == ["beach"]
    assert "Maragogi" in [item["name"] for item in turn.items]


def test_general_fallback_without_model() -> None:
    turn = TravelAgent(None).respond(
        "Who is Kenny Wallace?", filters=empty_filters(), language="en"
    )
    assert turn.intent == "smalltalk"
    assert turn.catalog_changed is False
    assert "reliable source" in turn.reply


def test_llm_failure_degrades_to_local_reply() -> None:
    class Broken:
        def generate(self, messages):  # noqa: ANN001
            raise RuntimeError("no gpu")

    agent = TravelAgent(Broken())  # type: ignore[arg-type]
    turn = agent.respond(
        "A romantic mountain escape for two",
        filters=empty_filters(),
        language="en",
    )
    assert turn.fallback_used is True
    assert turn.filters["tags"] == ["mountains", "romantic"]
    assert [item["name"] for item in turn.items] == ["Gramado"]
    assert "Gramado" in turn.reply


def test_mentioned_trip_is_included_for_follow_up() -> None:
    llm = _FakeLLM("Gramado is a cozy mountain town with wine and fondue.")
    agent = TravelAgent(llm)  # type: ignore[arg-type]
    filters = normalize_filters({"tags": ["budget"], "max_price": 1200})
    turn = agent.respond("Tell me more about Gramado", filters=filters, language="en")
    assert "Gramado" in [item["name"] for item in turn.items]
    assert turn.items[0]["name"] == "Gramado"


def test_reset_and_filter_clear() -> None:
    filters = normalize_filters({"tags": ["beach"], "max_price": 1500})
    turn = TravelAgent(None).respond("Show me everything again", filters=filters, language="en")
    assert turn.filters == empty_filters()
    assert turn.total == len(TRAVEL_ITEMS)


def test_clean_reply_removes_markdown_and_caps_length() -> None:
    assert _clean_reply("**Maragogi** is great.\n- Book it!") == "Maragogi is great. Book it!"
    long = " ".join(["word"] * 80)
    cleaned = _clean_reply(long, max_words=20)
    assert len(cleaned.split()) <= 20


def test_catalogue_payload_is_localized() -> None:
    en = catalogue_payload("en")
    pt = catalogue_payload("pt-BR")
    assert en["items"][0]["name"] == pt["items"][0]["name"]
    assert en["items"][0]["summary"] != pt["items"][0]["summary"]
    assert en["tags"]["beach"] == "Beach"
    assert pt["tags"]["beach"] == "Praia"


def test_llm_wrapper_type_is_constructible_without_loading() -> None:
    # A guard against accidentally turning the wrapper into a module-level load.
    assert QwenTravelLLM is not None
