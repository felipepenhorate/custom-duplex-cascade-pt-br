"""Bilingual travel agent + deterministic catalog filters.

The LLM is the conversational layer; it never owns the catalog.  On every turn
it returns a *complete* filter state plus a short reply, and the same filter
engine that powers the fallback applies it to the mock catalog.  That keeps the
spoken answer and the cards on screen in sync even when generation is slow or
malformed.

Two rules matter for the voice demo:

* the reply is always one to three short spoken sentences (the cards carry the
  detail);
* the language is decided by the latest user turn, so the UI and the TTS can
  switch from English to pt-BR in the middle of a conversation.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from .travel_data import (
    CLIMATES,
    PACES,
    REGIONS,
    TAGS,
    TRAVEL_ITEMS,
    localize,
)

# ---------------------------------------------------------------------------
# Filter schema and search
# ---------------------------------------------------------------------------

FILTER_KEYS = (
    "tags",
    "max_price",
    "min_price",
    "min_nights",
    "max_nights",
    "region",
    "climate",
    "pace",
)

EMPTY_FILTERS: dict[str, Any] = {
    "tags": [],
    "max_price": None,
    "min_price": None,
    "min_nights": None,
    "max_nights": None,
    "region": None,
    "climate": None,
    "pace": None,
}


def empty_filters() -> dict[str, Any]:
    return {key: ([] if key == "tags" else None) for key in FILTER_KEYS}


def _number(value: Any, low: float, high: float, *, integer: bool) -> int | float | None:
    if value is None or value == "":
        return None
    try:
        number = float(str(value).replace(",", "").replace("_", ""))
    except (TypeError, ValueError):
        return None
    if number < low or number > high:
        return None
    return int(round(number)) if integer else round(number, 2)


def normalize_filters(raw: Any, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate an LLM-produced filter object against the live schema.

    A key that is absent keeps the value from ``base`` (so a partial model
    answer cannot silently drop a constraint); an explicit empty list or
    ``null`` clears it.
    """
    out = empty_filters()
    if base:
        for key in FILTER_KEYS:
            if key in base:
                out[key] = list(base[key]) if key == "tags" else base[key]
    if not isinstance(raw, dict):
        return out

    if "tags" in raw:
        tags = raw.get("tags")
        if isinstance(tags, str):
            tags = [tags]
        if isinstance(tags, list):
            clean: list[str] = []
            for tag in tags:
                key = str(tag).strip().lower().replace(" ", "_").replace("-", "_")
                if key in TAGS and key not in clean:
                    clean.append(key)
            out["tags"] = clean[:6]
        elif tags is None:
            out["tags"] = []

    for key in ("max_price", "min_price", "min_nights", "max_nights"):
        if key not in raw:
            continue
        if raw[key] is None or raw[key] == "":
            out[key] = None
            continue
        value = _number(
            raw[key],
            300 if "price" in key else 1,
            12000 if "price" in key else 30,
            integer=True,
        )
        if value is not None:
            out[key] = value

    for key, vocab in (("region", REGIONS), ("climate", CLIMATES), ("pace", PACES)):
        if key not in raw:
            continue
        if raw[key] is None or raw[key] == "":
            out[key] = None
            continue
        value = str(raw[key]).strip().lower().replace(" ", "_")
        if value in vocab:
            out[key] = value

    if out["max_price"] and out["min_price"] and out["max_price"] < out["min_price"]:
        out["min_price"] = None
    if out["max_nights"] and out["min_nights"] and out["max_nights"] < out["min_nights"]:
        out["min_nights"] = None
    return out


def _hard_match(item: dict[str, Any], filters: dict[str, Any]) -> bool:
    if filters.get("max_price") and item["price"] > filters["max_price"]:
        return False
    if filters.get("min_price") and item["price"] < filters["min_price"]:
        return False
    if filters.get("min_nights") and item["nights"] < filters["min_nights"]:
        return False
    if filters.get("max_nights") and item["nights"] > filters["max_nights"]:
        return False
    if filters.get("region") and item["region"] != filters["region"]:
        return False
    if filters.get("climate") and item["climate"] != filters["climate"]:
        return False
    if filters.get("pace") and item["pace"] != filters["pace"]:
        return False
    return True


@dataclass
class SearchResult:
    items: list[dict[str, Any]]
    total: int
    relaxed: bool = False

    @property
    def count(self) -> int:
        return len(self.items)


def search(filters: dict[str, Any]) -> SearchResult:
    """Apply hard constraints and rank by tag overlap, rating and price.

    Tags are soft constraints: an item matching any selected tag is shown and
    ranked by how many tags it matches.  This avoids the empty result set that
    ``beach AND family AND food AND diving`` would produce while still making
    the first card feel like a direct answer.  If no item matches any tag, the
    hard-constraint set is returned and flagged as relaxed.
    """
    hard = [item for item in TRAVEL_ITEMS if _hard_match(item, filters)]
    tags = [tag for tag in filters.get("tags", []) if tag in TAGS]
    relaxed = False
    if tags:
        # First try the honest reading: every selected tag.  If that is empty,
        # fall back to any-tag matches and flag the set as relaxed so the UI can
        # say "closest matches" instead of silently claiming a perfect fit.
        tagged_all = [item for item in hard if all(tag in item["tags"] for tag in tags)]
        if tagged_all:
            hard = tagged_all
        else:
            tagged_any = [item for item in hard if set(item["tags"]) & set(tags)]
            hard = tagged_any
            relaxed = True
    ranked = sorted(
        hard,
        key=lambda item: (
            -len(set(item["tags"]) & set(tags)),
            -item["rating"],
            item["price"],
        ),
    )
    return SearchResult(items=ranked, total=len(ranked), relaxed=relaxed)


def match_reasons(item: dict[str, Any], filters: dict[str, Any], language: str) -> list[str]:
    lang = language if language in ("en", "pt-BR") else "en"
    reasons: list[str] = []
    for tag in filters.get("tags") or []:
        if tag in item["tags"] and tag in TAGS:
            reasons.append(TAGS[tag][lang])
    if filters.get("max_price") and item["price"] <= filters["max_price"]:
        reasons.append(
            "Under budget" if lang == "en" else "Dentro do orçamento"
        )
    if filters.get("climate") and item["climate"] == filters["climate"]:
        reasons.append(CLIMATES[item["climate"]][lang])
    if filters.get("region") and item["region"] == filters["region"]:
        reasons.append(REGIONS[item["region"]][lang])
    if filters.get("min_nights") and item["nights"] >= filters["min_nights"]:
        reasons.append(
            f"{item['nights']} nights" if lang == "en" else f"{item['nights']} noites"
        )
    # Keep the chips useful, not overwhelming.
    seen: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.append(reason)
    return seen[:4]


def client_items(items: list[dict[str, Any]], filters: dict[str, Any], language: str) -> list[dict[str, Any]]:
    out = []
    for item in items:
        payload = localize(item, language)
        payload["match_reasons"] = match_reasons(item, filters, language)
        out.append(payload)
    return out


# ---------------------------------------------------------------------------
# Language + local fallback parser
# ---------------------------------------------------------------------------

_PT_MARKERS = {
    "quero", "queria", "viagem", "praia", "montanha", "barato", "até", "ate",
    "com", "para", "uma", "um", "não", "nao", "sim", "minha", "meu", "gostaria",
    "familia", "família", "noite", "noites", "dias", "lugar", "melhor", "mais",
    "também", "tambem", "orçamento", "orcamento", "preciso", "pode", "você",
    "voce", "falar", "inglês", "ingles", "português", "portugues", "e",
    "quem", "qual", "quais", "quando", "onde", "porque", "porquê", "como",
    "é", "são", "foi", "era", "está", "estao", "estou", "tem", "tenho",
    "faz", "fazer", "vou", "vai", "eu", "nós", "nos", "eles", "elas", "ele",
    "ela", "isso", "esse", "essa", "do", "da", "dos", "das", "em", "no", "na",
    "por", "sobre", "conte", "saber", "favor", "olá", "ola", "oi", "obrigado",
    "obrigada", "bom", "boa", "tudo", "bem", "conhece", "significa", "capital",
    "tempo", "clima", "notícia", "noticia", "história", "historia", "piada",
}
_EN_MARKERS = {
    "want", "would", "like", "trip", "beach", "mountain", "cheap", "under",
    "with", "for", "the", "my", "family", "night", "nights", "days", "place",
    "best", "more", "also", "budget", "need", "can", "you", "speak", "english",
    "portuguese", "and", "a", "an", "to",
    "who", "what", "when", "where", "why", "how", "is", "are", "was", "were",
    "do", "does", "did", "could", "should", "will", "i", "me", "we", "they",
    "he", "she", "it", "this", "that", "of", "in", "on", "at", "about",
    "tell", "know", "please", "hello", "hi", "thanks", "thank", "good", "bad",
    "yes", "no", "capital", "weather", "news", "story", "joke", "mean",
}


def detect_language(text: str, fallback: str = "en") -> str:
    """A cheap language hint used only when the model does not provide one."""
    cleaned = text.lower()
    tokens = set(re.findall(r"[a-zà-ÿ']+", cleaned))
    if any(ch in cleaned for ch in "ãõçáéíóúâêôà"):
        return "pt-BR"
    pt = len(tokens & _PT_MARKERS)
    en = len(tokens & _EN_MARKERS)
    if pt > en:
        return "pt-BR"
    if en > pt:
        return "en"
    return fallback if fallback in ("en", "pt-BR") else "en"


def normalize_language(value: Any, fallback: str = "en") -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if text in {"pt", "pt-br", "ptbr", "portuguese", "português", "portugues", "brazilian portuguese"}:
        return "pt-BR"
    if text in {"en", "en-us", "en-gb", "english"}:
        return "en"
    return fallback if fallback in ("en", "pt-BR") else "en"


def strip_accents(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


_TAG_WORDS: dict[str, tuple[str, ...]] = {
    "beach": ("beach", "praia", "seaside", "litoral", "coast", "costa", "sand", "areia"),
    "islands": ("island", "ilha", "ilhas", "islands", "arquipelago", "arquipélago"),
    "mountains": ("mountain", "montanha", "serra", "alps", "alpes", "hill", "colina"),
    "city": ("city", "cidade", "urban", "urbano", "metropolis", "metropole"),
    "culture": ("culture", "cultura", "museum", "museu", "art", "arte", "theatre", "teatro"),
    "food": ("food", "comida", "gastronomy", "gastronomia", "restaurant", "restaurante", "culinary", "culinaria"),
    "nightlife": ("nightlife", "balada", "festa", "party", "night out", "vida noturna"),
    "nature": ("nature", "natureza", "wildlife", "vida selvagem", "forest", "floresta"),
    "family": ("family", "familia", "kids", "criancas", "children", "child", "crianca"),
    "romantic": ("romantic", "romantico", "romantica", "honeymoon", "lua de mel", "couple", "casal"),
    "adventure": ("adventure", "aventura", "hiking", "trilha", "trekking", "adrenaline", "adrenalina"),
    "budget": ("cheap", "barato", "budget", "economico", "affordable", "value", "custo-beneficio"),
    "luxury": ("luxury", "luxo", "premium", "5 star", "cinco estrelas", "high end", "expensive", "caro", "sofisticado"),
    "history": ("history", "historia", "historic", "historico", "heritage", "patrimonio"),
    "wildlife": ("wildlife", "vida selvagem", "animals", "animais", "safari"),
    "diving": ("diving", "mergulho", "snorkel", "snorkeling", "scuba"),
    "hiking": ("hiking", "trilha", "trekking", "trail"),
    "skiing": ("ski", "esqui", "snowboard", "snow", "neve"),
    "wine": ("wine", "vinho", "vinicola", "winery", "vineyard"),
    "theme_parks": ("theme park", "parque tematico", "disney", "universal", "roller coaster"),
    "shopping": ("shopping", "compras", "outlet", "mall", "shopping center"),
    "music": ("music", "musica", "samba", "tango", "concert", "concerto"),
}

_REGION_WORDS: dict[str, tuple[str, ...]] = {
    "south_america": ("south america", "america do sul", "brasil", "brazil", "argentina", "chile", "peru"),
    "north_america": ("north america", "america do norte", "mexico", "canada", "usa", "united states", "estados unidos"),
    "europe": ("europe", "europa", "portugal", "greece", "grecia", "spain", "espanha", "italy", "italia"),
    "asia": ("asia", "japan", "japao", "thailand", "tailandia", "china"),
    "africa": ("africa", "south africa", "africa do sul", "morocco", "marrocos"),
}

_RESET_WORDS = (
    "start over", "reset", "show all", "clear filters", "anywhere", "surprise me",
    "show me everything", "everything again", "all trips", "show everything",
    "comecar de novo", "começar de novo", "limpar filtros", "mostrar tudo",
    "mostrar tudo de novo", "tudo de novo", "todas as viagens", "qualquer lugar",
    "me surpreenda", "tanto faz", "reiniciar",
)

_LANGUAGE_SWITCH = re.compile(
    r"(?:speak|switch|change|talk|reply|respond|continue|use|falar|trocar|mudar|"
    r"continuar|responder|usar)[^.!?]{0,28}?(english|ingles|inglês|portuguese|portugues|português)",
    re.IGNORECASE,
)


def requested_language(text: str) -> str | None:
    """Detect an explicit request to continue in English or Portuguese."""
    match = _LANGUAGE_SWITCH.search(strip_accents(text.lower()))
    if not match:
        return None
    word = match.group(1)
    return "en" if word in ("english", "ingles") else "pt-BR"


# Words that make a turn a travel turn even when no filter field changed.
_TRAVEL_HINTS = re.compile(
    r"\b(trips?|travel|vacations?|holidays?|itinerary|hotels?|flights?|"
    r"destinations?|recommend|suggests?|suggestions?|options?|booking|book|"
    r"beaches?|mountains?|islands?|viagens?|viajar|ferias|férias|roteiros?|"
    r"hoteis|hotéis|voos?|destinos?|recomend|sugest|sugestoes|sugestões|"
    r"opcoes|opções|reservar|praias?|montanhas?|ilhas?)\b",
    re.IGNORECASE,
)

# A general/trivia request should not silently become a catalog filter just
# because a destination name or a tag appears in the sentence.
_GENERAL_REQUEST = re.compile(
    r"\b(joke|piada|story|historia|história|anecdote|who is|who was|"
    r"quem e|quem é|quem foi|what is the capital|capital of|qual e a capital|"
    r"qual é a capital|weather|forecast|clima|tempo em|news|noticia|notícia|"
    r"how are you|como voce esta|como você está|how do you|como funciona|"
    r"why is|why do|por que|porque)\b",
    re.IGNORECASE,
)

_GENERAL_TOPICS = re.compile(
    r"\b(movie|film|music|song|recipe|poem|actor|actress|singer|musician|"
    r"athlete|driver|president|capital|science|math|history of|weather|news|"
    r"game|sport|team|computer|python|code|filme|livro|música|musica|receita|"
    r"poema|ator|atriz|cantor|atleta|piloto|presidente|ciência|ciencia|"
    r"história de|historia de|jogo|esporte|time|computador|código|codigo)\b",
    re.IGNORECASE,
)

_EXPLICIT_TRAVEL = re.compile(
    r"\b(trips?|travel|vacations?|holidays?|itinerary|hotels?|flights?|"
    r"destinations?|booking|book|viagens?|viajar|ferias|férias|roteiros?|"
    r"hoteis|hotéis|voos?|destinos?|reservar)\b",
    re.IGNORECASE,
)


def _same_filters(a: dict[str, Any], b: dict[str, Any]) -> bool:
    for key in FILTER_KEYS:
        if key == "tags":
            if sorted(a.get("tags") or []) != sorted(b.get("tags") or []):
                return False
        elif a.get(key) != b.get(key):
            return False
    return True


def is_general_request(text: str) -> bool:
    return bool(_GENERAL_REQUEST.search(strip_accents(text.lower())))


def is_explicit_travel(text: str) -> bool:
    return bool(_EXPLICIT_TRAVEL.search(strip_accents(text.lower())))


def is_general_topic(text: str) -> bool:
    return bool(_GENERAL_TOPICS.search(strip_accents(text.lower())))


def has_travel_hint(text: str) -> bool:
    return bool(_TRAVEL_HINTS.search(strip_accents(text.lower())))


def _mentions(text: str, word: str) -> bool:
    """Word-boundary match that tolerates simple English/Portuguese plurals."""
    word = strip_accents(word.lower())
    pattern = r"\b" + re.escape(word) + r"(?:s|es)?\b"
    return re.search(pattern, text) is not None


def _parse_money(text: str) -> float | None:
    pattern = re.compile(
        r"(?:under|below|less than|up to|max(?:imum)?|budget(?: of)?|around|about|for|"
        r"ate|até|abaixo de|menos de|no maximo|no máximo|por|com|"
        r"aproximadamente|cerca de)\s*"
        r"(?:us\$|r\$|\$)?\s*([\d][\d.,]*)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        raw = match.group(1).rstrip(".,")
        # 1.500 / 1,500 / 1500.00 -> 1500
        if re.fullmatch(r"\d{1,3}[.,]\d{3}(?:[.,]\d{2})?", raw):
            raw = raw.replace(".", "").replace(",", "")
        else:
            raw = raw.replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            continue
        if 300 <= value <= 12000:
            return value
    # Bare "$1500" / "US$ 1,500" is a budget only when a spending verb is near.
    bare = re.search(r"(?:us\$|r\$|\$)\s*([\d][\d.,]*)", text, re.IGNORECASE)
    if bare:
        raw = bare.group(1).rstrip(".,")
        if re.fullmatch(r"\d{1,3}[.,]\d{3}(?:[.,]\d{2})?", raw):
            raw = raw.replace(".", "").replace(",", "")
        else:
            raw = raw.replace(",", ".")
        try:
            value = float(raw)
            if 300 <= value <= 12000:
                return value
        except ValueError:
            pass
    currency_amount = re.search(
        r"([\d][\d.,]*)\s*(?:dollars?|dólares?|dolares?|reais|usd)\b",
        strip_accents(text.lower()),
    )
    if currency_amount:
        raw = currency_amount.group(1).rstrip(".,")
        if re.fullmatch(r"\d{1,3}[.,]\d{3}(?:[.,]\d{2})?", raw):
            raw = raw.replace(".", "").replace(",", "")
        else:
            raw = raw.replace(",", ".")
        try:
            value = float(raw)
            if 300 <= value <= 12000:
                return value
        except ValueError:
            pass
    # Spoken amounts: "fifteen hundred dollars", "two thousand dollars".
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
        "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    }
    spoken = re.search(
        r"\b(" + "|".join(words) + r")\s+hundred\b", strip_accents(text.lower())
    )
    if spoken:
        value = words[spoken.group(1)] * 100
        if 300 <= value <= 12000:
            return value
    thousand = re.search(r"\b(\d+)\s+thousand\b", strip_accents(text.lower()))
    if thousand:
        value = int(thousand.group(1)) * 1000
        if 300 <= value <= 12000:
            return value
    # Portuguese spoken budgets ("mil e quinhentos", "dois mil").
    clean_pt = strip_accents(text.lower())
    hundreds = {
        "cem": 100, "cento": 100, "duzentos": 200, "trezentos": 300,
        "quatrocentos": 400, "quinhentos": 500, "seiscentos": 600,
        "setecentos": 700, "oitocentos": 800, "novecentos": 900,
    }
    mil_e = re.search(r"\bmil e (" + "|".join(hundreds) + r")\b", clean_pt)
    if mil_e:
        value = 1000 + hundreds[mil_e.group(1)]
        if 300 <= value <= 12000:
            return value
    mil = re.search(r"\b(dois|tres|quatro|cinco|seis|sete|oito|nove|dez)\s+mil\b", clean_pt)
    if mil:
        multipliers = {"dois": 2, "tres": 3, "quatro": 4, "cinco": 5, "seis": 6,
                       "sete": 7, "oito": 8, "nove": 9, "dez": 10}
        value = multipliers[mil.group(1)] * 1000
        if 300 <= value <= 12000:
            return value
    if re.search(r"\bmil\b", clean_pt):
        return 1000
    hundreds_only = re.search(r"\b(" + "|".join(k for k in hundreds if k not in ("cem", "cento")) + r")\b", clean_pt)
    if hundreds_only:
        value = hundreds[hundreds_only.group(1)]
        if 300 <= value <= 12000:
            return value
    return None


def _parse_nights(text: str) -> dict[str, int | None]:
    if re.search(r"\b(a week|one week|uma semana|1 semana|7 nights|7 noites)\b", text, re.IGNORECASE):
        return {"min_nights": None, "max_nights": None, "nights": 7}
    match = re.search(r"(\d+)\s*(?:-|\s)?\s*(?:nights?|noites?|days?|dias?)", text, re.IGNORECASE)
    if match:
        nights = int(match.group(1))
        if 1 <= nights <= 30:
            return {"min_nights": None, "max_nights": None, "nights": nights}
    return {"min_nights": None, "max_nights": None, "nights": None}


def fallback_plan(text: str, previous: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Extract filters without the LLM.  Returns ``(filters, reset)``."""
    clean = strip_accents(text.lower())
    if any(word in clean for word in (strip_accents(w) for w in _RESET_WORDS)):
        return empty_filters(), True

    tags: list[str] = []
    for tag, words in _TAG_WORDS.items():
        if any(_mentions(clean, word) for word in words):
            tags.append(tag)
    # Generic "coast" should also surface island escapes; keep it intentional.
    if "beach" in tags and "islands" not in tags and _mentions(clean, "island"):
        tags.append("islands")
    if re.search(r"\b(not too expensive|not expensive|nao muito caro|nao caro)\b", clean):
        tags = [tag for tag in tags if tag != "luxury"]

    filters = normalize_filters({}, previous)
    if tags:
        additive = bool(re.search(r"\b(also|as well|too|tambem|também|e também)\b", clean))
        if additive:
            filters["tags"] = list(dict.fromkeys((previous.get("tags") or []) + tags))
        else:
            filters["tags"] = tags

    price = _parse_money(text)
    if price:
        filters["max_price"] = int(round(price))
        filters["min_price"] = previous.get("min_price")
    elif "luxury" in tags:
        # "I want something expensive" should free the old upper bound.
        filters["max_price"] = None
    elif "budget" in tags:
        filters["max_price"] = int(previous.get("max_price") or 1200)
    if not price and re.search(r"\b(not too expensive|nao muito caro|não muito caro)\b", clean):
        filters["max_price"] = int(previous.get("max_price") or 1500)
    if not price and re.search(r"\b(cheaper|more affordable|mais barato|menos caro)\b", clean):
        current = previous.get("max_price")
        filters["max_price"] = int(current * 0.85) if current else 1200
    if re.search(
        r"\b(no budget|forget the budget|any budget|any price|money is no object|"
        r"sem orcamento|sem limite|qualquer preco|esqueca o orcamento)\b",
        clean,
    ):
        filters["max_price"] = None
        filters["min_price"] = None
    nights = _parse_nights(text)
    if nights["nights"]:
        filters["min_nights"] = nights["nights"]
        filters["max_nights"] = nights["nights"]

    for region, words in _REGION_WORDS.items():
        if any(_mentions(clean, word) for word in words):
            filters["region"] = region
            break
    if re.search(r"\b(warm|quente|calor|sunny|sol)\b", clean):
        filters["climate"] = "warm"
    elif re.search(r"\b(cold|frio|frio|snow|neve|winter|inverno)\b", clean):
        filters["climate"] = "cold"
    elif re.search(r"\b(mild|ameno|spring|primavera)\b", clean):
        filters["climate"] = "mild"
    if re.search(r"\b(relaxed|tranquil|tranquilo|slow|devagar)\b", clean):
        filters["pace"] = "relaxed"
    elif re.search(r"\b(adventure|aventura|active|ativo|hiking)\b", clean):
        filters["pace"] = "adventure"
    return normalize_filters(filters), False


def _fact_summary(item: dict[str, Any], language: str) -> str:
    lang = language if language in ("en", "pt-BR") else "en"
    if lang == "pt-BR":
        price = f"US$ {item['price']:,}".replace(",", ".")
        return f"{item['name']}, {item['nights']} noites, a partir de {price}"
    return f"{item['name']}, {item['nights']} nights, from US$ {item['price']:,}"


def fallback_reply(
    text: str,
    language: str,
    result: SearchResult,
    filters: dict[str, Any],
    *,
    reset: bool = False,
) -> str:
    lang = language if language in ("en", "pt-BR") else "en"
    if reset and not filters.get("tags") and not any(
        filters.get(key) for key in ("max_price", "min_price", "min_nights", "max_nights", "region", "climate", "pace")
    ):
        if lang == "pt-BR":
            return "Claro, recomecei o catálogo. Me diga o tipo de viagem que você imagina."
        return "Sure, I reset the catalog. Tell me what kind of trip you have in mind."

    if result.count == 0:
        if lang == "pt-BR":
            return "Não encontrei nenhuma opção com esses critérios. Quer que eu aumente o orçamento ou remova um filtro?"
        return "I couldn't find an option with all of those details. Want me to raise the budget or drop one filter?"

    top = result.items[0]
    if lang == "pt-BR":
        count = f"{result.count} opção" if result.count == 1 else f"{result.count} opções"
        if result.relaxed and filters.get("tags"):
            return (
                f"Não achei uma combinação exata, mas separei {count} próximas. "
                f"Minha primeira escolha é {_fact_summary(top, lang)}. "
                "Quer ajustar o orçamento ou o número de noites?"
            )
        return (
            f"Separei {count} para você. "
            f"Minha primeira escolha é {_fact_summary(top, lang)}. "
            "Quer que eu foque em praia, orçamento ou uma viagem mais romântica?"
        )
    count = f"{result.count} option" if result.count == 1 else f"{result.count} options"
    if result.relaxed and filters.get("tags"):
        return (
            f"I couldn't match every detail, but I pulled {count} close. "
            f"My first pick is {_fact_summary(top, lang)}. Want to adjust the budget or nights?"
        )
    return (
        f"I found {count} for you. "
        f"My first pick is {_fact_summary(top, lang)}. "
        "Want me to focus on beach, budget, or something more romantic?"
    )


# ---------------------------------------------------------------------------
# LLM wrapper
# ---------------------------------------------------------------------------

REPLY_SYSTEM = """You are Laya, the voice travel designer at Onda Travel. The website has already applied the customer's filters, and you are given only the matching trips. Write the spoken answer.

Rules:
- Reply in {language}.
- 1-2 short spoken sentences, at most 30 words. No markdown, emoji, bullet points or lists.
- Use only the MATCHING TRIPS below. Never invent a destination, price or detail.
- All prices are USD per person for the whole stay. Say "US$" or "dollars"; never say R$ or reais.
- Mention at most two trips and at most one price.
- If the matching list is empty, say there is no exact match and suggest relaxing one filter.
- If the customer asks about a specific trip, use its details.
- If the customer also asks a general or trivia question, answer that briefly and naturally before or alongside the travel suggestion.
- For live data such as weather, news or opening hours, say you cannot check it live.
- Sound like a warm human travel designer, not a catalog readout.
"""

GENERAL_SYSTEM = """You are Laya, a warm bilingual voice assistant at Onda Travel. You can make small talk, answer everyday questions and discuss general knowledge as well as travel.

Rules:
- Reply in {language}.
- 1-2 short spoken sentences, at most 35 words. No markdown, emoji, bullet points or lists.
- Answer the customer's question directly and naturally.
- If you do not know, or the answer needs live data (weather, news, current events), say so briefly and honestly.
- Never invent facts, people, dates or statistics. If you are not fully confident about a name, role, date or number, say that you may be mistaken instead of guessing.
- For questions about a specific person, hedge naturally ("As far as I know…") because your knowledge may be out of date.
- Do not force the conversation back to travel unless it fits naturally.
- If the customer is just saying hello, thanks or goodbye, respond warmly in one sentence.
"""


def general_fallback_reply(text: str, language: str) -> str:
    """A safe answer when the content model is unavailable."""
    lang = language if language in ("en", "pt-BR") else "en"
    clean = strip_accents(text.lower())
    if re.search(r"\b(hello|hi|hey|ola|olá|oi|bom dia|boa tarde|boa noite)\b", clean):
        return (
            "Oi! Estou por aqui. Quer conversar sobre uma viagem ou sobre outra coisa?"
            if lang == "pt-BR"
            else "Hi! I'm here. Want to talk about a trip, or about something else?"
        )
    if re.search(r"\b(thanks|thank you|obrigado|obrigada|valeu)\b", clean):
        return "De nada! Se precisar de mais alguma coisa, é só falar." if lang == "pt-BR" else "You're welcome! Just say the word if you need anything else."
    if re.search(r"\b(who is|who was|quem e|quem é|quem foi)\b", clean):
        return (
            "Não tenho uma fonte confiável sobre essa pessoa neste demo, então prefiro não inventar. Posso ajudar com uma viagem ou com outra pergunta simples."
            if lang == "pt-BR"
            else "I don't have a reliable source about that person in this demo, so I'd rather not make something up. I can help with a trip or another simple question."
        )
    return (
        "Consigo conversar sobre isso, mas meu conhecimento neste demo é limitado. Pode reformular ou me contar que tipo de viagem você imagina?"
        if lang == "pt-BR"
        else "I can chat about that, though my knowledge in this demo is limited. Could you rephrase, or tell me what kind of trip you're imagining?"
    )




def _extract_json(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(cleaned[start : end + 1])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    # Last resort: pull the two fields we need even if the JSON is broken by an
    # unescaped newline or a trailing sentence from the model.
    reply = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned, re.DOTALL)
    language = re.search(r'"language"\s*:\s*"([^"]+)"', cleaned)
    if reply:
        try:
            reply_text = json.loads(f'"{reply.group(1)}"')
        except json.JSONDecodeError:
            reply_text = reply.group(1)
        return {
            "reply": reply_text,
            "language": language.group(1) if language else None,
        }
    return None


def _history_text(history: list[dict[str, str]], limit: int = 8) -> str:
    lines = []
    for message in history[-limit:]:
        role = "Customer" if message.get("role") == "user" else "Laya"
        lines.append(f"{role}: {message.get('content', '')}")
    return "\n".join(lines) if lines else "(new conversation)"


def _language_name(language: str) -> str:
    return "Brazilian Portuguese" if language == "pt-BR" else "English"


def _clean_reply(text: str, *, max_words: int = 45) -> str:
    """Normalize a model answer for both the bubble and the TTS engine."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith("{"):
        parsed = _extract_json(cleaned)
        if parsed and parsed.get("reply"):
            cleaned = str(parsed["reply"])
    cleaned = re.sub(r"^```(?:\w+)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"[*_`#>]+", "", cleaned)
    # The catalog is USD; keep a stray BRL symbol from becoming a wrong spoken
    # price if the model mirrors the user's currency habit.
    cleaned = re.sub(r"\bR\$\s*", "US$ ", cleaned)
    cleaned = re.sub(r"\breais\b", "dólares", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[\r\n]+\s*[-•]\s*", " ", cleaned)
    cleaned = re.sub(r"^\s*[-•]\s*", "", cleaned)
    cleaned = re.sub(r"(?<=[.!?])\s*[-•]\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Two sentences are plenty for a voice turn; keep the first two and drop
    # any list-like trailing fragments.
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    if len(sentences) > 2:
        cleaned = " ".join(sentences[:2]).strip()
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words]).rstrip(" ,;:") + "."
    return cleaned.strip()


def _mentioned_items(text: str) -> list[dict[str, Any]]:
    clean = strip_accents(text.lower())
    found = []
    for item in TRAVEL_ITEMS:
        name = strip_accents(item["name"].lower())
        if name and name in clean:
            found.append(item)
    return found


def _prompt_items(items: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    lang = language if language in ("en", "pt-BR") else "en"
    return [
        {
            "id": item["id"],
            "name": item["name"],
            "country": item["country"][lang],
            "price_usd": item["price"],
            "nights": item["nights"],
            "rating": item["rating"],
            "tags": item["tags"],
            "climate": item["climate"],
            "region": item["region"],
            "summary": item["summary"][lang],
        }
        for item in items[:6]
    ]


@dataclass
class AssistantTurn:
    language: str
    reply: str
    filters: dict[str, Any]
    items: list[dict[str, Any]]
    total: int
    highlights: list[str] = field(default_factory=list)
    fallback_used: bool = False
    raw_output: str = ""
    relaxed: bool = False
    intent: str = "travel"
    catalog_changed: bool = True

    @property
    def empty(self) -> bool:
        return not self.items


class QwenTravelLLM:
    """Small wrapper around Qwen3-4B-Instruct-2507 for one JSON turn."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        max_new_tokens: int = 96,
    ) -> None:
        import torch

        self.torch = torch
        self.backend = "transformers"
        # Unsloth's patched Qwen kernels are measurably faster for the short
        # JSON turn this demo generates.  If the install lacks them, plain
        # transformers is a fully functional fallback.
        try:
            from unsloth import FastLanguageModel

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_path,
                max_seq_length=4096,
                dtype=torch.bfloat16,
                load_in_4bit=False,
            )
            FastLanguageModel.for_inference(self.model)
            self.backend = "unsloth"
        except Exception as exc:
            print(f"[agent] Unsloth Qwen load unavailable ({type(exc).__name__}: {exc}); using transformers", flush=True)
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            kwargs: dict[str, Any] = {
                "dtype": torch.bfloat16,
                "attn_implementation": "sdpa",
                "low_cpu_mem_usage": True,
            }
            if device == "cuda":
                kwargs["device_map"] = {"": 0}
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            self.model.eval()
        self.device = next(self.model.parameters()).device
        self.max_new_tokens = max_new_tokens

    def generate(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.03,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class TravelAgent:
    """Turns a user utterance into speech text + a new website filter state.

    Filters are deterministic: the same parser powers the no-LLM fallback and
    the live demo, so the cards can update even if generation is slow.  Qwen is
    asked for the *spoken reply only*, over the already-filtered shortlist.
    """

    def __init__(
        self,
        llm: QwenTravelLLM | None,
        *,
        fallback_language: str = "en",
    ) -> None:
        self.llm = llm
        self.fallback_language = fallback_language

    def _reply_prompt(
        self,
        user_text: str,
        filters: dict[str, Any],
        items: list[dict[str, Any]],
        history: list[dict[str, str]],
        language: str,
        total: int | None = None,
    ) -> list[dict[str, str]]:
        trips = json.dumps(_prompt_items(items, language), ensure_ascii=False, separators=(",", ":"))
        current = json.dumps(filters, ensure_ascii=False, separators=(",", ":"))
        total_line = f"TOTAL MATCHING TRIPS: {total}\n" if total is not None else ""
        user = (
            f"LANGUAGE: {_language_name(language)}\n"
            f"CUSTOMER MESSAGE:\n{user_text}\n\n"
            f"ACTIVE FILTERS:\n{current}\n\n"
            f"{total_line}"
            f"MATCHING TRIPS (up to six):\n{trips}\n\n"
            f"CONVERSATION:\n{_history_text(history)}\n\n"
            "Write the spoken reply now."
        )
        return [
            {"role": "system", "content": REPLY_SYSTEM.format(language=_language_name(language))},
            {"role": "user", "content": user},
        ]

    def _reply(
        self,
        user_text: str,
        filters: dict[str, Any],
        items: list[dict[str, Any]],
        history: list[dict[str, str]],
        language: str,
        total: int | None = None,
    ) -> tuple[str, str] | None:
        if self.llm is None:
            return None
        try:
            raw = self.llm.generate(
                self._reply_prompt(user_text, filters, items, history, language, total)
            )
        except Exception as exc:  # a demo should degrade, not die
            print(f"[agent] LLM failed: {type(exc).__name__}: {exc}", flush=True)
            return None
        reply = _clean_reply(raw)
        return (reply, raw) if reply else None

    def _general_prompt(
        self,
        user_text: str,
        history: list[dict[str, str]],
        language: str,
    ) -> list[dict[str, str]]:
        user = (
            f"LANGUAGE: {_language_name(language)}\n"
            f"CUSTOMER MESSAGE:\n{user_text}\n\n"
            f"CONVERSATION:\n{_history_text(history)}\n\n"
            "Write the spoken reply now."
        )
        return [
            {"role": "system", "content": GENERAL_SYSTEM.format(language=_language_name(language))},
            {"role": "user", "content": user},
        ]

    def _general_reply(
        self,
        user_text: str,
        history: list[dict[str, str]],
        language: str,
    ) -> tuple[str, str] | None:
        if self.llm is None:
            return None
        try:
            raw = self.llm.generate(self._general_prompt(user_text, history, language))
        except Exception as exc:
            print(f"[agent] general LLM failed: {type(exc).__name__}: {exc}", flush=True)
            return None
        reply = _clean_reply(raw)
        return (reply, raw) if reply else None

    def respond(
        self,
        user_text: str,
        *,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
        language: str | None = None,
    ) -> AssistantTurn:
        previous_filters = normalize_filters({}, filters or empty_filters())
        history = list(history or [])
        hint = language or detect_language(user_text, self.fallback_language)

        requested = requested_language(user_text)
        candidate, reset = fallback_plan(user_text, previous_filters)
        filter_changes = reset or not _same_filters(candidate, previous_filters)
        mentioned = _mentioned_items(user_text)
        general = is_general_request(user_text) or is_general_topic(user_text)
        explicit_travel = is_explicit_travel(user_text)
        travel_hint = has_travel_hint(user_text)
        travel_signal = reset or filter_changes or bool(mentioned) or travel_hint or explicit_travel

        if general and not explicit_travel:
            intent = "smalltalk"
        elif general and travel_signal:
            intent = "mixed"
        elif travel_signal:
            intent = "travel"
        else:
            intent = "smalltalk"
        if requested and intent == "smalltalk" and not travel_signal:
            intent = "language"

        turn_language = requested or detect_language(user_text, hint)
        filters = candidate if intent in ("travel", "mixed") else previous_filters
        result = search(filters)

        if intent in ("travel", "mixed") and mentioned:
            existing = {item["id"] for item in result.items}
            extra = [item for item in mentioned if item["id"] not in existing]
            if extra:
                result = SearchResult(
                    items=extra + result.items,
                    total=result.total + len(extra),
                    relaxed=result.relaxed,
                )
        highlights = [item["id"] for item in mentioned]
        if not highlights and result.items:
            highlights = [result.items[0]["id"]]

        reply = ""
        raw = ""
        fallback_used = True
        catalog_changed = intent in ("travel", "mixed")

        if intent == "language":
            # A language request should feel instant and must not be misread as
            # a new search.  Keep the filters, answer in the requested language.
            reply = (
                "Sure, we can continue in English. What kind of trip are you imagining?"
                if requested == "en"
                else "Claro, podemos continuar em português. Que tipo de viagem você está imaginando?"
            )
        elif intent == "smalltalk":
            generated = self._general_reply(user_text, history, turn_language)
            if generated:
                reply, raw = generated
                fallback_used = False
            else:
                reply = general_fallback_reply(user_text, turn_language)
        elif result.count == 0:
            # No matches: the deterministic answer is exact and immediate.
            reply = fallback_reply(user_text, turn_language, result, filters, reset=reset)
        else:
            generated = self._reply(
                user_text, filters, result.items, history, turn_language, result.total
            )
            if generated:
                reply, raw = generated
                fallback_used = False
            else:
                reply = fallback_reply(user_text, turn_language, result, filters, reset=reset)

        if reply:
            # If the model wrote in a different language than expected, keep the
            # UI and TTS consistent with what was actually said.
            detected = detect_language(reply, turn_language)
            if detected != turn_language and len(reply.split()) >= 4:
                turn_language = detected

        items = client_items(result.items, filters, turn_language)
        valid_ids = {item["id"] for item in items}
        highlights = [item_id for item_id in highlights if item_id in valid_ids]
        if not highlights and items:
            highlights = [items[0]["id"]]
        return AssistantTurn(
            language=turn_language,
            reply=reply,
            filters=filters,
            items=items,
            total=result.total,
            highlights=highlights[:3],
            fallback_used=fallback_used,
            raw_output=raw,
            relaxed=result.relaxed,
            intent=intent,
            catalog_changed=catalog_changed,
        )


def catalogue_payload(language: str) -> dict[str, Any]:
    """Initial state sent to a new browser session."""
    return {
        "items": client_items(TRAVEL_ITEMS, empty_filters(), language),
        "filters": empty_filters(),
        "tags": {key: value[language] if language in value else value["en"] for key, value in TAGS.items()},
        "regions": {key: value[language] if language in value else value["en"] for key, value in REGIONS.items()},
        "climates": {key: value[language] if language in value else value["en"] for key, value in CLIMATES.items()},
        "paces": {key: value[language] if language in value else value["en"] for key, value in PACES.items()},
    }
