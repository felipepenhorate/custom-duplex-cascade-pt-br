"""Mock travel catalog for the Onda Travel voice demo.

The catalog is deliberately small and hand-written: it is the single source of
truth the language model is allowed to recommend from, and the same objects are
sent to the browser so the cards, prices and tags always agree with what Laya
says out loud.

All prices are USD per person for the whole trip.  Every user-facing string is
a ``{"en": ..., "pt-BR": ...}`` pair so the page can switch language without a
network round trip.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

TAGS: dict[str, dict[str, str]] = {
    "beach": {"en": "Beach", "pt-BR": "Praia"},
    "islands": {"en": "Islands", "pt-BR": "Ilhas"},
    "mountains": {"en": "Mountains", "pt-BR": "Montanhas"},
    "city": {"en": "City", "pt-BR": "Cidade"},
    "culture": {"en": "Culture", "pt-BR": "Cultura"},
    "food": {"en": "Food", "pt-BR": "Gastronomia"},
    "nightlife": {"en": "Nightlife", "pt-BR": "Vida noturna"},
    "nature": {"en": "Nature", "pt-BR": "Natureza"},
    "family": {"en": "Family", "pt-BR": "Família"},
    "romantic": {"en": "Romantic", "pt-BR": "Romântico"},
    "adventure": {"en": "Adventure", "pt-BR": "Aventura"},
    "budget": {"en": "Budget", "pt-BR": "Econômico"},
    "luxury": {"en": "Luxury", "pt-BR": "Luxo"},
    "history": {"en": "History", "pt-BR": "História"},
    "wildlife": {"en": "Wildlife", "pt-BR": "Vida selvagem"},
    "diving": {"en": "Diving", "pt-BR": "Mergulho"},
    "hiking": {"en": "Hiking", "pt-BR": "Trilhas"},
    "skiing": {"en": "Skiing", "pt-BR": "Esqui"},
    "wine": {"en": "Wine", "pt-BR": "Vinhos"},
    "theme_parks": {"en": "Theme parks", "pt-BR": "Parques temáticos"},
    "shopping": {"en": "Shopping", "pt-BR": "Compras"},
    "music": {"en": "Music", "pt-BR": "Música"},
}

REGIONS: dict[str, dict[str, str]] = {
    "south_america": {"en": "South America", "pt-BR": "América do Sul"},
    "north_america": {"en": "North America", "pt-BR": "América do Norte"},
    "europe": {"en": "Europe", "pt-BR": "Europa"},
    "asia": {"en": "Asia", "pt-BR": "Ásia"},
    "africa": {"en": "Africa", "pt-BR": "África"},
}

CLIMATES: dict[str, dict[str, str]] = {
    "warm": {"en": "Warm", "pt-BR": "Quente"},
    "mild": {"en": "Mild", "pt-BR": "Ameno"},
    "cold": {"en": "Cold", "pt-BR": "Frio"},
}

PACES: dict[str, dict[str, str]] = {
    "relaxed": {"en": "Relaxed", "pt-BR": "Tranquilo"},
    "balanced": {"en": "Balanced", "pt-BR": "Equilibrado"},
    "adventure": {"en": "Adventure", "pt-BR": "Aventura"},
}


def _t(en: str, pt: str) -> dict[str, str]:
    return {"en": en, "pt-BR": pt}


def _l(en: list[str], pt: list[str]) -> dict[str, list[str]]:
    return {"en": en, "pt-BR": pt}


TRAVEL_ITEMS: list[dict[str, Any]] = [
    {
        "id": "maragogi",
        "name": "Maragogi",
        "country": _t("Brazil", "Brasil"),
        "region": "south_america",
        "price": 1280,
        "nights": 7,
        "rating": 4.8,
        "reviews": 412,
        "climate": "warm",
        "pace": "relaxed",
        "tags": ["beach", "islands", "family", "food", "diving", "nature"],
        "tagline": _t("Turquoise reef pools", "Piscinas naturais de água turquesa"),
        "summary": _t(
            "Seven slow days between coral reefs, mangrove rivers and a village "
            "that still feels like Brazil at its easiest.",
            "Sete dias tranquilos entre recifes de coral, rios de mangue e uma "
            "vila que ainda parece o Brasil mais fácil de viver.",
        ),
        "highlights": _l(
            ["Natural reef pools", "Boat trip to Galés", "Family-friendly resorts"],
            ["Piscinas naturais", "Passeio às Galés", "Resorts para famílias"],
        ),
        "best_for": _l(["Beach", "Families", "Snorkelling"], ["Praia", "Famílias", "Mergulho"]),
        "scene": "beach",
        "colors": ["#2dd4bf", "#0369a1"],
    },
    {
        "id": "noronha",
        "name": "Fernando de Noronha",
        "country": _t("Brazil", "Brasil"),
        "region": "south_america",
        "price": 2450,
        "nights": 6,
        "rating": 4.9,
        "reviews": 286,
        "climate": "warm",
        "pace": "balanced",
        "tags": ["beach", "islands", "nature", "diving", "romantic", "luxury", "wildlife"],
        "tagline": _t("A protected island universe", "Um universo de ilha preservado"),
        "summary": _t(
            "Volcanic peaks, empty beaches and water so clear that the boat seems "
            "to float over the reef. The premium choice for nature lovers.",
            "Picos vulcânicos, praias vazias e água tão clara que o barco parece "
            "flutuar sobre o recife. A escolha premium para quem ama natureza.",
        ),
        "highlights": _l(
            ["Dolphin Bay at sunrise", "Baía do Sancho", "Reef diving"],
            ["Baía dos Golfinhos ao amanhecer", "Baía do Sancho", "Mergulho no recife"],
        ),
        "best_for": _l(["Honeymoons", "Diving", "Nature"], ["Luas de mel", "Mergulho", "Natureza"]),
        "scene": "island",
        "colors": ["#22d3ee", "#155e75"],
    },
    {
        "id": "gramado",
        "name": "Gramado",
        "country": _t("Brazil", "Brasil"),
        "region": "south_america",
        "price": 980,
        "nights": 5,
        "rating": 4.7,
        "reviews": 538,
        "climate": "cold",
        "pace": "relaxed",
        "tags": ["mountains", "romantic", "food", "family", "wine", "city"],
        "tagline": _t("Chocolate, wine and alpine air", "Chocolate, vinho e ar de serra"),
        "summary": _t(
            "A storybook mountain town of fondue restaurants, hydrangea streets "
            "and vineyards just outside the centre.",
            "Uma cidade de serra de conto de fadas, com fondue, ruas de hortênsias "
            "e vinícolas logo ali fora do centro.",
        ),
        "highlights": _l(
            ["Lago Negro", "Serra Gaúcha wineries", "Winter lights festival"],
            ["Lago Negro", "Vinícolas da Serra Gaúcha", "Festival de inverno"],
        ),
        "best_for": _l(["Couples", "Winter", "Food"], ["Casaizinhos", "Inverno", "Gastronomia"]),
        "scene": "mountain",
        "colors": ["#a5b4fc", "#334155"],
    },
    {
        "id": "salvador",
        "name": "Salvador",
        "country": _t("Brazil", "Brasil"),
        "region": "south_america",
        "price": 1120,
        "nights": 6,
        "rating": 4.6,
        "reviews": 604,
        "climate": "warm",
        "pace": "balanced",
        "tags": ["city", "culture", "beach", "music", "food", "history", "budget"],
        "tagline": _t("Afro-Brazilian rhythm by the sea", "Ritmo afro-brasileiro à beira-mar"),
        "summary": _t(
            "Colourful Pelourinho streets in the morning, beach clubs in the "
            "afternoon and live samba until late.",
            "Ruas coloridas do Pelourinho pela manhã, beach clubs à tarde e samba "
            "ao vivo até tarde.",
        ),
        "highlights": _l(
            ["Pelourinho", "Barra beach", "Bahian cooking class"],
            ["Pelourinho", "Praia da Barra", "Aula de culinária baiana"],
        ),
        "best_for": _l(["Culture", "Music", "City breaks"], ["Cultura", "Música", "City breaks"]),
        "scene": "city",
        "colors": ["#fb923c", "#9a3412"],
    },
    {
        "id": "buenos-aires",
        "name": "Buenos Aires",
        "country": _t("Argentina", "Argentina"),
        "region": "south_america",
        "price": 890,
        "nights": 5,
        "rating": 4.7,
        "reviews": 721,
        "climate": "mild",
        "pace": "balanced",
        "tags": ["city", "culture", "food", "nightlife", "music", "budget", "shopping"],
        "tagline": _t("Tango, steakhouses and long nights", "Tango, parrillas e noites longas"),
        "summary": _t(
            "Grand boulevards, independent theatres and the best value steak "
            "dinner in South America.",
            "Avenidas elegantes, teatros independentes e o melhor jantar de "
            "carne da América do Sul pelo melhor preço.",
        ),
        "highlights": _l(
            ["San Telmo market", "Palermo food tour", "Tango in La Boca"],
            ["Feira de San Telmo", "Tour gastronômico em Palermo", "Tango em La Boca"],
        ),
        "best_for": _l(["Food", "Nightlife", "Value"], ["Gastronomia", "Vida noturna", "Custo-benefício"]),
        "scene": "city",
        "colors": ["#f472b6", "#7e22ce"],
    },
    {
        "id": "lisbon",
        "name": "Lisbon",
        "country": _t("Portugal", "Portugal"),
        "region": "europe",
        "price": 1650,
        "nights": 6,
        "rating": 4.8,
        "reviews": 934,
        "climate": "mild",
        "pace": "balanced",
        "tags": ["city", "culture", "food", "history", "romantic", "shopping"],
        "tagline": _t("Yellow trams and Atlantic light", "Elétricos amarelos e luz do Atlântico"),
        "summary": _t(
            "Hilltop viewpoints, tiled facades and pastéis de nata between "
            "neighbourhoods that reward wandering.",
            "Mirantes, fachadas de azulejo e pastéis de nata entre bairros que "
            "recompensam quem se perde.",
        ),
        "highlights": _l(
            ["Alfama at sunset", "Belém and pastéis", "Sintra day trip"],
            ["Alfama ao pôr do sol", "Belém e os pastéis", "Bate-volta a Sintra"],
        ),
        "best_for": _l(["Culture", "Food", "City breaks"], ["Cultura", "Gastronomia", "City breaks"]),
        "scene": "city",
        "colors": ["#fbbf24", "#b45309"],
    },
    {
        "id": "tulum",
        "name": "Tulum",
        "country": _t("Mexico", "México"),
        "region": "north_america",
        "price": 1420,
        "nights": 6,
        "rating": 4.6,
        "reviews": 688,
        "climate": "warm",
        "pace": "balanced",
        "tags": ["beach", "culture", "nature", "romantic", "nightlife", "food", "diving"],
        "tagline": _t("Mayan ruins above the sand", "Ruínas maias acima da areia"),
        "summary": _t(
            "Cliffside ruins, cenote swims and a beach strip that moves from "
            "yoga mornings to mezcal nights.",
            "Ruínas na falésia, mergulhos em cenotes e uma orla que vai do yoga "
            "pela manhã ao mezcal à noite.",
        ),
        "highlights": _l(
            ["Tulum ruins", "Gran Cenote", "Sian Ka'an biosphere"],
            ["Ruínas de Tulum", "Gran Cenote", "Biosfera de Sian Ka'an"],
        ),
        "best_for": _l(["Beach", "Couples", "Culture"], ["Praia", "Casaizinhos", "Cultura"]),
        "scene": "beach",
        "colors": ["#5eead4", "#0f766e"],
    },
    {
        "id": "banff",
        "name": "Banff",
        "country": _t("Canada", "Canadá"),
        "region": "north_america",
        "price": 1980,
        "nights": 7,
        "rating": 4.9,
        "reviews": 517,
        "climate": "cold",
        "pace": "adventure",
        "tags": ["mountains", "nature", "adventure", "hiking", "skiing", "wildlife", "luxury"],
        "tagline": _t("Glacier lakes and pine air", "Lagos glaciais e ar de pinheiro"),
        "summary": _t(
            "A week of turquoise lakes, gondola views and wildlife drives in the "
            "Rockies, with a cozy lodge to come home to.",
            "Uma semana de lagos turquesa, vistas de gondola e safáis de vida "
            "selvagem nas Rochosas, com um lodge aconchegante para voltar.",
        ),
        "highlights": _l(
            ["Lake Louise", "Moraine Lake", "Icefields Parkway"],
            ["Lago Louise", "Lago Moraine", "Icefields Parkway"],
        ),
        "best_for": _l(["Hiking", "Nature", "Adventure"], ["Trilhas", "Natureza", "Aventura"]),
        "scene": "mountain",
        "colors": ["#67e8f9", "#1e3a8a"],
    },
    {
        "id": "kyoto",
        "name": "Kyoto",
        "country": _t("Japan", "Japão"),
        "region": "asia",
        "price": 2300,
        "nights": 8,
        "rating": 4.9,
        "reviews": 803,
        "climate": "mild",
        "pace": "balanced",
        "tags": ["culture", "food", "history", "nature", "romantic", "luxury"],
        "tagline": _t("Temples, tea houses and quiet gardens", "Templos, casas de chá e jardins silenciosos"),
        "summary": _t(
            "Eight days of wooden temples, tiny bars and gardens designed for "
            "slowing down, with easy train hops from the city.",
            "Oito dias de templos de madeira, bares minúsculos e jardins feitos "
            "para desacelerar, com trens fáceis saindo da cidade.",
        ),
        "highlights": _l(
            ["Fushimi Inari at dawn", "Arashiyama bamboo grove", "Nishiki market"],
            ["Fushimi Inari ao amanhecer", "Bambuzal de Arashiyama", "Mercado Nishiki"],
        ),
        "best_for": _l(["Culture", "Food", "Couples"], ["Cultura", "Gastronomia", "Casaizinhos"]),
        "scene": "temple",
        "colors": ["#fda4af", "#9f1239"],
    },
    {
        "id": "cape-town",
        "name": "Cape Town",
        "country": _t("South Africa", "África do Sul"),
        "region": "africa",
        "price": 1890,
        "nights": 8,
        "rating": 4.8,
        "reviews": 466,
        "climate": "mild",
        "pace": "balanced",
        "tags": ["mountains", "beach", "city", "wine", "adventure", "nature", "food"],
        "tagline": _t("A mountain, two oceans and wine country", "Uma montanha, dois oceanos e vinícolas"),
        "summary": _t(
            "Hike Table Mountain in the morning, drive to the winelands in the "
            "afternoon and eat seafood by the harbour at night.",
            "Trilha na Table Mountain pela manhã, vinícolas à tarde e frutos do "
            "mar no porto à noite.",
        ),
        "highlights": _l(
            ["Table Mountain", "Cape Point", "Stellenbosch wine route"],
            ["Table Mountain", "Cape Point", "Rota dos vinhos de Stellenbosch"],
        ),
        "best_for": _l(["Adventure", "Wine", "Beach"], ["Aventura", "Vinhos", "Praia"]),
        "scene": "coast",
        "colors": ["#facc15", "#0f766e"],
    },
    {
        "id": "orlando",
        "name": "Orlando",
        "country": _t("United States", "Estados Unidos"),
        "region": "north_america",
        "price": 1540,
        "nights": 7,
        "rating": 4.5,
        "reviews": 1204,
        "climate": "warm",
        "pace": "adventure",
        "tags": ["family", "theme_parks", "city", "shopping", "adventure"],
        "tagline": _t("Theme parks, pools and easy days", "Parques, piscinas e dias fáceis"),
        "summary": _t(
            "A practical family week with park days, water parks and a hotel "
            "pool that saves the afternoon.",
            "Uma semana prática em família, com dias de parque, parques aquáticos "
            "e a piscina do hotel para salvar a tarde.",
        ),
        "highlights": _l(
            ["Magic Kingdom", "Universal Islands", "Outlet shopping"],
            ["Magic Kingdom", "Universal Islands", "Outlets"],
        ),
        "best_for": _l(["Families", "Theme parks", "Shopping"], ["Famílias", "Parques", "Compras"]),
        "scene": "park",
        "colors": ["#60a5fa", "#1d4ed8"],
    },
    {
        "id": "santorini",
        "name": "Santorini",
        "country": _t("Greece", "Grécia"),
        "region": "europe",
        "price": 2150,
        "nights": 7,
        "rating": 4.8,
        "reviews": 642,
        "climate": "warm",
        "pace": "relaxed",
        "tags": ["islands", "beach", "romantic", "luxury", "food", "culture"],
        "tagline": _t("White villages over a blue caldera", "Vilas brancas sobre a caldeira azul"),
        "summary": _t(
            "Cliffside sunsets, volcanic beaches and long lunches with a view "
            "that does not get old.",
            "Pôr do sol na falésia, praias vulcânicas e almoços longos com uma "
            "vista que não cansa.",
        ),
        "highlights": _l(
            ["Oia sunset", "Red Beach", "Caldera sailing"],
            ["Pôr do sol em Oia", "Praia Vermelha", "Velejo na caldeira"],
        ),
        "best_for": _l(["Couples", "Luxury", "Islands"], ["Casaizinhos", "Luxo", "Ilhas"]),
        "scene": "island",
        "colors": ["#a5f3fc", "#1e40af"],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def all_items() -> list[dict[str, Any]]:
    """A defensive copy so callers cannot mutate the catalog."""
    return deepcopy(TRAVEL_ITEMS)


def localize(item: dict[str, Any], language: str) -> dict[str, Any]:
    """Return one catalog item with the requested language picked out."""
    lang = language if language in ("en", "pt-BR") else "en"
    out = deepcopy(item)
    for key in ("country", "tagline", "summary"):
        value = out.get(key)
        if isinstance(value, dict):
            out[key] = value.get(lang) or value.get("en") or ""
    for key in ("highlights", "best_for"):
        value = out.get(key)
        if isinstance(value, dict):
            out[key] = value.get(lang) or value.get("en") or []
    out["tag_labels"] = [
        (TAGS.get(tag) or {}).get(lang) or tag.replace("_", " ") for tag in out["tags"]
    ]
    return out


def localized_catalog(language: str) -> list[dict[str, Any]]:
    return [localize(item, language) for item in TRAVEL_ITEMS]


def catalog_for_prompt() -> list[dict[str, Any]]:
    """Compact catalog for the LLM prompt.

    Tags, price and nights are enough for the model to choose and rank.  Long
    descriptions stay on the cards, where they are localized by the UI, which
    keeps the prompt (and therefore the first spoken word) fast.
    """
    return [
        {
            "id": item["id"],
            "name": item["name"],
            "country": item["country"]["en"],
            "region": item["region"],
            "price": item["price"],
            "nights": item["nights"],
            "rating": item["rating"],
            "climate": item["climate"],
            "pace": item["pace"],
            "tags": item["tags"],
        }
        for item in TRAVEL_ITEMS
    ]
