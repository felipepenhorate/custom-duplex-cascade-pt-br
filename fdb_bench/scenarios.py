"""pt-BR + en Full-Duplex-Bench scenarios (text domain).

Reproduces the *structure* of Full-Duplex-Bench v1.0's five subsets, at the
published sample counts, in two languages:

    ==========================  ======  =======================================
    subset                      FDB n   gold behaviour
    ==========================  ======  =======================================
    pause_candor                   216  stay silent through a natural pause
    pause_synthetic                137  stay silent through an inserted pause
    smooth_turn                    119  take the floor right after the turn ends
    backchannel_icc                 55  short backchannels, never a formal turn
    interruption_synthetic         200  yield, then answer the *interrupting* turn
    ==========================  ======  =======================================

Languages
---------
``pt`` is the in-distribution language of the three trained variants. ``en`` is
out-of-distribution for all of them (they were fine-tuned on pt-BR duplex data
only) and measures cross-lingual generalisation of the turn-taking protocol.
The protocol itself is language-neutral — the special tokens are English
(``<|user finish speaking|>``) and only the *content* changes — so the same
adapters drive both runs with a different system prompt and corpus.

``en`` additionally matches FDB's own language, so the interruption judge runs
FDB's verbatim rubric (see ``judge.py``) instead of a translation.

Design notes
------------
*Every* user-side event carries ``ends_turn``, mirroring what the real pipeline
knows from ASR/VAD. The **decision** the benchmark measures is whether the
model emits ``<|user finish speaking|>`` (takes the floor) at the right moment
— exactly the label the training data assigns at a user-turn boundary vs.
mid-turn (``build_duplex_dataset.py::user_phase``).

Pause scenarios are *terminal*: the user pauses and the stream ends, mirroring
FDB, where the pause is the last thing in the input audio. If the user resumed
and completed the turn, a correct model would legitimately speak and FDB's
single binary TOR could not tell "spoke too early" from "spoke correctly".
Recovery behaviour is covered by ``smooth_turn`` instead.

The backchannel GT distribution is the distribution of the *user's* sentence
endpoints: listeners backchannel where the speaker completes a phrase, which is
the structure FDB's ``icc_gt_distribution.json`` captures from real ICC
recordings. Emitted per scenario so the JSD metric has a real reference.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .chunking import ends_sentence, word_chunker

# FDB v1.0 published subset sizes.
SUBSET_SIZES = {
    "pause_candor": 216,
    "pause_synthetic": 137,
    "smooth_turn": 119,
    "backchannel_icc": 55,
    "interruption_synthetic": 200,
}

TASKS = tuple(SUBSET_SIZES)
LANGS = ("pt", "en")


# ---------------------------------------------------------- language packs --

@dataclass
class LangPack:
    """All natural-language material a scenario needs, for one language."""

    topics: list[tuple[str, list[str]]]
    user_backchannels: list[str]
    system_backchannchs: list[str]
    hesitations: list[str]
    interrupt_templates: list[str]
    followup_q: list[str]
    ask_opinion: str
    context_bridges: list[str]


PT = LangPack(
    topics=[
        ("cozinha", [
            "Comecei a aprender a cozinhar faz umas três semanas",
            "Comecei com coisas bem simples, tipo omelete e arroz",
            "O segredo do arroz é a proporção de água, uma e meia pra cada xícara",
            "Descobri que a carne precisa descansar antes de cortar",
            "Meu maior erro era ligar o fogo alto demais",
            "Peguei uma panela de ferro e mudou completamente o sabor",
            "Aprendi a temperar com alho, cebola e um pouco de páprica",
            "No domingo fiz um ensopado que durou a semana inteira",
        ]),
        ("viagem", [
            "Tô planejando uma viagem pra praia no fim do ano",
            "Já olhei as passagens e o preço tá bem salgado",
            "A melhor época pra ir é maio, porque chove menos",
            "Ficar numa pousada perto do centro sai bem mais barato",
            "Separei uma mala só com o essencial pra não pagar excesso",
            "Achei um hostel com nota alta e muitos comentários",
            "Quero evitar fins de semana porque fica tudo cheio",
            "Vou levar um caderninho pra anotar os lugares",
        ]),
        ("tecnologia", [
            "Meu notebook tá muito lento desde que instalei uma atualização nova",
            "Acontece bem quando abro vários programas ao mesmo tempo",
            "Tentei limpar os arquivos temporários e ajudou um pouco",
            "O problema maior é que a bateria acaba em duas horas",
            "Pensei em trocar, mas os outros modelos são caríssimos",
            "Um técnico falou que pode ser o cooler parado",
            "Vou levar numa assistência pra fazer um diagnóstico",
            "Quero aprender a usar melhor os atalhos de teclado",
        ]),
        ("saúde", [
            "Tenho sentado o dia inteiro por causa do trabalho remoto",
            "Minhas costas doem no fim do dia, principalmente a lombar",
            "Comecei a caminhar trinta minutos depois do almoço",
            "O médico disse que eu precisava alongar mais a musculatura",
            "Também falou que eu deveria dormir em outra posição",
            "Bebi mais água e a dor de cabeça diminuiu bastante",
            "Tentei um aplicativo de alongamento e sigo os exercícios",
            "Quero manter essa rotina pelo menos três vezes por semana",
        ]),
        ("finanças", [
            "Tô tentando organizar minhas contas do mês",
            "Fiz uma planilha pra ver pra onde vai o dinheiro",
            "O maior gasto é comida, principalmente delivery",
            "Resolvi cancelar dois serviços que eu não usava",
            "Coloquei uma parte da sobra num fundo de emergência",
            "Aprendi que guardar é tão importante quanto ganhar",
            "Ainda tenho umas parcelas de um empréstimo em aberto",
            "Quero quitar a dívida do cartão antes do fim do ano",
        ]),
        ("educação", [
            "Tô estudando de novo, depois de uns anos parado",
            "Escolhi programação porque parece que tem bastante vaga",
            "Começo o dia fazendo um exercício pequeno, nem que seja meia hora",
            "Travei em entender ponteiros, mas um exercício resolveu",
            "Gosto mais da parte prática do que da teoria",
            "Já fiz dois projetinhos pra colocar no portfólio",
            "Um colega me indicou um curso bem feito",
            "Meu plano é conseguir o primeiro emprego até o fim do ano",
        ]),
        ("esporte", [
            "Voltei a correr depois de uns dois anos parado",
            "Comecei devagar, andando mais do que correndo",
            "Marquei três corridas por semana pra não exagerar",
            "O joelho reclamou no começo, aí reduzi o ritmo",
            "Um técnico recomendou trocar o tênis, que já estava velho",
            "Já consigo correr cinco quilômetros sem parar",
            "Gosto de correr cedo, o ar da manhã tá bom",
            "No fim de semana compenso com uma caminhada mais longa",
        ]),
        ("jardinagem", [
            "Comecei uma horta pequena no quintal",
            "Plantei alface, tomate e manjericão",
            "Na primeira tentativa apareceram pulgões e perdi quase tudo",
            "Aprendi a não regar demais, que era o meu erro",
            "Agora uso adubo e as plantas estão bem mais fortes",
            "Um vizinho me deu mudas de hortelã",
            "Colher o que eu mesmo plantei muda o meu dia",
            "Quero montar uma estufa pequena para o inverno",
        ]),
        ("cinema", [
            "Tô procurando um filme bom pra ver no fim de semana",
            "Gosto de suspense, mas não dos muito longos",
            "Um filme curto e contemplativo sobre a cidade me prendeu",
            "Vi uma indicação bem estranha no começo",
            "Fui no cinema com um amigo que não conhecia a história",
            "A trilha sonora fez metade do trabalho",
            "No final tudo se encaixa de um jeito inesperado",
            "Esse tipo de filme me faz querer sair pensando",
        ]),
        ("música", [
            "Tô montando uma playlist pra trabalhar de casa",
            "Gosto de misturar música instrumental com as que têm letra",
            "Descobri um canal que faz versões de músicas antigas",
            "Quando estou cansado, toco mais o mesmo disco o dia todo",
            "Aprendi a ouvir álbuns inteiros, sem pular faixa nenhuma",
            "Uma indicação minha é a banda que abriu o show ontem",
            "É o tipo de música que combina com tarefa mecânica",
            "Ajuda bastante a manter o ritmo na parte chata do dia",
        ]),
        ("animais", [
            "Adotei um cachorro filhote no mês passado",
            "Ele chegou pequeno e hoje já puxa muito na guia",
            "O cachorro precisa de muito exercício por dia",
            "Comecei com ração e agora misturo um pouco de caseiro",
            "Ele faz três caminhadas curtas ao longo do dia",
            "Aprendi que ele entende bem o tom da voz",
            "De madrugada ele acorda só pra olhar pela janela",
            "Cada dia ele está menos medroso com as visitas",
        ]),
        ("rotina", [
            "Todo domingo eu preparo as refeições da semana",
            "Isso me poupa tempo e ainda economiza no fim do mês",
            "Faço um molho de tomate bem grosso pra congelar",
            "Uso o que sobrou do jantar no almoço do dia seguinte",
            "Só faço marmita de vidro que vai pro micro-ondas",
            "O segredo é não deixar comida crua na geladeira",
            "Minha cozinha é pequena, então organizo tudo em prateleiras",
            "Com o tempo essa rotina virou natural",
        ]),
    ],
    user_backchannels=[
        "ah", "entendi", "tá", "aham", "uhum", "é", "claro", "né", "pois é",
        "ah tá", "então", "hmm", "sim", "isso", "beleza", "verdade",
    ],
    system_backchannchs=["aham", "entendi", "tá", "uhum", "ah", "claro", "né"],
    hesitations=[
        "é... hum", "ah...", "quer dizer...", "enfim", "ah, peraí",
        "é... deixa eu ver", "hmm", "ah, isso é", "né, enfim",
    ],
    interrupt_templates=[
        "Espera, antes disso — {q}",
        "Ah, mas deixa eu te perguntar outra coisa: {q}",
        "Perá, mudando de assunto rapidinho — {q}",
        "Ei, pausa aí, eu queria te perguntar {q}",
        "Antes que eu esqueça, outra dúvida: {q}",
        "Ops, me interrompe aí — {q}",
    ],
    followup_q=[
        "quanto tempo leva pra dominar o básico",
        "qual é o primeiro passo que eu deveria fazer",
        "existe alguma versão gratuita disso",
        "como eu sei se tô fazendo certo",
        "vale a pena começar por isso ou por outra coisa",
        "o que eu devo evitar no começo",
        "qual material você indicaria pra um iniciante",
        "quanto isso costuma custar por mês",
        "isso funciona bem no celular também",
        "tem como eu acompanhar o progresso por algum aplicativo",
    ],
    ask_opinion=", o que você acha?",
    context_bridges=[],
)

EN = LangPack(
    topics=[
        ("cooking", [
            "I started learning to cook about three weeks ago",
            "I began with really simple things, like omelets and rice",
            "The secret to rice is the water ratio, one and a half cups per cup",
            "I found out the meat needs to rest before you slice it",
            "My biggest mistake was turning the heat way too high",
            "I got a cast iron pan and it changed the flavor completely",
            "I learned to season with garlic, onion and a little paprika",
            "On Sunday I made a stew that lasted the whole week",
        ]),
        ("travel", [
            "I'm planning a beach trip for the end of the year",
            "I already looked at the flights and the price is pretty steep",
            "The best time to go is May, because it rains less",
            "Staying at a guesthouse near downtown is much cheaper",
            "I packed a bag with only the essentials to avoid baggage fees",
            "I found a hostel with a high rating and lots of reviews",
            "I want to avoid weekends because everything gets crowded",
            "I'm bringing a little notebook to write down places",
        ]),
        ("technology", [
            "My laptop has been really slow since I installed a new update",
            "It happens mostly when I open several programs at once",
            "I tried clearing the temporary files and it helped a little",
            "The bigger problem is the battery only lasts two hours",
            "I thought about replacing it, but the other models are pricey",
            "A technician said it might be the fan",
            "I'm going to take it in for a proper diagnostic",
            "I want to get better at using keyboard shortcuts",
        ]),
        ("health", [
            "I've been sitting all day because of remote work",
            "My back hurts at the end of the day, mostly the lower back",
            "I started walking thirty minutes after lunch",
            "The doctor said I need to stretch my muscles more",
            "He also said I should try a different sleeping position",
            "Drinking more water really helped my headaches",
            "I tried a stretching app and I follow the exercises",
            "I want to keep this routine at least three times a week",
        ]),
        ("finances", [
            "I'm trying to organize my monthly bills",
            "I made a spreadsheet to see where the money actually goes",
            "The biggest expense is food, mostly delivery",
            "I cancelled two subscriptions I never used",
            "I put part of the savings into an emergency fund",
            "I learned that saving matters as much as earning",
            "I still have some installments left on a loan",
            "I want to pay off the card debt before the year ends",
        ]),
        ("education", [
            "I'm studying again, after a few years of not doing anything",
            "I picked programming because it seems to have a lot of openings",
            "I start the day with one small exercise, even if it's half an hour",
            "I got stuck on pointers, but one exercise cleared it up",
            "I like the practical part more than the theory",
            "I've already built two small projects for my portfolio",
            "A colleague recommended a really well made course",
            "My plan is to land a first job by the end of the year",
        ]),
        ("sport", [
            "I started running again after about two years off",
            "I began slowly, walking more than running",
            "I scheduled three runs a week so I wouldn't overdo it",
            "My knee complained at first, so I slowed down",
            "A coach recommended new shoes, the old ones were worn out",
            "I can run five kilometers without stopping now",
            "I like running early, the morning air is nice",
            "On weekends I make up for it with a longer walk",
        ]),
        ("gardening", [
            "I started a small vegetable garden in the backyard",
            "I planted lettuce, tomato and basil",
            "On the first try aphids showed up and I lost almost everything",
            "I learned not to overwater, that was my mistake",
            "Now I use fertilizer and the plants are much stronger",
            "A neighbor gave me some mint seedlings",
            "Harvesting what I planted myself changes my whole day",
            "I want to build a small greenhouse for winter",
        ]),
        ("cinema", [
            "I'm looking for a good movie to watch this weekend",
            "I like thrillers, but not the really long ones",
            "A short, quiet film about the city really held me",
            "I watched a recommendation that seemed odd at first",
            "I went to the cinema with a friend who didn't know the story",
            "The soundtrack did half the work",
            "In the end everything fits together in an unexpected way",
            "That kind of film makes me want to walk out thinking",
        ]),
        ("music", [
            "I'm putting together a playlist for working from home",
            "I like mixing instrumental music with songs that have lyrics",
            "I found a channel that does versions of older songs",
            "When I'm tired I replay the same album all day",
            "I learned to listen to whole albums without skipping tracks",
            "One recommendation is the band that opened the show last night",
            "It's the kind of music that goes well with a repetitive task",
            "It really helps me keep the pace through the boring part of the day",
        ]),
        ("pets", [
            "I adopted a puppy last month",
            "He arrived tiny and now he already pulls hard on the leash",
            "The dog needs a lot of exercise during the day",
            "I started with kibble and now mix in some homemade food",
            "He takes three short walks spread over the day",
            "I learned that he understands the tone of your voice",
            "In the middle of the night he wakes up just to look out the window",
            "Every day he's less afraid of visitors",
        ]),
        ("routine", [
            "Every Sunday I prepare the meals for the week",
            "That saves me time and still cuts the cost at month's end",
            "I make a thick tomato sauce to freeze",
            "I reuse what is left from dinner as the next day's lunch",
            "I only use glass containers that go in the microwave",
            "The trick is not leaving raw food in the fridge",
            "My kitchen is small, so I organize everything on shelves",
            "Over time this routine became second nature",
        ]),
    ],
    user_backchannels=[
        "uh-huh", "I see", "okay", "mm-hmm", "uh huh", "right", "sure", "yeah",
        "true", "okay then", "so", "hmm", "yes", "exactly", "fair", "alright",
    ],
    system_backchannchs=["mm-hmm", "I see", "okay", "uh-huh", "ah", "right", "yeah"],
    hesitations=[
        "um... er", "uh...", "I mean...", "anyway", "ah, hold on",
        "er... let me see", "hmm", "ah, that's", "you know, anyway",
    ],
    interrupt_templates=[
        "Wait, before that — {q}?",
        "Actually, let me ask you something else: {q}?",
        "Hold on, changing the subject quickly — {q}?",
        "Hey, pause there, I wanted to ask you {q}?",
        "Before I forget, another question: {q}?",
        "Oops, interrupt me — {q}?",
    ],
    followup_q=[
        "how long does it take to master the basics",
        "what's the first step I should take",
        "is there a free version of that",
        "how do I know if I'm doing it right",
        "is it worth starting with that or something else",
        "what should I avoid at the beginning",
        "what material would you recommend to a beginner",
        "how much does that usually cost per month",
        "does that work well on a phone too",
        "can I track my progress with some app",
    ],
    ask_opinion=", what do you think?",
    context_bridges=[],
)

LANG_PACKS: dict[str, LangPack] = {"pt": PT, "en": EN}


# ------------------------------------------------------------- dataclasses --

@dataclass
class UserEvent:
    """One user-side event on the simulated stream."""

    kind: str                      # "chunk" | "silence"
    text: str = ""
    ends_turn: bool = False        # ASR/VAD finalized the user turn here
    is_backchannel: bool = False   # the user produced a short backchannel
    ends_sentence: bool = False    # this piece ends a sentence (backchannel GT)
    # silence that follows a completed user turn: taking the floor here is
    # CORRECT (the user has stopped speaking), unlike a mid-turn pause. The
    # upstream release specifically needs this cue -- its server keeps firing
    # `<|no voice|>` ticks after speech and its model answers on one.
    after_turn_end: bool = False
    duration: float | None = None  # silence duration (silence events)


@dataclass
class Scenario:
    """A full FDB scenario: a user-side script plus the FDB annotations."""

    sid: str
    task: str
    lang: str = "pt"
    events: list[UserEvent] = field(default_factory=list)
    # interruption task
    context: str = ""          # the user turn before the interruption
    interrupt: str = ""        # the interrupting user turn
    interrupt_after_model_words: int = 0
    # backchannel task
    gt_backchannel_times: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- helpers ---

def _chunk(text: str, rng: random.Random,
           chunker: Callable[[str, random.Random], list[str]] | None = None) -> list[str]:
    """Turn one user utterance into the ASR-like stream the models were trained on.

    ``chunker`` defaults to the tokenizer-free approximation; real runs pass
    :class:`fdb_bench.chunking.TrainingChunker`, which reproduces the training
    stream exactly (1-7-token pieces, cut mid-sentence and mid-word). See
    ``chunking.py`` for why that matters: feeding whole sentences instead was
    the bench's largest fidelity bug.
    """
    fn = chunker or word_chunker
    pieces = [c for c in fn(text, rng) if c.strip()]
    return pieces or [text]


def _join_events(chunks: list[str], ends_turn: bool) -> list[UserEvent]:
    evs = [UserEvent(kind="chunk", text=c) for c in chunks]
    if evs:
        evs[-1].ends_turn = ends_turn
    return evs


# Measured over 3000 training records (see FINDINGS.md): a user turn is
# `mean 10.48 words`, `median 10`, `p90 14`, delivered as ~3.7 ASR chunks and
# containing `mean 1.51` sentence-final chunks. The turn end is then the LAST of
# those sentence-final chunks, and nothing in the text says which one it is —
# so this ratio is the task's built-in ambiguity, and it must be reproduced or
# the turn-end decision becomes artificially hard (or artificially easy).
TRAINING_TURN_WORDS = 10.48
TRAINING_SENT_FINAL_PER_TURN = 1.51


def _pick_topic(rng: random.Random, L: LangPack) -> tuple[str, list[str]]:
    return rng.choice(L.topics)


def _lang_of(L: LangPack) -> str:
    for code, pack in LANG_PACKS.items():
        if pack is L:
            return code
    raise KeyError("unknown LangPack instance")


def _training_sized_turn(rng: random.Random, clauses: list[str]) -> str:
    """Compose a user turn whose length matches the training distribution.

    Training turns are ~10.5 words with ~1.5 sentence-final chunks. The original
    generator joined 2-4 of the topic clauses, producing turns of 35-71 words
    with 3-6.6 sentence-final chunks — 3-7x the training length and 2-4x the
    ambiguity. Because sentence-final punctuation is the only reliable text cue
    for a turn end, every extra sentence-final chunk is an extra chance to fire
    `<|user finish speaking|>` early, which inflated pause/backchannel TOR
    (wanted: no fire) while inflating smooth/interruption TOR (wanted: fire).

    So: one clause, or two when both are short, targeting ~7-14 words.
    """
    order = list(clauses)
    rng.shuffle(order)
    turn = order[0].rstrip(".")
    if len(turn.split()) < 8 and len(order) > 1:
        second = order[1].rstrip(".")
        if len(turn.split()) + len(second.split()) <= 18:
            turn = f"{turn}. {second}"
    return turn + "."


def _long_turn(rng: random.Random, clauses: list[str]) -> str:
    """The other ablation arm: 2--4 sentences joined into one user turn.

    This is what the harness did before the turn-length bug was found: 35--71
    word turns carrying 3--7 sentence-final pieces, i.e. 2--4x the ambiguity of
    the real data. Kept as an explicit arm so the effect can be measured rather
    than argued about.
    """
    return _utterance(rng, clauses, rng.randint(2, 4))


def _turn(rng: random.Random, clauses: list[str], style: str = "training") -> str:
    return _long_turn(rng, clauses) if style == "long" else _training_sized_turn(rng, clauses)


def _utterance(
    rng: random.Random, clauses: list[str], n: int | None = None, close_last: bool = True,
) -> str:
    """Compose ``n`` distinct clauses into one utterance.

    Every clause but the last gets sentence-final punctuation (so clause-aware
    chunking can find boundaries); ``close_last=False`` leaves the trailing
    clause open, which is what a mid-utterance pause sounds like.
    """
    n = n or rng.randint(2, 3)
    picked = rng.sample(clauses, min(n, len(clauses)))
    seen: list[str] = []
    for c in picked:
        if c not in seen:
            seen.append(c)
    if not seen:
        return ""
    out = ". ".join(seen)
    if close_last:
        out += "."
    return out


# ------------------------------------------------------- subset generators --

def _gen_smooth_turn(sid: str, rng: random.Random, L: LangPack, chunker=None, style="training") -> Scenario:
    """A complete user turn; the model must take the floor."""
    topic, clauses = _pick_topic(rng, L)
    turn = _turn(rng, clauses, style).rstrip(".") + L.ask_opinion
    events = _join_events(_chunk(turn, rng, chunker), ends_turn=True)
    # Trailing silence: real audio never stops the instant the user's last word
    # does, and the upstream sbintuitions release relies on it — its server keeps
    # firing `<|no voice|>` ticks after the speech ends, and that is the cue its
    # model was trained to answer on. Without it the original never takes the
    # floor at all. The pt-BR fork fires on the last chunk instead, so this
    # costs those models nothing.
    events += [UserEvent(kind="silence", duration=rng.uniform(1.0, 1.5),
                         after_turn_end=True)]
    return Scenario(
        sid=sid, task="smooth_turn", lang=_lang_of(L),
        events=events, meta={"topic": topic, "turn": turn},
    )


def _gen_pause_synthetic(sid: str, rng: random.Random, L: LangPack, chunker=None, style="training") -> Scenario:
    """An inserted pause mid-sentence; the stream then ends."""
    topic, clauses = _pick_topic(rng, L)
    # last clause left open: the turn is genuinely unfinished when the pause hits
    turn = _turn(rng, clauses, style).rstrip(".")
    chunks = _chunk(turn, rng, chunker)
    # the pause lands mid-turn: with ASR-sized pieces the first piece is 1-2
    # words, so split around the middle of the turn instead
    split = max(1, min(len(chunks) - 1, int(round(len(chunks) * rng.uniform(0.35, 0.6)))))
    pause_s = rng.uniform(1.2, 4.0)
    # TERMINAL pause: the user goes quiet and the stream ends, mirroring FDB,
    # where the pause is the last thing in the input audio.
    #
    # An earlier version let the user RESUME after the pause (`chunks[split:]`
    # then another silence). That broke the measurement: speech the model
    # produced on the resumed content also landed after the pause onset, so the
    # metric could not tell "took the floor too early" (a failure) from "took the
    # floor once the user carried on" (correct). It inflated pause TOR from
    # ~0.30 to ~0.87 for both variants.
    events = (
        _join_events(chunks[:split], ends_turn=False)
        + [UserEvent(kind="silence", duration=pause_s + rng.uniform(2.0, 4.0))]
    )
    return Scenario(
        sid=sid, task="pause_synthetic", lang=_lang_of(L), events=events,
        meta={
            "topic": topic,
            # only what the user actually says: the pause is terminal, so the
            # rest of the utterance never reaches the model and is recorded
            # separately so the "everything is emitted" invariant stays true
            "turn": " ".join(chunks[:split]),
            "unspoken_tail": " ".join(chunks[split:]),
            "pause_s": round(pause_s, 2),
        },
    )


def _gen_pause_candor(sid: str, rng: random.Random, L: LangPack, chunker=None, style="training") -> Scenario:
    """A natural hesitation: the utterance trails off with a filler."""
    topic, clauses = _pick_topic(rng, L)
    # the trailing clause is left open — candor trailing off mid-thought
    turn = _turn(rng, clauses, style).rstrip(".")
    chunks = _chunk(turn, rng, chunker)
    split_at = max(1, min(len(chunks) - 1,
                          int(round(len(chunks) * rng.uniform(0.4, 0.7)))))
    hes = rng.choice(L.hesitations)
    pause_s = rng.uniform(1.8, 6.0)
    # candor: a natural hesitation, then terminal silence. The filler is spoken
    # BEFORE the pause (that is what a hesitation is); the earlier version put it
    # after, which made the "pause" a resumption and broke the metric the same
    # way the synthetic task was broken.
    events = (
        _join_events(chunks[:split_at], ends_turn=False)
        + [UserEvent(kind="chunk", text=hes)]
        + [UserEvent(kind="silence", duration=pause_s + rng.uniform(2.0, 4.0))]
    )
    spoken = " ".join(chunks[:split_at] + [hes])
    return Scenario(
        sid=sid, task="pause_candor", lang=_lang_of(L), events=events,
        meta={
            "topic": topic,
            # what the user actually says before trailing off. ``unspoken_tail``
            # is the rest of the thought they never get to — it is deliberately
            # absent from the stream (the pause is terminal), so it is kept out
            # of ``turn`` to keep the "every word is emitted" invariant true for
            # every task.
            "turn": spoken,
            "unspoken_tail": " ".join(chunks[split_at:]),  # never spoken
            "pause_s": round(pause_s, 2),
            "hesitation": hes,
        },
    )


def _gen_backchannel_icc(sid: str, rng: random.Random, L: LangPack, chunker=None, style="training") -> Scenario:
    """A monologue with sentence endpoints; the user never yields the floor."""
    topic, clauses = _pick_topic(rng, L)
    picked = rng.sample(clauses, rng.randint(2, min(4, len(clauses))))
    picked = [c.rstrip(".") + "." for c in picked]   # each clause is a full sentence
    events: list[UserEvent] = []
    gt_event_idx: list[int] = []
    for clause in picked:
        for c in _chunk(clause, rng, chunker):
            events.append(UserEvent(kind="chunk", text=c, ends_turn=False,
                                    ends_sentence=ends_sentence(c)))
        # the listener-backchannel target is the moment the speaker *finishes a
        # sentence*. With ASR-sized pieces the sentence end lands inside some
        # piece, so find it by content rather than assuming clause boundaries.
        sent_end = max((i for i, e in enumerate(events) if e.ends_sentence),
                       default=len(events) - 1)
        gt_event_idx.append(sent_end)
        if rng.random() < 0.55:
            events.append(UserEvent(
                kind="chunk", text=rng.choice(L.user_backchannels),
                is_backchannel=True,
            ))
    events.append(UserEvent(kind="silence", duration=rng.uniform(1.5, 3.0)))
    return Scenario(
        sid=sid, task="backchannel_icc", lang=_lang_of(L), events=events,
        gt_backchannel_times=gt_event_idx,   # converted to seconds by the clock
        meta={"topic": topic, "n_clauses": len(picked)},
    )


def _gen_interruption(sid: str, rng: random.Random, L: LangPack, chunker=None, style="training") -> Scenario:
    """The user asks, the model answers, the user barges in."""
    topic, clauses = _pick_topic(rng, L)
    context = _turn(rng, clauses, style)
    context_q = context.rstrip(".!?") + L.ask_opinion
    interrupt_q = rng.choice(L.followup_q)
    interrupt = rng.choice(L.interrupt_templates).format(q=interrupt_q)
    # the interrupting utterance must be multi-chunk: with a single chunk the
    # duplex stream would produce two consecutive assistant messages, which
    # build_duplex_dataset.py explicitly refuses to emit.
    interrupt = interrupt.rstrip("?.!") + "?"

    events = _join_events(_chunk(context_q, rng, chunker), ends_turn=True)
    # the model answers the context question, so it needs the same trailing
    # silence cue before the interrupting turn arrives (see _gen_smooth_turn)
    events += [UserEvent(kind="silence", duration=rng.uniform(1.0, 1.5),
                         after_turn_end=True)]
    events += _join_events(_chunk(interrupt, rng, chunker), ends_turn=True)
    events += [UserEvent(kind="silence", duration=rng.uniform(1.0, 1.5),
                         after_turn_end=True)]
    return Scenario(
        sid=sid, task="interruption_synthetic", lang=_lang_of(L), events=events,
        context=context_q, interrupt=interrupt,
        interrupt_after_model_words=rng.randint(8, 40),
        meta={"topic": topic},
    )


_GENERATORS = {
    "smooth_turn": _gen_smooth_turn,
    "pause_synthetic": _gen_pause_synthetic,
    "pause_candor": _gen_pause_candor,
    "backchannel_icc": _gen_backchannel_icc,
    "interruption_synthetic": _gen_interruption,
}

def build_scenarios(
    task: str, limit: int | None = None, seed: int = 1234, lang: str = "pt",
    chunker: Callable[[str, random.Random], list[str]] | None = None,
    turn_style: str = "training",
) -> list[Scenario]:
    """Build ``limit`` scenarios for ``task`` (default: the FDB v1.0 size).

    ``turn_style`` is an ablation axis: ``training`` matches the measured
    distribution, ``long`` reproduces the multi-sentence turns that the harness
    used before the turn-length bug was found (see FINDINGS.md).
    """
    if task not in _GENERATORS:
        raise KeyError(f"unknown task {task!r}; expected one of {TASKS}")
    if lang not in LANG_PACKS:
        raise KeyError(f"unknown lang {lang!r}; expected one of {LANGS}")
    L = LANG_PACKS[lang]
    n = SUBSET_SIZES[task] if limit is None else min(limit, SUBSET_SIZES[task])
    # per-task seed offset keeps each subset independent yet reproducible
    rng = random.Random(seed + TASKS.index(task) * 7919 + LANGS.index(lang) * 104729)
    return [_GENERATORS[task](f"{task}_{i:04d}", rng, L, chunker, turn_style)
            for i in range(n)]


def build_all(
    limit: int | None = None, seed: int = 1234, lang: str = "pt",
    chunker: Callable[[str, random.Random], list[str]] | None = None,
    turn_style: str = "training",
) -> list[Scenario]:
    out: list[Scenario] = []
    for task in TASKS:
        out.extend(build_scenarios(task, limit, seed, lang, chunker, turn_style))
    return out
