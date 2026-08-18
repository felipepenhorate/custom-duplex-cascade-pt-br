"""M1 - pt-BR multi-turn dialogue generator (stage 1 of the data pipeline).

Generates colloquial Brazilian-Portuguese multi-turn dialogues through a
llama.cpp instance exposing an OpenAI-compatible API (the same model stack
that will later serve the fine-tuned duplex model). No local model weights
are loaded: every prompt is sent as a /v1/chat/completions request, with
thinking mode disabled (Qwen3-style reasoning would otherwise consume the
generation budget). Output: JSONL with per-dialogue dedup/validation.

Usage:
  python data/build_dialogues.py --n-dialogues 2000 --out data/dialogues_pt.jsonl
  python data/build_dialogues.py --n-dialogues 50000   # full run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.common import sha256_dedup_key, is_portuguese

# ---------------------------------------------------------------------------
# pt-BR scenario bank: (topic, context) - extends variety for the 50k run.
# ---------------------------------------------------------------------------
SCENARIOS: list[tuple[str, str]] = [
    ("viagem para o Rio de Janeiro", "organizando uma viagem de férias, decidindo roteiro, hospedagem e o que fazer"),
    ("estudar para um concurso público", "conversa sobre rotina de estudos, matérias difíceis e motivação"),
    ("pedir comida por aplicativo", "o usuário quer sugestões de restaurantes e reclama da demora da entrega"),
    ("problemas com o celular", "o celular está travando, sem bateria e com memória cheia"),
    ("planejando um churrasco de domingo", "definindo quem traz o quê, carnes, bebidas e horário"),
    ("aprendendo a cozinhar", "receitas simples, dicas de tempero e erros comuns na cozinha"),
    ("primeiro dia em um emprego novo", "dúvidas sobre a rotina, o time e como se comportar"),
    ("organizando as finanças", "dívidas no cartão, como economizar e montar uma reserva"),
    ("treinar na academia", "montando um treino, alimentação e metas de peso"),
    ("assistir séries brasileiras", "recomendando séries, discutindo personagens e finais"),
    ("cuidar do jardim de casa", "plantas que não vingam, pragas e época de podar"),
    ("adotar um cachorro", "escolhendo a raça, custos, cuidados e adaptação do animal"),
    ("marcar uma consulta médica", "pelo SUS e por plano, sintomas e exames"),
    ("economizar na conta de luz", "dicas de consumo, eletrodomésticos e hábitos"),
    ("planejar uma festa de aniversário", "lista de convidados, bolo, música e decoração"),
    ("trabalho remoto", "vantagens, dificuldades de concentração e rotina em casa"),
    ("fazer a prova do Enem", "estratégia de prova, redação e gestão de tempo"),
    ("pegar a estrada para o interior", "trânsito, paradas, combustível e lanches"),
    ("café da manhã saudável", "opções rápidas, receitas e substituições"),
    ("comprar um carro usado", "marcas, negociação, cautelas e financiamento"),
    ("organizar o guarda-roupa", "doação, o que manter e como dobrar"),
    ("conhecer um lugar novo na cidade", "parques, museus, feiras e restaurantes"),
    ("trancar a faculdade", "prós e contras, alternativas e planos"),
    ("vender doces para renda extra", "receitas, custos, precificação e vendas online"),
    ("mudar de apartamento", "bairro, aluguel, condomínio e mudança"),
    ("dias de chuva no fim de semana", "programas dentro de casa e passeios cobertos"),
    ("aprender inglês sozinho", "métodos, aplicativos, constância e metas"),
    ("problemas com o vizinho barulhento", "como conversar, condomínio e limites"),
    ("montar um home office simples", "mesa, cadeira, iluminação e organização"),
    ("ir ao mercado gastando menos", "listas, ofertas, marcas e desperdício"),
    ("escolher o presente certo", "aniversário do amigo, orçamento e ideias criativas"),
    ("fazer exercícios em casa", "sem equipamento, rotina curta e consistência"),
    ("dieta com pouco tempo", "refeições rápidas, marmita e lanches"),
    ("briga com o irmão", "como se reaproximar e resolver a encrenca"),
    ("fotografia amadora", "câmera do celular, luz, enquadramento e edição"),
    ("jogar futebol no fim de semana", "pelada com amigos, escalação e lesões"),
    ("escolher um curso técnico", "opções, mercado de trabalho e tempo"),
    ("reclamar de atraso no serviço", "telefone, paciência e direitos do consumidor"),
    ("planejar o chá de bebê", "brincadeiras, lembrancinhas e espaço"),
    ("decidir entre Netflix e cinema", "programa a dois, pipoca e estacionamento"),
    ("dor de cabeça constante", "hidratação, sono, telas e quando procurar médico"),
    ("tirar a carteira de motorista", "autoescola, aulas práticas e nervosismo"),
    ("bate-papo sobre futebol", "campeonato brasileiro, times e jogadores"),
    ("pegar dicas de segurança", "aplicativos, horários e cuidados na rua"),
    ("sustentabilidade em casa", "reciclagem, compostagem e consumo consciente"),
    ("passeio com crianças no fim de semana", "parques, lanchonetes e horários"),
    ("investir o primeiro salário", "renda fixa, inflação e objetivos"),
    ("baixar e organizar aplicativos", "limpeza de tela, armazenamento e notificações"),
]

DIALECT_HINT = (
    "Use português brasileiro coloquial, com naturalidade de uma conversa real: "
    "frases curtas, gírias leves, hesitações ocasionais e respostas curtas do "
    "Assistente. Nada de textos longos, listas ou cabeçalhos."
)

_FORMAT_RULES = (
    "Cada fala deve ocupar exatamente uma linha, começando com 'Usuário:' ou "
    "'Assistente:', nesse formato nítido e alternado. Nada além das falas. "
    "Nenhum conteúdo fora desse formato."
)

PROMPT_SYSTEM = (
    "Você é um roteirista de diálogos em português brasileiro. Você cria "
    "conversas realistas de chat entre um Usuário e um Assistente. "
    "Responda direto, sem raciocinar em voz alta."
)

PROMPT_USER = (
    "Crie uma conversa de chat em português brasileiro com exatamente {n_turns} "
    "falas ({n_usr} do Usuário e {n_asst} do Assistente) sobre este assunto: {topic}.\n"
    "Contexto da conversa: {context}.\n"
    "{dialect}\n{format}"
)


def build_prompt(topic: str, context: str, n_turns: int, rng: random.Random) -> list[dict]:
    n_usr = (n_turns + 1) // 2
    n_asst = n_turns - n_usr
    user_msg = PROMPT_USER.format(
        n_turns=n_turns,
        n_usr=n_usr,
        n_asst=n_asst,
        topic=topic,
        context=context,
        dialect=rng.choice(
            [
                DIALECT_HINT,
                DIALECT_HINT
                + " O Usuário fala de forma informal, em tom de bate-papo.",
                DIALECT_HINT + " A conversa deve ter algumas hesitações e perguntas de volta.",
            ]
        ),
        format=_FORMAT_RULES,
    )
    return [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------
_PREFIXES = {"Usuário:", "Usuario:", "Assistente:"}
_ROLE_MAP = {"Usuário:": "user", "Usuario:": "user", "Assistente:": "assistant"}


def parse_dialogue(text: str) -> list[dict] | None:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        for prefix in _PREFIXES:
            if line.startswith(prefix):
                content = line[len(prefix):].strip()
                lines.append({"role": _ROLE_MAP[prefix], "content": content})
                break
    if len(lines) < 6:
        return None
    # strict alternation must hold
    for expected, line in zip(
        ("user", "assistant") * ((len(lines) + 1) // 2), lines
    ):
        if line["role"] != expected:
            return None
    return lines


def validate_dialogue(lines: list[dict]) -> bool:
    if not lines:
        return False
    # a dialogue must end with the assistant: the duplex walker pairs every
    # user micro-turn with a system reply (odd/user-final outputs are rejected)
    if lines[-1]["role"] != "assistant":
        return False
    joined_user = " ".join(x["content"] for x in lines if x["role"] == "user")
    joined_asst = " ".join(x["content"] for x in lines if x["role"] == "assistant")
    for text in (joined_user, joined_asst):
        if not is_portuguese(text):
            return False
    for line in lines:
        w = len(line["content"].split())
        if w < 1 or w > 180:
            return False
    # reject degenerate loops: same user line repeated 3+ times
    counts: dict[str, int] = {}
    for line in lines:
        key = line["content"].lower()
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= 3:
            return False
    return True


# ---------------------------------------------------------------------------
# llama.cpp OpenAI-compatible client
# ---------------------------------------------------------------------------
class LlamaCppClient:
    """Thin wrapper around a llama.cpp server's OpenAI-compatible API.

    Generation runs server-side (Q4_K GGUF, thinking disabled so the whole
    budget goes into the dialogue text). The HTTP layer is plain `requests`.
    """

    def __init__(self, api_base: str, model: str | None = None, timeout: int = 600):
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.model = model or self._detect_model()

    def _detect_model(self) -> str:
        r = requests.get(f"{self.api_base}/models", timeout=30)
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", [])]
        if not ids:
            raise RuntimeError(f"no model served at {self.api_base}/models")
        if len(ids) > 1:
            print(f"[llamacpp] multiple models served, using first: {ids[0]}")
        return ids[0]

    def complete(
        self,
        messages: list[dict],
        temperature: float,
        top_p: float,
        max_tokens: int,
        max_retries: int = 3,
    ) -> str:
        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
            # Qwen3-family: keep all generation budget in the answer text
            "chat_template_kwargs": {"enable_thinking": False},
        }
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                r = requests.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                if r.status_code in (429, 503):  # slot busy: retry with backoff
                    time.sleep(1 + attempt * 2)
                    continue
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                return msg.get("content") or ""
            except requests.RequestException as e:
                last_err = e
                time.sleep(1 + attempt * 2)
        raise RuntimeError(f"llama.cpp API failed after {max_retries} tries: {last_err}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api-base", default="http://127.0.0.1:8080/v1",
                   help="llama.cpp OpenAI-compatible endpoint")
    p.add_argument("--model", default=None,
                   help="served model id (default: first model from /v1/models)")
    p.add_argument("--n-dialogues", type=int, default=50000)
    p.add_argument("--workers", type=int, default=4,
                   help="concurrent requests (needs --parallel slots on the server)")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=900)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/dialogues_pt.jsonl")
    p.add_argument("--max-retries", type=int, default=3, help="retries per row on validation failure")
    args = p.parse_args()

    rng = random.Random(args.seed)
    client = LlamaCppClient(args.api_base, args.model)
    print(f"[dialogues] served model: {client.model}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # resume: count records already written
    written = 0
    seen_keys: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                written += 1
                try:
                    rec = json.loads(line)
                    seen_keys.add(rec["dedup_key"])
                except Exception:
                    pass
    needed = max(0, args.n_dialogues - written)
    print(f"[dialogues] resuming with {written} done; generating {needed} more", flush=True)

    def sample_meta() -> dict:
        topic, context = rng.choice(SCENARIOS)
        return {"topic": topic, "context": context, "n_turns": rng.randint(6, 14)}

    def generate_one(meta: dict) -> dict | None:
        for attempt in range(args.max_retries + 1):
            prompt = build_prompt(meta["topic"], meta["context"], meta["n_turns"], rng)
            text = client.complete(
                prompt, args.temperature, args.top_p, args.max_new_tokens
            )
            lines = parse_dialogue(text)
            if lines is not None and validate_dialogue(lines):
                return lines
            time.sleep(0.5 * attempt)
        return None

    stats: Counter[str] = Counter()
    lock = threading.Lock()
    accepted = 0
    rejected = 0

    def emit(rec: dict | None, meta: dict) -> None:
        nonlocal accepted, rejected
        with lock:
            if rec is None:
                stats["rejected_raw"] += 1
                rejected += 1
                return
            key = sha256_dedup_key(" ".join(x["content"] for x in rec))
            if key in seen_keys:
                stats["rejected_dup"] += 1
                rejected += 1
                return
            seen_keys.add(key)
            accepted += 1
            out = {
                "id": f"pt-dlg-{accepted:07d}",
                "dedup_key": key,
                "scenario": {"topic": meta["topic"], "context": meta["context"]},
                "messages": rec,
                "meta": {
                    "model": client.model,
                    "generator": "llama.cpp-openai",
                    "api_base": client.api_base,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "n_turns": len(rec),
                },
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    start = time.time()
    from concurrent.futures import FIRST_COMPLETED, Future, wait

    inflight: dict[Future, dict] = {}
    attempts = 0

    def launch() -> None:
        nonlocal attempts
        meta = sample_meta()
        inflight[ex.submit(generate_one, meta)] = meta
        attempts += 1

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        with tqdm(total=needed, desc="dialogues", unit="d") as bar:
            exhausted = False
            while accepted < needed and not exhausted:
                while len(inflight) < args.workers:
                    launch()
                done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                for fut in done:
                    meta = inflight.pop(fut)
                    rec = None
                    try:
                        rec = fut.result()
                    except Exception as e:
                        rec = None
                        print(f"[dialogues] worker error: {e}", flush=True)
                    if rec is None:
                        if attempts < needed * 20:
                            launch()  # refill: keep trying until the target is met
                        else:
                            exhausted = True
                            print(f"[dialogues] attempt cap hit at {accepted}/{needed}", flush=True)
                    else:
                        emit(rec, meta)
                bar.n = accepted
                bar.refresh()

    elapsed = time.time() - start
    print(
        f"[dialogues] done: {accepted} accepted, {rejected} rejected in {elapsed/60:.1f} min "
        f"({accepted/max(elapsed, 1e-9)*60:.1f} dialogues/min) -> {out_path}",
        flush=True,
    )
    print(f"[dialogues] breakdown: {dict(stats)}")


if __name__ == "__main__":
    main()