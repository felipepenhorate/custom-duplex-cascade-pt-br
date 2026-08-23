"""M1 (optional) — capability-anchor prompt set for the D2 variant (SPEC §4.3/§6).

A small, curated set of NON-duplex general-capability prompts whose teacher
logits get distilled alongside the duplex sequences during training, so the
LoRA delta is pinned to ≈ 0 on general prompts. This is the single-stage,
concurrent equivalent of the paper's "Re-invoke" and the most direct guard
against the reported function-calling regression.

Categories:
  * function-calling (pt-BR): OpenAI-style tool definition + user request;
    the frozen base model must still answer these exactly as before.
  * general QA / reasoning: pt-BR factual and everyday-reasoning questions.

The set is built from template banks so it can be regenerated with a different
seed/size; an external --input JSONL (e.g. from
../function_calling_is_all_you_need/) can be merged in as-is. Output is the
plain ChatML shape M2 will tokenize + score:

  {"id", "category", "system", "user"}

Usage:
  python data/build_anchor_prompts.py --n-per-category 150 --out data/anchor_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Function-calling bank (pt-BR)
# ---------------------------------------------------------------------------
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "buscar_previsao_do_tempo",
            "description": "Busca a previsão do tempo para uma cidade.",
            "parameters": {
                "type": "object",
                "properties": {"cidade": {"type": "string"}},
                "required": ["cidade"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agendar_compromisso",
            "description": "Agenda um compromisso no calendário do usuário.",
            "parameters": {
                "type": "object",
                "properties": {
                    "titulo": {"type": "string"},
                    "data": {"type": "string", "format": "date"},
                    "hora": {"type": "string", "format": "time"},
                },
                "required": ["titulo", "data", "hora"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_frete",
            "description": "Calcula o frete de uma entrega dado o CEP de origem e destino.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cep_origem": {"type": "string"},
                    "cep_destino": {"type": "string"},
                    "peso_kg": {"type": "number"},
                },
                "required": ["cep_origem", "cep_destino", "peso_kg"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enviar_email",
            "description": "Envia um e-mail para um destinatário.",
            "parameters": {
                "type": "object",
                "properties": {
                    "para": {"type": "string"},
                    "assunto": {"type": "string"},
                    "corpo": {"type": "string"},
                },
                "required": ["para", "assunto", "corpo"],
            },
        },
    },
]

FC_REQUESTS = [
    "Qual vai ser a previsão do tempo em São Paulo amanhã?",
    "Quero saber o tempo em Curitiba hoje.",
    "Me agenda uma reunião com a equipe na próxima terça às 14h.",
    "Preciso agendar uma consulta médica para o dia 15 do próximo mês de manhã.",
    "Quanto custa mandar um pacote de 3kg de Campinas para o Rio?",
    "Calcula o frete do CEP 01310-100 até o CEP 22040-001 com 1.5kg.",
    "Envia um e-mail para o cliente avisando que o pedido saiu para entrega.",
    "Manda um e-mail pra maria@exemplo.com confirmando a reunião de amanhã.",
]

FC_SYSTEM = (
    "Você é um assistente com acesso a ferramentas. Quando o pedido do usuário "
    "corresponder a uma das funções disponíveis, responda apenas com a chamada de "
    "função no formato correto, preenchendo todos os parâmetros obrigatórios."
)

# ---------------------------------------------------------------------------
# General QA / reasoning bank (pt-BR)
# ---------------------------------------------------------------------------
QA_SYSTEM = "Você é um assistente útil em português brasileiro. Responda de forma direta e correta."

QA_QUESTIONS = [
    "Qual é a capital do Brasil e em que região ela fica?",
    "Explique em duas frases o que é inflação.",
    "Se uma receita serve 4 pessoas e eu quero servir 6, por quanto devo multiplicar os ingredientes?",
    "Qual a diferença entre juros simples e juros compostos?",
    "O que significa a expressão 'água mole em pedra dura, tanto bate até que fura'?",
    "Quanto é 15% de 240?",
    "Cite três dicas para economizar energia elétrica em casa.",
    "Qual é a maior floresta tropical do mundo e em quais países ela se estende?",
    "Como se calcula a área de um triângulo?",
    "Explique brevemente o que faz um servidor DNS.",
    "O que é o efeito estufa e por que ele importa para o clima?",
    "Qual time ganhou o campeonato brasileiro de futebol em 2024?",
    "Diferencie hardware e software com um exemplo de cada.",
    "Se hoje é quarta-feira, que dia da semana será daqui a 10 dias?",
    "O que é fotossíntese e qual gás as plantas liberam durante esse processo?",
]


def build_function_calling(rng: random.Random, n: int) -> list[dict]:
    rows = []
    while len(rows) < n:
        tool = rng.choice(TOOL_DEFS)
        req = rng.choice(FC_REQUESTS)
        rows.append(
            {
                "id": f"anchor-fc-{len(rows):04d}",
                "category": "function_calling",
                "tool": tool,
                "system": FC_SYSTEM,
                "user": req,
            }
        )
    return rows


def build_general_qa(rng: random.Random, n: int) -> list[dict]:
    rows = []
    while len(rows) < n:
        q = rng.choice(QA_QUESTIONS)
        rows.append(
            {
                "id": f"anchor-qa-{len(rows):04d}",
                "category": "general_qa",
                "system": QA_SYSTEM,
                "user": q,
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/anchor_prompts.jsonl")
    p.add_argument("--n-per-category", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input", default=None,
                   help="optional external JSONL (id/category/system/user) to merge")
    args = p.parse_args()

    rng = random.Random(args.seed)
    rows = build_function_calling(rng, args.n_per_category)
    rows += build_general_qa(rng, args.n_per_category)

    if args.input:
        in_path = Path(args.input)
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rows.append(
                    {
                        "id": rec.get("id", f"anchor-ext-{len(rows):04d}"),
                        "category": rec.get("category", "external"),
                        "system": rec.get("system", ""),
                        "user": rec["user"],
                        **({k: v for k, v in rec.items() if k not in ("id", "category", "system", "user")}),
                    }
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cats: dict[str, int] = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print(f"[anchors] {len(rows)} prompts written -> {out_path}")
    for k, v in sorted(cats.items()):
        print(f"   {k:18s} {v}")


if __name__ == "__main__":
    main()