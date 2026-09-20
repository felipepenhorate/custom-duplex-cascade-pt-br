"""Eval the trained policy companion (SPEC 8.1) + controller probes.

Two parts:
  1. TAG-LEVEL EVAL on the held-out duplex eval split: per-tag
     precision/recall/F1 via restricted argmax over the protocol
     vocabulary at boundary positions.
  2. BARGE-IN PROBES (the controller test): hand-built scenarios where the
     assistant is mid-utterance and the user (a) barges in -> must emit
     <|user interruption|>, (b) says a backchannel -> <|user backchannel|>,
     (c) is silent -> no tag (keep talking), (d) finishes a turn ->
     <|user finish speaking|>.

Usage:
  python eval/eval_policy.py --model /mnt/f/duplex_cascade_runs/mtp_like/runs/final/merged
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.common import (
    IM_END,
    IM_END_NL,
    IM_START_ASSISTANT,
    IM_START_USER,
    NO_TAG,
    TOKEN,
)
from training.train_policy import POLICY_WEIGHTS

TAG_VOCAB = ["no_tag"] + list(POLICY_WEIGHTS)


def load_policy(model_dir: str):
    cfg = json.loads((Path(model_dir) / "policy_cfg.json").read_text())
    tok = AutoTokenizer.from_pretrained(model_dir, token=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="cuda", token=False,
    ).eval()
    tag_vocab = cfg["tag_vocab"]  # ["no_tag"] + tag names, decode order
    vocab_ids = [cfg["tag_ids"][name] for name in tag_vocab]
    return model, tok, vocab_ids, cfg, tag_vocab


def predict(model, tok, vocab_ids, tag_vocab, text: str) -> tuple[str, torch.Tensor]:
    ids = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**ids).logits[0, -1, vocab_ids]  # [K]
    probs = torch.softmax(logits.float(), dim=-1)
    idx = int(logits.argmax())
    return tag_vocab[idx], probs


def eval_split(model, tok, vocab_ids, tag_vocab, ds, limit: int = 1015) -> dict:
    tag_ids = vocab_ids
    total = Counter()
    correct = Counter()
    conf = Counter()
    for r in ds.select(range(min(limit, len(ds)))):
        ids = r["input_ids"]
        labels = r["labels"]
        with torch.no_grad():
            logits = model(
                input_ids=torch.tensor([ids], device="cuda")
            ).logits[0]
        sub = logits[:, tag_ids].argmax(dim=-1).cpu().tolist()
        true = [i for i, l in enumerate(labels) if l != -100]
        for pos in true:
            if labels[pos] not in tag_ids:
                continue  # label from a NEWER vocab (e.g. the v6 system
                # tags when evaluating v5) — cannot score it
            gold = tag_vocab[tag_ids.index(labels[pos])]
            pred = tag_vocab[sub[pos]]
            total[gold] += 1
            conf[(gold, pred)] += 1
    n = sum(total.values())
    out = {"n": n}
    for name in tag_vocab:
        tp = conf[(name, name)]
        fp = sum(c for (g, p), c in conf.items() if p == name and g != name)
        fn = sum(c for (g, p), c in conf.items() if g == name and p != name)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out[f"{name}_p"] = round(p, 3)
        out[f"{name}_r"] = round(r, 3)
        out[f"{name}_f1"] = round(f1, 3)
    out["acc"] = round(sum(conf[(g, g)] for g in tag_vocab) / n, 3)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/mnt/f/duplex_cascade_runs/mtp_like/runs/final_v2/merged")
    p.add_argument("--dataset", default="/mnt/f/duplex_cascade_runs/mtp_like/prepped_policy")
    p.add_argument("--limit", type=int, default=1015)
    args = p.parse_args()

    model, tok, vocab_ids, cfg, tag_vocab = load_policy(args.model)
    print(f"[eval] model {args.model} (base {cfg['base_model']})")

    ds = load_from_disk(args.dataset)["eval"]
    print("[eval] tag-level results on the duplex eval split:")
    for k, v in eval_split(model, tok, vocab_ids, tag_vocab, ds, args.limit).items():
        print(f"   {k:24s} {v}")

    # controller probes using the RUNTIME query path (guards + thresholds)
    from policy.duplex_policy import DuplexPolicy
    pol = DuplexPolicy(args.model)
    NOV = TOKEN["no_voice"]

    print("\n[eval] BARGE-IN / controller probes (runtime query path):")

    def probe(desc: str, items: list[dict], expected: str) -> None:
        got, probs = pol.query(items)
        mark = "OK " if got == expected else "MISS"
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        detail = ", ".join(f"{k}={v:.2f}" for k, v in top)
        print(f"   [{mark}] {desc:58s} -> {got:24s} (esperado: {expected:24s}) {detail}")

    # (a) user is speaking (words arriving, mid-question)
    probe("user speaking (mid-question, no punctuation)",
          [{"role": "user", "text": "me explica como funciona o sistema de", "partial": True}],
          "user_is_speaking")

    # (b) user finishes a question (full chunk + silence -> START ANSWER)
    probe("user finished question + silence",
          [{"role": "user", "text": "qual é a capital da Austrália?"},
           {"role": "user", "text": NOV, "partial": True}],
          "user_finish_speaking")

    # (c) assistant mid-utterance + user barges in
    probe("assistant mid-utterance + user barges in",
          [{"role": "user", "text": "me conta sobre o Rio de Janeiro"},
           {"role": "assistant", "text": "Ah, o Rio é incrível! A praia de Copacabana, o Pão de Açúcar e a Lagoa são lindos. Se você", "partial": True},
           {"role": "user", "text": "espera aí, eu quis dizer o interior", "partial": True}],
          "user_interruption")

    # (d) assistant mid-utterance + partial barge-in ('pera')
    probe("assistant mid-utterance + partial barge-in",
          [{"role": "user", "text": "qual a melhor época pra viajar?"},
           {"role": "assistant", "text": "Bom, depende do que você quer. No verão as praias ficam lotadas, mas se você prefere clima tranquilo, o segundo semestre é", "partial": True},
           {"role": "user", "text": "pera", "partial": True}],
          "user_interruption")

    # (e) assistant mid-utterance + backchannel ('aham') -> keep talking
    probe("assistant mid-utterance + backchannel 'aham'",
          [{"role": "user", "text": "você conhece algum restaurante bom?"},
           {"role": "assistant", "text": "Conheço vários! O melhor é o da esquina, que serve uma feijoada maravilhosa, e também tem aquele sushi", "partial": True},
           {"role": "user", "text": "aham", "partial": True}],
          "user_backchannel")

    # (f) assistant mid-utterance + silence -> keep talking (guard test)
    probe("assistant mid-utterance + silence (no voice)",
          [{"role": "user", "text": "como funciona o transporte público aí?"},
           {"role": "assistant", "text": "O metrô é bem eficiente, cobre boa parte da cidade e funciona até meia-noite, mas nos horários de pico ele", "partial": True},
           {"role": "user", "text": NOV, "partial": True}],
          "no_tag")

    # (g) assistant finished its answer + silence -> user thinking
    probe("assistant answered + silence (user thinking)",
          [{"role": "user", "text": "você gosta de café?"},
           {"role": "assistant", "text": "Adoro! Tomo pelo menos três xícaras por dia."}],
          "user_is_thinking")

    # ---- promptable-controller probes (v6): the companion reads the
    # system instruction and fires the system tags ----
    RULE_CPF = (
        "No CPF, apenas números são permitidos. Se o cliente disser uma "
        "letra em vez de um número, interrompa e avise que apenas números "
        "são aceitos."
    )
    RULE_INVEST = (
        "Se o cliente falar sobre investimentos, avise que isso está com o "
        "gerente e que ele será chamado."
    )
    NOV = TOKEN["no_voice"]
    SP = TOKEN["user_is_speaking"]

    def rule_probe(desc: str, rule: str, items: list[dict], expected: str) -> None:
        ctx = []
        if rule:
            ctx.append({"role": "system", "text": rule})
        ctx.extend(items)
        got, probs = pol.query(ctx)
        mark = "OK " if got == expected else "MISS"
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        detail = ", ".join(f"{k}={v:.2f}" for k, v in top)
        print(f"   [{mark}] {desc:58s} -> {got:24s} (esperado: {expected:24s}) {detail}")

    print("\n[eval] PROMPTABLE-CONTROLLER probes (v6):")
    rule_probe("rule CPF + digit chunks + LETTER 'a'",
               RULE_CPF,
               [{"role": "user", "text": "um"},
                {"role": "assistant", "text": SP},
                {"role": "user", "text": "dois"},
                {"role": "assistant", "text": SP},
                {"role": "user", "text": "três"},
                {"role": "assistant", "text": SP},
                {"role": "user", "text": "a", "partial": True}],
               "system_take_floor")
    rule_probe("rule CPF + digit chunks only (no letter)",
               RULE_CPF,
               [{"role": "user", "text": "um"},
                {"role": "assistant", "text": SP},
                {"role": "user", "text": "dois"},
                {"role": "assistant", "text": SP},
                {"role": "user", "text": "três", "partial": True}],
               "user_is_speaking")
    rule_probe("rule CPF + ONE digit + letter (live dictation shape)",
               RULE_CPF,
               [{"role": "user", "text": "um,"},
                {"role": "assistant", "text": SP},
                {"role": "user", "text": "a,", "partial": True}],
               "system_take_floor")
    rule_probe("rule CPF + rich digits + UNSEEN letter 'x' (concept)",
               RULE_CPF,
               [{"role": "user", "text": "zero,"},
                {"role": "assistant", "text": SP},
                {"role": "user", "text": "sete,"},
                {"role": "assistant", "text": SP},
                {"role": "user", "text": "x,", "partial": True}],
               "system_take_floor")
    rule_probe("rule CPF + rich digits only (no letter, concept)",
               RULE_CPF,
               [{"role": "user", "text": "zero,"},
                {"role": "assistant", "text": SP},
                {"role": "user", "text": "nove,"},
                {"role": "assistant", "text": SP},
                {"role": "user", "text": "vinte,", "partial": True}],
               "user_is_speaking")
    rule_probe("rule CPF + digit chunks + NON-letter word 'meu'",
               RULE_CPF,
               [{"role": "user", "text": "um,"},
                {"role": "assistant", "text": SP},
                {"role": "user", "text": "meu,", "partial": True}],
               "user_is_speaking")
    rule_probe("NO rule + letter 'a' (must NOT fire system tag)",
               "",
               [{"role": "user", "text": "um"},
                {"role": "assistant", "text": SP},
                {"role": "user", "text": "a", "partial": True}],
               "user_is_speaking")
    rule_probe("handover rule + 'investimentos'",
               RULE_INVEST,
               [{"role": "user", "text": "quero saber sobre"},
                {"role": "assistant", "text": SP},
                {"role": "user", "text": "investimentos", "partial": True}],
               "system_handover")
    rule_probe("handover rule + unrelated words (no fire)",
               RULE_INVEST,
               [{"role": "user", "text": "quero saber sobre"},
                {"role": "assistant", "text": SP},
                {"role": "user", "text": "o cartão", "partial": True}],
               "user_is_speaking")
    rule_probe("NO rule + 'investimentos' (must NOT fire)",
               "",
               [{"role": "user", "text": "quero saber sobre"},
                {"role": "assistant", "text": SP},
                {"role": "user", "text": "investimentos", "partial": True}],
               "user_is_speaking")


if __name__ == "__main__":
    main()