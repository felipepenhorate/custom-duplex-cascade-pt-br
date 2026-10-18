"""Which language did the model actually answer in?

The cross-lingual runs feed the pt-BR models English input. The benchmark's
relevance column then depends on the *reply* being in English too, and the
audio run has an extra constraint the text run does not: the scorer's ASR
(parakeet-tdt-0.6b-v2) is English-only, so a Portuguese reply reaches the judge
as a transliteration-like English transcript. This module measures how often that
happens, so the result is reported rather than inferred from a hand-picked
example.

Classification is deliberately dependency-free and conservative: strongly
discriminative function words plus Portuguese diacritics, with a margin rule and
an ``unclear`` bucket for short or ambiguous turns. Checked by hand against the
turns it labels.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

PT_WORDS = {
    "que", "não", "nao", "você", "voce", "uma", "com", "para", "isso", "está",
    "esta", "muito", "bem", "também", "tambem", "então", "entao", "eu", "acho",
    "aqui", "tudo", "mas", "por", "sobre", "quando", "obrigado", "obrigada",
    "sim", "não", "ser", "são", "sao", "foi", "tem", "vai", "pode", "como",
    "mais", "muito", "de", "da", "do", "em", "um", "se", "os", "as", "ao",
    "dos", "das", "pelo", "pela", "isso", "aquilo", "essa", "esse", "este",
    "esta", "nosso", "nossa", "meu", "minha", "seu", "sua", "estou", "estava",
    "quero", "preciso", "gosto", "vamos", "entendi", "legal", "certo", "beleza",
    "opções", "opcoes", "ninguém", "alguém", "porque", "porquê", "qual",
    "quais", "onde", "quando", "quanto", "quantos", "tudo", "nada", "sempre",
    "nunca", "talvez", "depois", "antes", "agora", "hoje", "aí", "ai", "né",
}
EN_WORDS = {
    "the", "you", "your", "and", "that", "this", "with", "have", "are", "was",
    "were", "what", "when", "where", "which", "how", "can", "could", "would",
    "should", "will", "there", "their", "them", "they", "it", "its", "is",
    "of", "to", "in", "for", "on", "as", "at", "be", "been", "do", "does",
    "did", "not", "but", "or", "if", "so", "from", "about", "like", "just",
    "some", "any", "more", "most", "very", "also", "than", "then", "because",
    "into", "over", "after", "before", "well", "yeah", "okay", "sure", "right",
}
_PT_DIACRITICS = re.compile(r"[ãõçáéíóúâêôà]", re.IGNORECASE)
_WORD = re.compile(r"[a-zà-ÿ']+", re.IGNORECASE)


def classify(text: str, *, min_markers: int = 2, margin: float = 1.5) -> str:
    """'pt', 'en' or 'unclear' for one assistant turn."""
    toks = _WORD.findall((text or "").lower())
    if not toks:
        return "unclear"
    counts = Counter(toks)
    pt = sum(counts[w] for w in PT_WORDS if w in counts)
    en = sum(counts[w] for w in EN_WORDS if w in counts)
    pt += 2 * len(_PT_DIACRITICS.findall(text or ""))
    if pt + en < min_markers:
        return "unclear"
    if pt >= en * margin:
        return "pt"
    if en >= pt * margin:
        return "en"
    return "unclear"


def _summarise(turns: list[str]) -> dict:
    langs = [classify(t) for t in turns]
    words = [len((t or "").split()) for t in turns]
    n = len(turns)
    out = {
        "n_turns": n,
        "n_words": sum(words),
        "pt_turns": langs.count("pt"),
        "en_turns": langs.count("en"),
        "unclear_turns": langs.count("unclear"),
        "pt_word_share": round(
            sum(w for w, l in zip(words, langs) if l == "pt") / max(1, sum(words)), 4),
    }
    out["pt_turn_share"] = round(out["pt_turns"] / max(1, n - out["unclear_turns"]), 4)
    return out


def audio(summary: dict, model: str, root: Path) -> dict:
    turns: list[str] = []
    by_subset: dict[str, list[str]] = {}
    scenarios_with_pt = 0
    n_scenarios = 0
    for t in (root / model).glob("*/*/trace.json"):
        r = json.loads(t.read_text(encoding="utf-8"))
        texts = [x["text"] for x in r.get("turns") or [] if x.get("text")]
        if not texts:
            continue
        n_scenarios += 1
        if any(classify(x) == "pt" for x in texts):
            scenarios_with_pt += 1
        turns.extend(texts)
        by_subset.setdefault(r["subset"], []).extend(texts)
    res = _summarise(turns)
    res["scenarios_with_pt_share"] = round(scenarios_with_pt / max(1, n_scenarios), 4)
    res["n_scenarios"] = n_scenarios
    res["by_subset"] = {k: _summarise(v)["pt_turn_share"] for k, v in sorted(by_subset.items())}
    return res


def text_run(model_dir: Path) -> dict:
    """Same statistic from the text-domain run's raw.jsonl."""
    f = model_dir / "raw.jsonl"
    if not f.is_file():
        return {}
    turns: list[str] = []
    for line in f.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        for m in r.get("model_turns") or []:
            if isinstance(m, dict) and m.get("text"):
                turns.append(m["text"])
        for m in r.get("turns") or []:
            if isinstance(m, dict) and m.get("text"):
                turns.append(m["text"])
    return _summarise(turns) if turns else {}


LABELS = {
    "original_dc": "Original (upstream, EN)", "sft": "SFT (pt-BR)",
    "distill": "Distill (pt-BR)", "mtp_like": "MTP companion (pt-BR)",
    "base": "Base (control)",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-audio", type=Path, default=Path("results_audio/decision"))
    ap.add_argument("--results-text", type=Path, default=Path("results/en"))
    ap.add_argument("--out", type=Path, default=Path("results_audio/decision/language.json"))
    args = ap.parse_args()

    doc: dict = {"audio": {}, "text_en": {}}
    print("audio run (English input, models trained on pt-BR):")
    for m in sorted([q.name for q in args.results_audio.iterdir() if q.is_dir()]):
        r = audio({}, m, args.results_audio)
        if not r.get("n_turns"):
            continue
        doc["audio"][m] = r
        print(f"  {LABELS.get(m, m):26s} turns={r['n_turns']:5d} words={r['n_words']:6d} "
              f"pt turns={r['pt_turn_share']:.3f} pt words={r['pt_word_share']:.3f} "
              f"scenarios w/ pt={r['scenarios_with_pt_share']:.3f}")
        print(f"      by subset (pt turn share): {r.get('by_subset')}")
    print("\ntext-domain run (en), same measure:")
    for m in sorted(p.name for p in args.results_text.iterdir() if p.is_dir()):
        r = text_run(args.results_text / m)
        if not r.get("n_turns"):
            continue
        doc["text_en"][m] = r
        print(f"  {LABELS.get(m, m):26s} turns={r['n_turns']:5d} "
              f"pt turns={r['pt_turn_share']:.3f} pt words={r['pt_word_share']:.3f}")
    args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
