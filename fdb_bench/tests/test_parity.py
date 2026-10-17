"""Parity tests: the bench must match the code it is measuring.

The harness re-implements the duplex prompt construction and the per-variant
system prompts instead of importing the live bridges (which pull in websockets,
numpy and a running STT/TTS). These tests assert byte equality against the real
files, so the two can never silently drift:

    /home/penhfel/unsloth_uv/bin/python -m pytest fdb_bench/tests -q
    # or, without pytest:
    /home/penhfel/unsloth_uv/bin/python fdb_bench/tests/test_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fdb_bench import protocol as P  # noqa: E402
from fdb_bench import scenarios as S  # noqa: E402
from fdb_bench.clock import SpeechClock  # noqa: E402

SFT_BRIDGE = P.VARIANT_DIRS["sft"] / "DuplexCascade" / "server.py"
DISTILL_BRIDGE = P.VARIANT_DIRS["distill"] / "DuplexCascade" / "server.py"


def test_build_prompt_matches_sft_bridge() -> None:
    live = P.bridge_method(SFT_BRIDGE, "BridgeServer", "build_prompt")
    cases = [
        ([{"role": "system", "content": "S"}], ""),
        ([{"role": "system", "content": "S"},
          {"role": "user", "content": "oi"}], "<|user finish speaking|>"),
        ([{"role": "user", "content": "a"}, {"role": "assistant", "content": "<|x|>b"},
          {"role": "user", "content": "<|no voice|>"}], ""),
        ([{"role": "user", "content": "a"}] * 5, "<|user is speaking|>"),
    ]
    for hist, prime in cases:
        assert P.build_prompt(hist, prime) == live(None, hist, prime), (hist, prime)


def test_build_prompt_matches_distill_bridge() -> None:
    live = P.bridge_method(DISTILL_BRIDGE, "BridgeServer", "build_prompt")
    hist = [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}]
    assert P.build_prompt(hist, "<|user finish speaking|>") == live(
        None, hist, "<|user finish speaking|>")


def test_sft_system_prompt_matches_bridge() -> None:
    """The sft bridge inlines the prompt; check our copy of the two sentences."""
    src = SFT_BRIDGE.read_text(encoding="utf-8")
    assert "Você é um assistente de voz em português." in src
    assert "Responda de forma curta e natural." in src
    assert P.SYSTEM_PROMPTS["sft"] == (
        "Você é um assistente de voz em português. Responda de forma curta e natural."
    )


def test_distill_system_prompt_matches_bridge() -> None:
    live = P.extract_bridge_symbol(DISTILL_BRIDGE, "SYSTEM_PROMPT")
    assert P.DISTILL_SYSTEM == live, "distill SYSTEM_PROMPT drifted"


def test_tokens_match_training_data() -> None:
    for variant in ("sft", "distill", "mtp_like"):
        toks = P.load_tokens(variant)
        for key in P.TAG_KEYS:
            assert key in toks, f"{variant} missing {key}"
        assert toks["user_finish_speaking"] == "<|user finish speaking|>"
    # sft and distill data/common.py are documented as byte-identical
    assert P.load_tokens("sft") == P.load_tokens("distill")
    # mtp_like adds the two controller tags
    assert set(P.load_tokens("mtp_like")) - set(P.load_tokens("sft")) == set(P.MTP_EXTRA_KEYS)


def test_leading_special_and_text_of() -> None:
    assert P.leading_special("<|user finish speaking|>Olá") == "<|user finish speaking|>"
    assert P.text_of("<|user finish speaking|>Olá <|user is thinking|>") == "Olá"
    # content before any tag means the model opened without deciding
    assert P.leading_special("Olá <|user is thinking|>") is None
    assert P.strip_trailing_eos("abc<|im_end|>def") == "abc"


def test_clock_thresholds_match_fdb() -> None:
    c = SpeechClock(words_per_second=3.0)
    # >= 1 s of speech is a formal turn (5 words = 1.67 s)
    assert c.is_formal_turn("um dois tres quatro cinco")
    # sub-1 s AND <= 3 words is a non-takeover (2 words = 0.67 s)
    assert not c.is_formal_turn("um dois")
    # at exactly 3 words/s the two FDB conditions coincide (3 words = 1.0 s),
    # so the duration branch decides; a property of the nominal rate, not a
    # bug -- see clock.py.
    assert c.is_formal_turn("um dois tres")
    # backchannel: <= 3 s and <= 3 words
    assert c.is_backchannel("aham")
    assert not c.is_backchannel("a b c d e f g h i j k l m n o p")


def test_scenario_sizes_match_fdb() -> None:
    assert S.SUBSET_SIZES == {
        "pause_candor": 216, "pause_synthetic": 137, "smooth_turn": 119,
        "backchannel_icc": 55, "interruption_synthetic": 200,
    }
    assert sum(S.SUBSET_SIZES.values()) == 727


def test_scenarios_are_reproducible_and_bilingual() -> None:
    a = S.build_scenarios("smooth_turn", limit=5, lang="pt")
    b = S.build_scenarios("smooth_turn", limit=5, lang="pt")
    assert [x.events for x in a] == [x.events for x in b]
    en = S.build_scenarios("smooth_turn", limit=5, lang="en")
    assert [x.events for x in en] != [x.events for x in a]
    assert all(x.lang == "en" for x in en)


def test_pause_scenarios_never_end_the_turn() -> None:
    """Pause tasks are terminal: no chunk may be flagged ends_turn."""
    for task in ("pause_synthetic", "pause_candor"):
        for sc in S.build_scenarios(task, limit=20, lang="pt"):
            assert not any(e.ends_turn for e in sc.events), sc.sid


def test_smooth_turn_scenarios_always_end_the_turn() -> None:
    """The last *chunk* ends the turn; a trailing silence follows it.

    Scenarios deliberately end with a short silence because real audio always
    does, and the upstream release depends on that cue to answer at all (see
    _gen_smooth_turn), so the invariant is about the final chunk.
    """
    for sc in S.build_scenarios("smooth_turn", limit=20, lang="pt"):
        chunks = [e for e in sc.events if e.kind == "chunk"]
        assert chunks and chunks[-1].ends_turn, sc.sid
        tail = sc.events[len(sc.events) - len([e for e in sc.events if e.kind == "silence"]):]
        assert any(e.kind == "silence" and e.after_turn_end for e in sc.events), sc.sid
        _ = tail


def test_interruption_turn_is_multi_chunk() -> None:
    for lang in ("pt", "en"):
        for sc in S.build_scenarios("interruption_synthetic", limit=30, lang=lang):
            chunks = [e for e in sc.events if e.kind == "chunk"]
            # context turn + at least 2 chunks for the interrupting turn
            assert len(chunks) >= 4, sc.sid
            assert chunks[-1].ends_turn


def test_prompt_is_independent_of_turn_end_metadata() -> None:
    """The turn-end flag must never leak into the prompt.

    The simulator hands adapters ``{"role", "text", "ends_turn"}`` items so
    stub/test policies can be scripted. The real adapters build their prompt
    from role+text only, so the model is forced to *infer* the turn end from the
    text — which is the entire VAD-free claim. If this test fails, a model is
    being told the answer.
    """
    from fdb_bench.adapters import LlamaDuplexAdapter

    leaked = {"role": "user", "text": "oi", "ends_turn": True}
    honest = {"role": "user", "text": "oi"}
    ad = LlamaDuplexAdapter.__new__(LlamaDuplexAdapter)  # no client needed
    from fdb_bench import protocol as _P
    ad.system_prompt = _P.SFT_SYSTEM
    assert (ad._to_history([leaked]) == ad._to_history([honest])
            == [{"role": "system", "content": _P.SFT_SYSTEM},
                {"role": "user", "content": "oi"}])


def test_policy_adapter_strips_protocol_tags_from_content_model() -> None:
    """The mtp_like content LLM must only ever see natural dialogue."""
    from fdb_bench import protocol as _P

    class _FakePolicy:
        no_voice = _P.load_tokens("mtp_like")["no_voice"]
        tag_vocab = list(_P.TAG_KEYS)

    class _A:
        pass

    ad = _A()
    ad.policy = _FakePolicy()
    from fdb_bench.adapters import PolicyAdapter
    PolicyAdapter.__init__  # noqa: B018
    items = [
        {"role": "user", "text": "parte um"},
        {"role": "assistant", "text": "<|user finish speaking|>"},
        {"role": "user", "text": "parte dois"},
        {"role": "user", "text": _FakePolicy.no_voice},
    ]
    hist = PolicyAdapter._dialogue_history(ad, items)
    assert hist == [("user", "parte um parte dois")], hist


def test_chunking_is_lossless_for_every_scenario() -> None:
    """Every character of the user's turn must reach the model.

    An earlier chunker *sampled* sentences when there were more of them than
    requested chunks, silently deleting the user's actual question — which made
    the model look like it never took the floor. This guards that class of bug
    for every subset, in both languages, under both chunkers.

    The invariant is character-level (whitespace aside), not word-level: the
    training chunker cuts *tokens*, so a piece may end mid-word, exactly as in
    training (`'Colher o que eu mesmo pl'` / `'antei muda'`).
    """
    import re

    from fdb_bench.chunking import word_chunker

    def norm(t: str) -> str:
        return re.sub(r"\s+", "", t)

    for chunker in (word_chunker, None):
        label = type(chunker).__name__ if chunker else "default"
        for lang in S.LANGS:
            for task in S.TASKS:
                for sc in S.build_scenarios(task, lang=lang, chunker=chunker):
                    said = norm("".join(e.text for e in sc.events if e.kind == "chunk"))
                    want = norm(sc.meta.get("turn") or sc.context)
                    # the tokenizer round-trip is idempotent, so equality holds
                    assert said == want, (
                        f"[{label}] {sc.sid}: {len(want) - len(said)} chars lost")


def test_chunks_match_the_training_length_distribution() -> None:
    """ASR pieces must be training-sized (<= 7 words), not whole sentences.

    The models were trained on 1-7-token user pieces (measured mean 2.19 words).
    Feeding whole sentences was a 5-10x input shift and the bench's largest
    fidelity bug, so it gets a regression test.
    """
    from fdb_bench.chunking import USER_CHUNK_MAX

    lengths = []
    for lang in S.LANGS:
        for task in S.TASKS:
            for sc in S.build_scenarios(task, lang=lang):
                lengths.extend(len(e.text.split())
                               for e in sc.events if e.kind == "chunk")
    assert lengths, "no chunks produced"
    assert max(lengths) <= USER_CHUNK_MAX, f"max chunk {max(lengths)} words"
    mean = sum(lengths) / len(lengths)
    assert mean <= 3.5, f"mean chunk {mean:.2f} words — too sentence-like"


def test_training_chunker_reproduces_the_reference_lengths() -> None:
    """The tokenizer chunker must produce the training regime, not sentences."""
    import random

    try:
        from transformers import AutoTokenizer
    except Exception:  # pragma: no cover
        print("SKIP  transformers unavailable")
        return
    from fdb_bench.chunking import TrainingChunker
    from fdb_bench.protocol import MODEL_REGISTRY

    merged = MODEL_REGISTRY["sft"]["gguf"].parent.parent / "merged_bf16"
    if not merged.is_dir():
        print(f"SKIP  {merged} not present")
        return
    tok = AutoTokenizer.from_pretrained(str(merged), token=False)
    tc = TrainingChunker(tok)
    rng = random.Random(0)
    lengths = []
    for task in S.TASKS:
        for sc in S.build_scenarios(task, lang="pt", chunker=tc):
            lengths.extend(len(e.text.split()) for e in sc.events if e.kind == "chunk")
    assert max(lengths) <= 7, f"max {max(lengths)}"
    assert sum(lengths) / len(lengths) <= 4.0, sum(lengths) / len(lengths)


def test_prompt_is_independent_of_turn_end_metadata() -> None:
    """The turn-end flag must never leak into the prompt.

    The simulator hands adapters ``{"role", "text", "ends_turn"}`` items so
    stub/test policies can be scripted. The real adapters build their prompt
    from role+text only, so the model is forced to *infer* the turn end from the
    text — which is the entire VAD-free claim. If this test fails, a model is
    being told the answer.
    """
    from fdb_bench.adapters import LlamaDuplexAdapter

    leaked = {"role": "user", "text": "oi", "ends_turn": True}
    honest = {"role": "user", "text": "oi"}
    ad = LlamaDuplexAdapter.__new__(LlamaDuplexAdapter)  # no client needed
    from fdb_bench import protocol as _P
    ad.system_prompt = _P.SFT_SYSTEM
    assert (ad._to_history([leaked]) == ad._to_history([honest])
            == [{"role": "system", "content": _P.SFT_SYSTEM},
                {"role": "user", "content": "oi"}])


def test_policy_adapter_strips_protocol_tags_from_content_model() -> None:
    """The mtp_like content LLM must only ever see natural dialogue."""
    from fdb_bench import protocol as _P

    class _FakePolicy:
        no_voice = _P.load_tokens("mtp_like")["no_voice"]
        tag_vocab = list(_P.TAG_KEYS)

    class _A:
        pass

    ad = _A()
    ad.policy = _FakePolicy()
    from fdb_bench.adapters import PolicyAdapter
    PolicyAdapter.__init__  # noqa: B018
    items = [
        {"role": "user", "text": "parte um"},
        {"role": "assistant", "text": "<|user finish speaking|>"},
        {"role": "user", "text": "parte dois"},
        {"role": "user", "text": _FakePolicy.no_voice},
    ]
    hist = PolicyAdapter._dialogue_history(ad, items)
    assert hist == [("user", "parte um parte dois")], hist


def test_chunking_is_lossless_for_every_scenario() -> None:
    """Every word of the user's turn must reach the model.

    A chunker that sampled sentences silently deleted the user's actual
    question, which made the model look like it never took the floor. This
    guards against that class of bug for every subset in both languages.
    """
    import collections

    for lang in S.LANGS:
        for task in S.TASKS:
            for sc in S.build_scenarios(task, lang=lang):
                said = collections.Counter(
                    " ".join(e.text for e in sc.events if e.kind == "chunk").split()
                )
                want = collections.Counter(
                    (sc.meta.get("turn") or sc.context).split()
                )
                assert not (want - said), f"{sc.sid}: dropped {(want - said)}"


def test_chunks_are_non_empty_and_ordered() -> None:
    for lang in S.LANGS:
        for task in S.TASKS:
            for sc in S.build_scenarios(task, lang=lang):
                chunks = [e.text for e in sc.events if e.kind == "chunk"]
                assert chunks, sc.sid
                assert all(c.strip() for c in chunks), sc.sid


def test_turns_match_the_training_length_distribution() -> None:
    """User turns must be training-sized, not concatenated multi-sentence essays.

    Training turns are ~10.5 words with ~1.5 sentence-final chunks; the turn end
    is the LAST of those, and nothing in the text says which. Joining 2-4 clauses
    produced 35-71-word turns with 3-6.6 sentence-final chunks — 3-7x the
    ambiguity, which inflated pause/backchannel TOR (should not fire) and
    smooth/interruption TOR (should fire). This pins the shape.

    `pause_*` and `smooth_turn` are held to the training range. `backchannel_icc`
    and `interruption_synthetic` legitimately exceed it (a monologue is longer
    than any single turn, and the interruption task stacks two turns), so they
    are only checked for not being wildly off.
    """
    import re
    from fdb_bench.chunking import TrainingChunker
    from fdb_bench.protocol import MODEL_REGISTRY

    try:
        from transformers import AutoTokenizer
    except Exception:  # pragma: no cover
        print("SKIP  transformers unavailable")
        return
    merged = MODEL_REGISTRY["sft"]["gguf"].parent.parent / "merged_bf16"
    if not merged.is_dir():
        print(f"SKIP  {merged} not present")
        return
    tok = AutoTokenizer.from_pretrained(str(merged), token=False)
    tc = TrainingChunker(tok)
    term = re.compile(r"[.!?…]\s*$")

    budget = {"pause_synthetic": 12, "pause_candor": 14, "smooth_turn": 22,
              "interruption_synthetic": 40, "backchannel_icc": 45}
    sf_budget = {"pause_synthetic": 0.8, "pause_candor": 1.0, "smooth_turn": 2.0,
                 "interruption_synthetic": 3.2, "backchannel_icc": 4.0}
    for task in S.TASKS:
        words, sf = [], []
        for sc in S.build_scenarios(task, limit=80, lang="pt", chunker=tc):
            chunks = [e for e in sc.events if e.kind == "chunk"]
            if not chunks:
                continue
            words.append(len(" ".join(e.text for e in chunks).split()))
            sf.append(sum(1 for e in chunks if term.search(e.text)))
        mean_words = sum(words) / len(words)
        mean_sf = sum(sf) / len(sf)
        assert mean_words <= budget[task], (
            f"{task}: {mean_words:.1f} words/turn (training is 10.5, budget {budget[task]})")
        assert mean_sf <= sf_budget[task], (
            f"{task}: {mean_sf:.2f} sentence-final chunks/turn (budget {sf_budget[task]})")


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
