"""Adapters: one class per model family the bench can drive.

============================  ==============================================
adapter                       drives
============================  ==============================================
:class:`LlamaDuplexAdapter`   the tag-emitting duplex LLMs — ``sft`` (original
                              recipe), ``distill``, and the untrained ``base``
                              control. Speaks to a ``llama-server`` via the raw
                              ``/v1/completions`` endpoint with ``-sp``, because
                              the chat template mangles ``<|...|>`` tokens in
                              assistant history (see ``server.py``).
:class:`PolicyAdapter`        the ``mtp_like`` companion — a 0.8B model that
                              emits *only* a tag, plus a stock Qwen3-4B that
                              writes the content.
============================  ==============================================

Both expose the same contract (:class:`fdb_bench.simulate.Adapter`): given the
duplex ``items`` list, return ``(tag_key, spoken_text)`` and append the
assistant's own output to ``items``.

The micro-turn loop lives here, copied from ``llm_tick_task`` in
``sft/DuplexCascade/server.py``: the assistant delivers its answer in chunks,
each chunk being a separate assistant item separated by a ``<|no voice|>`` user
turn, and the turn closes when the model emits a turn-taking tag.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from . import protocol as P
from .system_prompts_en import EN_SFT_SYSTEM, TAG_PERMITTING_SYSTEM_EN

# Generation settings.
#
# Greedy (temperature 0) on purpose. The live bridges sample at 0.3/0.9, but a
# benchmark must be reproducible and comparable: at 0.3 the *same* scenario
# flipped between ``<|user finish speaking|>`` and ``<|user is speaking|>``
# across runs, which moved TOR by whole tenths and made two models impossible to
# separate from run-to-run noise. The repo's own capability eval
# (distill/eval/eval_capabilities.py) also scores with do_sample=False.
#
# repeat_penalty keeps each bridge's shipped value (distill's run_services.sh
# sets 1.15; sft's leaves it at 1.0) so that difference is still represented.
GENERATION = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 96, "repeat_penalty": 1.0}
SFT_GEN = dict(GENERATION)
DISTILL_GEN = {**GENERATION, "repeat_penalty": 1.15}
# decision-only probes need very few tokens: the tag comes first
TAG_MAX_TOKENS = 24

MAX_MICRO_TURNS = 10          # live bridge: max_micro_turns = 10


class LlamaServerError(RuntimeError):
    pass


# ----------------------------------------------------------------- client ---

class LlamaClient:
    """Minimal blocking client for a ``llama-server`` ``/v1/completions``."""

    def __init__(self, api_base: str, model: str | None = None, timeout: float = 90.0):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Generations the server refused to return. These are COUNTED, not
        # raised: see ``complete``.
        self.n_rejected = 0

    def complete(self, prompt: str, **gen: Any) -> str:
        payload: dict[str, Any] = {"prompt": prompt, "stream": False}
        payload.update(gen)
        if self.model:
            payload["model"] = self.model
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.api_base}/completions", data=data,
            headers={"Content-Type": "application/json"},
        )
        # HTTP 500 from llama-server is a *property of the generation*, not a
        # transport failure: it returns
        #   "The model produced output that does not match the expected
        #    Content-only format"
        # when the fine-tune emits ChatML control tokens of its own (header
        # imitation), which these models do when they degenerate. Raising here
        # was actively harmful: the scenario was dropped from the metrics, and
        # because the degenerate cases are also the worst-behaved ones, the
        # surviving sample was biased *in the model's favour* (61/727 scenarios
        # lost for distill in English, 58 of them pause scenarios).
        #
        # So a rejected generation is reported as an EMPTY completion and
        # counted. The adapter's normal path then applies: no leading tag means
        # "no decision" (a wait), which is what a model that produced nothing
        # usable actually did.
        last: Exception | None = None
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    obj = json.loads(r.read())
                return (obj.get("choices") or [{}])[0].get("text") or ""
            except urllib.error.HTTPError as e:
                if e.code == 500:
                    self.n_rejected += 1
                    return ""
                last = e
                time.sleep(1.5)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last = e
                time.sleep(1.5)
        raise LlamaServerError(f"llama-server request failed: {last}")

    def chat(self, messages: list[dict], **gen: Any) -> str:
        """Chat-style request, for the stock big model and for the judge.

        Two quirks handled here:

        * **thinking models put everything in ``reasoning_content``.** Qwen3.5
          (the judge) leaves ``content`` empty until its chain of thought
          finishes, so a short ``max_tokens`` yields ``""``. Thinking is
          disabled via ``chat_template_kwargs`` (needs ``--jinja`` on the
          server), and ``reasoning_content`` is kept as a fallback so a judge
          that thinks anyway is still usable.
        * ``enable_thinking`` is only understood by templates that support it;
          a server that rejects the kwarg falls back to a plain request.
        """
        base: dict[str, Any] = {"messages": messages, "stream": False}
        base.update(gen)
        if self.model:
            base["model"] = self.model

        for payload in (
            {**base, "chat_template_kwargs": {"enable_thinking": False}},
            base,
        ):
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.api_base}/chat/completions", data=data,
                headers={"Content-Type": "application/json"},
            )
            last: Exception | None = None
            for _ in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        obj = json.loads(r.read())
                    msg = (obj.get("choices") or [{}])[0].get("message") or {}
                    text = msg.get("content") or ""
                    if not text.strip():
                        text = msg.get("reasoning_content") or ""
                    return text
                except urllib.error.HTTPError as e:
                    if e.code == 400:
                        raise  # bad request: a retry will not help
                    last = e
                    time.sleep(1.5)
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                    last = e
                    time.sleep(1.5)
            if isinstance(last, urllib.error.HTTPError):
                break
        raise LlamaServerError(f"llama-server chat request failed: {last}")


# -------------------------------------------------- tag-emitting duplex LLM ---

class LlamaDuplexAdapter:
    """Drives a fine-tuned duplex LLM (sft / distill / untrained base control).

    Mirrors ``llm_tick_task`` from ``DuplexCascade/server.py``:

    * the prompt is built by :func:`protocol.build_prompt` (raw ChatML),
    * the **first** micro-turn of a turn is primed with
      ``<|user finish speaking|>`` in the *deployed* pipeline — but that is a
      VAD-free violation: it hands the model the answer. The bench therefore
      primes with ``""`` and lets the model *decide*, which is what the paper's
      VAD-free claim requires and what FDB measures. ``prime_mode`` switches
      between the two (see ``README.md`` → "Prime mode"),
    * the assistant continues its turn by feeding ``<|no voice|>`` until it
      emits a turn-closing tag.
    """

    def __init__(
        self,
        client: LlamaClient,
        variant: str,
        gen: dict[str, Any],
        *,
        name: str = "duplex",
        system_prompt: str | None = None,
        prime_mode: str = "decision",
        prompt_candidate: str = "none",
        max_micro_turns: int = MAX_MICRO_TURNS,
        # Upstream semantics (sbintuitions/DuplexCascade): the model SPEAKS by
        # emitting text and stays quiet by emitting a tag with no text. Its
        # server streams any generated text to TTS and uses the tags only to
        # stop (`assistant_backchannel`, `user is thinking`). This fork is the
        # opposite: `<|user finish speaking|>` is the speak signal and content
        # follows it. So the two need different floor rules.
        floor_by_text: bool = False,
        prepend_bos: bool = False,
    ) -> None:
        self.client = client
        self.variant = variant
        self.gen = dict(gen)
        self.name = name
        self.tokens = P.load_tokens(variant)
        self.floor_by_text = floor_by_text
        self.prepend_bos = prepend_bos
        # None = fall back to the variant default; "" = no system item at all
        self.system_prompt = (P.SYSTEM_PROMPTS[variant]
                              if system_prompt is None else system_prompt)
        self.prompt_candidate = prompt_candidate
        self.prime_mode = prime_mode
        self.max_micro_turns = max_micro_turns
        self.last_call_s = 0.0

        # reverse lookup: tag string -> key. Must be injective: two keys sharing
        # a tag string would silently mis-map one of them (that mistake turned
        # every `<|user is talking|>` into a system backchannel).
        self._key = {}
        for k, v in self.tokens.items():
            if v in self._key:
                raise ValueError(
                    f"{variant}: tag {v!r} is mapped to both "
                    f"{self._key[v]!r} and {k!r}; the tag set must be injective"
                )
            self._key[v] = k

    # -- Adapter protocol --------------------------------------------------
    @property
    def no_voice_marker(self) -> str:
        return self.tokens["no_voice"]

    def default_backchannel(self) -> str:
        return "aham"

    def reset(self) -> None:
        self.last_call_s = 0.0

    def act(self, items: list[dict]) -> tuple[str, str]:
        """One decision point: ask the model what to do, return (tag_key, text)."""
        history = self._to_history(items)
        prime = (
            self.tokens["user_finish_speaking"]
            if self.prime_mode == "prime" else ""
        )

        t0 = time.time()
        chunk = self._gen(history, prime, tag_only=True)
        self.last_call_s = time.time() - t0

        # strip the trailing EOS; split off the leading decision tag
        body = P.strip_trailing_eos(chunk)
        special = P.leading_special(body)
        text = P.text_of(body)
        tag_key = self._key.get(special or "", "user_is_speaking") if special else "user_is_speaking"

        # upstream rule: emitting speech is the decision to take the floor
        if self.floor_by_text and text.strip() and tag_key not in (
                "user_interruption",):
            tag_key = "user_finish_speaking"

        if tag_key == "user_finish_speaking":
            # the model took the floor: deliver the rest of the turn through the
            # micro-turn loop, exactly as llm_tick_task does
            full = self._micro_turn_loop(history, first_body=body)
            spoken = full["text"]
            for extra in full["items"]:
                items.append(extra)
            if spoken:
                items.append({"role": "assistant", "text": spoken})
            return "user_finish_speaking", spoken

        # a non-floor tag: the assistant says the tag and nothing else
        items.append({"role": "assistant", "text": special or text})
        return tag_key, text

    # -- internals ---------------------------------------------------------
    def _to_history(self, items: list[dict]) -> list[dict]:
        """Build the message list for ``build_prompt``.

        ``system_prompt`` empty means *no* system item at all, which is the
        duplex training distribution (``data/duplex_train*.jsonl`` has no system
        message) and the only setting under which the protocol is comparable
        across variants.
        """
        out: list[dict] = []
        if self.system_prompt:
            out.append({"role": "system", "content": self.system_prompt})
        for it in items:
            if it["role"] == "system":
                continue
            out.append({"role": it["role"], "content": it["text"]})
        return out

    def _gen(self, history: list[dict], prime: str, *, tag_only: bool) -> str:
        gen = dict(self.gen)
        if tag_only:
            gen["max_tokens"] = min(gen.get("max_tokens", 96), TAG_MAX_TOKENS)
        return self.client.complete(
            P.build_prompt(history, prime=prime, bos=self.prepend_bos), **gen)

    def _micro_turn_loop(self, history: list[dict], first_body: str) -> dict[str, Any]:
        """Continue the assistant turn across ``<|no voice|>`` round trips.

        Copied from ``llm_tick_task`` in the live bridge, with two additions the
        demo loop does not need:

        * **stop on any turn-closing tag**, checked on *every* generated chunk,
          not just the first — without this a model that never closes its turn
          burns the whole iteration budget re-reading the same context;
        * **stop on a repeated chunk**, the signature of the degeneration these
          fine-tunes fall into when pushed off-distribution (the distill model
          looping one sentence, or slipping into another language). Detecting it
          keeps one bad scenario from becoming ten full generations.

        Returns ``{"text": full spoken text, "items": [assistant items]}``.
        """
        spoken_parts: list[str] = [P.text_of(first_body)]
        assistant_items: list[dict] = [{"role": "assistant", "text": first_body.strip()}]
        hist = list(history) + [{"role": "assistant", "content": first_body.strip()}]
        closing = tuple(self.tokens[k] for k in P.TURN_END_TOKENS)
        seen: set[str] = set()

        for _ in range(self.max_micro_turns):
            if P.has_any(first_body, closing):
                break
            # the user stays silent so the model continues its own turn
            hist.append({"role": "user", "content": self.tokens["no_voice"]})
            t0 = time.time()
            nxt = self._gen(hist, prime="", tag_only=False)
            self.last_call_s += time.time() - t0
            nxt_body = P.strip_trailing_eos(nxt)
            if not nxt_body.strip():
                break
            key = " ".join(P.text_of(nxt_body).split()[:8]).lower()
            if key and key in seen:
                break               # degeneration guard
            if key:
                seen.add(key)
            txt = P.text_of(nxt_body)
            assistant_items.append({"role": "assistant", "text": nxt_body.strip()})
            hist.append({"role": "assistant", "content": nxt_body.strip()})
            if txt:
                spoken_parts.append(txt)
            if P.has_any(nxt_body, closing):
                break
        return {"text": " ".join(p for p in spoken_parts if p).strip(),
                "items": assistant_items}


class HFDuplexAdapter:
    """In-process twin of :class:`LlamaDuplexAdapter` for HF checkpoints.

    Needed for the original DuplexCascade release, which ships as a raw
    ``model_state.safetensors`` state dict rather than a servable GGUF, so it
    cannot go through ``llama-server``. Same prompt construction, same
    micro-turn loop, same tag semantics — only the generation backend differs,
    which keeps the comparison against the llama.cpp-served variants fair
    (identical decoding settings, no server-side sampling differences).
    """

    def __init__(
        self,
        model,
        tokenizer,
        variant: str,
        *,
        name: str = "original_dc",
        system_prompt: str | None = None,
        prime_mode: str = "decision",
        gen: dict[str, Any] | None = None,
        max_micro_turns: int = MAX_MICRO_TURNS,
    ) -> None:
        self.model = model
        self.tok = tokenizer
        self.variant = variant
        self.name = name
        self.tokens = P.load_tokens(variant)
        self.system_prompt = system_prompt or P.SYSTEM_PROMPTS[variant]
        self.prime_mode = prime_mode
        self.gen = gen or SFT_GEN
        self.max_micro_turns = max_micro_turns
        self.last_call_s = 0.0
        self._key = {v: k for k, v in self.tokens.items()}

    @property
    def no_voice_marker(self) -> str:
        return self.tokens["no_voice"]

    def default_backchannel(self) -> str:
        return "aham"

    def reset(self) -> None:
        self.last_call_s = 0.0

    def act(self, items: list[dict]) -> tuple[str, str]:
        history = self._to_history(items)
        prime = self.tokens["user_finish_speaking"] if self.prime_mode == "prime" else ""
        t0 = time.time()
        chunk = self._gen(history, prime, tag_only=True)
        self.last_call_s = time.time() - t0

        body = P.strip_trailing_eos(chunk)
        special = P.leading_special(body)
        text = P.text_of(body)
        tag_key = self._key.get(special or "", "user_is_speaking") if special else "user_is_speaking"

        # upstream rule: emitting speech is the decision to take the floor
        if self.floor_by_text and text.strip() and tag_key not in (
                "user_interruption",):
            tag_key = "user_finish_speaking"

        if tag_key == "user_finish_speaking":
            full = self._micro_turn_loop(history, first_body=body)
            for extra in full["items"]:
                items.append(extra)
            if full["text"]:
                items.append({"role": "assistant", "text": full["text"]})
            return "user_finish_speaking", full["text"]
        items.append({"role": "assistant", "text": special or text})
        return tag_key, text

    def _to_history(self, items: list[dict]) -> list[dict]:
        out: list[dict] = []
        if self.system_prompt:
            out.append({"role": "system", "content": self.system_prompt})
        for it in items:
            if it["role"] == "system":
                continue
            out.append({"role": it["role"], "content": it["text"]})
        return out

    def _gen(self, history: list[dict], prime: str, *, tag_only: bool) -> str:
        import torch

        prompt = P.build_prompt(history, prime=prime)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        gen = dict(self.gen)
        max_new = gen.pop("max_tokens", 96)
        if tag_only:
            max_new = min(max_new, TAG_MAX_TOKENS)
        gen.pop("repeat_penalty", None)   # handled via repetition_penalty below
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new,
                do_sample=gen.get("temperature", 0.0) > 0,
                temperature=gen.get("temperature", 0.0) or None,
                top_p=gen.get("top_p", 1.0),
                repetition_penalty=self.gen.get("repeat_penalty", 1.0),
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)

    def _micro_turn_loop(self, history: list[dict], first_body: str) -> dict[str, Any]:
        """Same loop and same guards as :meth:`LlamaDuplexAdapter._micro_turn_loop`."""
        spoken_parts = [P.text_of(first_body)]
        assistant_items = [{"role": "assistant", "text": first_body.strip()}]
        hist = list(history) + [{"role": "assistant", "content": first_body.strip()}]
        closing = tuple(self.tokens[k] for k in P.TURN_END_TOKENS)
        seen: set[str] = set()
        for _ in range(self.max_micro_turns):
            if P.has_any(first_body, closing):
                break
            hist.append({"role": "user", "content": self.tokens["no_voice"]})
            t0 = time.time()
            nxt = self._gen(hist, prime="", tag_only=False)
            self.last_call_s += time.time() - t0
            nxt_body = P.strip_trailing_eos(nxt)
            if not nxt_body.strip():
                break
            key = " ".join(P.text_of(nxt_body).split()[:8]).lower()
            if key and key in seen:
                break
            if key:
                seen.add(key)
            txt = P.text_of(nxt_body)
            assistant_items.append({"role": "assistant", "text": nxt_body.strip()})
            hist.append({"role": "assistant", "content": nxt_body.strip()})
            if txt:
                spoken_parts.append(txt)
            if P.has_any(nxt_body, closing):
                break
        return {"text": " ".join(p for p in spoken_parts if p).strip(),
                "items": assistant_items}


# -------------------------------------------------- mtp_like policy companion ---

class PolicyAdapter:
    """Drives the ``mtp_like`` companion (0.8B tag classifier + stock big LLM).

    Unlike the duplex LLMs, the companion never writes content: it is queried
    once per stream event and returns a single protocol tag
    (``mtp_like/policy/duplex_policy.py::DuplexPolicy.query``). When it fires
    ``user_finish_speaking`` the *stock* Qwen3-4B writes the answer — the whole
    architectural point of the variant ("big LLM untouched").

    The companion runs in-process via transformers; the stock big model is
    served by ``llama-server`` (same binary, same GGUF as the ``base`` control).
    """

    def __init__(
        self,
        policy,
        big_client: LlamaClient,
        *,
        name: str = "mtp_like",
        system_prompt: str = "",
        big_system: str | None = None,
        gen: dict[str, Any] | None = None,
        max_content_words: int = 48,
        prompt_candidate: str = "instruction",
    ) -> None:
        self.policy = policy
        self.big_client = big_client
        self.name = name
        self.system_prompt = system_prompt
        self.big_system = big_system or self._default_big_system(system_prompt)
        self.gen = gen or {"temperature": 0.3, "top_p": 0.9, "max_tokens": 96}
        self.max_content_words = max_content_words
        self.last_call_s = 0.0
        self.prompt_candidate = prompt_candidate
        # companion tag vocabulary: the list of tag *names* it can emit
        self.tag_vocab = list(getattr(self.policy, "tag_vocab", []))

    @staticmethod
    def _default_big_system(system_prompt: str) -> str:
        # The persona's content LLM. Uses the *variant's* own system prompt so
        # the content style matches the duplex LLMs being compared against.
        return system_prompt or "Você é um assistente de voz em português. Responda de forma curta e natural."

    @property
    def no_voice_marker(self) -> str:
        return self.policy.no_voice

    def default_backchannel(self) -> str:
        return "aham"

    def reset(self) -> None:
        self.last_call_s = 0.0

    def act(self, items: list[dict]) -> tuple[str, str]:
        """One companion query, plus a content turn when the floor is taken.

        Faithful to ``orchestrator.py::_apply``:

        * ``user_interruption`` hard-stops the answer and **does not** call the
          content model — the companion is the controller, the big LLM is only
          consulted once the turn is actually the assistant's;
        * ``system_backchannel`` emits the canned listener backchannel and stays
          quiet otherwise;
        * ``user_finish_speaking`` / ``system_take_floor`` hand the floor to the
          stock content model;
        * ``no_tag`` and the wait tags emit nothing.
        """
        t0 = time.time()
        tag, _probs = self.policy.query(self._policy_items(items))
        self.last_call_s = time.time() - t0

        if tag in ("no_tag", "user_is_speaking", "user_is_thinking",
                   "user_backchannel"):
            return tag, ""

        if tag == "user_interruption":
            return tag, ""          # yield now; no content generated

        if tag == "system_backchannel":
            bc = self.default_backchannel()
            items.append({"role": "assistant", "text": bc})
            return tag, bc

        if tag in ("user_finish_speaking", "system_take_floor", "system_handover"):
            t1 = time.time()
            spoken = self._big_reply(items)
            self.last_call_s += time.time() - t1
            if spoken.strip():
                items.append({"role": "assistant", "text": spoken})
                return "user_finish_speaking", spoken
            return "no_tag", ""      # handover the big model declined

        return tag, ""

    def _policy_items(self, items: list[dict]) -> list[dict]:
        """Adapt the simulator's items to the companion's expected shape.

        The simulator already seeds ``items`` with the system item, so nothing is
        prepended here: doing so produced TWO system items, which is off the
        companion's training distribution (``render_items`` emits at most one)
        and made it misfire on every decision.
        """
        return [{"role": it["role"], "text": it["text"]}
                for it in items if it["text"].strip()]

    def _big_reply(self, items: list[dict]) -> str:
        """Ask the stock big LLM for the answer, given the duplex history.

        Mirrors ``Orchestrator._dialogue_history``: protocol tag chunks and
        ``<|no voice|>`` markers are stripped and consecutive user chunks are
        merged, so the stock LLM only ever sees natural dialogue.
        """
        msgs: list[dict] = [{"role": "system", "content": self.big_system}]
        for role, content in self._dialogue_history(items):
            msgs.append({"role": role, "content": content})
        gen = dict(self.gen)
        gen["max_tokens"] = self.max_content_words
        try:
            return self.big_client.chat(msgs, **gen).strip()
        except LlamaServerError:
            return ""

    def _dialogue_history(self, items: list[dict]) -> list[tuple[str, str]]:
        """(role, text) pairs of real dialogue only — see
        ``mtp_like/policy/orchestrator.py::Orchestrator._dialogue_history``."""
        tag_texts = {self.policy.no_voice}
        try:
            tokens = P.load_tokens("mtp_like")
            tag_texts.update(tokens[k] for k in P.TAG_KEYS)
        except Exception:
            pass
        out: list[dict] = []
        for it in items:
            text = it["text"].strip()
            # protocol tags, silence markers and the controller's own system
            # item are not dialogue: the stock model must never see them (and
            # the content model gets its own system message separately)
            if not text or text in tag_texts or text.startswith("<|"):
                continue
            if it["role"] == "system":
                continue
            if out and out[-1]["role"] == it["role"]:
                out[-1]["content"] += " " + text
            else:
                out.append({"role": it["role"], "content": text})
        return [(o["role"], o["content"]) for o in out]


# ------------------------------------------------------------------ factory ---

# Per-variant prompt candidates. Each entry is a *named way to run that
# variant*; the name is what gets recorded in meta.json, so a table can always
# be traced back to the configuration that produced it.
#
# "duplex" is the variant's own prompt that asks for the full-duplex protocol —
# the configuration under which measuring turn-taking is meaningful. See
# fdb_bench/calibrate_prompts.py, which measures these candidates instead of
# arguing about them.
PROMPT_CANDIDATES: dict[str, list[str]] = {
    #                          none, bridge prompt, tag-permissive, duplex trigger
    "sft":       ["none", "deployed", "tag"],
    "sft_m2":    ["none", "deployed", "tag"],
    "distill":   ["none", "deployed", "tag", "trigger", "trigger_tags"],
    "mtp_like":  ["none", "instruction"],
    "base":      ["none", "deployed", "tag"],
    "original_dc": ["none", "deployed"],
    "original_dc_hf": ["none", "deployed"],
}

# The candidate each variant uses by default. Chosen by
# `calibrate_prompts.py` on tag-decision accuracy + content sanity; see
# fdb_bench/PROMPTS.md for the measurements behind these choices.
DEFAULT_PROMPT_CANDIDATE: dict[str, str] = {
    # The PRIMARY configuration is the *training distribution* for both duplex
    # LLMs: `prepare`/`build_duplex_dataset.py` emit no system item at all, so
    # running without one measures the protocol where it was learned and
    # requires no bench-invented prompt and no selection on the evaluation
    # metrics. `PROMPTS.md` shows the variants are extremely prompt-sensitive,
    # so the prompt is treated as an experimental factor and the sensitivity
    # runs are reported alongside — see run_all.sh and PROMPTS.md.
    #
    # mtp_like is the exception: its companion's training data *does* carry a
    # system instruction on ~50% of records (policy/prep_policy_dataset.py:120),
    # and the measured difference from `none` is nil, so it keeps the
    # on-distribution instruction.
    "sft": "none",
    "sft_m2": "none",
    "distill": "none",
    "mtp_like": "instruction",
    "base": "none",
    # the original runs with no system item by default, matching how the
    # three pt-BR variants are measured; "deployed" (a neutral prompt) is
    # run as an extra, since the upstream model may expect one
    "original_dc": "none",
    "original_dc_hf": "none",
}


def resolve_system_prompt(
    key: str, lang: str, candidate: str, override: str | None = None,
) -> str:
    """Resolve the system item for one *named* configuration.

    ====================  =====================================================
    candidate             system item
    ====================  =====================================================
    ``none``              none at all — the duplex *training* distribution
                          (``data/duplex_train*.jsonl`` carries no system
                          message)
    ``deployed``          the exact prompt that variant's live bridge ships
    ``tag``               a prompt that permits the protocol tags
    ``trigger``           distill only: ``TRIGGER_SYSTEM`` from the training
                          code — names the duplex role and the tags, says
                          nothing about the speech
    ``trigger_tags``      distill only: ``TRIGGER_SYSTEM_ALLOW_TAGS`` — asks for
                          the protocol *and* coherent content
    ``instruction``       mtp_like only: the companion's on-distribution generic
                          instruction ("manage the conversation normally")
    ====================  =====================================================

    A candidate that does not apply to a variant raises, so a typo in a sweep
    cannot silently fall back to a different prompt and produce a mislabelled
    table row.
    """
    if override is not None:
        return override

    spec = P.MODEL_REGISTRY[key]
    kind = spec["kind"]
    if candidate not in PROMPT_CANDIDATES[key]:
        raise KeyError(
            f"candidate {candidate!r} is not valid for {key!r}; "
            f"expected one of {PROMPT_CANDIDATES[key]}"
        )

    if candidate == "none":
        return ""

    if kind == "policy_companion":
        # the companion's training data carries a system instruction on ~50% of
        # records, so "instruction" is its on-distribution setting
        return P.MTP_COMPANION_SYSTEM if lang == "pt" else P.MTP_COMPANION_SYSTEM_EN

    if candidate == "tag":
        return P.TAG_PERMITTING_SYSTEM if lang == "pt" else TAG_PERMITTING_SYSTEM_EN

    if candidate in ("trigger", "trigger_tags"):
        tp = P.distill_teacher_prompts()
        base = tp["TRIGGER_SYSTEM"] if candidate == "trigger" else tp["TRIGGER_SYSTEM_ALLOW_TAGS"]
        if lang == "pt":
            return base
        from .system_prompts_en import DISTILL_TRIGGER_EN, DISTILL_TRIGGER_TAGS_EN
        return DISTILL_TRIGGER_EN if candidate == "trigger" else DISTILL_TRIGGER_TAGS_EN

    # deployed
    if lang == "en":
        from .system_prompts_en import for_model
        return for_model(key, spec["system"])
    return spec["system"]


def build_adapter(
    key: str,
    *,
    api_base: str,
    big_api_base: str | None = None,
    device: str = "cuda",
    prime_mode: str = "decision",
    prompt_candidate: str | None = None,
    system_override: str | None = None,
    lang: str = "pt",
) -> Any:
    """Instantiate the adapter for a :data:`protocol.MODEL_REGISTRY` key.

    ``lang`` selects the language pack and translates the system prompt for
    ``en`` (the duplex models were trained pt-BR only, so ``en`` measures
    cross-lingual transfer of the *protocol*).
    """
    spec = P.MODEL_REGISTRY[key]
    kind = spec["kind"]
    candidate = (prompt_candidate or DEFAULT_PROMPT_CANDIDATE.get(key, "none"))
    system = resolve_system_prompt(key, lang, candidate, system_override)

    if kind == "llama_duplex":
        variant = spec["variant"]
        gen = DISTILL_GEN if key == "distill" else SFT_GEN
        adapter = LlamaDuplexAdapter(
            LlamaClient(api_base), variant, gen, name=key,
            system_prompt=system, prime_mode=prime_mode,
            prompt_candidate=candidate,
            floor_by_text=(variant == "original_dc"),
            prepend_bos=(variant == "original_dc"),
        )
        return adapter

    if kind == "policy_companion":
        from .policy_companion import load_policy   # lazy: heavy import
        policy = load_policy(spec["companion"], device=device)
        # The content LLM is stock, so it gets the *shared* short voice prompt
        # rather than the companion's protocol instruction: the point of this
        # variant is that the big model never sees the tags.
        big_system = P.SFT_SYSTEM if lang == "pt" else EN_SFT_SYSTEM
        return PolicyAdapter(
            policy, LlamaClient(big_api_base or api_base), name=key,
            system_prompt=system or "", big_system=big_system,
            prompt_candidate=candidate,
        )

    if kind == "original_dc":
        from .original_dc import build_original_adapter
        return build_original_adapter(spec, api_base=api_base, lang=lang,
                                      prime_mode=prime_mode,
                                      prompt_candidate=candidate)

    raise KeyError(f"no adapter for model kind {kind!r}")
