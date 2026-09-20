"""M3 — the duplex orchestrator (SPEC 4.2): the companion is the controller.

Event-driven full-duplex loop over:
  * user ASR word events      -> query companion with the open user item
  * silence (word-gap timer)  -> insert <|no voice|> and query
  * assistant sentence boundary -> close the assistant chunk and query

The companion's tag drives the system (SPEC 4.2):
  * <|user finish speaking|>  -> start the big model's answer
  * <|user interruption|>     -> hard-stop the answer + TTS
  * <|user is speaking|>      -> user mid-utterance: pause / keep waiting
  * <|user is thinking|>      -> user composing: keep waiting
  * <|user backchannel|>      -> user acknowledged: keep talking
  * <|system backchannel|>    -> assistant should backchannel (canned)
  * no tag                    -> nothing (emit-nothing case)

No VAD: user-speech state comes from ASR word events + word-gap timer +
the learned tags. The big model is stock (never sees the tags).

Usage (CLI demo — type to talk, pause to let silence fire):
  python policy/orchestrator.py --companion /mnt/f/duplex_cascade_runs/mtp_like/runs/final_v2/merged \
      --llm transformers --big-model Qwen/Qwen3-4B-Instruct-2507
"""

from __future__ import annotations

import argparse
import re
import select
import sys
import time
from dataclasses import dataclass, field
from typing import Iterator

import torch

from data.common import IM_END, IM_END_NL, IM_START_ASSISTANT, IM_START_USER, TOKEN
from policy.duplex_policy import DuplexPolicy

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")

_TAG_TEXT = {
    "user_is_speaking": TOKEN["user_is_speaking"],
    "user_interruption": TOKEN["user_interruption"],
    "user_backchannel": TOKEN["user_backchannel"],
    "user_is_thinking": TOKEN["user_is_thinking"],
    "system_backchannel": TOKEN["system_backchannel"],
    "user_finish_speaking": TOKEN["user_finish_speaking"],
    "system_take_floor": TOKEN["system_take_floor"],
    "system_handover": TOKEN["system_handover"],
}

BIG_SYSTEM = (
    "Você é o assistente virtual do Banco Penha, um banco brasileiro. Atenda "
    "os clientes com educação e profissionalismo, em português brasileiro, "
    "como numa conversa falada. Responda em UMA ou DUAS frases curtas e "
    "naturais, sem listas, sem tópicos, sem emojis e sem repetir a pergunta. "
    "Quando o cliente pedir um cartão de crédito, avise que para dar "
    "continuidade é preciso do CPF dele e peça os onze números."
)


# ---------------------------------------------------------------------------
# Big model backends (stock LLMs — no duplex tokens in their vocabulary)
# ---------------------------------------------------------------------------

class BaseLLM:
    name = "base"

    def iter_sentences(self, user_turn: str,
                       history: list[dict] | None = None) -> Iterator[str]:
        raise NotImplementedError


class TransformersLLM(BaseLLM):
    name = "transformers"

    def __init__(self, model_name: str, device: str = "cuda", max_len: int = 96):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name, token=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map=device, token=False,
        ).eval()
        self.max_len = max_len
        self._stop = False

    def stop(self) -> None:
        """Ask the active generator to stop (checked between chunks)."""
        self._stop = True

    def speak_response(self, instruction: str, max_new_tokens: int = 48) -> str:
        """Generate ONE short assistant response to a control instruction
        (the domain agent describes the situation; the model answers in its
        own voice under the bank persona)."""
        prompt = self._prompt(instruction)
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(
            out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def _prompt(self, user_turn: str, history: list[dict] | None = None) -> str:
        msgs = [{"role": "system", "content": BIG_SYSTEM}]
        if history:
            msgs.extend(history)
        msgs.append({"role": "user", "content": user_turn})
        try:
            return self.tok.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
        except Exception:
            # fallback: manual ChatML + direct-answer directive
            return (
                f"{IM_START_USER}{BIG_SYSTEM} Responda direto, "
                f"sem raciocínio.{IM_END_NL}"
                + "".join(
                    f"{IM_START_USER}{m['content']}{IM_END_NL}"
                    if m["role"] == "user"
                    else f"{IM_START_ASSISTANT}{m['content']}{IM_END_NL}"
                    for m in msgs[1:]
                )
                + IM_START_ASSISTANT
            )

    def iter_sentences(self, user_turn: str,
                       history: list[dict] | None = None) -> Iterator[str]:
        prompt = self._prompt(user_turn, history)
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        seq: torch.Tensor = enc["input_ids"]
        attn: torch.Tensor = enc["attention_mask"]
        self._stop = False
        new_tokens = 0
        buf = ""
        max_len = self.max_len
        past_len = seq.shape[1]
        while new_tokens < max_len:
            if self._stop:
                break
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=seq, attention_mask=attn,
                    max_new_tokens=32, do_sample=False,
                    eos_token_id=self.tok.eos_token_id, use_cache=True,
                )
            new_tokens += out.shape[1] - seq.shape[1]
            seq, attn = out, torch.ones_like(out)
            text = self.tok.decode(out[0][past_len:], skip_special_tokens=True)
            past_len = out.shape[1]
            buf += text
            while True:
                m = _SENT_SPLIT.search(buf)
                if m is None:
                    break
                sent = buf[: m.end()].strip()
                buf = buf[m.end():]
                if sent:
                    yield sent
            if out[0, -1] == self.tok.eos_token_id:
                break
        if buf.strip():
            yield buf.strip()


class LlamaCppLLM(BaseLLM):
    """OpenAI-compatible llama.cpp server (like the sft stack)."""

    name = "llamacpp"

    def __init__(self, api_base: str = "http://127.0.0.1:8080/v1", **kw):
        import requests

        self.requests = requests
        self.api_base = api_base

    def iter_sentences(self, user_turn: str) -> Iterator[str]:
        # stream the completion and split sentences
        payload = {
            "prompt": user_turn,
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": True,
        }
        with self.requests.post(
            f"{self.api_base}/completions", json=payload, stream=True,
        ) as r:
            buf = ""
            for line in r.iter_lines():
                if not line or not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                import json

                chunk = json.loads(data)["choices"][0]["text"]
                buf += chunk
                while True:
                    m = _SENT_SPLIT.search(buf)
                    if m is None:
                        break
                    sent = buf[: m.end()].strip()
                    buf = buf[m.end():]
                    if sent:
                        yield sent
            if buf.strip():
                yield buf.strip()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class Orchestrator:
    policy: DuplexPolicy
    llm: BaseLLM
    word_gap_s: float = 0.6
    tts: object | None = None
    items: list = field(default_factory=list)
    pending_user: str = ""
    _turn_parts: list = field(default_factory=list)  # closed chunks of the current turn
    _turn_start: int = 0  # items index where the current user turn begins
    state: str = "listening"  # listening | answering
    answer_gen: Iterator[str] | None = None
    verbose: bool = True
    last_tag: str = "no_tag"  # last companion decision (test hook)
    emit_cb: object | None = None  # Callable[[str, str], None] called on events
    rule: str = ""  # prompt rule (system instruction) for the promptable controller

    # -- context ----------------------------------------------------------

    @property
    def current_user_text(self) -> str:
        """The user's full turn so far (closed chunks + the open one)."""
        parts = list(self._turn_parts)
        if self.pending_user.strip():
            parts.append(self.pending_user.strip())
        return " ".join(parts)

    def _context(self, trigger: str) -> list[dict]:
        ctx = list(self.items)
        if self.rule.strip():
            ctx.insert(0, {"role": "system", "text": self.rule})
        if trigger == "silence":
            # the user's words so far are still in pending_user (never
            # closed): include them closed, then the no-voice marker —
            # exactly the training silence-record shape
            if self.pending_user.strip():
                ctx.append({"role": "user", "text": self.pending_user.strip()})
            ctx.append({"role": "user", "text": TOKEN["no_voice"], "partial": True})
        elif self.pending_user.strip():
            ctx.append({"role": "user", "text": self.pending_user, "partial": True})
        return ctx

    # -- events -----------------------------------------------------------

    def reset_turn(self) -> None:
        """Abandon the current user turn (e.g., invalid input rejected by a
        domain agent). The user starts the next utterance fresh."""
        self.pending_user = ""
        self._turn_parts = []

    def reset(self) -> None:
        """Start a fresh session: clear the duplex context and state."""
        self.items = []
        self.pending_user = ""
        self._turn_parts = []
        self.state = "listening"
        self.answer_gen = None
        self.last_tag = "no_tag"

    def on_user_words(self, words: str) -> None:
        w = words.strip()
        if not w:
            return
        # split long utterances into ~6-word pieces so each piece forms a
        # chunk of the training shape ([u chunk, a tag, ...]) — a single
        # giant chunk + silence is out of the training distribution.
        # Dictations come back comma-separated ("1, 2, 3, a, b") — chunk on
        # commas too, so each digit/letter is its own [u word, a tag] pair.
        comma_parts = re.split(r"(?<=,)\s*|\s+(?=,)", w)
        for part in comma_parts:
            part = part.strip()
            if not part:
                continue
            words_in = part.split()
            if len(words_in) <= 6:
                self._on_word_event(part)
            else:
                for i in range(0, len(words_in), 6):
                    self._on_word_event(" ".join(words_in[i:i + 6]))

    def _on_word_event(self, w: str) -> None:
        if self.state == "answering":
            self._close_answer()  # user talked over us; decide below
        if not self.pending_user:
            # first word of a new utterance: mark where its items begin so
            # on_user_utterance can wipe the partial-fed chunks
            self._turn_start = len(self.items)
        if self.pending_user.strip():
            # a NEW word batch arrived: close the previous chunk and attach
            # the tag it received (training stream: u chunk -> a tag -> ...)
            self.items.append({"role": "user", "text": self.pending_user.strip()})
            self._turn_parts.append(self.pending_user.strip())
            if self.last_tag != "no_tag":
                self.items.append(
                    {"role": "assistant", "text": _TAG_TEXT.get(self.last_tag, "")}
                )
        self.pending_user = w
        self._emit("user_asr", w)
        self._query_and_apply("user")

    def on_user_utterance(self, text: str) -> None:
        """The FINAL ASR transcript of the user's utterance (authoritative):
        wipe any partial-fed chunks of this turn and feed the final text."""
        del self.items[self._turn_start:]
        self._turn_start = len(self.items)
        self._turn_parts = []
        self.pending_user = ""
        self.on_user_words(text)

    def on_silence(self) -> None:
        if not (self.pending_user or self.state == "answering"):
            return  # fully idle; nothing to decide
        self._query_and_apply("silence")
        # Turn-end robustness: the full-context query can stay stuck in
        # "speaking" when the turn was chunked word-by-word (out-of-training
        # density of [u word, a tag] pairs). The companion fires "finish"
        # reliably on the MINIMAL shape [u final-chunk, u no voice] (trained
        # with weight 5) — use it as a second opinion.
        if self.last_tag != "user_finish_speaking" and self.pending_user.strip():
            tag2, probs2 = self.policy.query(
                [
                    {"role": "user", "text": self.pending_user.strip()},
                    {"role": "user", "text": TOKEN["no_voice"], "partial": True},
                ]
            )
            if (
                tag2 == "user_finish_speaking"
                and probs2["user_finish_speaking"] >= 0.5
            ):
                self.last_tag = "user_finish_speaking"
                self._emit("policy", "user_finish_speaking (minimal-shape confirm)")
                self._apply("user_finish_speaking")
                return
            # final fallback: sentence-final punctuation + silence = turn end
            if self.pending_user.rstrip().endswith((".", "?", "!", "…")):
                self.last_tag = "user_finish_speaking"
                self._emit("policy", "user_finish_speaking (punctuation fallback)")
                self._apply("user_finish_speaking")

    # -- companion queries ------------------------------------------------

    def _query_and_apply(self, trigger: str) -> None:
        tag, probs = self.policy.query(self._context(trigger))
        self.last_tag = tag
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        self._emit("policy", f"{tag}  ({', '.join(f'{k}={v:.2f}' for k, v in top)})")
        self._apply(tag)

    def _apply(self, tag: str) -> None:
        if tag == "no_tag":
            return  # keep the current user chunk open
        if tag == "user_finish_speaking":
            self._start_answer()  # closes the user item FIRST
            self.items.append(
                {"role": "assistant", "text": TOKEN["user_finish_speaking"]}
            )
            return
        if tag == "system_take_floor":
            # the prompt rule fired: the SYSTEM interrupts the user and
            # speaks now (the bridge responds INSTANTLY — the big model is
            # too slow to cut the user mid-speech)
            self._close_answer()
            user_text = self.current_user_text  # capture BEFORE the reset
            self.reset_turn()
            self._emit("system_take_floor",
                       f"{self.rule} | user: {user_text}")
            return
        if tag == "system_handover":
            # the prompt rule fired: hand control to the big model, which
            # decides what to do (maybe nothing). The user's turn is NOT
            # reset here — only if the big model chooses to respond.
            self._emit("system_handover",
                       f"{self.rule} | user: {self.current_user_text}")
            return
        # speaking / thinking / backchannel / interruption: the current
        # chunk stays OPEN (it is closed when new words arrive); the tag is
        # attached to it then. Only interruption acts immediately.
        if tag == "user_interruption":
            self._hard_stop()
        elif tag == "system_backchannel":
            self._emit("tts", "aham")  # canned backchannel

    # -- answer control ----------------------------------------------------

    def _start_answer(self) -> None:
        if not self.pending_user.strip():
            return
        self.items.append({"role": "user", "text": self.pending_user.strip()})
        self._turn_parts.append(self.pending_user.strip())
        # attach the tag the closing chunk received (speaking/backchannel/
        # interruption); finish is appended by _apply afterwards
        if self.last_tag not in ("no_tag", "user_finish_speaking"):
            self.items.append(
                {"role": "assistant", "text": _TAG_TEXT.get(self.last_tag, "")}
            )
        self.pending_user = ""
        # the big model gets the FULL user turn (all chunks of this turn),
        # not just the last word — it must answer the whole question
        user_turn = " ".join(self._turn_parts).strip()
        self._turn_parts = []
        self.state = "answering"
        self._emit("state", f"answering: {user_turn!r}")
        # the big model must SEE the conversation: history = everything
        # before the current turn (the finish tag + tag chunks + no-voice
        # markers are protocol, not dialogue). Without it, each answer is
        # stateless — e.g. the bank assistant repeats the CPF ask.
        history = self._dialogue_history(self.items[: self._turn_start])
        self.answer_gen = self.llm.iter_sentences(user_turn, history)

    @staticmethod
    def _dialogue_history(items: list[dict]) -> list[dict]:
        """items -> user/assistant messages for the big model: skips the
        protocol tag chunks (<|user is speaking|>, finish, no-voice, ...)
        and merges consecutive user chunks into one message."""
        tag_texts = set(_TAG_TEXT.values())
        tag_texts.add(TOKEN["no_voice"])
        history: list[dict] = []
        for it in items:
            text = it["text"].strip()
            if text in tag_texts:
                continue
            if history and history[-1]["role"] == it["role"]:
                history[-1]["content"] += " " + text
            else:
                history.append({"role": it["role"], "content": text})
        return history

    def _hard_stop(self) -> None:
        if self.state == "answering":
            self._emit("state", "interrupted")
            self.state = "listening"
            self.answer_gen = None

    def _close_answer(self) -> None:
        if self.state == "answering":
            self.answer_gen = None
            self.state = "listening"
            if hasattr(self.llm, "stop"):
                self.llm.stop()  # stop burning GPU on the abandoned answer
        # any chunks of the interrupted answer's turn are consumed; the
        # user's next words start a FRESH turn
        self._turn_parts = []

    def barge_in(self) -> None:
        """Instant barge-in signalled by the bridge from raw mic energy
        (no VAD — the bridge already sees the PCM stream). Stops generation
        immediately; the user's words arrive from the STT afterwards and the
        companion fires the interruption tag from content."""
        self._close_answer()
        self._emit("state", "barge-in (energy)")

    def _assistant_sentence(self, sent: str) -> None:
        self.items.append({"role": "assistant", "text": sent})
        self._emit("assistant_text", sent)
        # sentence boundary event: what next?
        self._query_and_apply("assistant")

    def note_assistant(self, text: str) -> None:
        """Append an assistant item to the transcript WITHOUT a policy query
        (used by the bridge for instant canned responses like the
        take_floor interrupt phrase — the transcript must reflect it, but
        no decision is needed)."""
        text = text.strip()
        if text:
            self.items.append({"role": "assistant", "text": text})
            self._emit("assistant_text", text)

    # -- main loop ----------------------------------------------------------

    def run_loop(self, prompt: str = "> ") -> None:
        print("full-duplex loop — type to talk; silence fires the gap timer; Ctrl-D to quit")
        while True:
            if self.state == "answering" and self.answer_gen is not None:
                # drive the big model; check for user input between sentences
                for sent in self.answer_gen:
                    if self.state != "answering":
                        break
                    self._assistant_sentence(sent)
                    ready, _, _ = select.select([sys.stdin], [], [], 0)
                    if ready:
                        self.state = "listening"
                        self.answer_gen = None
                        break
                self._close_answer()
                # after finishing the answer, close with a thinking state query
                self._query_and_apply("silence")
            # wait for input with a word-gap timeout
            ready, _, _ = select.select([sys.stdin], [], [], self.word_gap_s)
            if ready:
                line = sys.stdin.readline()
                if not line:
                    break
                self.on_user_words(line)
            else:
                self.on_silence()

    def _emit(self, kind: str, msg: str) -> None:
        if self.emit_cb is not None:
            try:
                self.emit_cb(kind, msg)
            except Exception:
                pass
        if not self.verbose:
            return
        tag = {
            "user_asr": "USER",
            "assistant_text": "ASSISTANT",
            "policy": "  POLICY",
            "state": "  STATE",
            "tts": "  TTS",
        }.get(kind, kind)
        print(f"[{tag:9s}] {msg}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--companion", default="/mnt/f/duplex_cascade_runs/mtp_like/runs/final_v2/merged")
    p.add_argument("--llm", choices=["transformers", "llamacpp"], default="transformers")
    p.add_argument("--big-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--api-base", default="http://127.0.0.1:8080/v1")
    p.add_argument("--word-gap-s", type=float, default=0.6)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    policy = DuplexPolicy(args.companion)
    if args.llm == "transformers":
        llm = TransformersLLM(args.big_model)
    else:
        llm = LlamaCppLLM(args.api_base)
    orch = Orchestrator(policy=policy, llm=llm, word_gap_s=args.word_gap_s,
                        verbose=not args.quiet)
    print(f"[orch] companion={args.companion} llm={args.llm} ({args.big_model})")
    orch.run_loop()


if __name__ == "__main__":
    main()