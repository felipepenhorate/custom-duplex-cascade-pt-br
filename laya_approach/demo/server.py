#!/usr/bin/env python3
"""Onda Travel — a voice-first, bilingual demo for the Laya companion.

The browser streams 24 kHz float32 microphone frames over ``/ws``.  A per-session
faster-whisper endpointing loop turns that stream into incremental ASR pieces;
Laya decides *when* the turn is over and *whether* speech is a barge-in.  Only
on ``user_finish_speaking`` does the Qwen 3 4B travel agent run.  The answer and
the new catalog filter state are sent back together, and Phonon streams the
spoken reply while the browser reshuffles the cards.

Run::

    source /path/to/.venv-unsloth/bin/activate
    python -m laya_approach.demo.server \
        --laya-model models/laya-companion \
        --llm-model models/qwen3-4b-instruct-2507 \
        --tts-model models/phonon \
        --voices-dir models/phonon-voices

Then open http://localhost:7860.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .audio import (
    SAMPLE_RATE,
    PhononTTS,
    STTEngine,
    StreamingASR,
    strip_for_tts,
)
from .travel_agent import (
    AssistantTurn,
    QwenTravelLLM,
    TravelAgent,
    catalogue_payload,
    client_items,
    empty_filters,
    normalize_filters,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = Path(__file__).resolve().parent / "web"
DEFAULT_LAYA = PROJECT_ROOT / "models" / "laya-companion"
DEFAULT_LLM = PROJECT_ROOT / "models" / "qwen3-4b-instruct-2507"
DEFAULT_TTS = PROJECT_ROOT / "models" / "phonon"
DEFAULT_VOICES = PROJECT_ROOT / "models" / "phonon-voices"

BACKCHANNELS = {
    "aham", "uhum", "hum", "hmm", "tá", "ta", "sim", "certo", "ok", "okay",
    "entendi", "beleza", "legal", "claro", "pois é", "pois e", "né", "ne",
    "yeah", "yes", "yep", "sure", "right", "got it", "cool", "nice", "thanks",
    "thank you", "mm-hmm", "uh-huh", "alright",
}


def _is_backchannel(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().strip().strip(".,!?…").split())
    return normalized in BACKCHANNELS


@dataclass
class RuntimeConfig:
    laya_model: str
    llm_model: str
    tts_model: str
    voices_dir: str | None
    device: str
    laya_stream: str
    stt_model: str
    stt_device: str
    stt_compute: str
    tts_voice: str
    endpoint_silence: float
    final_watchdog: float
    enable_policy: bool
    enable_llm: bool
    enable_tts: bool
    enable_stt: bool
    host: str = "0.0.0.0"
    port: int = 7860
    strict: bool = False


class Runtime:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        self.policy = None
        self.llm: QwenTravelLLM | None = None
        self.agent = TravelAgent(None)
        self.stt: STTEngine | None = None
        self.tts: PhononTTS | None = None
        self.ready = asyncio.Event()
        self.error: str | None = None
        self.warnings: list[str] = []
        self.policy_lock = threading.Lock()
        self.llm_lock = asyncio.Lock()
        self.tts_lock = asyncio.Lock()
        self.status = "loading"
        self.started_at = time.time()

    async def start(self) -> None:
        try:
            await asyncio.to_thread(self._load_blocking)
            self.status = "ready"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "error"
            print(f"[demo] startup failed: {self.error}", flush=True)
        finally:
            self.ready.set()

    def _attempt(self, label: str, loader, fallback=None):
        """Load one optional model; keep the rest of the demo alive on failure."""
        t0 = time.time()
        try:
            value = loader()
            print(f"[demo] {label} ready in {time.time() - t0:.1f}s", flush=True)
            return value
        except Exception as exc:
            message = f"{label} failed: {type(exc).__name__}: {exc}"
            print(f"[demo] {message}", flush=True)
            self.warnings.append(message)
            if self.cfg.strict:
                raise
            print(f"[demo] continuing without {label}; see /api/health warnings", flush=True)
            return fallback

    def _load_blocking(self) -> None:
        cfg = self.cfg
        if cfg.enable_policy or cfg.enable_llm:
            # Unsloth patches transformers in place and must be imported before
            # the Qwen/Transformers path is imported.
            try:
                import unsloth  # noqa: F401
            except Exception as exc:
                message = f"Unsloth import failed: {type(exc).__name__}: {exc}"
                print(f"[demo] {message}", flush=True)
                self.warnings.append(message)
                if cfg.strict:
                    raise

        if cfg.enable_policy:
            print(f"[demo] loading Laya policy {cfg.laya_model} ({cfg.laya_stream}) ...", flush=True)

            def load_laya():
                from laya_approach.policy import load as load_policy

                return load_policy(cfg.laya_model, device=cfg.device, stream=cfg.laya_stream)

            self.policy = self._attempt("Laya policy", load_laya, None)
        if cfg.enable_llm:
            print(f"[demo] loading Qwen travel LLM {cfg.llm_model} ...", flush=True)
            self.llm = self._attempt(
                "Qwen LLM",
                lambda: QwenTravelLLM(cfg.llm_model, device=cfg.device),
                None,
            )
        self.agent = TravelAgent(self.llm)
        if cfg.enable_stt:
            print(f"[demo] loading faster-whisper {cfg.stt_model} ...", flush=True)
            self.stt = self._attempt(
                "faster-whisper",
                lambda: STTEngine(cfg.stt_model, device=cfg.stt_device, compute_type=cfg.stt_compute),
                None,
            )
        if cfg.enable_tts:
            print(f"[demo] loading Phonon TTS {cfg.tts_model} ...", flush=True)
            self.tts = self._attempt(
                "Phonon TTS",
                lambda: PhononTTS(cfg.tts_model, voices_dir=cfg.voices_dir, voice=cfg.tts_voice),
                None,
            )
        if self.warnings:
            print(
                "[demo] started in degraded mode. /api/health lists the missing pieces; "
                "use --strict to make load failures fatal.",
                flush=True,
            )

    async def decide(self, items: list[dict[str, Any]]) -> tuple[str, dict[str, float], float]:
        """Return (tag, probabilities, latency_ms).  Falls back to a tiny heuristic."""
        if self.policy is None:
            event = items[-1]
            if event.get("system_speaking"):
                tag = "user_backchannel" if _is_backchannel(event.get("text", "")) else "user_interruption"
            elif event.get("kind") == "final" and event.get("ends_turn"):
                tag = "user_finish_speaking"
            elif event.get("kind") == "silence":
                tag = "user_finish_speaking"
            elif event.get("kind") == "partial":
                tag = "user_is_speaking"
            else:
                tag = "no_tag"
            return tag, {tag: 1.0}, 0.0

        def run() -> tuple[str, dict[str, float]]:
            with self.policy_lock:
                return self.policy(items)

        t0 = time.perf_counter()
        tag, probs = await asyncio.to_thread(run)
        latency = (time.perf_counter() - t0) * 1000.0
        event = items[-1]
        # The v7 model has a known partial-ack gap: a short acknowledgement
        # split across a partial can look like an interruption.  The runtime
        # guard is deliberately conservative here.
        if (
            tag == "user_interruption"
            and event.get("system_speaking")
            and _is_backchannel(event.get("text", ""))
        ):
            tag = "user_backchannel"
        return tag, probs, latency

    async def ask_agent(
        self,
        user_text: str,
        filters: dict[str, Any],
        history: list[dict[str, str]],
        language: str,
    ) -> AssistantTurn:
        async with self.llm_lock:
            return await asyncio.to_thread(
                self.agent.respond,
                user_text,
                filters=filters,
                history=history,
                language=language,
            )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session:
    def __init__(self, runtime: Runtime, ws: WebSocket, sid: str) -> None:
        self.runtime = runtime
        self.ws = ws
        self.id = sid
        self.language = "pt-BR"
        self.filters = empty_filters()
        self.history: list[dict[str, str]] = []
        self.laya_history: list[dict[str, Any]] = []
        self.asr: StreamingASR | None = None
        self.mic_on = False
        self.assistant_speaking = False
        self.thinking = False
        self.response_seq = 0
        self.pending_final: dict[str, Any] | None = None
        self.silence_task: asyncio.Task | None = None
        self.answer_task: asyncio.Task | None = None
        self.tts_cancel = threading.Event()
        self.send_lock = asyncio.Lock()
        self.closed = False
        self.last_user_text = ""
        self.current_partial = ""
        self.stt_language: str | None = None
        self.highlights: list[str] = []

    # ------------------------------------------------------------- transport
    async def send_json(self, payload: dict[str, Any]) -> None:
        if self.closed:
            return
        try:
            async with self.send_lock:
                await self.ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            self.closed = True

    async def send_audio(self, pcm: np.ndarray) -> None:
        if self.closed:
            return
        payload = np.asarray(pcm, dtype="<f4").reshape(-1).tobytes()
        try:
            async with self.send_lock:
                await self.ws.send_bytes(payload)
        except Exception:
            self.closed = True

    async def state(self, value: str, detail: str = "") -> None:
        await self.send_json({"type": "state", "state": value, "detail": detail})

    # --------------------------------------------------------------- startup
    async def ready(self) -> None:
        if self.runtime.error:
            await self.send_json(
                {"type": "error", "message": self.runtime.error, "fatal": True}
            )
            return
        payload = catalogue_payload(self.language)
        await self.send_json(
            {
                "type": "ready",
                "session_id": self.id,
                "language": self.language,
                "catalog": payload["items"],
                "filters": payload["filters"],
                "tags": payload["tags"],
                "regions": payload["regions"],
                "climates": payload["climates"],
                "paces": payload["paces"],
                "capabilities": {
                    "policy": self.runtime.policy is not None,
                    "llm": self.runtime.llm is not None,
                    "stt": self.runtime.stt is not None,
                    "tts": self.runtime.tts is not None,
                    "laya_stream": self.runtime.cfg.laya_stream,
                    "policy_model": Path(self.runtime.cfg.laya_model).name,
                    "llm_model": Path(self.runtime.cfg.llm_model).name,
                    "tts_model": Path(self.runtime.cfg.tts_model).name,
                },
                "warnings": self.runtime.warnings,
                "greeting": self._greeting(),
            }
        )
        await self.state("idle")

    def _greeting(self) -> str:
        if self.language == "pt-BR":
            return (
                "Oi! Eu sou a Laya, sua consultora de viagens. "
                "Me conte que tipo de viagem você imagina, e eu ajusto o catálogo "
                "enquanto conversamos."
            )
        return (
            "Hi! I'm Laya, your travel designer. "
            "Tell me what kind of trip you're imagining, and I'll shape the "
            "catalog while we talk."
        )

    # ------------------------------------------------------------------- mic
    async def start_mic(self) -> None:
        if self.runtime.stt is None:
            await self.send_json(
                {
                    "type": "error",
                    "message": "Speech recognition is disabled on this server. Use the text box.",
                    "fatal": False,
                }
            )
            return
        if self.mic_on:
            return
        self.asr = StreamingASR(
            self.runtime.stt,
            on_partial=self.on_partial,
            on_final=self.on_final,
            language=self._stt_language(),
            endpoint_silence=self.runtime.cfg.endpoint_silence,
        )
        self.asr.set_level_callback(self._on_level)
        self.asr.start()
        self.mic_on = True
        await self.state("listening")
        await self.send_json({"type": "mic", "on": True})

    async def stop_mic(self) -> None:
        self.mic_on = False
        if self.asr is not None:
            await self.asr.close()
            self.asr = None
        await self.send_json({"type": "mic", "on": False})
        await self.state("idle")

    def _stt_language(self) -> str | None:
        return self.stt_language

    def _on_level(self, level: float, speaking: bool) -> None:
        # The browser already draws its own local meter; this hook is useful
        # for server logs and future adaptive gain.
        _ = level, speaking

    async def feed_audio(self, pcm: np.ndarray) -> None:
        if not self.mic_on or self.asr is None:
            return
        self.asr.feed(pcm)

    # ------------------------------------------------------------- ASR events
    async def on_partial(self, piece: str, full_text: str) -> None:
        if not piece:
            return
        self.current_partial = full_text
        # New speech after a final means the user kept thinking out loud; the
        # watchdog must not answer the previous fragment.
        if self.pending_final is not None and not self.assistant_speaking and not self.thinking:
            self._cancel_silence()
            self.pending_final = None
        await self.send_json({"type": "user_partial", "text": full_text})
        if self.assistant_speaking or self.thinking:
            await self._on_user_speech_while_busy(piece, full_text, final=False)
            return
        await self._append_laya(
            {
                "role": "user",
                "text": piece,
                "kind": "partial",
                "ends_turn": False,
                "system_speaking": False,
            }
        )
        await self._decide_and_react(final=False)

    async def on_final(self, full_text: str, final_piece: str, detected: str | None) -> None:
        full_text = (full_text or "").strip()
        if not full_text:
            return
        self.current_partial = ""
        self.last_user_text = full_text
        if detected in ("pt", "en"):
            self.stt_language = detected
            if self.asr is not None:
                self.asr.language = detected
        print(
            f"[{self.id}] user ({detected or 'auto'}): {full_text!r}",
            flush=True,
        )
        await self.send_json(
            {
                "type": "user_final",
                "text": full_text,
                "language": "pt-BR" if detected == "pt" else ("en" if detected == "en" else self.language),
                "input": "voice",
            }
        )
        if detected in ("pt", "en"):
            detected_language = "pt-BR" if detected == "pt" else "en"
            if detected_language != self.language:
                # Whisper has already identified the spoken language; switch
                # the page before waiting for the content model.
                await self.set_language(detected_language)
        if self.assistant_speaking or self.thinking:
            await self._on_user_speech_while_busy(final_piece or full_text, full_text, final=True)
            return
        await self._start_final_turn(full_text, final_piece or full_text)

    async def _start_final_turn(self, full_text: str, piece: str) -> None:
        self._cancel_silence()
        self.pending_final = {"text": full_text, "piece": piece, "seq": self.response_seq + 1}
        await self._append_laya(
            {
                "role": "user",
                "text": piece,
                "kind": "final",
                "ends_turn": True,
                "system_speaking": False,
            }
        )
        tag, probs, latency = await self._decide_and_react(final=True)
        if tag == "user_finish_speaking":
            self._spawn_answer(full_text)
            return
        # Some turns need the post-final silence tick(s) the audio contract was
        # trained with.  The watchdog keeps a live demo from ever hanging.
        self.silence_task = asyncio.create_task(self._watch_final())

    async def _watch_final(self) -> None:
        pending = self.pending_final
        if not pending:
            return
        started = time.monotonic()
        ticks = 0
        try:
            while not self.closed and self.pending_final is pending:
                await asyncio.sleep(0.24)
                ticks += 1
                if self.pending_final is not pending:
                    return
                max_ticks = 2 if self.runtime.cfg.laya_stream == "audio" else 1
                if not self.assistant_speaking and not self.thinking and ticks <= max_ticks:
                    await self._append_laya(
                        {
                            "role": "user",
                            "text": "",
                            "kind": "silence",
                            "ends_turn": False,
                            "system_speaking": False,
                        }
                    )
                    tag, probs, latency = await self._decide_and_react(final=False)
                    if tag == "user_finish_speaking":
                        self._spawn_answer(pending["text"])
                        return
                elapsed = time.monotonic() - started
                # Keep feeding the contract's silence ticks, but only override
                # the model after the configured grace period.
                if ticks < max_ticks:
                    continue
                if elapsed >= self.runtime.cfg.final_watchdog:
                    print(
                        f"[{self.id}] watchdog answered after {elapsed:.2f}s "
                        f"(stream={self.runtime.cfg.laya_stream})",
                        flush=True,
                    )
                    await self.send_json(
                        {"type": "turn", "forced": True, "reason": "endpoint-watchdog"}
                    )
                    self._spawn_answer(pending["text"])
                    return
        except asyncio.CancelledError:
            return

    async def _on_user_speech_while_busy(self, piece: str, full_text: str, *, final: bool) -> None:
        """Barge-in / backchannel while the assistant is speaking or thinking."""
        event = {
            "role": "user",
            "text": piece,
            "kind": "final" if final else "partial",
            "ends_turn": bool(final),
            "system_speaking": True,
        }
        await self._append_laya(event)
        tag, probs, latency = await self._decide_and_react(final=final, emit=not final)
        if tag == "user_backchannel" or _is_backchannel(piece):
            if final:
                await self.send_json(
                    {"type": "user_backchannel", "text": full_text}
                )
            return
        if self.thinking and not final:
            # The model is still generating; do not let it start speaking over
            # the user.  Invalidate the pending answer and keep listening.
            await self._interrupt_and_listen(piece)
            return
        if tag == "user_interruption" or (final and len(full_text.split()) >= 2):
            await self._interrupt_and_listen(full_text if final else piece)
            if final:
                # The final event was already appended and judged above; answer
                # it directly instead of feeding a duplicate final to Laya.
                self._spawn_answer(full_text)

    async def _interrupt_and_listen(self, heard: str = "") -> None:
        self.response_seq += 1
        self.pending_final = None
        self._cancel_silence()
        self.thinking = False
        if self.assistant_speaking:
            print(f"[{self.id}] barge-in: {heard[:80]!r}", flush=True)
            self.tts_cancel.set()
            await self.send_json({"type": "audio_control", "action": "stop", "reason": "barge-in"})
            await self.send_json({"type": "assistant_interrupted"})
            self.assistant_speaking = False
        # The in-flight LLM result is discarded by the response_seq check; the
        # TTS stream stops on the cancel event.  Leaving the thread alone keeps
        # the GPU serialized for the next answer.
        await self.state("listening")

    # --------------------------------------------------------------- policy
    async def _append_laya(self, event: dict[str, Any]) -> None:
        self.laya_history.append(event)
        if len(self.laya_history) > 20:
            self.laya_history = self.laya_history[-20:]

    async def _decide_and_react(self, *, final: bool, emit: bool = True) -> tuple[str, dict[str, float], float]:
        event = self.laya_history[-1] if self.laya_history else {
            "role": "user", "text": "", "kind": "silence", "ends_turn": False
        }
        items = self.laya_history[-12:]
        tag, probs, latency = await self.runtime.decide(items)
        if emit:
            print(
                f"[{self.id}] policy {tag} ({latency:.0f} ms) "
                f"event={event.get('kind')} sys={bool(event.get('system_speaking'))} "
                f"text={event.get('text', '')!r}",
                flush=True,
            )
            await self.send_json(
                {
                    "type": "policy",
                    "tag": tag,
                    "probs": {key: round(float(value), 4) for key, value in probs.items()},
                    "latency_ms": round(latency, 1),
                    "kind": event.get("kind"),
                    "text": event.get("text", ""),
                    "system_speaking": bool(event.get("system_speaking")),
                    "stream": self.runtime.cfg.laya_stream,
                }
            )
        return tag, probs, latency

    # ----------------------------------------------------------------- turn
    def _spawn_answer(self, user_text: str) -> asyncio.Task:
        """Run the LLM/TTS answer off the ASR task.

        Keeping the answer in its own task is what makes the demo truly
        duplex: while Qwen thinks and Phonon speaks, the ASR loop keeps reading
        the microphone, so a barge-in can be detected immediately instead of
        after the generated audio has finished.
        """
        # Do not cancel an in-flight LLM thread here: asyncio cannot stop a
        # generation already running in a worker, and releasing the lock would
        # let two generations touch the GPU at once.  The response_seq check in
        # _answer_now makes the stale result a no-op.
        self.answer_task = asyncio.create_task(self._answer_now(user_text))
        return self.answer_task

    async def _answer_now(self, user_text: str) -> None:
        self._cancel_silence()
        self.pending_final = None
        self.response_seq += 1
        seq = self.response_seq
        self.thinking = True
        language_before = self.language
        await self.state("thinking")
        await self.send_json({"type": "assistant_thinking", "language": self.language})

        self.history.append({"role": "user", "content": user_text})
        history_for_prompt = self.history[:-1]

        try:
            turn = await self.runtime.ask_agent(
                user_text,
                self.filters,
                history_for_prompt,
                self.language,
            )
        except Exception as exc:
            print(f"[{self.id}] agent failed: {type(exc).__name__}: {exc}", flush=True)
            await self.send_json(
                {
                    "type": "error",
                    "message": "The travel agent hit a snag. Please try again.",
                    "fatal": False,
                }
            )
            self.thinking = False
            await self.state("listening" if self.mic_on else "idle")
            return

        if seq != self.response_seq or self.closed:
            return  # the user interrupted while the model was thinking

        self.thinking = False
        self.filters = normalize_filters(turn.filters)
        self.language = turn.language
        language_changed = turn.language != language_before
        self._sync_stt_language()
        self.history.append({"role": "assistant", "content": turn.reply})
        if len(self.history) > 16:
            self.history = self.history[-16:]

        print(
            f"[{self.id}] assistant intent={turn.intent} lang={turn.language} "
            f"catalog_changed={turn.catalog_changed} fallback={turn.fallback_used} "
            f"reply={turn.reply!r}",
            flush=True,
        )
        if language_changed:
            await self.send_json({"type": "language", "language": self.language})
        await self.send_json(
            {
                "type": "assistant_start",
                "language": self.language,
                "fallback": turn.fallback_used,
            }
        )
        await self.send_json(
            {
                "type": "assistant_text",
                "text": turn.reply,
                "language": self.language,
                "fallback": turn.fallback_used,
            }
        )
        if turn.catalog_changed:
            await self._send_catalog(turn)
        elif language_changed:
            # Small talk does not change the filters, but the cards still have
            # to switch language with the page.
            await self._send_catalog(turn, highlights=self.highlights)
        await self._append_laya({"role": "assistant", "text": turn.reply})
        await self.send_json(
            {
                "type": "turn",
                "forced": False,
                "empty": turn.empty,
                "relaxed": turn.relaxed,
                "fallback": turn.fallback_used,
                "intent": turn.intent,
                "catalog_changed": turn.catalog_changed,
                "language_changed": language_changed,
            }
        )
        await self._speak(turn, seq)

    async def _send_catalog(
        self,
        turn: AssistantTurn,
        *,
        highlights: list[str] | None = None,
    ) -> None:
        chosen = list(turn.highlights if highlights is None else highlights)
        valid_ids = {item["id"] for item in turn.items}
        chosen = [item_id for item_id in chosen if item_id in valid_ids]
        self.highlights = chosen
        await self.send_json(
            {
                "type": "catalog",
                "items": turn.items,
                "filters": turn.filters,
                "highlights": chosen,
                "total": turn.total,
                "language": turn.language,
                "relaxed": turn.relaxed,
            }
        )

    async def speak_welcome(self) -> None:
        """Speak the welcome greeting through the normal Phonon path.

        The browser requests this only after a user gesture, so autoplay
        policies do not silently discard it.  It is not a conversational turn:
        it does not touch history, filters or the catalog.
        """
        if self.assistant_speaking or self.thinking or self.closed:
            return
        greeting = self._greeting()
        print(f"[{self.id}] welcome ({self.language})", flush=True)
        await self.send_json(
            {"type": "welcome", "text": greeting, "language": self.language}
        )
        turn = AssistantTurn(
            language=self.language,
            reply=greeting,
            filters=self.filters,
            items=[],
            total=0,
            intent="welcome",
            catalog_changed=False,
        )
        await self._speak(turn, self.response_seq, kind="welcome")

    async def _speak(self, turn: AssistantTurn, seq: int, *, kind: str = "answer") -> None:
        reply = strip_for_tts(turn.reply)
        if not reply or self.runtime.tts is None:
            self.assistant_speaking = False
            await self.send_json(
                {
                    "type": "audio_start",
                    "kind": kind,
                    "language": turn.language,
                    "sample_rate": SAMPLE_RATE,
                    "enabled": False,
                    "seq": seq,
                }
            )
            await self.send_json(
                {
                    "type": "audio_end",
                    "kind": kind,
                    "enabled": False,
                    "audio_seconds": 0,
                    "spoken_seconds": 0,
                    "seq": seq,
                }
            )
            await self.state("listening" if self.mic_on else "idle")
            return

        self.tts_cancel = threading.Event()
        cancel = self.tts_cancel
        self.assistant_speaking = True
        await self.send_json(
            {
                "type": "audio_start",
                "kind": kind,
                "language": turn.language,
                "sample_rate": SAMPLE_RATE,
                "enabled": True,
                "format": "f32le",
                "seq": seq,
            }
        )
        await self.state("speaking")
        started = time.monotonic()
        sent_audio = False
        samples_sent = 0
        try:
            async with self.runtime.tts_lock:
                async for chunk in self.runtime.tts.stream(
                    reply, language=turn.language, cancel=cancel
                ):
                    if seq != self.response_seq or cancel.is_set() or self.closed:
                        break
                    sent_audio = True
                    samples_sent += int(chunk.size)
                    await self.send_audio(chunk)
        except Exception as exc:
            print(f"[{self.id}] TTS failed: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if seq == self.response_seq and not self.closed:
                await self.send_json(
                    {
                        "type": "audio_end",
                        "kind": kind,
                        "enabled": True,
                        "spoken_seconds": round(time.monotonic() - started, 2),
                        "audio_seconds": round(samples_sent / SAMPLE_RATE, 2),
                        "had_audio": sent_audio,
                        "seq": seq,
                    }
                )
                if not sent_audio:
                    self.assistant_speaking = False
                    await self.state("listening" if self.mic_on else "idle")
                else:
                    # The browser sends playback_done when its scheduled audio
                    # drains.  Keep a server-side fallback so a disconnected tab
                    # cannot leave system_speaking stuck on.
                    asyncio.create_task(self._release_speaking_after(seq, started, sent_audio))

    async def _release_speaking_after(self, seq: int, started: float, sent_audio: bool) -> None:
        # Generous upper bound: stream generation time + playback time.
        await asyncio.sleep(2.0 if not sent_audio else 30.0)
        if seq == self.response_seq and self.assistant_speaking:
            self.assistant_speaking = False
            await self.state("listening" if self.mic_on else "idle")

    async def playback_done(self, seq: int | None = None) -> None:
        if seq is not None and seq != self.response_seq:
            return
        if self.assistant_speaking:
            self.assistant_speaking = False
            await self.state("listening" if self.mic_on else "idle")

    # ------------------------------------------------------------- text turn
    async def text_turn(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if self.assistant_speaking or self.thinking:
            await self._interrupt_and_listen(text)
        self._cancel_silence()
        self.pending_final = None
        if self.asr is not None:
            self.asr.reset()
        self.last_user_text = text
        print(f"[{self.id}] user (typed): {text!r}", flush=True)
        await self.send_json({"type": "user_final", "text": text, "input": "text", "language": self.language})
        await self._append_laya(
            {
                "role": "user",
                "text": text,
                "kind": "final",
                "ends_turn": True,
                "system_speaking": False,
            }
        )
        tag, probs, latency = await self._decide_and_react(final=True)
        # Typed input is deliberately direct in a text-only environment, but
        # the policy decision is still emitted and visible in the debug panel.
        self._spawn_answer(text)

    # ----------------------------------------------------------------- reset
    async def reset(self) -> None:
        self.response_seq += 1
        self._cancel_silence()
        self.pending_final = None
        self.tts_cancel.set()
        self.assistant_speaking = False
        self.thinking = False
        self.filters = empty_filters()
        self.history.clear()
        self.laya_history.clear()
        self.highlights = []
        self.last_user_text = ""
        if self.asr is not None:
            self.asr.reset()
        await self.send_json({"type": "audio_control", "action": "stop", "reason": "reset"})
        await self.send_json({"type": "reset"})
        payload = catalogue_payload(self.language)
        await self.send_json(
            {
                "type": "catalog",
                "items": payload["items"],
                "filters": payload["filters"],
                "highlights": [],
                "total": len(payload["items"]),
                "language": self.language,
                "relaxed": False,
            }
        )
        await self.state("listening" if self.mic_on else "idle")

    async def set_filters(self, filters: dict[str, Any]) -> None:
        """Direct website filtering (chip removal / clear) without a voice turn."""
        from .travel_agent import search

        self.filters = normalize_filters(filters)
        result = search(self.filters)
        items = client_items(result.items, self.filters, self.language)
        active = bool(self.filters.get("tags")) or any(
            self.filters.get(key) is not None
            for key in ("max_price", "min_price", "min_nights", "max_nights", "region", "climate", "pace")
        )
        self.highlights = [item["id"] for item in items[:1]] if active else []
        await self.send_json(
            {
                "type": "catalog",
                "items": items,
                "filters": self.filters,
                "highlights": self.highlights,
                "total": result.total,
                "language": self.language,
                "relaxed": result.relaxed,
            }
        )

    async def set_language(self, language: str) -> None:
        self.language = "pt-BR" if str(language).lower().startswith("pt") else "en"
        self._sync_stt_language()
        await self.send_json({"type": "language", "language": self.language})
        # Re-localize the current result set without changing the filters.
        from .travel_agent import search

        result = search(self.filters)
        items = client_items(result.items, self.filters, self.language)
        valid = {item["id"] for item in items}
        highlights = [item_id for item_id in self.highlights if item_id in valid]
        await self.send_json(
            {
                "type": "catalog",
                "items": items,
                "filters": self.filters,
                "highlights": highlights,
                "total": result.total,
                "language": self.language,
                "relaxed": result.relaxed,
            }
        )

    def _sync_stt_language(self) -> None:
        # Use the agent's language as the next partial-decoding hint.  Finals
        # always auto-detect, so a user can still switch mid-conversation.
        self.stt_language = "pt" if self.language == "pt-BR" else "en"
        if self.asr is not None:
            self.asr.language = self.stt_language

    def _cancel_silence(self) -> None:
        if self.silence_task and not self.silence_task.done():
            self.silence_task.cancel()
        self.silence_task = None

    async def close(self) -> None:
        self.closed = True
        self.tts_cancel.set()
        self._cancel_silence()
        if self.asr is not None:
            await self.asr.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def build_app(cfg: RuntimeConfig) -> FastAPI:
    runtime = Runtime(cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = asyncio.create_task(runtime.start())
        yield
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    app = FastAPI(title="Onda Travel · Laya duplex demo", lifespan=lifespan)
    app.state.runtime = runtime

    @app.get("/api/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": runtime.status,
                "error": runtime.error,
                "warnings": runtime.warnings,
                "policy": runtime.policy is not None,
                "llm": runtime.llm is not None,
                "stt": runtime.stt is not None,
                "tts": runtime.tts is not None,
                "laya_stream": cfg.laya_stream,
                "models": {
                    "laya": cfg.laya_model,
                    "llm": cfg.llm_model,
                    "tts": cfg.tts_model,
                    "stt": cfg.stt_model,
                },
                "uptime_s": round(time.time() - runtime.started_at, 1),
            }
        )

    @app.websocket("/ws")
    async def websocket(ws: WebSocket) -> None:
        await ws.accept()
        sid = secrets.token_hex(4)
        session = Session(runtime, ws, sid)
        print(f"[{sid}] session connected", flush=True)
        try:
            await runtime.ready.wait()
            if runtime.error:
                await session.send_json({"type": "error", "message": runtime.error, "fatal": True})
                await ws.close()
                return
            await session.ready()
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    pcm = np.frombuffer(message["bytes"], dtype="<f4").astype(np.float32, copy=False)
                    await session.feed_audio(pcm)
                    continue
                raw = message.get("text")
                if raw is None:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = payload.get("type")
                if kind == "start":
                    await session.start_mic()
                elif kind == "stop":
                    await session.stop_mic()
                elif kind == "text":
                    await session.text_turn(str(payload.get("text", "")))
                elif kind == "language":
                    await session.set_language(str(payload.get("language", "en")))
                elif kind == "filters":
                    await session.set_filters(payload.get("filters") or {})
                elif kind == "reset":
                    await session.reset()
                elif kind == "welcome":
                    await session.speak_welcome()
                elif kind == "playback_done":
                    await session.playback_done(payload.get("seq"))
                elif kind == "stop_audio":
                    await session._interrupt_and_listen()
                elif kind == "ping":
                    await session.send_json({"type": "pong", "t": time.time()})
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            print(f"[{sid}] session error: {type(exc).__name__}: {exc}", flush=True)
            try:
                await session.send_json(
                    {"type": "error", "message": f"{type(exc).__name__}: {exc}", "fatal": False}
                )
            except Exception:
                pass
        finally:
            await session.close()
            print(f"[{sid}] session closed", flush=True)

    # Static files last, so /api and /ws win.
    app.mount("/", StaticFiles(directory=str(WEB_ROOT), html=True), name="web")
    return app


def parse_args(argv: list[str] | None = None) -> RuntimeConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument(
        "--laya-model",
        default=os.environ.get("LAYA_MODEL", str(DEFAULT_LAYA)),
        help="Local merged Laya checkpoint (from lumierenoir/laya-duplex-cascade-pt-br)",
    )
    p.add_argument(
        "--llm-model",
        default=os.environ.get("TRAVEL_LLM_MODEL", str(DEFAULT_LLM)),
        help="Qwen 4B instruct checkpoint or Hub repo id",
    )
    p.add_argument(
        "--tts-model",
        default=os.environ.get("TRAVEL_TTS_MODEL", str(DEFAULT_TTS)),
        help="Phonon checkpoint directory, or hf://gradium/phonon-40m-multilingual",
    )
    p.add_argument(
        "--voices-dir",
        default=os.environ.get("PHONON_VOICES", str(DEFAULT_VOICES)),
        help="Directory containing Phonon voices.jsonl and voices/*.safetensors",
    )
    p.add_argument("--tts-voice", default="Manuela-Lu")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--laya-stream",
        choices=("text", "audio"),
        default="text",
        help="Laya contract.  The demo uses text because STT emits a semantic final; "
        "audio is available for raw VAD-style streams.",
    )
    p.add_argument("--stt-model", default="small")
    p.add_argument("--stt-device", default="cuda")
    p.add_argument("--stt-compute", default="float16")
    p.add_argument("--endpoint-silence", type=float, default=0.85)
    p.add_argument("--final-watchdog", type=float, default=1.5)
    p.add_argument("--no-policy", action="store_true", help="Run without the Laya checkpoint.")
    p.add_argument("--no-llm", action="store_true", help="Use the deterministic bilingual fallback.")
    p.add_argument("--no-tts", action="store_true", help="Text-only mode; the browser still animates.")
    p.add_argument("--no-stt", action="store_true", help="Disable the microphone; typed chat only.")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail startup if a requested model cannot load (default: degrade to text/deterministic).",
    )
    p.add_argument("--mock", action="store_true", help="No models at all: text chat + deterministic search.")
    args = p.parse_args(argv)
    if args.mock:
        args.no_policy = args.no_llm = args.no_tts = args.no_stt = True
    return RuntimeConfig(
        laya_model=args.laya_model,
        llm_model=args.llm_model,
        tts_model=args.tts_model,
        voices_dir=args.voices_dir,
        device=args.device,
        laya_stream=args.laya_stream,
        stt_model=args.stt_model,
        stt_device=args.stt_device,
        stt_compute=args.stt_compute,
        tts_voice=args.tts_voice,
        endpoint_silence=args.endpoint_silence,
        final_watchdog=args.final_watchdog,
        enable_policy=not args.no_policy,
        enable_llm=not args.no_llm,
        enable_tts=not args.no_tts,
        enable_stt=not args.no_stt,
        host=args.host,
        port=args.port,
        strict=args.strict,
    )


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    app = build_app(cfg)
    import uvicorn

    print(f"[demo] Onda Travel on http://{cfg.host}:{cfg.port}", flush=True)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
