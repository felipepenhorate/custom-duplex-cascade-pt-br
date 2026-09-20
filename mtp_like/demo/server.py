#!/usr/bin/env python3
"""MTP-like browser demo bridge (M3).

Bridges the browser demo to the MTP-like duplex stack:

  browser (24 kHz f32 PCM)  <->  this server (:31606)  <->  STT (:31607)
                                                  |
                          companion (DuplexPolicy) + stock big model
                                                  |
                                            TTS (:31608)

The companion is the controller (SPEC 4): the bridge feeds STT word events
into the Orchestrator (policy/orchestrator.py), which queries the trained
companion on every event and drives the stock big model (TransformersLLM)
to produce answer sentences. Tags fire TTS/browser controls.

Backends:
  * STT: reuse sft/services/stt_service.py  (faster-whisper, pt, :31607)
  * TTS: reuse sft/services/tts_service.py   (pocket-tts, pt, :31608)
  * big model: in-process transformers (Qwen3-4B-Instruct-2507 default;
    any text LLM, incl. Qwen/Qwen3.5-4B via --big-model)
  * companion: the trained merged bf16 policy model (--companion)

Browser protocol (same as the sft bridge): binary f32 PCM frames in, binary
f32 PCM out, JSON {user_asr|assistant_text|assistant_special|audio_control},
strings Reset/Done.

Usage:
  python ../sft/services/stt_service.py --port 31607 --model small &
  python ../sft/services/tts_service.py --port 31608 &
  python demo/server.py --port 31606 --companion /mnt/f/duplex_cascade_runs/mtp_like/runs/final_v5/merged
  # open http://localhost:31606
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import http
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import msgpack  # type: ignore
import numpy as np
import websockets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policy.duplex_policy import DuplexPolicy  # noqa: E402
from policy.orchestrator import Orchestrator, TransformersLLM  # noqa: E402
from demo.bank_agent import BankAgent  # noqa: E402

WEB_ROOT = PROJECT_ROOT / "demo" / "web"

# tags that cut the assistant's TTS immediately (a REAL user barge-in).
# "user is speaking" / "user is thinking" from silence queries are NOT
# cut signals — cutting on them stops playback mid-answer (the timing bug).
CUT_TAGS = {"user_interruption"}

_SENTINEL = object()

# digits as spoken words: the companion's rule scenarios are trained on
# "um dois três" — real dictations come back from the STT as "1 2 3".
# Normalize before feeding so the runtime matches the training shape.
DIGIT_WORDS = {
    "0": "zero", "1": "um", "2": "dois", "3": "três", "4": "quatro",
    "5": "cinco", "6": "seis", "7": "sete", "8": "oito", "9": "nove",
}


def digit_words(text: str) -> str:
    """Replace every digit sequence with its spoken words ("123" ->
    "um dois três"), keeping punctuation; idempotent on already-normal
    text."""
    return re.sub(r"\d+", lambda m: " ".join(DIGIT_WORDS[d] for d in m.group()),
                  text)

# default prompt rule for the promptable controller (v6): the companion
# reads this system instruction and fires <|system take floor|> when the
# user dictates a letter inside the CPF. The credit-card ask is NOT here:
# the big model handles it in its answers (BIG_SYSTEM persona). The rule
# text is the exact training template.
DEFAULT_RULE = (
    "No CPF, apenas números são permitidos. Se o cliente disser uma letra "
    "em vez de um número, interrompa e avise que apenas números são "
    "aceitos."
)

# spoken INSTANTLY when <|system take floor|> fires (the big model is too
# slow to cut the user mid-speech; the interrupt needs sub-second latency)
DEFAULT_INTERRUPT = "Apenas números são aceitos no CPF."


def _is_emoji_only(text: str) -> bool:
    import unicodedata

    chars = [c for c in text if not c.isspace()]
    if not chars:
        return True
    return all(unicodedata.category(c) == "So" for c in chars)


async def _ws_connect(url: str):
    return await websockets.connect(url)


class BridgeServer:
    def __init__(
        self,
        *,
        stt_ws: str,
        tts_ws: str,
        companion_dir: str,
        big_model: str,
        barge_rms: float,
        barge_frames: int,
        rule: str = DEFAULT_RULE,
        bank_agent: bool = False,
        interrupt_phrase: str = DEFAULT_INTERRUPT,
    ) -> None:
        self.doc_root = str(WEB_ROOT.resolve())
        self.stt_ws = stt_ws
        self.tts_ws = tts_ws
        self.barge_rms = float(barge_rms)
        self.barge_frames = int(barge_frames)
        self.rule = rule
        self.bank_agent = bank_agent
        self.interrupt_phrase = interrupt_phrase
        print(f"[bridge] loading companion {companion_dir} ...", flush=True)
        self.policy = DuplexPolicy(companion_dir)
        print(f"[bridge] loading big model {big_model} ...", flush=True)
        self.llm = TransformersLLM(big_model)
        print("[bridge] models ready", flush=True)

    @staticmethod
    def filter_period_for_tts(text: str) -> str:
        import re

        return re.sub(r"(?<=\S)\.(?=\s|$)", "", text)

    # ---------------------------------------------------------------- static
    async def process_request(self, connection, request):
        from websockets.datastructures import Headers
        from websockets.http11 import Response

        def _resp(status, reason, mime, body: bytes):
            return Response(status, reason, Headers({"Content-Type": mime}), body)

        if "sec-websocket-key" in request.headers:
            return None  # websocket upgrade
        path = request.path
        if path == "/":
            path = "/index.html"
        fs_path = os.path.join(self.doc_root, path.lstrip("/"))
        if not os.path.isfile(fs_path):
            return _resp(http.HTTPStatus.NOT_FOUND, "Not Found",
                         "text/plain; charset=utf-8", b"Not found")
        ext = os.path.splitext(fs_path)[1].lower()
        mime = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".svg": "image/svg+xml",
        }.get(ext, "application/octet-stream")
        with open(fs_path, "rb") as f:
            return _resp(http.HTTPStatus.OK, "OK", mime, f.read())

    async def run(self, port: int) -> None:
        print(f"[bridge] listening on 0.0.0.0:{port}", flush=True)
        async with websockets.serve(
            self.handle_connection,
            host="",
            port=port,
            max_size=8 << 20,
            max_queue=32,
            process_request=self.process_request,
        ):
            await asyncio.Future()

    async def handle_connection(self, ws) -> None:
        try:
            await self.handle_connection_impl(ws)
        except websockets.exceptions.ConnectionClosedError:
            return
        except Exception as e:
            print(f"[bridge] session error: {type(e).__name__}: {e}", flush=True)
            return

    # ---------------------------------------------------------------- main
    async def handle_connection_impl(self, ws) -> None:
        send_lock = asyncio.Lock()

        async def send_json(obj: dict) -> None:
            async with send_lock:
                await ws.send(json.dumps(obj, ensure_ascii=False))

        async def send_pcm_f32le(pcm: np.ndarray) -> None:
            if pcm is None or pcm.size == 0:
                return
            async with send_lock:
                await ws.send(
                    np.asarray(pcm, dtype=np.float32).reshape(-1)
                    .astype("<f4", copy=False).tobytes()
                )

        async def send_audio_control(action: str, reason: str = "") -> None:
            payload: dict = {"type": "audio_control", "action": action}
            if reason:
                payload["reason"] = reason
            await send_json(payload)

        # ---- orchestrator (companion = controller) ----
        orch = Orchestrator(policy=self.policy, llm=self.llm, verbose=True,
                            rule=self.rule)
        orch_lock = asyncio.Lock()
        bank = BankAgent() if self.bank_agent else None
        # set when the companion took the floor mid-utterance: the final
        # UtteranceEnd transcript must NOT re-feed (it would re-trigger)
        interrupted_turn = {"v": False}

        async def orch_call(fn, *args):
            async with orch_lock:
                return await asyncio.to_thread(fn, *args)

        async def handle_system_floor(rule: str, user_text: str) -> None:
            """<|system take floor|>: the prompt rule fired — the SYSTEM
            interrupts the user MID-SPEECH. The response must be INSTANT
            (sub-second): the big model takes 2-4s to generate, which would
            never cut the user off. Speak the rule's correction phrase
            directly via the local TTS."""
            print(f"[bridge] system take_floor: {user_text!r}", flush=True)
            interrupted_turn["v"] = True
            await orch_call(orch.reset_turn)
            await send_json({"type": "user_commit"})
            await send_json({"type": "assistant_start"})
            await send_json({"type": "assistant_text",
                             "text": self.interrupt_phrase})
            await orch_call(orch.note_assistant, self.interrupt_phrase)
            await tts_client.send_text(
                self.filter_period_for_tts(self.interrupt_phrase))
            await send_audio_control("stop", reason="system-take-floor")

        async def handle_system_handover(rule: str, user_text: str) -> None:
            """<|system handover|>: the rule fired; the BIG MODEL decides
            what to do — respond now, or do nothing (NADA)."""
            print(f"[bridge] system handover: {user_text!r}", flush=True)
            prompt = (
                f"Regra do sistema acionada: {rule}. O cliente disse: "
                f"{user_text}. Se houver algo a fazer agora, responda "
                f"naturalmente; caso contrário responda apenas NADA."
            )
            msg = await asyncio.to_thread(self.llm.speak_response, prompt, 64)
            if not msg.strip() or "NADA" in msg.strip().upper():
                print("[bridge] handover: big model decided nothing",
                      flush=True)
                return
            interrupted_turn["v"] = True
            await orch_call(orch.reset_turn)
            await send_json({"type": "user_commit"})
            await send_json({"type": "assistant_start"})
            await send_json({"type": "assistant_text", "text": msg})
            await tts_client.send_text(self.filter_period_for_tts(msg))

        async def handle_bank_action(action: tuple) -> None:
            """The Banco Penha agent took the floor (system barge-in):
            abandon the user's turn, let the BIG model (bank persona)
            generate the response, speak it."""
            kind = action[0]
            print(f"[bridge] bank action: {kind}", flush=True)
            await orch_call(orch.reset_turn)
            await send_json({"type": "user_commit"})
            msg = BankAgent.message(action)
            try:
                generated = await asyncio.to_thread(
                    self.llm.speak_response, BankAgent.BANK_INSTRUCTIONS[kind]
                )
                if generated.strip():
                    msg = generated
            except Exception as e:
                print(f"[bridge] bank generation failed: {e}", flush=True)
            await send_json({"type": "assistant_start"})
            await send_json({"type": "assistant_text", "text": msg})
            if msg.strip():
                await tts_client.send_text(self.filter_period_for_tts(msg))
            if kind == "cpf_letters":
                await send_audio_control("stop", reason="bank-barge-in")

        # events emitted from worker threads (policy decisions, sentences)
        events: collections.deque = collections.deque()
        orch.emit_cb = lambda kind, msg: events.append((kind, msg))

        # ---- browser audio -> STT ----
        audio_q: asyncio.Queue = asyncio.Queue(maxsize=64)
        ctrl_q: asyncio.Queue = asyncio.Queue(maxsize=8)

        async def browser_rx_task() -> None:
            try:
                async for msg in ws:
                    if isinstance(msg, str):
                        s = msg.strip()
                        if s == "Reset":
                            await ctrl_q.put("reset")
                        elif s == "Done":
                            await ctrl_q.put("done")
                            return
                    else:
                        if not msg:
                            continue
                        pcm = np.frombuffer(msg, dtype="<f4").astype(
                            np.float32, copy=False)
                        if pcm.size == 0:
                            continue
                        await barge_check(pcm)
                        if audio_q.full():
                            try:
                                audio_q.get_nowait()
                            except Exception:
                                pass
                        await audio_q.put(pcm)
            except websockets.exceptions.ConnectionClosed:
                return

        browser_rx = asyncio.create_task(browser_rx_task())
        assistant_playing_until = 0.0

        # instant barge-in state: raw mic energy gated against the echo
        # level of the TTS the bridge itself is playing (no VAD model)
        barge_state = {"consecutive": 0, "echo_floor": 0.0}

        async def barge_check(pcm: np.ndarray) -> None:
            """Sustained mic energy while the assistant is speaking =
            the user took the floor: stop the answer + TTS immediately.

            Echo-safe: while the assistant plays, the mic is dominated by
            the echo; track its level with a fast-rise EMA (self-calibrating
            per session) and only trigger at 2x that floor — the user's
            voice over the assistant's audio exceeds it, the echo does not.
            """
            rms = float(np.sqrt((pcm**2).mean()))
            playing = (orch.state == "answering"
                       or time.time() < assistant_playing_until)
            if not playing:
                barge_state["consecutive"] = 0
                barge_state["echo_floor"] *= 0.99  # decay back to idle
                return
            barge_state["echo_floor"] = (
                0.95 * barge_state["echo_floor"] + 0.05 * rms
            )
            threshold = max(self.barge_rms, 1.5 * barge_state["echo_floor"])
            if rms > threshold:
                barge_state["consecutive"] += 1
            else:
                barge_state["consecutive"] = 0
            if barge_state["consecutive"] >= self.barge_frames:
                barge_state["consecutive"] = 0
                print("[bridge] barge-in detected (energy)", flush=True)
                await orch_call(orch.barge_in)
                await send_audio_control("stop", reason="barge-in")

        # ---- TTS client ----
        class TTSClient:
            def __init__(self_obj):
                self_obj.ws = None
                self_obj.rx_task = None
                self_obj.lock = asyncio.Lock()

            async def connect(self_obj):
                await self_obj.close_internal()
                try:
                    self_obj.ws = await _ws_connect(self.tts_ws)
                    if self_obj.ws:
                        self_obj.rx_task = asyncio.create_task(
                            self_obj.rx_loop(self_obj.ws))
                except Exception:
                    self_obj.ws = None
                    self_obj.rx_task = None

            async def close_internal(self_obj):
                if self_obj.rx_task:
                    self_obj.rx_task.cancel()
                    try:
                        await self_obj.rx_task
                    except Exception:
                        pass
                    self_obj.rx_task = None
                if self_obj.ws:
                    try:
                        await self_obj.ws.close()
                    except Exception:
                        pass
                    self_obj.ws = None

            async def send_text(self_obj, text):
                async with self_obj.lock:
                    if self_obj.ws is None:
                        await self_obj.connect()
                    if self_obj.ws:
                        try:
                            await self_obj.ws.send(msgpack.packb(
                                {"type": "Text", "text": text},
                                use_bin_type=True))
                            await self_obj.ws.send(msgpack.packb(
                                {"type": "Eos"}, use_bin_type=True))
                        except Exception:
                            await self_obj.close_internal()

            async def rx_loop(self_obj, ws_ref):
                nonlocal assistant_playing_until
                try:
                    async for msg_bytes in ws_ref:
                        try:
                            msg = msgpack.unpackb(msg_bytes, raw=False)
                            if msg.get("type") == "Audio":
                                pcm = np.asarray(msg.get("pcm", []),
                                                 dtype=np.float32)
                                if pcm.size:
                                    assistant_playing_until = time.time() + 1.0
                                    # track the played level: the mic echo
                                    # scales with it (barge-in threshold)
                                    barge_state["echo_floor"] = max(
                                        float(np.sqrt((pcm**2).mean())),
                                        0.8 * barge_state["echo_floor"],
                                    )
                                    await send_pcm_f32le(pcm)
                            elif msg.get("type") == "Eos":
                                assistant_playing_until = time.time() + 0.6
                        except Exception:
                            continue
                except Exception:
                    pass

        tts_client = TTSClient()
        await tts_client.connect()

        # ---- STT ----
        stt_ref: dict[str, Any] = {"ws": await _ws_connect(self.stt_ws)}
        stt_send_lock = asyncio.Lock()

        async def stt_sender_task() -> None:
            nonlocal assistant_playing_until
            try:
                while True:
                    pcm = await audio_q.get()
                    if pcm is None:
                        return
                    # echo gate: drop mic audio while the assistant's TTS plays
                    if time.time() < assistant_playing_until:
                        audio_q.task_done()
                        continue
                    offset = 0
                    while offset < pcm.size:
                        chunk = pcm[offset: offset + 1920]
                        offset += 1920
                        if chunk.size == 0:
                            continue
                        packed = msgpack.packb(
                            {"type": "Audio", "pcm": [float(x) for x in chunk]},
                            use_bin_type=True, use_single_float=True)
                        try:
                            async with stt_send_lock:
                                await stt_ref["ws"].send(packed)
                        except Exception:
                            try:
                                async with stt_send_lock:
                                    await stt_ref["ws"].close()
                            except Exception:
                                pass
                            async with stt_send_lock:
                                stt_ref["ws"] = await _ws_connect(self.stt_ws)
                                await stt_ref["ws"].send(packed)
                    audio_q.task_done()
            except (websockets.exceptions.ConnectionClosed, Exception):
                return

        async def stt_receiver_task() -> None:
            last_partial = ""
            utterance_words: list[str] = []
            bank_busy = False
            try:
                while True:
                    ws = stt_ref["ws"]
                    try:
                        msg_bytes = await ws.recv()
                    except websockets.exceptions.ConnectionClosed:
                        if stt_ref["ws"] is ws:
                            return
                        await asyncio.sleep(0.1)
                        continue
                    except Exception:
                        await asyncio.sleep(0.1)
                        continue
                    msg = msgpack.unpackb(msg_bytes, raw=False)
                    if not isinstance(msg, dict):
                        continue
                    t = msg.get("type")
                    if t == "PartialText":
                        partial = digit_words(str(msg.get("text", "")).strip())
                        if partial:
                            await send_json({"type": "user_asr", "text": partial})
                            if partial.startswith(last_partial):
                                new = partial[len(last_partial):].strip()
                                if new:
                                    # while the companion already took the
                                    # floor this utterance, keep the mic
                                    # flowing but don't feed (the letters in
                                    # the context would re-trigger the rule
                                    # for every following word)
                                    if not interrupted_turn["v"]:
                                        # Banco Penha agent watches EVERY word
                                        if bank is not None and not bank_busy:
                                            action = bank.on_words(new)
                                            if action:
                                                bank_busy = True
                                                await handle_bank_action(action)
                                        # feed new settled words to the
                                        # companion in BOTH states: while
                                        # answering it is barge-in, while
                                        # listening it is the prompt-rule
                                        # take_floor — the trained shape is
                                        # [u chunk, a tag] per word, and the
                                        # final UtteranceEnd transcript is
                                        # authoritative (it replaces
                                        # partial-fed chunks)
                                        if not bank_busy:
                                            await orch_call(orch.on_user_words, new)
                            last_partial = partial
                    elif t == "Word":
                        w = digit_words(str(msg.get("text", "")).strip())
                        if w:
                            utterance_words.append(w)
                            if bank is not None and not bank_busy:
                                action = bank.on_words(" ".join(utterance_words))
                                if action:
                                    bank_busy = True
                                    await handle_bank_action(action)
                            await send_json({"type": "user_asr",
                                             "text": " ".join(utterance_words)})
                    elif t == "UtteranceEnd":
                        print("[bridge] stt_rx: UtteranceEnd -> policy query",
                              flush=True)
                        last_partial = ""
                        final_text = " ".join(utterance_words).strip()
                        utterance_words = []
                        if bank_busy:
                            # the bank agent consumed this utterance
                            await orch_call(orch.reset_turn)
                            bank_busy = False
                        elif interrupted_turn["v"]:
                            # the companion already took the floor
                            # mid-utterance (take_floor / handover): the
                            # final transcript would just re-trigger the
                            # rule — drop it
                            interrupted_turn["v"] = False
                            await orch_call(orch.reset_turn)
                        else:
                            if final_text:
                                await orch_call(orch.on_user_utterance, final_text)
                            await orch_call(orch.on_silence)
                        if bank is not None:
                            bank.reset_utterance()
            except Exception:
                return

        # ---- answer driver: pulls sentences from the orchestrator ----
        # NOTE: must loop FOREVER — the first answer's generator ending must
        # NOT kill the driver, or every later answer is silently orphaned.
        async def answer_driver() -> None:
            saw_sentence = False
            try:
                while True:
                    gen = orch.answer_gen
                    if orch.state != "answering" or gen is None:
                        await asyncio.sleep(0.05)
                        continue
                    # next(gen, sentinel): StopIteration cannot propagate
                    # through asyncio.to_thread (PEP 479)
                    sent = await asyncio.to_thread(lambda: next(gen, _SENTINEL))
                    if sent is _SENTINEL:
                        # answer complete: close the state and run the
                        # post-answer silence query (thinking/wait), then
                        # keep serving future answers
                        if saw_sentence:
                            await send_json({"type": "assistant_commit"})
                            saw_sentence = False
                        await orch_call(orch._close_answer)
                        await orch_call(orch._query_and_apply, "silence")
                        continue
                    # drop stale sentences: the answer may have been closed
                    # while we were generating (user barged in)
                    if orch.state != "answering" or orch.answer_gen is not gen:
                        continue
                    await orch_call(orch._assistant_sentence, sent)
                    if sent.strip() and not _is_emoji_only(sent):
                        if not saw_sentence:
                            await send_json({"type": "assistant_start"})
                            saw_sentence = True
                        await send_json({"type": "assistant_text", "text": sent})
                        await tts_client.send_text(
                            self.filter_period_for_tts(sent))
            except Exception:
                return

        # ---- event dispatcher (policy tags / TTS controls) ----
        # NOTE: policy tags are NOT sent to the browser — the client's
        # assistant_special handler clears the assistant bubble, which
        # would wipe the transcript on every tag.
        async def event_dispatcher() -> None:
            while True:
                if not events:
                    await asyncio.sleep(0.02)
                    continue
                kind, msg = events.popleft()
                try:
                    if kind == "policy":
                        tag = msg.split()[0]
                        # companion state panel: forward EVERY decision
                        await send_json({"type": "companion_state",
                                         "text": msg})
                        if tag == "user_interruption":
                            await send_audio_control("stop", reason=tag)
                        elif tag == "system_backchannel":
                            await tts_client.send_text("aham")
                    elif kind == "tts":
                        await tts_client.send_text(msg)
                    elif kind == "system_take_floor":
                        rule, _, user_text = msg.partition(" | user: ")
                        await handle_system_floor(rule, user_text)
                    elif kind == "system_handover":
                        rule, _, user_text = msg.partition(" | user: ")
                        await handle_system_handover(rule, user_text)
                    elif kind == "state":
                        await send_json({"type": "companion_state",
                                         "text": msg})
                        if msg.startswith("answering"):
                            # the user's turn is committed: the browser
                            # closes its user bubble
                            await send_json({"type": "user_commit"})
                except Exception:
                    continue

        stt_sender = asyncio.create_task(stt_sender_task())
        stt_receiver = asyncio.create_task(stt_receiver_task())
        answer_driver_task = asyncio.create_task(answer_driver())
        events_task = asyncio.create_task(event_dispatcher())

        print("[bridge] browser session connected", flush=True)
        try:
            while True:
                if browser_rx.done():
                    break
                try:
                    ctrl = await asyncio.wait_for(ctrl_q.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                if ctrl == "reset":
                    await orch_call(orch.reset)
                    await send_json({"type": "user_asr", "text": ""})
                    await send_json({"type": "assistant_text", "text": ""})
                    await send_json({"type": "assistant_special", "text": ""})
                    continue
                break
        finally:
            for t in (stt_sender, stt_receiver, answer_driver_task, events_task):
                t.cancel()
            await tts_client.close_internal()
            try:
                await stt_ref["ws"].close()
            except Exception:
                pass
            try:
                await asyncio.gather(stt_sender, stt_receiver, answer_driver_task,
                                     events_task, return_exceptions=True)
            except Exception:
                pass
        try:
            browser_rx.cancel()
            await asyncio.gather(browser_rx, return_exceptions=True)
        except Exception:
            pass


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=31606)
    p.add_argument("--stt-ws", default="ws://127.0.0.1:31607")
    p.add_argument("--tts-ws", default="ws://127.0.0.1:31608")
    p.add_argument("--companion", default="/mnt/f/duplex_cascade_runs/mtp_like/runs/final_v5/merged")
    p.add_argument("--big-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--barge-rms", type=float, default=0.006,
                   help="mic RMS that counts as speech for instant barge-in")
    p.add_argument("--barge-frames", type=int, default=3,
                   help="consecutive frames (80 ms each) of speech to trigger")
    p.add_argument("--policy-rule", default=DEFAULT_RULE,
                   help="prompt rule for the companion (system instruction); "
                        "'' disables the promptable controller")
    p.add_argument("--bank-agent", action="store_true",
                   help="also run the deterministic Banco Penha agent "
                        "(redundant when --policy-rule covers the flow)")
    p.add_argument("--interrupt-phrase", default=DEFAULT_INTERRUPT,
                   help="phrase spoken instantly on <|system take floor|>")
    return p.parse_args()


def main() -> None:
    args = get_args()
    server = BridgeServer(
        stt_ws=args.stt_ws,
        tts_ws=args.tts_ws,
        companion_dir=args.companion,
        big_model=args.big_model,
        barge_rms=args.barge_rms,
        barge_frames=args.barge_frames,
        rule=args.policy_rule,
        bank_agent=args.bank_agent,
        interrupt_phrase=args.interrupt_phrase,
    )
    asyncio.run(server.run(args.port))


if __name__ == "__main__":
    main()