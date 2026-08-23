#!/usr/bin/env python3
"""DuplexCascade-Distill bridge server (M4).

Bridges the browser demo to the three backends:
  * STT: services/stt_service.py  (faster-whisper, ws://127.0.0.1:31607)
  * LLM: llama.cpp OpenAI-compatible server (http://127.0.0.1:8080/v1),
        serving the self-distilled DuplexCascade-Distill GGUF.
  * TTS: services/tts_service.py   (pocket-tts, ws://127.0.0.1:31608)

Browser protocol (unchanged from DuplexCascade):
  browser --(24k f32 PCM, 1920-sample frames)--> bridge --> STT
  STT --Word--> bridge --> browser {type:"user_asr", text}
  bridge --> llama.cpp (streaming chat) --text deltas--> bridge
  bridge --> browser {type:"assistant_text"} / {type:"assistant_special"}
  bridge --> TTS (Text/Eos) --Audio 24k--> bridge --> browser (binary f32 PCM)

The LLM is reached over llama.cpp's OpenAI-compatible API. The duplex special
tokens are emitted by the fine-tuned model as single tokens; streaming deltas
are split on `<|...|>` boundaries to drive TTS + control.

This file is based on the custom-duplex-cascade-pt-br server (the most updated
line of the GUI), plus the distill adaptations:
  * SYSTEM_PROMPT = the DuplexCascade trigger prompt used to generate the teacher
    content (data/teacher_prompts.py, content-only variant) + an explicit
    turn-close instruction. The distill model's content is anchored to the frozen
    base under that prompt (a short generic prompt drifts it), and unlike the SFT
    baseline it does NOT spontaneously emit <|user is thinking|> to end a turn,
    so the instruction makes it close turns reliably.
  * labels/title -> DuplexCascade-Distill.
"""

from __future__ import annotations

import argparse
import asyncio
import http
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import msgpack  # type: ignore
import numpy as np
import websockets
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
WEB_ROOT = PROJECT_ROOT / "web"

# pt-BR duplex special tokens (paper §3.2). The fine-tuned model emits these
# as single tokens; <|system backchannel|> (beta) is accepted but unused in base.
SPECIAL_TOKENS = [
    "<|no voice|>",
    "<|user is speaking|>",
    "<|user finish speaking|>",
    "<|user is thinking|>",
    "<|user interruption|>",
    "<|user backchannel|>",
    "<|system backchannel|>",
]

# tokens that gate TTS: finish speaking -> speak; thinking -> pause; interruption -> stop
FINISH_TOKENS = {"<|user finish speaking|>"}
THINK_TOKENS = {"<|user is thinking|>"}
INTERRUPT_TOKENS = {"<|user interruption|>", "<|user is speaking|>"}
SPEAK_BACKCHANNEL = {"<|system backchannel|>"}

# Distill adaptation: the teacher (frozen base) generated the assistant content
# under this trigger system prompt (data/teacher_prompts.py, content-only
# variant). Using the same prompt at inference keeps the distill model's content
# on-distribution (a short generic prompt drifts it, e.g. stray foreign tokens /
# markdown). The final sentence is a second distill adaptation: the distill model
# does NOT spontaneously emit <|user is thinking|> to end a turn in free
# generation (the SFT baseline does); the explicit instruction makes it close
# turns reliably.
SYSTEM_PROMPT = (
    "Você é um assistente de voz full-duplex do sistema DuplexCascade. "
    "Numa conversa duplex, sistema e usuário falam em micro-turnos curtos e "
    "naturais, com sobreposição, como numa conversa de voz real. "
    "Você gerencia a tomada de turno com tags especiais: "
    "<|user is speaking|> (usuário está falando), "
    "<|user finish speaking|> (usuário terminou de falar), "
    "<|user is thinking|> (usuário está pensando), "
    "<|user interruption|> (usuário interrompeu), "
    "<|user backchannel|> (usuário faz um backchannel), "
    "<|system backchannel|> (backchannel do sistema) e "
    "<|no voice|> (silêncio). "
    "Responda de forma natural e completa, em português brasileiro coloquial, "
    "como uma fala falada em voz alta. Responda apenas com a fala — "
    "não imprima nenhuma tag especial. Não se repita: diga cada informação "
    "uma única vez. "
    "Quando terminar de responder, emita <|user is thinking|> para indicar "
    "que seu turno terminou."
)


def _build_tts_ws_url(base_ws_url: str) -> str:
    return base_ws_url


async def _ws_connect(url: str, headers: dict) -> websockets.WebSocketClientProtocol:
    sig = __import__("inspect").signature(websockets.connect)
    kwargs = {}
    if headers:
        if "additional_headers" in sig.parameters:
            kwargs["additional_headers"] = headers
        elif "extra_headers" in sig.parameters:
            kwargs["extra_headers"] = headers
    return await websockets.connect(url, **kwargs)


def _split_special(text: str) -> list[tuple[str, str]]:
    """Split text into (kind, payload) pairs: kind in {special, text}."""
    parts: list[tuple[str, str]] = []
    pattern = re.compile(r"(<\|[^|>]+\|>)")
    for tok in pattern.split(text):
        if not tok:
            continue
        if tok.startswith("<|") and tok.endswith("|>"):
            parts.append(("special", tok))
        else:
            parts.append(("text", tok))
    return parts


# Short pt-BR conversational fillers that the duplex model tends to emit
# between content chunks (e.g. "Opa!", "Entendi!", "Tranquilo!"). Consecutive
# repeats of these are stripped so the final answer reads/sounds clean.
_FILLER_RE = re.compile(r"^(opa|entendi|tranquilo|ok|okay|tá|é|ah|hmm|e aí|claro|bom|legal|boa|isso|beleza|show|perfeito|então|bora|vamos lá|né|pois é|ah tá|aham|uhum|sei|vai dar certo|que legal|força aí)([.!…]*)$", re.IGNORECASE)


def _is_filler(text: str) -> bool:
    return bool(_FILLER_RE.match(text.strip()))


# A single word/char repeated >= 4 times at the END of a micro-turn is a
# degenerate sampling loop (llama.cpp without repeat_penalty on a truncated
# answer can collapse to "ő ő ő ő..." or "é é é é"). Strip it so it never
# reaches TTS/history; the model usually follows with a proper close.
def _strip_degenerate_tail(text: str) -> str:
    """Remove degenerate-repetition tails:
      * a trailing run of identical 1-2 char tokens (e.g. 'ő ő ő ő', 'é é é'), and
      * a single foreign-letter token the model emits as a close-substitute
        ('...sentir. ő') - never real pt-BR speech.
    Conservative on purpose: emoji, punctuation and real words are kept.
    """
    words = text.split()
    # 1) trailing identical short-token run
    while len(words) >= 2 and words[-1] == words[-2] and len(words[-1]) <= 2:
        words.pop()
    # 2) single short foreign-letter token following a sentence end
    if len(words) >= 2 and len(words[-1]) <= 2 and words[-1].isalpha() and not words[-1].isascii():
        rest = " ".join(words[:-1])
        if rest and not rest[-1].isalnum() and not rest[-1].isspace():
            words = words[:-1]
    out = " ".join(words).strip()
    if out and out != text:
        print(f"[bridge] _strip_degenerate_tail: dropped {len(text.split()) - len(out.split())} tokens", flush=True)
    return out


class BridgeServer:
    def __init__(
        self,
        *,
        stt_ws: str,
        tts_ws: str,
        llm_api_base: str,
        llm_model: str,
        overlap_window_s: float,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repeat_penalty: float,
        thinking: bool,
    ) -> None:
        self.doc_root = str(WEB_ROOT.resolve())
        self.stt_ws = stt_ws
        self.tts_ws = tts_ws
        self.llm_api_base = llm_api_base.rstrip("/")
        self.llm_model = llm_model
        self.overlap_window_s = float(overlap_window_s)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.repeat_penalty = float(repeat_penalty)
        self.enable_thinking = bool(thinking)

    def filter_period_for_tts(self, text: str) -> str:
        """pt-BR-safe: strip only sentence-final periods (keep abbreviations
        and decimals like 1.5, 10h30)."""
        return re.sub(r"(?<=\S)\.(?=\s|$)", "", text)

    # ---------------------------------------------------------------- static
    async def process_request(self, connection, request):
        """Serve static web files for plain HTTP (websockets >= 17 API:
        process_request(connection, request) -> Response | None)."""
        from websockets.http11 import Response
        from websockets.datastructures import Headers

        def _resp(status, reason, mime, body: bytes):
            return Response(status, reason, Headers({"Content-Type": mime}), body)

        # detect websocket upgrade: the web client always sends the key header
        if "sec-websocket-key" in request.headers:
            return None  # let websockets continue the handshake
        path = request.path
        if path == "/":
            path = "/index.html"
        rel = path.lstrip("/")
        fs_path = os.path.join(self.doc_root, rel)
        if not os.path.isfile(fs_path):
            return _resp(http.HTTPStatus.NOT_FOUND, "Not Found",
                         "text/plain; charset=utf-8", b"Not found")
        ext = os.path.splitext(fs_path)[1].lower()
        mime = "application/octet-stream"
        if ext == ".html":
            mime = "text/html; charset=utf-8"
        elif ext == ".css":
            mime = "text/css; charset=utf-8"
        elif ext == ".js":
            mime = "application/javascript; charset=utf-8"
        elif ext == ".json":
            mime = "application/json; charset=utf-8"
        elif ext in (".png",):
            mime = "image/png"
        elif ext in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        elif ext == ".svg":
            mime = "image/svg+xml"
        with open(fs_path, "rb") as f:
            data = f.read()
        return _resp(http.HTTPStatus.OK, "OK", mime, data)

    async def run(self, port: int) -> None:
        print(f"[DuplexCascade-Distill] listening on 0.0.0.0:{port}", flush=True)
        async with websockets.serve(
            self.handle_connection,
            host="",
            port=port,
            max_size=8 << 20,
            max_queue=32,
            process_request=self.process_request,
        ):
            await asyncio.Future()

    async def handle_connection(self, ws: websockets.WebSocketServerProtocol) -> None:
        try:
            await self.handle_connection_impl(ws)
        except websockets.exceptions.ConnectionClosedError:
            return
        except Exception as e:
            print(f"[bridge] session error: {type(e).__name__}: {e}", flush=True)
            return

    # ---------------------------------------------------------------- LLM
    def build_prompt(self, history: list[dict], prime: str = "") -> str:
        """Build the ChatML duplex micro-turn prompt (matches training format).

        Assistant turns carry the duplex special tokens; the model is fine-tuned
        to continue this token stream. Uses /v1/completions (raw) because the
        chat template mangles <|...|> tokens in assistant history.
        `prime` (e.g. "<|user finish speaking|>") is appended right after the
        assistant header so the model continues with content instead of
        deciding the turn is still open."""
        parts: list[str] = []
        for m in history:
            role = m.get("role", "user")
            content = str(m.get("content", ""))
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            else:
                parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n" + prime)
        return "".join(parts)

    async def llm_stream(self, history: list[dict], prime: str = "") -> tuple[list[str], str]:
        """Generate the next assistant micro-turn via llama.cpp (raw completions).

        Async (httpx) so the call never blocks the event loop — essential for
        the live browser session (blocking urllib would stall send_json/ASR).
        Returns (text deltas, finish_reason); `prime` seeds the assistant turn
        start. `finish_reason` is "stop" when the model ended the turn itself
        (emitted <|im_end|>/EOS) vs "length" when max_tokens cut it off."""
        import httpx

        prompt = self.build_prompt(history, prime=prime)
        payload = {
            "prompt": prompt,
            "stream": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "max_tokens": self.max_new_tokens,
        }
        deltas: list[str] = []
        finish_reason: str = ""
        timeout = httpx.Timeout(60.0, connect=5.0)
        print(f"[bridge] llm_stream: entering, api={self.llm_api_base}", flush=True)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", f"{self.llm_api_base}/completions", json=payload
                ) as resp:
                    print(f"[bridge] llm_stream: connected, status={resp.status_code}", flush=True)
                    async for raw in resp.aiter_lines():
                        if not raw.startswith("data:"):
                            continue
                        data = raw[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except Exception:
                            continue
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        fr = choices[0].get("finish_reason")
                        if fr:
                            finish_reason = str(fr)
                        text = choices[0].get("text") or ""
                        if text:
                            deltas.append(text)
        except Exception:
            deltas = []

        if not deltas:
            # fallback: non-streaming (stream mode may drop special-token-only
            # responses on some llama.cpp builds)
            payload["stream"] = False
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    r = await client.post(
                        f"{self.llm_api_base}/completions", json=payload
                    )
                obj = r.json()
                text = (obj.get("choices") or [{}])[0].get("text") or ""
                if text:
                    deltas.append(text)
                fr = (obj.get("choices") or [{}])[0].get("finish_reason")
                if fr:
                    finish_reason = str(fr)
            except Exception:
                pass
        return deltas, finish_reason

    # ---------------------------------------------------------------- main
    async def handle_connection_impl(self, ws: websockets.WebSocketServerProtocol) -> None:
        send_lock = asyncio.Lock()

        async def send_json(obj: Dict[str, Any]) -> None:
            async with send_lock:
                await ws.send(json.dumps(obj, ensure_ascii=False))

        async def send_pcm_f32le(pcm: np.ndarray) -> None:
            if pcm is None or pcm.size == 0:
                return
            pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
            async with send_lock:
                await ws.send(pcm.astype("<f4", copy=False).tobytes())

        async def send_audio_control(action: str, reason: str = "") -> None:
            payload: Dict[str, Any] = {"type": "audio_control", "action": str(action)}
            if reason:
                payload["reason"] = str(reason)
            await send_json(payload)

        audio_q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=64)
        ctrl_q: asyncio.Queue[str] = asyncio.Queue(maxsize=8)

        async def browser_rx_task() -> None:
            n_frames = 0
            last_log = time.time()
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
                        pcm = np.frombuffer(msg, dtype="<f4").astype(np.float32, copy=False)
                        if pcm.size == 0:
                            continue
                        n_frames += 1
                        if n_frames == 1:
                            print(f"[bridge] rx: first audio frame from browser ({pcm.size} samples)", flush=True)
                        elif time.time() - last_log >= 2.0:
                            print(f"[bridge] rx: {n_frames} frames so far", flush=True)
                            last_log = time.time()
                        if audio_q.full():
                            try:
                                audio_q.get_nowait()
                            except Exception:
                                pass
                        await audio_q.put(pcm)
            except websockets.exceptions.ConnectionClosed:
                print(f"[bridge] rx: browser disconnected after {n_frames} frames", flush=True)
                return

        print("[bridge] browser session connected", flush=True)
        browser_rx = asyncio.create_task(browser_rx_task())

        first_session = True
        while True:
            # drain leftover audio between sessions (Reset), but NOT on the first
            # iteration — the user's very first frames must not be discarded
            if not first_session:
                while not audio_q.empty():
                    try:
                        audio_q.get_nowait()
                    except Exception:
                        break
            first_session = False

            first_text_received_event = asyncio.Event()
            asr_buffer_words: list[str] = []
            asr_buffer_lock = asyncio.Lock()
            # serializes writes to the shared STT websocket. stt_sender_task
            # streams audio AND can reconnect/swap stt_ref["ws"], while
            # llm_tick_task sends Eos turn boundaries. Without this lock the
            # two tasks write to the same websocket concurrently and the send
            # can hang forever (the bridge then stops answering).
            stt_send_lock = asyncio.Lock()

            # conversation history (message format for the duplex micro-turn stream)
            history: list[dict] = [{
                "role": "system",
                "content": SYSTEM_PROMPT,
            }]

            # ---- TTS client ----
            class TTSClient:
                def __init__(self_obj):
                    self_obj.ws = None
                    self_obj.rx_task = None
                    self_obj.lock = asyncio.Lock()

                async def connect(self_obj):
                    await self_obj.close_internal()
                    try:
                        self_obj.ws = await _ws_connect(self.tts_ws, {})
                        if self_obj.ws:
                            self_obj.rx_task = asyncio.create_task(self_obj.rx_loop(self_obj.ws))
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

                async def send_text(self_obj, text, send_eos=False):
                    async with self_obj.lock:
                        if self_obj.ws is None:
                            await self_obj.connect()
                        if self_obj.ws:
                            try:
                                await self_obj.ws.send(msgpack.packb({"type": "Text", "text": text}, use_bin_type=True))
                                if send_eos:
                                    await self_obj.ws.send(msgpack.packb({"type": "Eos"}, use_bin_type=True))
                            except Exception:
                                await self_obj.close_internal()

                async def rx_loop(self_obj, ws_ref):
                    """Forward TTS audio frames to the browser (24k f32 PCM)."""
                    nonlocal assistant_playing_until
                    try:
                        async for msg_bytes in ws_ref:
                            try:
                                msg = msgpack.unpackb(msg_bytes, raw=False)
                                if msg.get("type") == "Audio":
                                    pcm = np.asarray(msg.get("pcm", []), dtype=np.float32)
                                    if pcm.size:
                                        # echo gate: keep blocking mic->STT during and
                                        # just after playback (self-decaying window)
                                        assistant_playing_until = time.time() + 1.0
                                        await send_pcm_f32le(pcm)
                                elif msg.get("type") == "Eos":
                                    # normal completion: keep the echo gate up
                                    # briefly but do NOT send "stop" (the browser
                                    # must be allowed to drain its play queue);
                                    # "stop" is reserved for interruptions
                                    assistant_playing_until = time.time() + 0.6
                            except Exception:
                                continue
                    except Exception:
                        pass

            tts_client = TTSClient()
            await tts_client.connect()

            # echo gating: while the assistant's audio is playing, the open mic
            # picks it up; drop those frames so they don't re-enter ASR/LLM.
            assistant_playing_until = 0.0

            # ---- STT ----
            # shared holder so sender and receiver always use the same socket,
            # even when the sender reconnects on a stale connection
            stt_ref: dict[str, Any] = {"ws": await _ws_connect(self.stt_ws, {})}

            async def stt_sender_task() -> None:
                nonlocal assistant_playing_until
                n_fwd = 0
                last_log = time.time()
                try:
                    while True:
                        pcm = await audio_q.get()
                        if pcm is None:
                            return
                        n_fwd += 1
                        if n_fwd == 1:
                            print("[bridge] stt_send: first frame forwarded to STT", flush=True)
                        elif time.time() - last_log >= 2.0:
                            print(f"[bridge] stt_send: {n_fwd} frames forwarded", flush=True)
                            last_log = time.time()
                        # echo gate: drop mic audio while the assistant's TTS was
                        # recently playing (self-decaying time window)
                        if time.time() < assistant_playing_until:
                            audio_q.task_done()
                            continue
                        offset = 0
                        while offset < pcm.size:
                            chunk = pcm[offset : offset + 1920]
                            offset += 1920
                            if chunk.size == 0:
                                continue
                            stt_msg = {"type": "Audio", "pcm": [float(x) for x in chunk]}
                            packed = msgpack.packb(stt_msg, use_bin_type=True, use_single_float=True)
                            try:
                                async with stt_send_lock:
                                    await stt_ref["ws"].send(packed)
                            except Exception:
                                # stale connection: reconnect and resend
                                try:
                                    async with stt_send_lock:
                                        await stt_ref["ws"].close()
                                except Exception:
                                    pass
                                async with stt_send_lock:
                                    stt_ref["ws"] = await _ws_connect(self.stt_ws, {})
                                    await stt_ref["ws"].send(packed)
                        audio_q.task_done()
                except websockets.exceptions.ConnectionClosed:
                    return
                except Exception:
                    return

            async def stt_receiver_task() -> None:
                try:
                    while True:
                        ws = stt_ref["ws"]  # follow reconnects made by the sender
                        try:
                            msg_bytes = await ws.recv()
                        except websockets.exceptions.ConnectionClosed:
                            # connection may have been swapped by the sender
                            if stt_ref["ws"] is ws:
                                print("[bridge] stt receiver conn closed", flush=True)
                                return
                            await asyncio.sleep(0.1)
                            continue
                        except Exception:
                            # don't spin on a dead socket: back off and retry
                            await asyncio.sleep(0.1)
                            continue
                        msg = msgpack.unpackb(msg_bytes, raw=False)
                        if not isinstance(msg, dict):
                            continue
                        t = msg.get("type")
                        if t == "PartialText":
                            # live feedback while speaking: show it (replace) but do
                            # NOT feed the LLM buffers — the final Words are authoritative
                            partial = str(msg.get("text", "")).strip()
                            if partial:
                                await send_json({"type": "user_asr", "text": partial})
                        elif t == "Word":
                            w = str(msg.get("text", "")).strip()
                            if w:
                                async with asr_buffer_lock:
                                    asr_buffer_words.append(w)
                                    current_asr_text = " ".join(asr_buffer_words)
                                first_text_received_event.set()
                                print(f"[bridge] stt_rx: Word {w!r} -> asr_text {current_asr_text!r}", flush=True)
                                await send_json({"type": "user_asr", "text": current_asr_text})
                        elif t == "Eos":
                            # turn boundary from the tick: reset this turn's words
                            async with asr_buffer_lock:
                                asr_buffer_words.clear()
                        elif t == "UtteranceEnd":
                            # STT finalized the utterance (silence detected): answer
                            print("[bridge] stt_rx: UtteranceEnd -> triggering answer", flush=True)
                            utterance_end_event.set()
                except Exception as e:
                    print(f"[bridge] stt receiver err: {type(e).__name__}: {e}", flush=True)
                    return

            # ---- LLM tick ----
            # set when the STT finalizes an utterance (silence detected there)
            utterance_end_event = asyncio.Event()

            async def llm_tick_task() -> None:
                await first_text_received_event.wait()
                # accumulate the user's utterance, then answer when the STT
                # signals the user has finished (utterance_end_event)
                user_words: list[str] = []

                try:
                    while True:
                        # pull any new ASR words (incremental streaming for the UI)
                        async with asr_buffer_lock:
                            if asr_buffer_words:
                                user_words.extend(asr_buffer_words)
                                asr_buffer_words.clear()
                        await asyncio.sleep(0.05)

                        # answer when the STT finalized the utterance
                        if user_words and utterance_end_event.is_set():
                            utterance_end_event.clear()
                            # drain any words that arrived while we slept (the STT
                            # sends all Words BEFORE UtteranceEnd, so this catches
                            # the complete transcript, not a stale prefix)
                            async with asr_buffer_lock:
                                if asr_buffer_words:
                                    user_words.extend(asr_buffer_words)
                                    asr_buffer_words.clear()
                            user_text = " ".join(user_words)
                            print(f"[bridge] tick: ANSWERING with user_text {user_text!r}", flush=True)

                            # explicit turn boundary for STT: reset its buffer so the
                            # next utterance starts fresh (avoids cross-turn bleed).
                            # Non-blocking on purpose - a stuck STT websocket send
                            # must never block the answer path.
                            try:
                                async with stt_send_lock:
                                    await asyncio.wait_for(
                                        stt_ref["ws"].send(msgpack.packb({"type": "Eos"}, use_bin_type=True)),
                                        timeout=2.0,
                                    )
                            except Exception:
                                pass
                            print("[bridge] tick: sent Eos to STT", flush=True)

                            history.append({"role": "user", "content": user_text})
                            # prime the assistant turn: the user finished speaking, so
                            # the model continues with <|user finish speaking|> + content
                            llm_t0 = time.time()
                            print("[bridge] tick: calling llm_stream", flush=True)

                            # Duplex micro-turn loop: the fine-tuned model was trained
                            # on CHUNKED assistant turns - each chunk is a separate
                            # assistant message separated by <|no voice|> user turns
                            # (and the turn ends with <|user is thinking|>). A single
                            # completion yields only the FIRST chunk because <|im_end|>
                            # is the EOS token. To get the FULL response we must loop:
                            #   call LLM -> collect text chunk -> if the model hasn't
                            #   signaled turn-complete, feed <|no voice|> as the user
                            #   turn and call again.
                            max_micro_turns = 10
                            micro_history = list(history)  # working copy for the loop
                            assistant_parts: list[str] = []
                            turn_done = False
                            last_sent_text: str = ""
                            for _ in range(max_micro_turns):
                                # only the FIRST micro-turn is primed with
                                # <|user finish speaking|> (the user just spoke).
                                # Subsequent chunks continue after a <|no voice|>
                                # user turn - re-priming with the finish token on
                                # every turn confuses the model (it thinks the
                                # user interrupted to speak again) and breaks the
                                # natural content->thinking transition.
                                prime = "<|user finish speaking|>" if _ == 0 else ""
                                deltas, finish_reason = await asyncio.wait_for(
                                    self.llm_stream(micro_history, prime=prime), timeout=30.0
                                )
                                chunk = "".join(deltas)
                                print(f"[bridge] tick: LLM micro-turn ({time.time()-llm_t0:.2f}s, fr={finish_reason or '?'}): {chunk!r}", flush=True)

                                # notify UI of the finish cue (it was in the prompt)
                                if _ == 0:
                                    await send_json({"type": "assistant_special", "text": "<|user finish speaking|>"})

                                # strip the trailing EOS so history stays clean
                                chunk_body = chunk.split("<|im_end|>")[0]

                                # strip a degenerate-repetition tail ("ő ő ő ő")
                                # so the loop never reaches TTS/history; if the
                                # WHOLE chunk was repetition, break the turn.
                                stripped = _strip_degenerate_tail(chunk_body)
                                if not stripped.strip():
                                    print(f"[bridge] tick: degenerate-repetition-only chunk, ending turn: {chunk_body!r}", flush=True)
                                    turn_done = True
                                    break
                                if stripped != chunk_body:
                                    chunk_body = stripped

                                # strip consecutive repeat fillers ("Opa! Opa!" or
                                # "Entendi!" repeated right after the same filler) so
                                # the answer doesn't stutter. Compare against the last
                                # non-filler text AND the previous chunk.
                                chunk_text = "".join(
                                    p for kind, p in _split_special(chunk_body) if kind == "text"
                                ).strip()
                                if chunk_text and _is_filler(chunk_text) and _is_filler(last_sent_text):
                                    print(f"[bridge] tick: skipping repeated filler {chunk_text!r}", flush=True)
                                    # still feed it to history so generation context stays
                                    # consistent, but don't send to UI/TTS
                                    if chunk_body.strip():
                                        assistant_parts.append(chunk_body)
                                        micro_history.append({"role": "assistant", "content": chunk_body})
                                    # continue the loop (a filler is not turn-end)
                                    if not any(t in chunk for t in ("<|user is thinking|>", "<|user is speaking|>", "<|user interruption|>")):
                                        micro_history.append({"role": "user", "content": "<|no voice|>"})
                                    continue

                                # process text + special tokens in this chunk
                                pending_text = ""
                                for kind, part in _split_special(chunk_body):
                                    if kind == "text":
                                        pending_text += part
                                    else:
                                        if pending_text:
                                            clean = pending_text.strip()
                                            if clean:
                                                await send_json({"type": "assistant_text", "text": pending_text})
                                                await tts_client.send_text(self.filter_period_for_tts(pending_text))
                                                last_sent_text = clean
                                            pending_text = ""
                                        await self.handle_special(part, send_json, tts_client, send_audio_control)
                                if pending_text:
                                    clean = pending_text.strip()
                                    if clean:
                                        print(f"[bridge] tick: sending assistant_text {pending_text!r}", flush=True)
                                        await send_json({"type": "assistant_text", "text": pending_text})
                                        await tts_client.send_text(self.filter_period_for_tts(pending_text))
                                        last_sent_text = clean

                                if chunk_body.strip():
                                    assistant_parts.append(chunk_body)
                                    micro_history.append({"role": "assistant", "content": chunk_body})

                                # the turn ends when the model emits a turn-taking
                                # special (<|user is thinking|> = thinking pairs close a
                                # completed turn; interruption/speaking = user took over)
                                if any(t in chunk for t in ("<|user is thinking|>", "<|user is speaking|>", "<|user interruption|>")):
                                    turn_done = True
                                    break
                                # A micro-turn that ended NATURALLY (the model emitted
                                # <|im_end|>/EOS, finish_reason="stop") IS a completed
                                # turn even without an explicit thinking tag - do NOT
                                # re-prompt it with <|no voice|>, or the model rambles
                                # on ("vou esperar sua próxima palavra..." fillers) and
                                # pollutes the history with degenerate tails. Only a
                                # max_tokens cut-off ("length") means the answer was
                                # truncated and should be continued.
                                if finish_reason == "stop":
                                    turn_done = True
                                    break
                                # otherwise keep the duplex loop going: user stays
                                # silent (<|no voice|>) so the model continues its turn
                                micro_history.append({"role": "user", "content": "<|no voice|>"})

                            assistant_text = " ".join(p for p in assistant_parts if p.strip())
                            print(f"[bridge] tick: LLM returned {len(assistant_text)} chars in "
                                  f"{time.time() - llm_t0:.2f}s (done={turn_done})", flush=True)

                            if assistant_text.strip():
                                history.append({"role": "assistant", "content": assistant_text})
                                # keep the prompt within the 8192-token context: drop the
                                # OLDEST full turn pairs (but always keep the system msg at
                                # the front). Unbounded history grows past the window and the
                                # model degenerates into header-imitation ("assistant
                                # assistant assistant") on long sessions.
                                MAX_HISTORY_MSGS = 30  # ~15 user+assistant turns
                                if len(history) > MAX_HISTORY_MSGS:
                                    # mutate in place (never reassign `history`):
                                    # reassigning inside llm_tick_task makes Python
                                    # treat `history` as a local and the first
                                    # answer then fails with UnboundLocalError.
                                    # Keep index 0 (system) + the last 29 msgs.
                                    del history[1 : len(history) - (MAX_HISTORY_MSGS - 1)]
                                print(f"[bridge] tick: turn complete, history now {len(history)} msgs", flush=True)

                            # turn complete: clear both buffers so the next
                            # utterance starts a fresh turn (multi-turn)
                            async with asr_buffer_lock:
                                asr_buffer_words.clear()
                            user_words.clear()

                except asyncio.CancelledError:
                    return
                except Exception:
                    traceback.print_exc()
                    return

            stt_sender = asyncio.create_task(stt_sender_task())
            stt_receiver = asyncio.create_task(stt_receiver_task())
            llm_tick = asyncio.create_task(llm_tick_task())

            reset_or_done: Optional[str] = None
            try:
                while True:
                    if browser_rx.done():
                        reset_or_done = "done"
                        break
                    try:
                        reset_or_done = await asyncio.wait_for(ctrl_q.get(), timeout=0.2)
                        break
                    except asyncio.TimeoutError:
                        continue
            finally:
                for t in (stt_sender, stt_receiver, llm_tick):
                    t.cancel()
                await tts_client.close_internal()
                try:
                    await stt_ref["ws"].close()
                except Exception:
                    pass
                await asyncio.gather(stt_sender, stt_receiver, llm_tick, return_exceptions=True)

            if reset_or_done == "reset":
                try:
                    await send_json({"type": "user_asr", "text": ""})
                    await send_json({"type": "assistant_text", "text": ""})
                    await send_json({"type": "assistant_special", "text": ""})
                except Exception:
                    pass
                continue
            break

        try:
            browser_rx.cancel()
            await asyncio.gather(browser_rx, return_exceptions=True)
        except Exception:
            pass

    async def handle_special(self, token: str, send_json, tts_client, send_audio_control) -> None:
        """React to a duplex special token: notify the UI and steer TTS."""
        if token not in SPECIAL_TOKENS:
            return
        await send_json({"type": "assistant_special", "text": token})
        if token in FINISH_TOKENS:
            # user finished speaking -> finalize TTS of the accumulated response
            await tts_client.send_text("", send_eos=True)
        elif token in THINK_TOKENS or token in INTERRUPT_TOKENS:
            # user thinking / interrupted -> stop current playback
            await send_audio_control("stop", reason=token)


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=31606)
    p.add_argument("--stt-ws", type=str, default="ws://127.0.0.1:31607")
    p.add_argument("--tts-ws", type=str, default="ws://127.0.0.1:31608")
    p.add_argument("--llm-api-base", type=str, default="http://127.0.0.1:8080/v1")
    p.add_argument("--llm-model", type=str, default=None,
                   help="model id served by llama.cpp (default: auto-detect)")
    p.add_argument("--overlap-window-s", type=float, default=1.2,
                   help="silence (s) before completing the user turn")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="per micro-turn token budget. Must be big enough for a "
                        "substantive answer to COMPLETE (and emit the turn-close "
                        "tag) in one micro-turn; 96 truncates real answers and the "
                        "no-voice continuation then degenerates into token loops.")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repeat-penalty", type=float, default=1.15,
                   help="llama.cpp repeat penalty: suppresses degenerate-repetition "
                        "loops (e.g. the model repeating a filler token 'X X X X...')")
    p.add_argument("--thinking", action="store_true",
                   help="enable thinking mode (fine-tuned model is trained without it)")
    return p.parse_args()


def _auto_detect_model(api_base: str) -> str:
    with urllib.request.urlopen(f"{api_base.rstrip('/')}/models", timeout=15) as r:
        data = json.loads(r.read().decode("utf-8"))
    ids = [m.get("id", "") for m in data.get("data", [])]
    if not ids:
        raise RuntimeError(f"no model served at {api_base}/models")
    return ids[0]


def main() -> None:
    args = get_args()
    if not args.llm_model:
        args.llm_model = _auto_detect_model(args.llm_api_base)
    print(f"[DuplexCascade-Distill] LLM backend: {args.llm_model} @ {args.llm_api_base}", flush=True)

    server = BridgeServer(
        stt_ws=args.stt_ws,
        tts_ws=args.tts_ws,
        llm_api_base=args.llm_api_base,
        llm_model=args.llm_model,
        overlap_window_s=args.overlap_window_s,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        thinking=args.thinking,
    )
    asyncio.run(server.run(args.port))


if __name__ == "__main__":
    main()