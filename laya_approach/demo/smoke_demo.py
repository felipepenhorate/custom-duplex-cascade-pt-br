#!/usr/bin/env python3
"""Small websocket smoke test for the Onda Travel demo.

Text turn::

    python -m laya_approach.demo.smoke_demo --text "I want a beach under $1,500"

Microphone path (requires soundfile)::

    python -m laya_approach.demo.smoke_demo --audio /path/to/24k_user.wav

The script prints Laya's decision, the spoken reply and the resulting catalog,
then exits as soon as the answer's audio ends.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import numpy as np
import websockets


async def _ready(ws) -> dict:
    while True:
        message = await ws.recv()
        if isinstance(message, str):
            payload = json.loads(message)
            if payload.get("type") == "ready":
                return payload
            if payload.get("type") == "error":
                raise SystemExit(payload.get("message", "server error"))


async def _drain_until_audio_end(ws, timeout: float = 40.0) -> None:
    while True:
        message = await asyncio.wait_for(ws.recv(), timeout=timeout)
        if isinstance(message, (bytes, bytearray)):
            continue
        payload = json.loads(message)
        kind = payload.get("type")
        if kind == "policy":
            print(f"  policy   {payload['tag']} ({payload['latency_ms']} ms)")
        elif kind == "user_partial":
            print(f"  partial  {payload['text']}")
        elif kind == "user_final":
            print(f"  final    {payload['text']}")
        elif kind == "assistant_text":
            print(f"  reply    {payload['text']}")
        elif kind == "catalog":
            active = {key: value for key, value in payload.get("filters", {}).items() if value}
            names = [item["name"] for item in payload.get("items", [])[:5]]
            print(f"  catalog  {payload['total']} trips  filters={active}  {names}")
        elif kind == "audio_start":
            print(f"  audio    start ({payload.get('language')})")
        elif kind == "audio_end":
            print(
                "  audio    end "
                f"({payload.get('audio_seconds', 0)} s of audio, "
                f"{payload.get('spoken_seconds', 0)} s to generate)"
            )
            await ws.send(json.dumps({"type": "playback_done", "seq": payload.get("seq")}))
            return
        elif kind == "error":
            print(f"  error    {payload.get('message')}")
            return


async def run(url: str, text: str | None, audio: str | None) -> None:
    async with websockets.connect(url, max_size=8 << 20) as ws:
        ready = await _ready(ws)
        print(f"ready: {ready['language']}  capabilities={ready['capabilities']}")
        if audio:
            import soundfile as sf

            pcm, sample_rate = sf.read(audio, dtype="float32")
            if sample_rate != 24_000:
                raise SystemExit(f"expected 24 kHz audio, got {sample_rate}")
            await ws.send(json.dumps({"type": "start"}))
            started = time.monotonic()
            for offset in range(0, len(pcm), 480):
                frame = pcm[offset : offset + 480].astype("<f4", copy=False).tobytes()
                await ws.send(frame)
                await asyncio.sleep(0.02)
            for _ in range(90):  # 1.8 s of silence to endpoint the turn
                await ws.send(np.zeros(480, dtype="<f4").tobytes())
                await asyncio.sleep(0.02)
            print(f"streamed {len(pcm) / 24_000:.2f} s in {time.monotonic() - started:.2f} s")
        else:
            if not text:
                raise SystemExit("pass --text or --audio")
            await ws.send(json.dumps({"type": "text", "text": text}))
            print(f"sent: {text}")
        await _drain_until_audio_end(ws)
        await ws.send(json.dumps({"type": "stop"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:7860/ws")
    parser.add_argument("--text")
    parser.add_argument("--audio", help="24 kHz mono wav to stream as the user")
    args = parser.parse_args()
    asyncio.run(run(args.url, args.text, args.audio))


if __name__ == "__main__":
    main()
