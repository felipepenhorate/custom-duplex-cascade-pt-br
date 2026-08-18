"""M3 - TTS service: streaming pt-BR speech with pocket-tts.

Receives msgpack `{"type":"Text","text":...}` (+ optional `{"type":"Eos"}`) and
streams back `{"type":"Audio","pcm":[...]}` frames at 24 kHz (80 ms / 1920
samples, matching the browser + bridge frame size), then `{"type":"Eos"}`.

Implementation (SPEC 7.2): pocket-tts `TTSModel.load_model(language="portuguese")`,
`generate_audio_stream(state, text)` yields 24 kHz chunks as decoded. Frames are
split into 1920-sample messages for protocol compatibility.

Usage:
  python services/tts_service.py --port 31608
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import msgpack
import numpy as np
import torch
import websockets

try:
    from pocket_tts import TTSModel
except ImportError:  # pragma: no cover
    print("[tts] run from the venv with pocket-tts installed")
    sys.exit(1)

FRAME_SAMPLES = 1920  # 80 ms @ 24k (browser frame size)


class TTSService:
    def __init__(self, language: str, device: str, quantize: bool, voice: str):
        self.language = language
        self.device = device
        self.quantize = quantize
        self.voice = voice
        self.model = TTSModel.load_model(language=language, quantize=quantize)
        self.model.to(device)
        self.model.eval()
        # voice-conditioned starting state (predefined catalog voice, no cloning)
        self.init_state = self.model.get_state_for_audio_prompt(voice)

    async def handle(self, ws: websockets.WebSocketServerProtocol) -> None:
        try:
            async for raw in ws:
                if isinstance(raw, str):
                    continue
                try:
                    msg = msgpack.unpackb(raw, raw=False)
                except Exception:
                    continue
                if not isinstance(msg, dict):
                    continue
                t = msg.get("type")
                if t == "Text":
                    text = str(msg.get("text", ""))
                    if not text.strip():
                        continue
                    await self.speak(ws, text)
                elif t == "Eos":
                    await ws.send(msgpack.packb({"type": "Eos"}, use_bin_type=True))
        except websockets.exceptions.ConnectionClosed:
            pass

    def _generate_frames(self, text: str) -> list[np.ndarray]:
        """Run pocket-tts generation (blocking, CPU/GPU-bound) in a worker
        thread so the event loop stays free for other connections."""
        gen = self.model.generate_audio_stream(self.init_state, text)
        frames: list[np.ndarray] = []
        for chunk in gen:
            if isinstance(chunk, torch.Tensor):
                chunk = chunk.detach().cpu().float().numpy()
            chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
            offset = 0
            while offset < chunk.size:
                frame = chunk[offset : offset + FRAME_SAMPLES]
                offset += FRAME_SAMPLES
                if frame.size:
                    frames.append(frame)
        return frames

    async def speak(self, ws, text: str) -> None:
        frames = await asyncio.to_thread(self._generate_frames, text)
        for frame in frames:
            pcm = [float(x) for x in frame]
            await ws.send(msgpack.packb({"type": "Audio", "pcm": pcm}, use_bin_type=True, use_single_float=True))
        # signal end of speech so the bridge can release its echo gate
        await ws.send(msgpack.packb({"type": "Eos"}, use_bin_type=True))
        print(f"[tts] spoke {len(frames)} frames ({len(frames) * FRAME_SAMPLES / 24000:.2f}s)", flush=True)


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=31608)
    p.add_argument("--language", default="portuguese",
                   help="pocket-tts language (portuguese, english, ...)")
    p.add_argument("--voice", default=None,
                   help="predefined voice (default: language default, e.g. rafael for pt)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quantize", action="store_true", help="quantize TTS for lower VRAM")
    args = p.parse_args()

    if not args.voice:
        from pocket_tts.default_parameters import get_default_voice_for_language
        args.voice = get_default_voice_for_language(args.language)
    svc = TTSService(args.language, args.device, args.quantize, args.voice)
    print(f"[tts] pocket-tts '{args.language}' voice={args.voice} ready on :{args.port} (24 kHz stream)", flush=True)
    async with websockets.serve(svc.handle, host="", port=args.port, max_size=8 << 20):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
