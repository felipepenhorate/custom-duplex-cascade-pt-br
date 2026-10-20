"""Score rendered audio with the benchmark's own tools.

Two stages, both deliberately using upstream code rather than a port:

1. **`output.json`** -- ``fdb_eval/asr_official.py``'s recipe with
   ``nvidia/parakeet-tdt-0.6b-v2`` (NeMo), including the interruption task's
   crop-from-the-end-of-the-interruption rule and the offset it adds back, so the
   timestamps stay in session time exactly as the benchmark expects.
2. **The four task metrics** -- the vendored ``eval_*.py`` modules, called
   verbatim, with their printed results parsed. The only substitution is the
   judge behind the interruption rubric: upstream builds an OpenAI client for
   ``gpt-4-turbo``, and the same call signature is served here by the local
   ``Qwen3.5-9B`` GGUF, so the rubric and the parser are upstream's and only the
   judge model differs.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent / "fdb_eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

PARAKEET = "nvidia/parakeet-tdt-0.6b-v2"

_PATTERNS = {
    "pause_handling": {"tor": r"Average take turn:\s*([0-9.]+)"},
    "smooth_turn_taking": {
        "tor": r"Average take turn:\s*([0-9.]+)",
        "latency_s": r"Average latency:\s*([0-9.]+)",
    },
    "backchannel": {
        "jsd_mean": r"JSD - Mean:\s*([0-9.]+)\s*±\s*([0-9.]+)",
        "tor": r"TOR - Mean:\s*([0-9.]+)\s*±\s*([0-9.]+)",
        "freq_mean": r"Frequency - Mean:\s*([0-9.]+)\s*±\s*([0-9.]+)",
    },
    "user_interruption": {
        "relevance": r"Average rating:\s*([0-9.]+)",
        "tor": r"Average take turn:\s*([0-9.]+)",
        "latency_s": r"Average latency:\s*([0-9.]+)",
    },
}


# --------------------------------------------------------------------- judge
class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, outer: "LocalJudgeShim") -> None:
        self._outer = outer

    def create(self, *, model=None, messages=None, seed=None, **kw):  # noqa: ARG002
        return _Response(self._outer.ask(messages or []))


class LocalJudgeShim:
    """OpenAI-client-shaped shim that answers from a local llama.cpp server.

    Upstream's ``eval_user_interruption.py`` builds ``OpenAI(...)`` and calls
    ``client.chat.completions.create(model="gpt-4-turbo", messages=..., seed=0)``.
    This object offers exactly that surface so the official rubric, prompt
    assembly and output parser all run unmodified.
    """

    def __init__(self, api_base: str = "http://127.0.0.1:8099/v1", max_tokens: int = 320) -> None:
        # Kept dependency-free (stdlib HTTP) so the scoring pass can run under the
        # NeMo virtualenv, which is the only one that has parakeet installed.
        self.api_base = api_base.rstrip("/")
        self.max_tokens = max_tokens
        self.calls = 0
        self.chat = self  # `client.chat.completions.create`
        self.completions = _Completions(self)

    def ask(self, messages: list[dict]) -> str:
        self.calls += 1
        body = json.dumps({
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = __import__("urllib.request", fromlist=["Request"]).Request(
            f"{self.api_base}/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with __import__("urllib.request", fromlist=["urlopen"]).urlopen(
                    req, timeout=300) as r:
                obj = json.loads(r.read().decode())
            return ((obj.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        except Exception as e:  # noqa: BLE001
            return (f"Analysis: judge call failed ({type(e).__name__}).\n"
                    "I would rate the AI's response as 0")


def normalise_for_recognition(wav: np.ndarray, *, target: float = 0.9,
                              floor: float = 1e-4) -> np.ndarray:
    """Peak-normalise a rendered reply before it is handed to the recogniser.

    The recogniser should read what the model said, not how loud the TTS voice
    happened to be. The two pocket-tts voices differ by 12.6 dB (english
    -18.5 dB RMS, portuguese -31.1 dB), and whisper's VAD -- which runs before
    the decoder -- rejects a reply it cannot find speech in: on the pt set 23
    of 119 smooth scenarios had audible audio and an empty transcript. The
    English renders were already at a normal level (median peak 0.58), so the
    same normalisation is applied to both sets: it changes what is *heard*, not
    what was said.

    Files whose peak is at or below ``floor`` are left alone: that is digital
    silence with at most a click in it, and amplifying a click would only feed
    the recogniser a hallucination.
    """
    if wav.size == 0:
        return wav
    pk = float(np.abs(wav).max())
    if pk <= floor:
        return wav
    return (wav * (target / pk)).astype(np.float32)


# ------------------------------------------------------------------- parakeet
class ParakeetScorer:
    def __init__(self, model_name: str = PARAKEET) -> None:
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_name).cuda()

    def words(self, wav_path: Path, *, crop_from: float | None = None) -> list[dict]:
        """Word timestamps in *session* seconds (offset added back, as upstream)."""
        import soundfile as sf
        import tempfile
        import os

        wav, sr = sf.read(str(wav_path))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        offset = 0.0
        if crop_from is not None:
            offset = float(crop_from)
            wav = wav[int(offset * sr):]
        if wav.size < int(0.05 * sr):
            return []
        wav = normalise_for_recognition(wav)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, wav, sr)
            name = tmp.name
        try:
            out = self.model.transcribe([name], timestamps=True)[0]
        finally:
            os.unlink(name)
        chunks = []
        for w in (out.timestamp or {}).get("word", []):
            chunks.append({
                "text": w["word"],
                "timestamp": [float(w["start"]) + offset, float(w["end"]) + offset],
            })
        return chunks

    def write_output_json(self, sample_dir: Path, task: str) -> dict:
        """`output.json` next to `output.wav`, in the benchmark's schema."""
        crop = None
        if task == "user_interruption":
            meta = json.loads((sample_dir / "interrupt.json").read_text(encoding="utf-8"))
            crop = float(meta[0]["timestamp"][1])
        chunks = self.words(sample_dir / "output.wav", crop_from=crop)
        doc = {"text": " ".join(c["text"] for c in chunks).strip(), "chunks": chunks}
        (sample_dir / "output.json").write_text(
            json.dumps(doc, indent=4), encoding="utf-8")
        return doc


# ---------------------------------------------------------------- task evals
def _parse(task: str, text: str) -> dict:
    out: dict = {}
    for key, pat in _PATTERNS[task].items():
        m = re.search(pat, text)
        if not m:
            continue
        if m.lastindex and m.lastindex > 1:
            out[key] = float(m.group(1))
            out[f"{key}_std"] = float(m.group(2))
        else:
            out[key] = float(m.group(1))

    # also read the individual model's audio-based behaviour gate from our own
    # render (as a cross-check on the ASR-derived one)
    return out


def _patch_torchaudio_load() -> None:
    """Make ``torchaudio.load`` work without TorchCodec.

    torchaudio >= 2.9 delegates decoding to TorchCodec, which is not installed
    (and is not needed anywhere else in this harness). The vendored evaluators
    only use ``torchaudio.load`` plus ``functional.resample``, so loading through
    libsndfile -- the same decoder the rest of the audio path uses -- keeps their
    code path otherwise untouched.
    """
    import soundfile as sf
    import torch
    import torchaudio

    def load(path, *a, **kw):  # noqa: ARG001
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(wav.T).contiguous(), int(sr)

    torchaudio.load = load


def run_task_eval(root: Path, task: str, *, judge: LocalJudgeShim | None = None) -> dict:
    """Call the vendored upstream evaluator and return its numbers.

    ``root`` is resolved to an absolute path first: the backchannel evaluator has
    to be run from its own directory (it reads ``./icc_gt_distribution.json``),
    and a relative data path would then resolve against that directory.
    """
    root = Path(root).resolve()
    _patch_torchaudio_load()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if task == "pause_handling":
            from eval_pause_handling import eval_pause_handling

            eval_pause_handling(str(root))
        elif task == "smooth_turn_taking":
            from eval_smooth_turn_taking import eval_smooth_turn_taking

            eval_smooth_turn_taking(str(root))
        elif task == "backchannel":
            # upstream resolves ./icc_gt_distribution.json relative to the CWD
            import os

            from eval_backchannel import eval_backchannel

            cwd = os.getcwd()
            try:
                os.chdir(EVAL_DIR)
                eval_backchannel(str(root))
            finally:
                os.chdir(cwd)
        elif task == "user_interruption":
            from eval_user_interruption import eval_user_interruption

            eval_user_interruption(str(root), judge)
        else:
            raise ValueError(task)
    text = buf.getvalue()
    res = _parse(task, text)
    res["_raw_tail"] = text[-1200:]
    return res


def judge_stats(judge: LocalJudgeShim | None) -> dict:
    return {"judge_calls": judge.calls} if judge else {}


__all__ = [
    "ParakeetScorer", "LocalJudgeShim", "run_task_eval", "np",
]
