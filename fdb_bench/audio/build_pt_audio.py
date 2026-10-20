"""Build a pt-BR version of the Full-Duplex-Bench audio set.

The English WAVs are re-voiced in Brazilian Portuguese while keeping the
benchmark's *timeline* intact, so every annotation transfers verbatim: the pause
intervals, the annotated turn ends and the interruption spans all keep their
original seconds, and the pt-BR models can be measured on in-distribution
language with the same GT the English run used.

How it works, per scenario:

1. speech spans come from silero VAD on the original audio -- the same detector
   the benchmark's own scripts use;
2. each span's words are taken from the benchmark's word-level transcript when it
   ships one, and from parakeet-tdt-0.6b-v2 otherwise (``--stage asr``);
3. the spans' texts are translated in one Qwen3.5-9B call per scenario, numbered,
   with an instruction to keep the casual register, the fillers and the length;
4. each translated span is spoken by pocket-tts and **time-stretched to its
   original slot** (pitch-preserving phase vocoder), so speech and silence land
   where they did in English;
5. ``input.wav`` (24 kHz) is rebuilt, the annotations are copied unchanged, and a
   derived word-level ``transcription.json`` plus a per-scenario
   ``translation.json`` are written for audit.

Voice: cloning needs the gated ``kyutai/pocket-tts`` weights (accept the terms at
huggingface.co/kyutai/pocket-tts, then re-run with ``--voice-source clone`` to
condition on the original speaker's audio). Until then the default catalog voice
for the language is used, and every scenario records which voice it used.

Two stages, because parakeet lives in the NeMo virtualenv and pocket-tts does not:

    # once, for the subsets with no shipped transcript
    /home/penhfel/nemo_uv/bin/python -m fdb_bench.audio.build_pt_audio --stage asr
    # then
    /home/penhfel/unsloth_uv/bin/python -m fdb_bench.audio.build_pt_audio --stage build

Resume-safe: a scenario is skipped when its ``input.wav`` and
``translation.json`` both exist.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dataset import TASKS, iter_scenarios
from .tts import AssistantVoice

SR = 24000
EN_ROOT = Path("/mnt/f/fdb_data/v1.0")
OUT_ROOT = Path("/mnt/f/fdb_data/pt_v1.0")

TRANSLATE_SYSTEM = (
    "You translate spoken English into natural spoken Brazilian Portuguese for a "
    "speech corpus. Rules: output only the translation, never commentary; keep "
    "the casual register, the hesitations and the fillers (\"like\", \"I don't "
    "know\") as their natural pt-BR equivalents (\"tipo\", \"sei lá\"); keep the "
    "length similar to the English so the timing still fits; never add or drop "
    "information; do not translate proper names. The input is a numbered list and "
    "you must answer with exactly the same numbering, one line per segment, "
    "formatted `N: <translation>`."
)


@dataclass
class Segment:
    """One speech span of the original audio, with its text and its slot."""

    index: int
    start: float
    end: float
    text_en: str = ""
    text_pt: str = ""
    synth_s: float = 0.0
    fitted_s: float = 0.0
    stretch: float = 1.0       # rate applied (synth -> slot); 1.0 = untouched
    gain: float = 1.0          # level normalisation applied to the synthesis
    clamped: bool = False
    padded: bool = False      # silence added because the synthesis was short
    truncated: bool = False   # cut at the slot end rather than overflow a pause
    error: str | None = None

    @property
    def slot_s(self) -> float:
        return max(0.0, self.end - self.start)


# --------------------------------------------------------------------- helpers
def _resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    from scipy.signal import resample_poly

    if sr_in == sr_out:
        return x.astype(np.float32)
    g = np.gcd(int(sr_in), sr_out)
    return resample_poly(x, sr_out // g, int(sr_in) // g).astype(np.float32)


def load_wav(path: Path, sr_out: int = SR) -> tuple[np.ndarray, int]:
    import soundfile as sf

    w, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=1)
    return _resample(w, sr, sr_out), sr_out


def speech_spans(wav24: np.ndarray) -> list[tuple[float, float]]:
    from .stt import silero_spans

    return silero_spans(wav24)


def spans_to_segments(wav24: np.ndarray, words: list[dict]) -> list[Segment]:
    """One segment per silero span, carrying the words that fall inside it."""
    segs: list[Segment] = []
    for i, (s, e) in enumerate(speech_spans(wav24)):
        inside = [
            w for w in words
            if w.get("timestamp") and (w["timestamp"][0] + w["timestamp"][1]) / 2 >= s
            and (w["timestamp"][0] + w["timestamp"][1]) / 2 < e
        ]
        segs.append(Segment(index=i, start=float(s), end=float(e),
                            text_en=" ".join(w["text"] for w in inside).strip()))
    return [g for g in segs if g.text_en]


def qwen_chat(system: str, user: str, *, api_base: str, max_tokens: int = 1200) -> str:
    """One chat completion against the local llama-server."""
    import urllib.request

    body = json.dumps({
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"]


def load_cached_translations(cache_dir: Path, subset: str, sid: str):
    """The previous build's translations for one scenario, keyed by segment index.

    The cache stores the English source next to each translation, so a cache
    that does not match the audio is detected instead of silently producing a
    mismatched pair. Returns ``None`` when no cache is in use at all.
    """
    if cache_dir is None:
        return None
    f = Path(cache_dir) / subset / sid / "translation.json"
    if not f.is_file():
        return {}
    doc = json.loads(f.read_text(encoding="utf-8"))
    out = {}
    for sg in doc.get("segments", []):
        out[int(sg["index"])] = (sg.get("text_en", ""), sg.get("text_pt", ""))
    return out


def apply_cached_translations(segments: list[Segment], cached: dict) -> None:
    """Fill in ``text_pt`` from the cache, keeping only entries that still match."""
    for sg in segments:
        entry = cached.get(sg.index)
        if entry is None:
            continue
        en, pt = entry
        if en and sg.text_en and en.strip() != sg.text_en.strip():
            continue          # the audio changed under the cache: do not reuse it
        if pt:
            sg.text_pt = pt


_PT_MARKERS = set("ãõçáéíóúâêôãõüàèìòù")


def translate_interrupt_texts(doc, *, api_base: str) -> dict:
    """Put the interrupting turn and its context in the language of the audio.

    The relevance rubric quotes both verbatim (``Contextual user turn``,
    ``User interrupting turn``). On the re-voiced set the audio is Portuguese, so
    leaving them in English makes the relevance column measure a language
    mismatch rather than the reply.
    """
    for item in doc:
        for key in ("context", "interrupt"):
            txt = item.get(key)
            if not txt or not isinstance(txt, str):
                continue
            if any(ch in _PT_MARKERS for ch in txt):
                continue                      # already Portuguese
            try:
                pt = qwen_chat(
                    TRANSLATE_SYSTEM,
                    f"1: {txt}", api_base=api_base, max_tokens=300)
            except Exception:                # noqa: BLE001 - keep the English
                continue
            pt = re.sub(r"^\s*1\s*[:.)-]\s*", "", (pt or "").strip()).strip()
            if pt:
                item[key] = pt
    return doc


def translate_segments(segments: list[Segment], *, api_base: str,
                       max_tokens: int = 1200) -> None:
    """One numbered Qwen call per scenario, with per-segment retry fallback."""
    def qwen(system: str, user: str) -> str:
        return qwen_chat(system, user, api_base=api_base, max_tokens=max_tokens)

    listing = "\n".join(f"{i + 1}: {s.text_en}" for i, s in enumerate(segments))
    out = qwen(TRANSLATE_SYSTEM, listing)
    parsed: dict[int, str] = {}
    for line in (out or "").splitlines():
        m = re.match(r"\s*(\d+)\s*[:.)-]\s*(.+)", line)
        if m:
            parsed[int(m.group(1))] = m.group(2).strip()
    missing = []
    for i, s in enumerate(segments):
        s.text_pt = parsed.get(i + 1, "").strip()
        if not s.text_pt:
            missing.append(s)
    # segments the batch answer dropped: ask for them one at a time
    for s in missing:
        try:
            s.text_pt = qwen(TRANSLATE_SYSTEM, f"1: {s.text_en}")
            s.text_pt = re.sub(r"^\s*1\s*[:.)-]\s*", "", s.text_pt).strip()
        except Exception as e:  # noqa: BLE001
            s.error = f"translate:{type(e).__name__}"


def time_fit(audio: np.ndarray, slot_s: float, *, cap: float = 2.0):
    """Stretch ``audio`` to fill ``slot_s`` exactly (pitch-preserving WSOLA).

    No librosa: its phase vocoder needs numba, and the numba wheel here refuses
    the installed numpy. ``stretch.fit_to_slot`` is dependency-free, keeps the
    pitch and returns exactly the slot's length.

    **Not used any more.** Filling the slot is what put silence *inside* the
    user's turn: the Portuguese TTS is shorter than the English span in 61% of
    the segments (median 0.91 of the slot, and 30.4% of them needed more than
    1.33x of expansion), so the remainder was padded with digital silence. The
    harness's turn-end signal is a 0.6 s silence gate, so every padded gap was a
    new invitation to take the floor -- 303 of them landed in silence the
    benchmark does not annotate as a pause, against 58 in the English recording,
    and at every class of invitation every model took the floor 1.2-2.0x more
    often on the re-voiced audio, including the English-trained upstream model.
    See :func:`plan_timeline` for what replaces it.
    """
    from .stretch import fit_to_slot

    return fit_to_slot(audio, slot_s, sr=SR, cap=cap)


# --------------------------------------------------------------- the timeline
def plan_timeline(segments: list[Segment], durs: list[float]) -> list[float]:
    """Where each synthesised span starts in the re-timed Portuguese scene.

    Nothing is stretched and nothing is padded. The Portuguese speaker says the
    same thing at their own speed; the gap between two consecutive spans -- the
    real pause the benchmark annotates -- keeps its length exactly; and the
    scene shifts by the accumulated difference between the Portuguese speech and
    the English span it replaces.

    The result is a scene with the same *structure* as the original recording
    (same spans, same gaps, same annotations after remapping) and a speaker who
    is simply a Portuguese speaker. The alternative -- stretching each span to
    fill its English slot -- was rejected on measurement: 30.4% of the segments
    would have needed more than 1.33x of expansion and 10.8% more than 2x, which
    is a larger distortion than a uniform rate difference.
    """
    starts: list[float] = []
    for i, s in enumerate(segments):
        if i == 0:
            starts.append(s.start)
        else:
            gap = max(0.0, s.start - segments[i - 1].end)
            starts.append(starts[i - 1] + durs[i - 1] + gap)
    return starts


def make_remap(segments: list[Segment], starts: list[float], durs: list[float]):
    """A continuous map from English audio time to Portuguese audio time.

    Inside a span the map scales (the words keep their relative position); in a
    gap it shifts (the gap keeps its length). It is continuous at the span
    boundaries, so every annotation the benchmark ships transfers.
    """
    n = len(segments)

    def remap(t: float) -> float:
        for i, s in enumerate(segments):
            if t < s.start:                       # in the gap before span i
                if i == 0:
                    return t
                prev = segments[i - 1]
                return starts[i - 1] + durs[i - 1] + (t - prev.end)
            if t < s.end:                         # inside span i
                f = durs[i] / max(1e-6, s.slot_s)
                return starts[i] + (t - s.start) * f
        return starts[n - 1] + durs[n - 1] + (t - segments[n - 1].end)

    return remap


def remap_annotation(doc, remap) -> dict:
    """Re-time an annotation document (pause / turn_taking / interrupt)."""
    out = json.loads(json.dumps(doc))
    for item in out:
        ts = item.get("timestamp")
        if isinstance(ts, list) and len(ts) >= 2:
            item["timestamp"] = [round(remap(float(ts[0])), 3),
                                 round(remap(float(ts[1])), 3)]
    return out


def peak_normalise(audio: np.ndarray, target: float) -> tuple[np.ndarray, float]:
    """Scale to a target peak, returning the gain applied.

    Cloned voices conditioned on telephone-band conversation audio come out much
    hotter than the catalog ones (measured: peak 0.93 against 0.31 for the same
    sentence), and the output is written as PCM16, so without this the rebuilt set
    clips. The gain is recorded per segment instead of being applied silently.
    """
    if audio.size == 0:
        return audio, 1.0
    pk = float(np.abs(audio).max())
    if pk < 1e-6:
        return audio, 1.0
    g = float(target / pk)
    return (audio * g).astype(np.float32), g


def derived_transcription(segments: list[Segment], starts: list[float],
                          durs: list[float]) -> list[dict]:
    """Word-level transcript for the new audio, distributed by word length.

    Nothing downstream depends on it (the benchmark's evaluators read the model's
    *output*, and this harness uses the GT only as a reference), so the words are
    spread proportionally inside their segment rather than force-aligned. The
    segment boundaries themselves are exact, and they are the *new* ones.
    """
    out: list[dict] = []
    for i, s in enumerate(segments):
        words = s.text_pt.split()
        if not words:
            continue
        weights = np.array([len(w) + 1 for w in words], dtype=np.float64)
        weights /= weights.sum()
        t = starts[i]
        for w, frac in zip(words, weights):
            d = durs[i] * float(frac)
            out.append({"text": w, "timestamp": [round(t, 3), round(t + d, 3)]})
            t += d
    return out


# ----------------------------------------------------------------------- stage
def stage_asr(scenarios, out_root: Path, *, limit: int | None) -> None:
    """Word-level English ASR for subsets that ship no transcript."""
    from .score import ParakeetScorer

    todo = []
    for sc in scenarios:
        d = out_root / sc.subset / sc.sid
        if (d / "asr_en.json").is_file():
            continue
        if sc.subset in ("candor_pause_handling", "synthetic_pause_handling",
                         "icc_backchannel") and sc.gt_words:
            continue
        todo.append(sc)
    if limit:
        todo = todo[:limit]
    print(f"[asr] {len(todo)} scenarios need ASR", flush=True)
    if not todo:
        return
    scorer = ParakeetScorer()
    for n, sc in enumerate(todo, 1):
        d = out_root / sc.subset / sc.sid
        d.mkdir(parents=True, exist_ok=True)
        words = scorer.words(sc.input_wav)
        (d / "asr_en.json").write_text(
            json.dumps(words, ensure_ascii=False), encoding="utf-8")
        if n % 25 == 0 or n <= 2:
            print(f"[asr] {n}/{len(todo)} {sc.subset}/{sc.sid} "
                  f"{len(words)} words", flush=True)


def stage_build(scenarios, out_root: Path, *, limit: int | None, voice_name: str,
                voice_source: str, api_base: str, dry_run: bool,
                max_words: int = 15, translations: Path | None = None) -> None:
    import soundfile as sf

    if limit:
        scenarios = scenarios[:limit]
    voice = None
    if not dry_run:
        voice = AssistantVoice("portuguese", voice=voice_name)
        print(f"[build] voice={voice.voice} source={voice_source}", flush=True)

    n_done = n_skip = 0
    t0 = time.time()
    for sc in scenarios:
        src_dir = sc.root
        out_dir = out_root / sc.subset / sc.sid
        if (out_dir / "input.wav").is_file() and (out_dir / "translation.json").is_file():
            n_skip += 1
            continue

        wav24, _ = load_wav(sc.input_wav)
        words = list(sc.gt_words)
        if not words:
            p = out_dir / "asr_en.json"
            if not p.is_file():
                print(f"[build] SKIP {sc.key}: no transcript and no asr_en.json (run --stage asr)")
                continue
            words = json.loads(p.read_text(encoding="utf-8"))
        segments = spans_to_segments(wav24, words)
        if not segments:
            print(f"[build] SKIP {sc.key}: no speech spans")
            continue

        cached = load_cached_translations(translations, sc.subset, sc.sid)
        if cached is not None:
            apply_cached_translations(segments, cached)
            missing = [sg for sg in segments if not sg.text_pt]
            if missing:
                print(f"[build] {sc.key}: {len(missing)} segment(s) missing from the "
                      f"cache -> translating", flush=True)
                translate_segments(missing, api_base=api_base)
        else:
            translate_segments(segments, api_base=api_base)
        if dry_run:
            n_done += 1
            continue

        # onset: clone from the original speaker when asked and possible
        state = None
        used_voice = voice.voice
        if voice_source == "clone":
            try:
                seg = max(range(len(segments)), key=lambda i: segments[i].slot_s)
                a = int(segments[seg].start * SR)
                b = min(wav24.size, a + int(6 * SR))
                import tempfile

                prompt, _ = peak_normalise(wav24[a:b], 0.6)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
                    sf.write(t.name, prompt, SR)
                    state = voice.model.get_state_for_audio_prompt(t.name)
                used_voice = f"clone:{sc.sid}"
            except Exception as e:  # noqa: BLE001 - fall back to the catalog voice
                state = None
                used_voice = f"{voice.voice} (clone unavailable: {type(e).__name__})"

        # Synthesise at the TTS's natural rate: no time-fit, no padding.
        audios: list[np.ndarray] = []
        for sg in segments:
            if not sg.text_pt:
                audios.append(np.zeros(0, dtype=np.float32))
                sg.error = sg.error or "empty_translation"
                continue
            chunk = (voice.speak(sg.text_pt, max_words=max_words) if state is None
                     else _speak_with_state(voice, state, sg.text_pt, max_words))
            levelled, gain = peak_normalise(chunk.audio, 0.7)
            sg.gain = round(gain, 3)
            sg.synth_s = round(levelled.size / SR, 3)
            sg.fitted_s = sg.synth_s
            sg.stretch = 1.0
            audios.append(levelled)

        # A segment with no translation would leave a hole in the user's turn --
        # the exact defect this rebuild exists to remove. Fail loudly instead.
        empty = [sg for sg in segments if not sg.text_pt]
        if empty:
            raise RuntimeError(
                f"{sc.key}: {len(empty)} segment(s) have no translation; the audio "
                f"would carry a silence inside the user's turn. Fix the "
                f"translation and rebuild.")

        durs = [a.size / SR for a in audios]
        starts = plan_timeline(segments, durs)
        remap = make_remap(segments, starts, durs)

        total = starts[-1] + durs[-1] + max(0.0, wav24.size / SR - segments[-1].end)
        out = np.zeros(max(1, int(round(total * SR))), dtype=np.float32)
        for i, a in enumerate(audios):
            if a.size == 0:
                continue
            p0 = int(round(starts[i] * SR))
            p1 = min(out.size, p0 + a.size)
            if p1 > p0:
                out[p0:p1] += a[: p1 - p0]

        sf.write(str(out_dir / "input.wav"), np.clip(out, -1.0, 1.0), SR, subtype="PCM_16")
        # annotations transfer, re-timed: same events, new seconds
        for name in ("pause.json", "turn_taking.json", "interrupt.json"):
            src = src_dir / name
            if src.is_file():
                doc = json.loads(src.read_text(encoding="utf-8"))
                if name == "interrupt.json" and api_base:
                    doc = translate_interrupt_texts(doc, api_base=api_base)
                (out_dir / name).write_text(
                    json.dumps(remap_annotation(doc, remap), ensure_ascii=False, indent=1),
                    encoding="utf-8")
        # interruption subset also ships its pieces as separate files
        if sc.interrupt_span:
            i0, i1 = sc.interrupt_span
            n0, n1 = remap(i0), remap(i1)
            sf.write(str(out_dir / "context.wav"),
                     np.clip(out[: int(n0 * SR)], -1, 1), SR, subtype="PCM_16")
            sf.write(str(out_dir / "interrupt.wav"),
                     np.clip(out[int(n0 * SR): int(n1 * SR)], -1, 1), SR, subtype="PCM_16")
        (out_dir / "transcription.json").write_text(
            json.dumps(derived_transcription(segments, starts, durs),
                       ensure_ascii=False, indent=1),
            encoding="utf-8")
        (out_dir / "translation.json").write_text(json.dumps({
            "subset": sc.subset, "sid": sc.sid, "sr": SR,
            "voice": used_voice, "voice_source": voice_source,
            "text_source": "gt_transcription" if sc.gt_words else "parakeet",
            "source_duration_s": round(wav24.size / SR, 3),
            "built_duration_s": round(total, 3),
            "policy": "natural rate, gaps preserved, no padding, annotations re-timed",
            "segments": [{
                "index": sg.index, "start": round(sg.start, 3), "end": round(sg.end, 3),
                "slot_s": round(sg.slot_s, 3), "synth_s": sg.synth_s,
                "fitted_s": sg.fitted_s, "stretch": sg.stretch, "clamped": False,
                "padded": False, "truncated": False,
                "new_start": round(starts[i], 3),
                "new_end": round(starts[i] + durs[i], 3),
                "gain": sg.gain,
                "text_en": sg.text_en, "text_pt": sg.text_pt, "error": sg.error,
            } for i, sg in enumerate(segments)],
        }, ensure_ascii=False, indent=1), encoding="utf-8")

        n_done += 1
        if n_done % 20 == 0 or n_done <= 3:
            rate = (time.time() - t0) / max(1, n_done)
            print(f"[build] {n_done} built ({rate:.1f}s/scenario) "
                  f"{sc.subset}/{sc.sid}: {len(segments)} segments", flush=True)

    print(f"[build] done: {n_done} built, {n_skip} already present -> {out_root}", flush=True)


def _speak_with_state(voice, state, text: str, max_words: int = 15):
    """pocket-tts synthesis with an explicit (cloned) conditioning state."""
    import numpy as np

    from .tts import SpokenChunk, split_for_tts, strip_for_speech

    t0 = time.monotonic()
    parts: list[np.ndarray] = []
    for piece in split_for_tts(strip_for_speech(text), max_words=max_words):
        try:
            for chunk in voice.model.generate_audio_stream(state, piece):
                if hasattr(chunk, "detach"):
                    chunk = chunk.detach().cpu().float().numpy()
                parts.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
        except Exception:  # noqa: BLE001
            continue
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return SpokenChunk(text, audio, audio.size / SR, time.monotonic() - t0)


def stage_verify(scenarios, out_root: Path, *, limit: int | None = None) -> None:
    """Re-recognise what was built and flag held-vowel degeneration.

    Prevention (the word cap above) is the first line, but the failure mode is
    silent in the audio and only visible in what a recogniser makes of it -- a
    held vowel transcribes as a run of one character. So the check is the same
    one that found the problem: ASR the built input and look for long runs of a
    repeated character.
    """
    import re

    from faster_whisper import WhisperModel

    bad_re = re.compile(r"(.)\1{5,}")
    # medium was the wrong judge for this: given language="pt" it *translates*
    # English audio into fluent Portuguese (verified on the English control), so a
    # degenerate transcript said nothing reliable. large-v3 transcribes instead of
    # translating and is the model used for the check that found the real bug.
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    rows, bad = [], 0
    if limit:
        scenarios = scenarios[:limit]
    for n, sc in enumerate(scenarios, 1):
        d = out_root / sc.subset / sc.sid
        wav = d / "input.wav"
        if not wav.is_file():
            continue
        segs, _ = model.transcribe(str(wav), language="pt", beam_size=1,
                                   temperature=0.0, word_timestamps=False)
        text = " ".join(s.text for s in segs)
        hit = bool(bad_re.search(text))
        bad += hit
        rows.append({"subset": sc.subset, "sid": sc.sid, "degraded": hit,
                     "text": text[:300]})
        if n % 50 == 0 or n <= 3:
            print(f"[verify] {n} checked, {bad} degraded", flush=True)
    (out_root / "verify.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(f"[verify] {bad}/{len(rows)} scenarios show held-vowel degeneration "
          f"-> {out_root / 'verify.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="build", choices=["asr", "build", "verify"])
    ap.add_argument("--data-root", type=Path, default=EN_ROOT)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--tasks", default="all",
                    help="comma-separated subset names, or 'all'")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--voice", default=None,
                    help="pocket-tts catalog voice (default: the pt-BR default)")
    ap.add_argument("--voice-source", default="catalog", choices=["catalog", "clone"])
    ap.add_argument("--max-words", type=int, default=15,
                    help="words per pocket-tts call. The 30-word default produced "
                         "50+ token chunks and a held-vowel degeneration on 57%% of "
                         "the first pt-BR build; 15 keeps it inside the limit.")
    ap.add_argument("--api-base", default="http://127.0.0.1:8099/v1",
                    help="llama-server used for translation (Qwen3.5-9B)")
    ap.add_argument("--translations", type=Path, default=None,
                    help="reuse the translations of a previous build "
                         "(<dir>/<subset>/<sid>/translation.json); the English "
                         "source is checked against the audio, so a stale cache "
                         "is refused rather than silently reused")
    ap.add_argument("--dry-run", action="store_true",
                    help="translate only, no audio (checks the translator)")
    args = ap.parse_args()

    tasks = list(TASKS) if args.tasks == "all" else args.tasks.split(",")
    scenarios = iter_scenarios(tasks, root=args.data_root)
    print(f"[pt-audio] {len(scenarios)} scenarios from {args.data_root} -> {args.out}",
          flush=True)
    for sc in scenarios:
        (args.out / sc.subset / sc.sid).mkdir(parents=True, exist_ok=True)
    if args.stage == "asr":
        stage_asr(scenarios, args.out, limit=args.limit)
    else:
        stage_build(scenarios, args.out, limit=args.limit, voice_name=args.voice,
                    voice_source=args.voice_source, api_base=args.api_base,
                    dry_run=args.dry_run, max_words=args.max_words,
                    translations=args.translations)
    if args.stage == "verify":
        stage_verify(scenarios, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
