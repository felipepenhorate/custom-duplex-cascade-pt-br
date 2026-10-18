"""FDB v1.0 on-disk layout -> scenario objects.

The benchmark ships five subsets as ``<task>/<id>/{input.wav, <annotation>.json,
transcription.json}``. Nothing about the audio or the annotations is touched; this
module only names the fields the harness needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# subset folder -> task key; the GT transcripts are word-level user transcripts
TASKS: dict[str, str] = {
    "candor_pause_handling": "pause_handling",
    "synthetic_pause_handling": "pause_handling",
    "candor_turn_taking": "smooth_turn_taking",
    "icc_backchannel": "backchannel",
    "synthetic_user_interruption": "user_interruption",
}

# subsets whose GT is "the model must stay silent through a mid-turn pause"
PAUSE_SUBSETS = ("candor_pause_handling", "synthetic_pause_handling")

DEFAULT_ROOT = Path("/mnt/f/fdb_data/v1.0")


@dataclass
class AudioScenario:
    subset: str
    task: str
    sid: str
    root: Path                      # sample directory
    input_wav: Path
    duration_s: float = 0.0
    # annotations, in the benchmark's own schema
    pause: dict | None = None
    turn_taking: dict | None = None
    interrupt: dict | None = None
    gt_words: list[dict] = field(default_factory=list)   # user GT transcription

    @property
    def key(self) -> str:
        return f"{self.subset}/{self.sid}"

    # -- benchmark reference points -------------------------------------
    @property
    def pause_interval(self) -> tuple[float, float] | None:
        if not self.pause:
            return None
        ts = self.pause[0]["timestamp"]
        return (float(ts[0]), float(ts[1]))

    @property
    def turn_end(self) -> float | None:
        if not self.turn_taking:
            return None
        return float(self.turn_taking[0]["timestamp"][0])

    @property
    def interrupt_span(self) -> tuple[float, float] | None:
        if not self.interrupt:
            return None
        ts = self.interrupt[0]["timestamp"]
        return (float(ts[0]), float(ts[1]))

    @property
    def interrupt_texts(self) -> tuple[str, str] | None:
        if not self.interrupt:
            return None
        return (self.interrupt[0]["context"], self.interrupt[0]["interrupt"])


def _sample_dirs(subset_root: Path) -> list[Path]:
    return sorted(
        [p for p in subset_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )


def load_scenario(subset: str, sample_dir: Path) -> AudioScenario:
    import soundfile as sf

    info = sf.info(str(sample_dir / "input.wav"))
    sc = AudioScenario(
        subset=subset,
        task=TASKS[subset],
        sid=sample_dir.name,
        root=sample_dir,
        input_wav=sample_dir / "input.wav",
        duration_s=info.frames / info.samplerate,
    )
    for name, attr in (("pause.json", "pause"), ("turn_taking.json", "turn_taking"),
                       ("interrupt.json", "interrupt")):
        p = sample_dir / name
        if p.is_file():
            setattr(sc, attr, json.loads(p.read_text(encoding="utf-8")))
    tr = sample_dir / "transcription.json"
    if tr.is_file():
        sc.gt_words = json.loads(tr.read_text(encoding="utf-8"))
    return sc


def iter_scenarios(
    tasks: list[str] | None = None,
    root: Path | str = DEFAULT_ROOT,
    limits: dict[str, int] | None = None,
) -> list[AudioScenario]:
    """All scenarios in the five v1.0 subsets, in a stable order.

    ``tasks`` filters by subset folder name; ``limits`` caps how many samples are
    taken from each subset (for smoke runs).
    """
    root = Path(root)
    subsets = tasks or list(TASKS)
    out: list[AudioScenario] = []
    for subset in subsets:
        base = root / subset / subset
        if not base.is_dir():
            raise FileNotFoundError(f"{base} not found (unpack the v1.0 zips)")
        dirs = _sample_dirs(base)
        if limits and subset in limits:
            dirs = dirs[: limits[subset]]
        out.extend(load_scenario(subset, d) for d in dirs)
    return out
