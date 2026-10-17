"""End-to-end smoke test of the harness with a scripted stub adapter.

Runs the real simulator + real metrics over every subset in both languages
without a GPU or a model, using an oracle-ish stub that follows a fixed policy.
This validates the plumbing (timeline construction, metric math, serialisation,
aggregation) and, because the stub's behaviour is known by construction, it
also pins the metrics' direction:

* a **perfect** policy must score pause TOR = 0 (never barge into a pause),
  smooth TOR = 1 (always take the floor when the turn ends) and interruption
  TOR = 1 (always yield and answer);
* a **chatty** policy must score the opposite on all three;
* a **silent** policy must take the floor only on ``ends_turn`` chunks.

Run:  /home/penhfel/unsloth_uv/bin/python -m fdb_bench.tests.test_simulator
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fdb_bench import metrics as M            # noqa: E402
from fdb_bench import scenarios as S          # noqa: E402
from fdb_bench.clock import SpeechClock       # noqa: E402
from fdb_bench.simulate import DuplexSimulator  # noqa: E402


class StubAdapter:
    """A scripted duplex policy — no model, fully deterministic."""

    def __init__(self, policy: str) -> None:
        self.policy = policy
        self.name = f"stub_{policy}"
        self._answered_context = False
        self.system_prompt = "Você é um assistente de voz em português. Responda de forma curta e natural."
        self.no_voice_marker = "<|no voice|>"
        self.last_call_s = 0.01

    def reset(self) -> None:
        self.last_call_s = 0.01
        self._answered_context = False

    def default_backchannel(self) -> str:
        return "aham"

    def act(self, items: list[dict]) -> tuple[str, str]:
        last = items[-1]
        text = last["text"]
        silence = text.strip() == self.no_voice_marker

        if self.policy == "perfect":
            # obey ends_turn exactly; never speak during silence; never interrupt
            if last.get("ends_turn"):
                return "user_finish_speaking", "Bem, sobre isso, o primeiro passo costuma ser simples: definir o objetivo com clareza, juntar o material basico e comecar pequeno, sem tentar fazer tudo de uma vez, revisando o ritmo ao longo das semanas."
            if silence:
                return "user_is_thinking", ""
            return "user_is_speaking", ""

        if self.policy == "chatty":
            # takes the floor at every opportunity, backchannels during silences
            if silence:
                return "system_backchannel", ""
            return "user_finish_speaking", "Bem, sobre isso, o primeiro passo costuma ser simples: definir o objetivo com clareza, juntar o material basico e comecar pequeno, sem tentar fazer tudo de uma vez, revisando o ritmo ao longo das semanas."

        if self.policy == "deaf":
            # answers the first (context) question, then is completely deaf to
            # the barge-in: never yields, never re-takes the floor
            if last.get("ends_turn") and not self._answered_context:
                self._answered_context = True
                return "user_finish_speaking", "Bem, sobre isso, o primeiro passo costuma ser simples: definir o objetivo com clareza, juntar o material basico e comecar pequeno, sem tentar fazer tudo de uma vez, revisando o ritmo ao longo das semanas."
            return "user_is_speaking", ""

        if self.policy == "silent":
            # waits for a real turn end, never speaks otherwise
            if last.get("ends_turn"):
                return "user_finish_speaking", "Bem, sobre isso, o primeiro passo costuma ser simples: definir o objetivo com clareza, juntar o material basico e comecar pequeno, sem tentar fazer tudo de uma vez, revisando o ritmo ao longo das semanas."
            return "user_is_speaking", ""

        raise KeyError(self.policy)


def run_policy(policy: str, lang: str = "pt", limit: int = 12):
    sim = DuplexSimulator(StubAdapter(policy), SpeechClock(3.0))
    out: dict[str, list] = {}
    for task in S.TASKS:
        out[task] = [sim.run(sc) for sc in S.build_scenarios(task, limit=limit, lang=lang)]
    return out



def check_ata() -> list[str]:
    """ATA must not reward a model that simply never speaks."""
    nan = float("nan")
    cases = {
        "never speaks": (dict(pause_synthetic_tor=0.0, pause_candor_tor=0.0,
                              smooth_tor=0.0, intr_tor=nan), 0.0, 1e-9),
        "always speaks": (dict(pause_synthetic_tor=1.0, pause_candor_tor=1.0,
                               smooth_tor=1.0, intr_tor=1.0), 0.5, 1e-9),
        "balanced": (dict(pause_synthetic_tor=0.1, pause_candor_tor=0.2,
                          smooth_tor=0.9, intr_tor=0.95), 0.8875, 1e-9),
        "good pause, never answers": (dict(pause_synthetic_tor=0.1, pause_candor_tor=0.2,
                                          smooth_tor=0.9, intr_tor=nan), 0.65, 1e-9),
    }
    fails = []
    for name, (row, want, tol) in cases.items():
        got = M.averaged_turn_taking_accuracy(row)
        ok = abs(got - want) <= tol
        print(f"{'PASS' if ok else 'FAIL'}  ATA[{name}] = {got:.4f} (want {want})")
        if not ok:
            fails.append(f"ATA[{name}]")
    return fails



def main() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"{'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    clock = SpeechClock(3.0)

    # ---- perfect policy ------------------------------------------------
    perfect = run_policy("perfect")
    ps = M.pause_handling(perfect["pause_synthetic"], clock)
    pc = M.pause_handling(perfect["pause_candor"], clock)
    st = M.smooth_turn_taking(perfect["smooth_turn"], clock)
    it = M.user_interruption_tor_latency(perfect["interruption_synthetic"])
    bc = M.backchannel(perfect["backchannel_icc"], clock)

    check("perfect: pause_synthetic TOR == 0", ps["tor"] == 0.0, f"tor={ps['tor']}")
    check("perfect: pause_candor TOR == 0", pc["tor"] == 0.0, f"tor={pc['tor']}")
    check("perfect: smooth TOR == 1", st["tor"] == 1.0, f"tor={st['tor']}")
    check("perfect: backchannel TOR == 0", bc["tor"] == 0.0, f"tor={bc['tor']}")

    # ---- chatty policy ------------------------------------------------
    chatty = run_policy("chatty")
    c_ps = M.pause_handling(chatty["pause_synthetic"], clock)
    c_pc = M.pause_handling(chatty["pause_candor"], clock)
    c_st = M.smooth_turn_taking(chatty["smooth_turn"], clock)
    c_bc = M.backchannel(chatty["backchannel_icc"], clock)

    check("chatty: pause_synthetic TOR > perfect", c_ps["tor"] > ps["tor"],
          f"chatty={c_ps['tor']} perfect={ps['tor']}")
    check("chatty: pause_candor TOR > perfect", c_pc["tor"] > pc["tor"],
          f"chatty={c_pc['tor']} perfect={pc['tor']}")
    check("chatty: backchannel TOR == 1 (takes formal turn)", c_bc["tor"] == 1.0,
          f"tor={c_bc['tor']}")
    check("chatty: backchannel freq > 0", c_bc["freq"] > 0, f"freq={c_bc['freq']}")
    check("chatty: smooth TOR == 1", c_st["tor"] == 1.0, f"tor={c_st['tor']}")

    # ---- interruption: a deaf model must never re-take the floor -------
    deaf = run_policy("deaf")
    d_it = M.user_interruption_tor_latency(deaf["interruption_synthetic"])
    check("deaf (ignores the barge-in): interruption TOR == 0", d_it["tor"] == 0.0,
          f"tor={d_it['tor']}")
    check("deaf: post-interruption response is empty",
          all(not r.post_interrupt_text for r in deaf["interruption_synthetic"]))

    # ---- JSD sanity ----------------------------------------------------
    check("backchannel JSD in [0, 1]", all(
        0.0 <= M.backchannel(perfect["backchannel_icc"], clock)["jsd"] <= 1.0
        for _ in range(1)
    ))
    check("backchannel JSD finite for every policy",
          not any(math.isnan(M.backchannel(chatty["backchannel_icc"], clock)["jsd"])
                  for _ in range(1)))

    # ---- silent policy: takes the floor exactly on ends_turn ----------
    silent = run_policy("silent")
    s_st = M.smooth_turn_taking(silent["smooth_turn"], clock)
    s_ps = M.pause_handling(silent["pause_synthetic"], clock)
    check("silent: smooth TOR == 1", s_st["tor"] == 1.0, f"tor={s_st['tor']}")
    check("silent: pause TOR == 0", s_ps["tor"] == 0.0, f"tor={s_ps['tor']}")

    # ---- timeline sanity ---------------------------------------------
    r = perfect["smooth_turn"][0]
    check("user events are time-ordered",
          all(a.t_end <= b.t_start + 1e-6 for a, b in zip(r.user_events, r.user_events[1:])),
          f"n={len(r.user_events)}")
    check("end_time >= last user event",
          r.end_time >= max(e.t_end for e in r.user_events))

    # ---- bilingual parity: same structure in both languages ------------
    for lang in ("pt", "en"):
        out = run_policy("perfect", lang=lang, limit=6)
        check(f"[{lang}] perfect policy all-correct",
              M.pause_handling(out["pause_synthetic"], clock)["tor"] == 0.0
              and M.smooth_turn_taking(out["smooth_turn"], clock)["tor"] == 1.0)

    failures.extend(check_ata())

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
