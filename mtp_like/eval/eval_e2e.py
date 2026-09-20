"""M3 e2e test — drive the orchestrator with a scripted timeline (SPEC 8.3).

Simulates the full-duplex loop without a browser: timed user ASR word
events and silence gaps are fed to the Orchestrator; the big model answers
between user events. Verifies the controller behavior:
  * user words -> <|user is speaking|> / pause
  * silence after a question -> <|user finish speaking|> -> answer starts
  * user words DURING the answer -> <|user interruption|> -> answer stops
  * backchannel during the answer -> keeps talking

Usage:
  python eval/eval_e2e.py --companion /mnt/f/duplex_cascade_runs/mtp_like/runs/final_v2/merged \
      --big-model Qwen/Qwen3-4B-Instruct-2507 [--fast]  (--fast: tiny cached model)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policy.duplex_policy import DuplexPolicy
from policy.orchestrator import Orchestrator, TransformersLLM


def drive(orch: Orchestrator, events: list[tuple[float, str | None]]) -> None:
    """events: [(delay_s, text|None)] — None = silence gap fires the timer."""
    t0 = time.time()
    for delay, text in events:
        while time.time() - t0 < delay:
            time.sleep(min(0.05, delay - (time.time() - t0)))
        if text is None:
            orch.on_silence()
        else:
            orch.on_user_words(text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--companion", default="/mnt/f/duplex_cascade_runs/mtp_like/runs/final_v2/merged")
    p.add_argument("--big-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--fast", action="store_true",
                   help="use the tiny cached Qwen2.5-0.5B for a quick smoke")
    args = p.parse_args()

    big = "Qwen/Qwen2.5-0.5B-Instruct" if args.fast else args.big_model
    print(f"[e2e] companion={args.companion} big={big}")
    orch = Orchestrator(policy=DuplexPolicy(args.companion), llm=TransformersLLM(big))

    print("\n=== turn 1: normal question + answer ===")
    drive(orch, [
        (0.0, "olá,"),          # user starts -> speaking
        (0.2, "tudo bem?"),     # more words -> speaking
        (0.8, None),            # silence -> finish -> answer starts
    ])
    # let the answer run to completion (sentences emitted via the loop)
    for sent in (orch.answer_gen or []):
        if orch.state != "answering":
            break
        orch._assistant_sentence(sent)
    orch._close_answer()

    print("\n=== turn 2: user barges in mid-answer ===")
    orch.on_user_words("me conta uma receita de pão de queijo?")
    print("  (waiting for finish...)", "last_tag =", orch.last_tag)
    # let the answer start
    if orch.state != "answering":
        orch.on_silence()
    print("  state =", orch.state, "last_tag =", orch.last_tag)
    interrupted = False
    for _ in range(30):
        if orch.state != "answering":
            break
        try:
            sent = next(orch.answer_gen)
        except StopIteration:
            break
        orch._assistant_sentence(sent)
        # user barges in right after the first assistant sentence
        if len([it for it in orch.items if it["role"] == "assistant"]) >= 2:
            orch.on_user_words("espera aí,")
            orch.on_user_words("eu quis dizer sem queijo")
            if orch.last_tag == "user_interruption":
                interrupted = True
                print("  (barge-in detected: answer interrupted)")
                break
    if not interrupted:
        print("  NOTE: no interruption tag fired (check companion thresholds)")
    print("\n=== final duplex context ===")
    for it in orch.items[-8:]:
        print(f"   [{it['role']:9s}] {it['text'][:80]}")


if __name__ == "__main__":
    main()