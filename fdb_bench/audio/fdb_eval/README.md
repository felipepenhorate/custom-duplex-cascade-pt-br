# Vendored Full-Duplex-Bench v1.0 evaluation scripts

Copied **verbatim** from `DanielLin94144/Full-Duplex-Bench`,
`v1_v1.5/evaluation/` (and `get_transcript/asr.py` as `asr_official.py`), so the
audio run reports numbers produced by the benchmark's own code rather than by a
re-implementation:

| file | what it is |
|---|---|
| `eval_pause_handling.py` | pause-handling TOR (`--task pause_handling`) |
| `eval_smooth_turn_taking.py` | smooth-turn-taking TOR + latency |
| `eval_backchannel.py` | backchannel TOR / JSD / frequency (needs `./icc_gt_distribution.json`) |
| `eval_user_interruption.py` | interruption TOR + latency + GPT-judged relevance |
| `asr_official.py` | parakeet-tdt-0.6b-v2 word-aligned `output.json` producer |
| `icc_gt_distribution.json` | ICC listener-backchannel ground truth |

Two local substitutions, both documented in `../README.md` and in the article:

1. **The judge.** `eval_user_interruption.py` builds an OpenAI client and asks
   `gpt-4-turbo`. It is driven here with a shim that has the same call signature
   but answers from the local `Qwen3.5-9B` GGUF, so the *rubric and the parser are
   the official ones* and only the model behind them differs.
2. **Nothing else.** The three rule-based metrics run unmodified.
