"""Full-Duplex-Bench (pt-BR, text-domain) evaluation harness for DuplexCascade.

Evaluates the three trained variants (``sft`` = original recipe, ``distill`` =
self-distillation, ``mtp_like`` = policy companion) against the four
Full-Duplex-Bench v1.0 dimensions (pause handling, backchannel, smooth
turn-taking, user interruption) and compares them with the published
DuplexCascade numbers.

See ``README.md`` for the metric definitions and ``REFERENCE.md`` for the
published numbers this harness reproduces.
"""
