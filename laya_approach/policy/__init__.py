"""Runtime Laya policy package.

Usage::

    from laya_approach.policy import load

    policy = load("models/laya-companion", stream="text")
    tag, probs = policy(items)
"""

from .laya_policy import LayaPolicy, load

__all__ = ["LayaPolicy", "load"]
