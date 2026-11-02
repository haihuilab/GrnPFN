"""Helpers shared by the evaluation scripts: spent cell-split seeds and pair-level scoring."""
from __future__ import annotations

from common import tc

SPENT = (193, 197, 199, 211)


def score(pred, obs, centre):
    """pair-specific and raw top-100 cosine per test pair (pair-specific centres predictions over test pairs)"""
    pc = pred - pred.mean(0)
    return [tc(pc[i], obs[i] - centre) for i in range(len(obs))], [tc(pred[i], obs[i]) for i in range(len(obs))]
