"""The within-round convergence -- iterative, reliability-weighted truth discovery.

Plain majority vote gives every agent one equal, fixed vote. That breaks when
several weak agents share the *same* wrong bias (correlated error): they
out-vote a smaller set of strong, independent agents.

stigma instead runs a short fixed-point loop. Each iteration:

1. score every candidate by the reliability-weighted mass of its signals;
2. re-estimate each agent's reliability from how well its picks agree with the
   current consensus.

Agents that cluster with the emerging consensus gain weight; lone or mutually
inconsistent agents lose it. Seeding the agent weights from the cross-run
:class:`~stigma.pheromone.PheromoneStore` lets the round start from what past
verification already taught the swarm.
"""

from __future__ import annotations

import math

from .signals import ConsensusResult, Signal


def _normalise(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    if total <= 0.0:
        n = len(scores) or 1
        return {c: 1.0 / n for c in scores}
    return {c: v / total for c, v in scores.items()}


def _agreement(distribution: dict[str, float]) -> float:
    """1 - normalised Shannon entropy: 1.0 = fully converged, 0.0 = uniform."""
    n = len(distribution)
    if n <= 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in distribution.values() if p > 0.0)
    return max(0.0, 1.0 - entropy / math.log(n))


def consense(
    signals: list[Signal],
    start_weights: dict[str, float],
    *,
    iterations: int = 15,
    damping: float = 0.5,
    abstain_margin: float = 0.05,
) -> ConsensusResult:
    """Aggregate ``signals`` into a consensus result.

    Args:
        signals: All votes for this round.
        start_weights: Initial per-agent reliability (typically pheromone levels).
        iterations: Fixed-point sweeps. More = sharper convergence.
        damping: Blend between the old and re-estimated agent weight each sweep,
            in ``[0, 1]``. Lower is more stable.
        abstain_margin: If the top-two consensus gap is under this, abstain.
    """
    if not signals:
        return ConsensusResult(None, [], {}, 0.0, True, {})

    candidates = sorted({s.candidate_id for s in signals})
    agents = sorted({s.agent_id for s in signals})

    # The cross-run pheromone is the ANCHOR: it is the only signal that can tell
    # "correlated wrong" from "correlated right", so it must not be overridden by
    # within-round agreement. The loop below only refines it within a bounded
    # band, so a strong learned prior always survives a colluding majority.
    prior = {a: max(start_weights.get(a, 1.0), 1e-6) for a in agents}
    weights = dict(prior)

    distribution: dict[str, float] = {}
    for _ in range(iterations):
        scores = {c: 0.0 for c in candidates}
        for s in signals:
            scores[s.candidate_id] += weights[s.agent_id] * s.weight
        distribution = _normalise(scores)
        peak = max(distribution.values()) or 1.0

        # Each agent's within-round alignment (how much of its vote mass landed
        # on high-consensus candidates) nudges its prior by a bounded factor in
        # [lo, hi] -- centred on 1.0, applied to the FIXED prior so it can't drift.
        updated: dict[str, float] = {}
        for a in agents:
            agree_mass = 0.0
            total_mass = 0.0
            for s in signals:
                if s.agent_id == a:
                    agree_mass += s.weight * distribution[s.candidate_id]
                    total_mass += s.weight
            aligned = (agree_mass / total_mass) / peak if total_mass else 0.0
            aligned = min(1.0, max(0.0, aligned))
            factor = 1.0 + damping * (2.0 * aligned - 1.0)  # in [lo, hi]
            updated[a] = prior[a] * factor
        weights = updated

    ranking = sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)
    top_gap = ranking[0][1] - (ranking[1][1] if len(ranking) > 1 else 0.0)
    abstain = top_gap < abstain_margin
    winner = None if abstain else ranking[0][0]

    return ConsensusResult(
        winner=winner,
        ranking=ranking,
        distribution=distribution,
        agreement=_agreement(distribution),
        abstain=abstain,
        agent_weights=weights,
    )
