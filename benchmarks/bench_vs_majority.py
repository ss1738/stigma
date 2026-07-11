"""stigma vs plain majority vote under *correlated* agent error.

The realistic failure mode for majority vote isn't random noise -- it's several
weak agents that share the same wrong bias (common training data, common
failure mode). Here 3 weak agents collude on a decoy answer more often than not,
against 2 strong independent agents. Majority vote counts heads and gets fooled;
stigma learns from verified outcomes which agents to trust and recovers.

Run: python benchmarks/bench_vs_majority.py
"""

from __future__ import annotations

import random
from collections import Counter

from stigma import Signal, Swarm

CHOICES = ["A", "B", "C", "D"]
N_QUESTIONS = 2000
SEED = 7


def make_round(rng: random.Random) -> tuple[list[Signal], str]:
    """One question: pick a truth, then simulate 5 agents voting."""
    truth = rng.choice(CHOICES)
    decoy = rng.choice([c for c in CHOICES if c != truth])
    signals: list[Signal] = []

    # Two strong, independent agents (75% correct).
    for name in ("good1", "good2"):
        pick = truth if rng.random() < 0.75 else rng.choice(CHOICES)
        signals.append(Signal(name, pick, 0.8))

    # Three weak agents that collude on the *same* decoy 60% of the time.
    for name in ("weak1", "weak2", "weak3"):
        pick = decoy if rng.random() < 0.60 else rng.choice(CHOICES)
        signals.append(Signal(name, pick, 0.9))

    return signals, truth


def majority_vote(signals: list[Signal]) -> str:
    tally: Counter[str] = Counter()
    for s in signals:
        tally[s.candidate_id] += 1  # one head, one vote
    return tally.most_common(1)[0][0]


def main() -> None:
    rng = random.Random(SEED)
    swarm = Swarm()

    maj_hits = 0
    stigma_hits = 0
    stigma_hits_late = 0  # accuracy over the final 25% (after learning)
    warmup = int(N_QUESTIONS * 0.75)

    for i in range(N_QUESTIONS):
        signals, truth = make_round(rng)

        if majority_vote(signals) == truth:
            maj_hits += 1

        result = swarm.consense(signals)
        if result.winner == truth:
            stigma_hits += 1
            if i >= warmup:
                stigma_hits_late += 1

        swarm.reward(truth)  # verifier reveals the answer

    late_n = N_QUESTIONS - warmup
    print(f"questions:                 {N_QUESTIONS}")
    print(f"majority vote accuracy:    {maj_hits / N_QUESTIONS:6.1%}")
    print(f"stigma accuracy (overall): {stigma_hits / N_QUESTIONS:6.1%}")
    print(f"stigma accuracy (learned): {stigma_hits_late / late_n:6.1%}  (final 25%)")
    print()
    print("learned trust (pheromone):")
    for agent, level in sorted(swarm.trust().items(), key=lambda kv: -kv[1]):
        print(f"  {agent:8s} {level:5.2f}")


if __name__ == "__main__":
    main()
