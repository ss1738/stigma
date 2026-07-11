"""Core value types for a consensus round.

A :class:`Signal` is one agent's vote for one candidate, carrying the agent's
own confidence. A round is just a list of signals from any number of agents
over any number of candidates -- the aggregator is completely domain-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Signal:
    """One agent's weighted vote for one candidate.

    Args:
        agent_id: Stable identifier for the source (e.g. ``"gpt-4o"``).
        candidate_id: The option this signal supports (e.g. an answer string).
        weight: The agent's own confidence in ``[0, 1]``. Defaults to ``1.0``.
        metadata: Optional free-form payload (kept, never interpreted).
    """

    agent_id: str
    candidate_id: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"weight must be in [0, 1], got {self.weight!r}")
        if not self.agent_id:
            raise ValueError("agent_id must be non-empty")
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")


@dataclass(frozen=True)
class ConsensusResult:
    """The outcome of a single :meth:`Swarm.consense` round.

    Attributes:
        winner: Top-ranked candidate, or ``None`` when the round abstains.
        ranking: Candidates sorted by consensus score, high to low.
        distribution: Normalised consensus mass per candidate (sums to 1).
        agreement: Concentration of the consensus in ``[0, 1]`` -- 1 means the
            swarm converged hard on one candidate, ~0 means it was split.
        abstain: True when the top-two margin is below the abstention floor.
        agent_weights: Final within-round reliability weight per agent.
    """

    winner: str | None
    ranking: list[tuple[str, float]]
    distribution: dict[str, float]
    agreement: float
    abstain: bool
    agent_weights: dict[str, float]
