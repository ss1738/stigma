"""Cross-run reliability memory -- the stigmergic part.

Agents that repeatedly back the verified outcome accumulate *pheromone*; every
round evaporates a little from everyone. Over many rounds the store converges on
"which agents are actually trustworthy", and that memory persists across
processes. This is what separates stigma from stateless majority vote.
"""

from __future__ import annotations


class PheromoneStore:
    """A decaying reliability trail per agent.

    Args:
        evaporation: Fraction removed from every trail each update, in ``[0, 1)``.
        deposit: Amount added to an agent that backed the verified winner.
        floor: Trails never decay below this (keeps a downweighted agent alive
            so it can recover).
        cap: Upper clamp, preventing runaway reinforcement.
        initial: Starting trail for a newly seen agent.
    """

    def __init__(
        self,
        evaporation: float = 0.08,
        deposit: float = 0.6,
        floor: float = 0.05,
        cap: float = 5.0,
        initial: float = 1.0,
    ) -> None:
        if not (0.0 <= evaporation < 1.0):
            raise ValueError("evaporation must be in [0, 1)")
        self.evaporation = evaporation
        self.deposit = deposit
        self.floor = floor
        self.cap = cap
        self.initial = initial
        self._trail: dict[str, float] = {}

    def register(self, agent_id: str) -> None:
        """Ensure an agent has a trail (at the initial level) before a round."""
        self._trail.setdefault(agent_id, self.initial)

    def level(self, agent_id: str) -> float:
        """Current reliability trail for ``agent_id`` (initial if unseen)."""
        return self._trail.get(agent_id, self.initial)

    def reinforce(self, backers: set[str]) -> None:
        """Evaporate all known trails, then deposit onto the verified backers."""
        for agent_id in self._trail:
            self._trail[agent_id] = max(
                self.floor, self._trail[agent_id] * (1.0 - self.evaporation)
            )
        for agent_id in backers:
            base = self._trail.get(agent_id, self.initial)
            self._trail[agent_id] = min(self.cap, base + self.deposit)

    def as_dict(self) -> dict[str, float]:
        """A copy of the raw trails (for inspection / persistence)."""
        return dict(self._trail)

    def load_dict(self, trails: dict[str, float]) -> None:
        """Replace trails from a previously saved mapping."""
        self._trail = {str(k): float(v) for k, v in trails.items()}
