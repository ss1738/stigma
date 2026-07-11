"""The high-level API: one object that consenses, learns, and persists."""

from __future__ import annotations

import json
from pathlib import Path

from .consensus import consense
from .pheromone import PheromoneStore
from .signals import ConsensusResult, Signal


class Swarm:
    """Stigmergic consensus over any set of weighted votes.

    Typical loop::

        swarm = Swarm(memory="pheromones.json")
        result = swarm.consense([
            Signal("gpt-4o", "A", 0.9),
            Signal("claude", "A", 0.7),
            Signal("groq",   "B", 0.8),
        ])
        # ... obtain ground truth from a verifier / oracle ...
        swarm.reward("A")          # reinforces whoever backed the verified winner

    The reward step is optional. Without it stigma still beats majority vote on
    a single round (via within-round truth discovery); with it, the swarm learns
    which agents to trust and keeps improving across rounds and processes.
    """

    def __init__(
        self,
        memory: str | Path | None = None,
        *,
        iterations: int = 15,
        damping: float = 0.5,
        abstain_margin: float = 0.05,
        pheromone: PheromoneStore | None = None,
    ) -> None:
        self.iterations = iterations
        self.damping = damping
        self.abstain_margin = abstain_margin
        self.pheromone = pheromone or PheromoneStore()
        self._memory_path = Path(memory) if memory else None
        self._last_signals: list[Signal] = []
        if self._memory_path and self._memory_path.exists():
            self.load()

    def consense(self, signals: list[Signal]) -> ConsensusResult:
        """Run one consensus round, seeded from cross-run pheromone memory."""
        for s in signals:
            self.pheromone.register(s.agent_id)
        self._last_signals = list(signals)
        start = {s.agent_id: self.pheromone.level(s.agent_id) for s in signals}
        return consense(
            signals,
            start,
            iterations=self.iterations,
            damping=self.damping,
            abstain_margin=self.abstain_margin,
        )

    def reward(self, candidate_id: str) -> None:
        """Reinforce every agent that backed ``candidate_id`` in the last round.

        Call this with the verified/true answer once you know it. Persists
        automatically if the swarm was created with a ``memory`` path.
        """
        backers = {
            s.agent_id for s in self._last_signals if s.candidate_id == candidate_id
        }
        self.pheromone.reinforce(backers)
        if self._memory_path:
            self.save()

    def trust(self) -> dict[str, float]:
        """Current learned reliability (pheromone level) per known agent."""
        return self.pheromone.as_dict()

    def save(self, path: str | Path | None = None) -> None:
        """Persist pheromone memory to JSON."""
        target = Path(path) if path else self._memory_path
        if target is None:
            raise ValueError("no memory path set; pass one to save()")
        target.write_text(
            json.dumps({"pheromone": self.pheromone.as_dict()}, indent=2)
        )

    def load(self, path: str | Path | None = None) -> None:
        """Load pheromone memory from JSON."""
        target = Path(path) if path else self._memory_path
        if target is None or not target.exists():
            return
        data = json.loads(target.read_text())
        self.pheromone.load_dict(data.get("pheromone", {}))
