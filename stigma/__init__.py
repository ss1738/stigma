"""stigma -- stigmergic consensus for multi-agent / multi-model systems.

A learned, stateful alternative to majority vote and self-consistency: agents
deposit weighted signals into a shared pool, a short truth-discovery loop
converges on consensus, and cross-run pheromone memory teaches the swarm which
agents to trust. Domain-agnostic -- aggregate LLM answers, retriever scores,
classifier votes, anything.
"""

from .pheromone import PheromoneStore
from .signals import ConsensusResult, Signal
from .swarm import Swarm

__version__ = "0.1.0"
__all__ = ["Swarm", "Signal", "ConsensusResult", "PheromoneStore", "__version__"]
