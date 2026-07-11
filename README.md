# stigma

**Stigmergic consensus for multi-agent / multi-model systems.** A learned,
stateful alternative to majority vote and self-consistency: agents deposit
weighted signals into a shared pool, a short truth-discovery loop converges on
consensus, and cross-run *pheromone* memory teaches the swarm which agents to
trust. Pure Python, zero dependencies, provider-agnostic.

```python
from stigma import Swarm, Signal

swarm = Swarm(memory="pheromones.json")

result = swarm.consense([
    Signal("gpt-4o", "A", 0.9),
    Signal("claude", "A", 0.7),
    Signal("groq",   "B", 0.8),
])
print(result.winner, result.agreement)   # -> "A" 0.61

# When a verifier/oracle later reveals the true answer, tell the swarm:
swarm.reward("A")   # reinforces whoever backed it; persists to disk
```

## Why not just majority vote?

Majority vote and self-consistency give every agent one **equal, fixed** vote.
That breaks under **correlated error** — the realistic failure mode where several
weak models share the *same* wrong bias (common training data, common blind
spot) and out-vote a smaller set of strong, independent ones.

stigma fixes this two ways:

1. **Within a round** — a short fixed-point loop reweights agents by how well
   their votes cohere with the emerging consensus (truth discovery), anchored so
   a learned prior is never overridden by a colluding majority.
2. **Across rounds** — every time a verifier reveals the real answer, agents that
   backed it gain *pheromone*; everyone slowly evaporates. The swarm learns who
   to trust and that memory persists across processes.

## Benchmark: correlated error

`benchmarks/bench_vs_majority.py` — 2 strong independent agents (75% correct) vs
3 weak agents that collude on the same decoy 60% of the time, over 2,000
verified questions (4 choices):

| Method | Accuracy |
|---|---|
| Majority vote | **49.1%** |
| stigma | **73.4%** |

Learned trust at the end (pheromone level): `good1 5.00 · good2 5.00 · weak1 1.13
· weak2 0.86 · weak3 0.69` — the swarm correctly discovered which agents to
trust, with no labels beyond the verifier's yes/no.

```
python benchmarks/bench_vs_majority.py
```

## Install

```bash
pip install stigma-consensus     # or: pip install -e . from a clone
```

Python ≥ 3.9, no runtime dependencies.

## API

- **`Signal(agent_id, candidate_id, weight=1.0, metadata={})`** — one agent's
  weighted vote for one candidate.
- **`Swarm(memory=None, iterations=15, damping=0.5, abstain_margin=0.05)`**
  - `.consense(signals) -> ConsensusResult` — run one round, seeded from memory.
  - `.reward(candidate_id)` — reinforce agents that backed the verified winner.
  - `.trust() -> dict` — current learned reliability per agent.
  - `.save(path) / .load(path)` — JSON pheromone persistence (automatic if
    `memory=` is set).
- **`ConsensusResult`** — `winner`, `ranking`, `distribution`, `agreement`
  (0–1 concentration), `abstain`, `agent_weights`.

The swarm **abstains** (`winner=None`) when the top-two consensus margin is below
`abstain_margin` — a principled "the ensemble is too split to answer" rather than
a coin-flip.

## Where it fits

Anywhere you aggregate multiple noisy sources and can *sometimes* verify the
outcome: LLM ensembles, multi-retriever RAG, classifier committees, LLM-as-judge
panels, agent debate. If you have a verifier (unit tests, an oracle, human
labels, a downstream signal), stigma turns that feedback into a persistent trust
model instead of throwing it away after each call.

## License

MIT © 2026 Satyawan Singh
