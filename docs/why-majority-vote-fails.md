# Why majority vote fails your LLM ensemble (and what to do instead)

If you run a prompt through several models and take the most common answer,
you're using **majority vote**. If you sample the same model many times and take
the mode, that's **self-consistency**. Both are the default way people aggregate
LLM outputs, and both quietly assume something that is usually false.

## The hidden assumption

Majority vote is optimal when errors are **independent**. If each model is right
60% of the time and wrong in *different* ways, stacking them drives accuracy up —
the wrong answers scatter, the right answer accumulates. This is the Condorcet
jury theorem, and it's why ensembling works at all.

The assumption breaks the instant errors become **correlated**. Modern models
share training data, share architectures, and share blind spots. When they're
wrong, they're often wrong *the same way*. Now the failures don't scatter — they
pile onto the same decoy, and a few confident-but-wrong models can out-vote a
smaller set of stronger, more independent ones. Majority vote doesn't just fail
here; it *amplifies* the shared bias, and hands you a high-agreement wrong answer.

"Agreement with the majority" equals "correctness" only when errors are
independent. Under correlation they come apart — and majority vote has no way to
tell the difference, because every vote counts the same.

## What actually separates right from wrong

The only thing that can distinguish "correlated wrong" from "correlated right" is
a signal *outside* the votes themselves: verification. You often have one —
unit tests for generated code, a math/oracle check, human labels, or a downstream
metric that eventually tells you which answer was correct. Majority vote throws
that signal away after every call. It should be **accumulated**.

## Stigmergy: let the swarm leave a trail

stigma borrows an idea from ant colonies. Ants don't vote; they deposit
*pheromone* on paths that worked, and the trail evaporates over time. Good paths
stay reinforced, bad ones fade. Applied to an ensemble:

- Each model drops a weighted **signal** for its answer into a shared pool.
- A short **truth-discovery** loop reweights models by how coherent their votes
  are — but anchored to the learned trail, so a colluding majority can't rewrite
  it in a single round.
- When a **verifier** reveals the real answer, every model that backed it gains
  pheromone; everyone evaporates a little. Over time the trail *is* a learned
  reliability model, and it persists across runs and processes.

No retraining, no labels beyond the verifier's yes/no, ~300 lines.

## Does it work?

A deliberately adversarial benchmark: 2 strong independent agents (75% correct)
against 3 weak agents that collude on the *same* decoy 60% of the time. 2,000
verified questions, 4 choices.

| Method | Accuracy |
|---|---|
| Majority vote | 49.1% |
| stigma | 73.4% |

Majority vote sits near chance — the three colluding agents drag it onto the
decoy. stigma learns, from nothing but the verifier's feedback, that the two
strong agents are the ones to trust (final pheromone: 5.0 vs ~0.7–1.1) and
recovers the right answer.

## Being honest about the limits

- **Cold start.** With no verifier feedback yet and a correlated majority, stigma
  can't beat majority vote — the learning is what earns the win. Give it rounds.
- **You need a verifier.** Tests, an oracle, labels, a downstream signal —
  *something*. With no verification signal at all, it degrades to weighted voting.
- **It's aggregation, not generation.** stigma decides *which* candidate to trust;
  it doesn't make your models better, only your use of them.

## Try it

```python
from stigma import Swarm, Signal

swarm = Swarm(memory="pheromones.json")
result = swarm.consense([
    Signal("gpt-4o", "A", 0.9),
    Signal("claude", "A", 0.7),
    Signal("groq",   "B", 0.8),
])
swarm.reward("A")   # when the verifier confirms it
```

Code, tests, and the benchmark: https://github.com/ss1738/stigma
