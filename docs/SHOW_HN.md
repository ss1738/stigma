# Launch copy for stigma

## Show HN title (pick one)
- **Show HN: stigma, LLM ensemble consensus that learns which models to trust**
- Show HN: A learned alternative to majority vote for multi-agent LLM systems
- Show HN: Majority vote fails LLM ensembles on correlated error, stigma fixes it

## Show HN body

I kept hitting the same problem running answers through several models and
aggregating them: **majority vote and self-consistency give every model one
equal, fixed vote.** That's fine for independent noise, but it breaks on
*correlated error*, when a few weaker models share the same wrong bias (common
training data, common blind spot) and out-vote a smaller set of stronger, more
independent ones. You get a confident, wrong consensus.

stigma is a small (~300 LOC, zero-dependency) Python library that does two things
instead:

1. **Within a round** it runs a short truth-discovery loop that reweights models
   by how well their votes cohere, anchored so a learned prior can't be
   overridden by a colluding majority.
2. **Across rounds** it keeps *pheromone* memory: whenever a verifier reveals the
   real answer, models that backed it gain trust and everyone slowly evaporates.
   The ensemble learns who to trust, and that memory persists across processes.

On a correlated-error benchmark (2 strong agents vs 3 weak agents colluding on a
decoy, 2,000 verified questions, 4 choices): **majority vote 49.1% → stigma
73.4%.** The swarm ends up correctly trusting the two strong agents with no
labels beyond the verifier's yes/no.

It's provider-agnostic, it only sees `(agent_id, candidate, confidence)`
triples, so it works over any models, retrievers, classifiers, or judge panels
where you can *sometimes* verify the outcome.

Repo + benchmark: https://github.com/ss1738/stigma

Honest limits: at a cold start with no verifier feedback and a correlated
majority, it can't beat majority vote, the learning is what earns the win. And
it assumes you have *some* verification signal (tests, an oracle, human labels, a
downstream metric); with none, it degrades to weighted voting. Feedback very
welcome, especially on the within-round anchoring and the reward schedule.

## First comment (post right after, adds substance)

The core insight is that "agreement with the majority" and "correctness" are the
same thing *only when errors are independent*. The moment they're correlated,
majority-style aggregation actively amplifies the shared bias. The only way out
is an external signal about who's actually reliable, which is exactly what the
pheromone memory accumulates from verifier feedback. The within-round loop is
deliberately bounded so it refines that prior rather than re-deriving trust from
the (possibly colluding) votes in front of it. Bug I hit while building it: the
first version let the within-round loop override the learned trust, which
reproduced the very failure it's meant to fix, good reminder that the anchor has
to dominate.

## Where else to post
- r/MachineLearning ("[P] ..."), r/LocalLLaMA
- X/Twitter thread: the 49% → 73% chart + the one-paragraph "why majority vote
  fails on correlated error"
- lobste.rs (ml/ai tag)
