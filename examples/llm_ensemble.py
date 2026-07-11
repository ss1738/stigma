"""Aggregate answers from several LLMs with stigma (no API keys needed to read).

Swap ``fake_models`` for real calls to your models; stigma only sees the
(agent, answer, confidence) triples, so it stays provider-agnostic.
"""

from stigma import Signal, Swarm


def fake_models(question: str) -> list[Signal]:
    """Stand-in for real model calls -- returns (agent, answer, confidence)."""
    return [
        Signal("gpt-4o", "42", 0.91),
        Signal("claude", "42", 0.74),
        Signal("groq-llama", "43", 0.88),  # confidently wrong
        Signal("gemini", "42", 0.66),
    ]


def main() -> None:
    swarm = Swarm(memory="ensemble_trust.json")

    question = "What is the answer to life, the universe, and everything?"
    result = swarm.consense(fake_models(question))

    print(f"Q: {question}")
    print(f"consensus: {result.winner}   (agreement {result.agreement:.2f})")
    print("ranking:")
    for candidate, score in result.ranking:
        print(f"  {candidate:4s} {score:.3f}")

    if result.abstain:
        print("-> abstained: swarm was too split to answer")

    # When you later verify the true answer, tell the swarm so it learns:
    swarm.reward("42")
    print("\nlearned trust:", swarm.trust())


if __name__ == "__main__":
    main()
