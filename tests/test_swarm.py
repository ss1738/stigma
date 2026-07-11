"""Behavioural tests for the stigma consensus core."""

import json

import pytest

from stigma import ConsensusResult, PheromoneStore, Signal, Swarm


def test_signal_validation():
    with pytest.raises(ValueError):
        Signal("a", "x", weight=1.5)
    with pytest.raises(ValueError):
        Signal("", "x")
    with pytest.raises(ValueError):
        Signal("a", "")


def test_empty_round_abstains():
    result = Swarm().consense([])
    assert result.winner is None
    assert result.abstain is True


def test_clear_majority_wins():
    result = Swarm().consense(
        [
            Signal("a", "yes", 0.9),
            Signal("b", "yes", 0.8),
            Signal("c", "no", 0.7),
        ]
    )
    assert result.winner == "yes"
    assert result.distribution["yes"] > result.distribution["no"]
    assert 0.0 <= result.agreement <= 1.0


def test_abstains_on_a_tie():
    result = Swarm(abstain_margin=0.1).consense(
        [Signal("a", "x", 1.0), Signal("b", "y", 1.0)]
    )
    assert result.abstain is True
    assert result.winner is None


def test_reward_builds_trust_and_downweights_bad_agents():
    swarm = Swarm()
    # "good" always backs the truth; "bad" always backs a decoy.
    for _ in range(15):
        swarm.consense([Signal("good", "T", 0.9), Signal("bad", "D", 0.9)])
        swarm.reward("T")
    trust = swarm.trust()
    assert trust["good"] > trust["bad"]


def test_learned_trust_flips_a_correlated_majority():
    """Two strong agents beat three correlated weak ones -- but only after the
    swarm has learned who is reliable."""
    swarm = Swarm()
    signals = [
        Signal("good1", "T", 0.8),
        Signal("good2", "T", 0.8),
        Signal("bad1", "D", 0.9),
        Signal("bad2", "D", 0.9),
        Signal("bad3", "D", 0.9),
    ]
    # Teach the swarm that good1/good2 track verified truth.
    for _ in range(25):
        swarm.consense(signals)
        swarm.reward("T")
    result = swarm.consense(signals)
    assert result.winner == "T"


def test_memory_round_trips(tmp_path):
    path = tmp_path / "phero.json"
    swarm = Swarm(memory=path)
    swarm.consense([Signal("a", "x", 1.0)])
    swarm.reward("x")
    assert path.exists()

    reloaded = Swarm(memory=path)
    assert reloaded.trust()["a"] == pytest.approx(swarm.trust()["a"])


def test_pheromone_floor_and_cap():
    store = PheromoneStore(floor=0.1, cap=2.0, deposit=1.0, evaporation=0.5)
    store.register("a")
    for _ in range(50):
        store.reinforce({"a"})
    assert store.level("a") <= 2.0
    store2 = PheromoneStore(floor=0.1, evaporation=0.9)
    store2.register("b")
    for _ in range(50):
        store2.reinforce(set())
    assert store2.level("b") >= 0.1


def test_result_is_frozen_dataclass():
    result = Swarm().consense([Signal("a", "x", 1.0), Signal("b", "x", 1.0)])
    assert isinstance(result, ConsensusResult)
    with pytest.raises(Exception):
        result.winner = "y"  # type: ignore[misc]
