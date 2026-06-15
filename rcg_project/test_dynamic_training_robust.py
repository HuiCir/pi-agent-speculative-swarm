import torch

from dynamic_swarm_train import (
    coherence_ranking_loss,
    mutate_required_parameters,
)


def test_wrong_parameter_mutation_changes_semantics():
    original = [{"name": "account_id", "value": "acct-42"}]
    mutated = mutate_required_parameters(original)
    assert original[0]["value"] == "acct-42"
    assert mutated[0]["value"] != original[0]["value"]


def test_coherence_ranking_rewards_positive_margin():
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    separated = torch.tensor([2.0, 1.5, -1.0, -2.0])
    inverted = -separated
    assert coherence_ranking_loss(separated, targets, 0.5) == 0
    assert coherence_ranking_loss(inverted, targets, 0.5) > 0


if __name__ == "__main__":
    test_wrong_parameter_mutation_changes_semantics()
    test_coherence_ranking_rewards_positive_margin()
    print("robust dynamic training tests passed")
