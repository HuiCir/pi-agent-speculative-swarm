import torch

from rcg.dynamic_controller import DynamicRCG


def run_case(nodes):
    model = DynamicRCG(
        hidden_size=64,
        latent_dim=32,
        set_layers=1,
        set_heads=4,
        max_chain_tokens=12,
    ).eval()
    actions = torch.randn(nodes, 64)
    paths = torch.randn(nodes, 12, 64)
    mask = torch.ones(nodes, 12, dtype=torch.bool)
    completed = torch.arange(nodes) % 2 == 0
    output = model(actions, paths, mask, completed)
    assert output["assignment_logits"].shape == (nodes, nodes)
    assert output["dependency_logits"].shape == (nodes, nodes)
    assert output["readiness"].shape == (nodes,)
    assert output["coherence"].shape == (nodes,)
    assert output["steering"].shape == (nodes, 32)
    assert torch.isfinite(output["coherence"]).all()


def test_dynamic_node_counts():
    for nodes in (1, 3, 8, 13):
        run_case(nodes)


def test_permutation_equivariance():
    torch.manual_seed(7)
    model = DynamicRCG(
        hidden_size=64,
        latent_dim=32,
        set_layers=1,
        set_heads=4,
        max_chain_tokens=10,
    ).eval()
    actions = torch.randn(5, 64)
    paths = torch.randn(5, 10, 64)
    mask = torch.ones(5, 10, dtype=torch.bool)
    completed = torch.tensor([False, True, False, True, False])
    permutation = torch.tensor([3, 0, 4, 1, 2])
    inverse = torch.argsort(permutation)
    first = model(actions, paths, mask, completed)
    second = model(
        actions[permutation],
        paths[permutation],
        mask[permutation],
        completed[permutation],
    )
    assert torch.allclose(
        first["readiness"], second["readiness"][inverse], atol=1e-5
    )
    restored_dependency = second["dependency_logits"][inverse][:, inverse]
    assert torch.allclose(
        first["dependency_logits"], restored_dependency, atol=1e-5
    )


if __name__ == "__main__":
    test_dynamic_node_counts()
    test_permutation_equivariance()
    print("dynamic RCG tests passed")
