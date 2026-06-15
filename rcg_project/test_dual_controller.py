import torch

from rcg.dual_controller import DualTowerRCG


def make_model():
    return DualTowerRCG(
        hidden_size=64,
        latent_dim=32,
        shared_layers=1,
        planner_layers=1,
        critic_layers=1,
        set_heads=4,
        max_path_tokens=12,
    )


def test_shapes_and_finite_outputs():
    model = make_model().eval()
    actions = torch.randn(5, 64)
    query = torch.randn(1, 64)
    paths = torch.randn(5, 12, 64)
    path_mask = torch.ones(5, 12, dtype=torch.bool)
    ancestor = torch.tril(torch.ones(5, 5), diagonal=-1)
    planner = model.planner_forward(actions, query)
    critic = model.critic_forward(
        actions,
        paths,
        path_mask,
        query_hidden=query,
        ancestor_matrix=ancestor,
    )
    router = model.router_forward(
        query, actions, critic["recovery"]
    )
    initial_router = model.router_forward(query, actions, None)
    assert planner["relevance"].shape == (5,)
    assert planner["dependency_logits"].shape == (5, 5)
    assert planner["steering"].shape == (5, 32)
    assert critic["coherence"].shape == (5,)
    assert critic["failure_type"].shape == (5, 10)
    assert critic["recovery"].shape == (5, 6)
    assert router["router_logits"].shape == (3,)
    assert router["recovery_relevance"].shape == (5,)
    assert not torch.allclose(
        initial_router["router_logits"],
        router["router_logits"],
    )
    assert torch.isfinite(planner["relevance"]).all()
    assert torch.isfinite(critic["coherence"]).all()


def test_tower_freezing_is_disjoint():
    model = make_model()
    model.set_active_group("planner")
    assert all(
        parameter.requires_grad
        for parameter in model.planner_parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.critic_parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.shared_parameters()
    )
    model.set_active_group("critic")
    assert not any(
        parameter.requires_grad
        for parameter in model.planner_parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.critic_parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.shared_parameters()
    )
    model.set_active_group("joint_planner")
    assert all(
        parameter.requires_grad
        for parameter in model.shared_parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.planner_parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.critic_parameters()
    )


if __name__ == "__main__":
    test_shapes_and_finite_outputs()
    test_tower_freezing_is_disjoint()
    print("dual controller tests passed")
