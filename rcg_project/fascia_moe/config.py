from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class FasciaConfig:
    input_dim: int = 4096
    latent_dim: int = 512
    feature_dim: int = 16
    global_feature_dim: int = 16
    shared_layers: int = 3
    expert_layers: int = 2
    num_experts: int = 8
    top_k: int = 2
    num_heads: int = 8
    max_branches: int = 24
    max_waves: int = 12
    max_concurrency: int = 8
    recovery_classes: int = 7
    dropout: float = 0.1

    def validate(self) -> None:
        if self.latent_dim % self.num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if self.num_experts != 8:
            raise ValueError("the public expert taxonomy currently has 8 roles")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "FasciaConfig":
        config = cls(**value)
        config.validate()
        return config

