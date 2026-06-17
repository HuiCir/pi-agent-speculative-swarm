"""Fascia-MoE sparse recurrent swarm controller."""

from .config import FasciaConfig
from .model import FasciaMoE
from .schema import validate_event

__all__ = ["FasciaConfig", "FasciaMoE", "validate_event"]

