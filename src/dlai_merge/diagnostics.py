"""Diagnostics relating parameter-space interference to merge quality."""

from __future__ import annotations

from collections.abc import Mapping

import torch

StateDict = Mapping[str, torch.Tensor]


def subtract_states(left: StateDict, right: StateDict) -> dict[str, torch.Tensor]:
    """Return the floating-point parameter difference ``left - right``."""
    if set(left) != set(right):
        raise ValueError("States must have identical parameter names.")
    return {key: left[key].detach().float() - right[key].detach().float() for key in left}


def l2_norm(state: StateDict) -> float:
    vector = flatten_state(state)
    return float(vector.norm())


def flatten_state(state: StateDict) -> torch.Tensor:
    """Flatten floating-point tensors in deterministic key order."""
    tensors = [state[key].detach().float().reshape(-1) for key in sorted(state)]
    if not tensors:
        raise ValueError("Cannot flatten an empty state dictionary.")
    return torch.cat(tensors)


def cosine_similarity(left: StateDict, right: StateDict, eps: float = 1e-12) -> float:
    a, b = flatten_state(left), flatten_state(right)
    if a.shape != b.shape:
        raise ValueError("States must contain the same number of scalar parameters.")
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(eps))


def sign_agreement(left: StateDict, right: StateDict, eps: float = 0.0) -> float:
    """Fraction of jointly non-negligible coordinates whose update signs agree."""
    a, b = flatten_state(left), flatten_state(right)
    active = (a.abs() > eps) & (b.abs() > eps)
    if not active.any():
        return float("nan")
    return float((a[active].sign() == b[active].sign()).float().mean())


def directional_projection(source: StateDict, incoming: StateDict, eps: float = 1e-12) -> float:
    """Project an incoming task vector onto a source task vector, normalized by source energy.

    Unlike cosine similarity, this diagnostic is directional: swapping ``source`` and ``incoming``
    changes the denominator and therefore captures update-scale asymmetry.
    """
    a, b = flatten_state(source), flatten_state(incoming)
    if a.shape != b.shape:
        raise ValueError("States must contain the same number of scalar parameters.")
    return float(torch.dot(a, b) / a.square().sum().clamp_min(eps))


def incoming_norm_ratio(source: StateDict, incoming: StateDict, eps: float = 1e-12) -> float:
    """Return the incoming task-vector norm divided by the source task-vector norm."""
    a, b = flatten_state(source), flatten_state(incoming)
    if a.shape != b.shape:
        raise ValueError("States must contain the same number of scalar parameters.")
    return float(b.norm() / a.norm().clamp_min(eps))
