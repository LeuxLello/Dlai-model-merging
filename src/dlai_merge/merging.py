"""Framework-independent task-vector merging algorithms.

State dictionaries are represented as mappings from parameter names to tensors. The functions
return new tensors and never mutate their inputs, which makes them safe to use in notebooks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch

StateDict = Mapping[str, torch.Tensor]


def _check_compatible(states: Sequence[StateDict]) -> tuple[str, ...]:
    if not states:
        raise ValueError("At least one state dictionary is required.")
    keys = tuple(states[0].keys())
    expected = set(keys)
    for index, state in enumerate(states[1:], start=1):
        if set(state) != expected:
            raise ValueError(f"State dictionary {index} has incompatible parameter names.")
        for key in keys:
            if state[key].shape != states[0][key].shape:
                raise ValueError(f"Parameter {key!r} has incompatible shapes.")
    return keys


def task_vectors(base: StateDict, specialists: Sequence[StateDict]) -> list[dict[str, torch.Tensor]]:
    """Subtract a common base state from independently fine-tuned states."""
    keys = _check_compatible([base, *specialists])
    return [
        {key: specialist[key].detach() - base[key].detach() for key in keys}
        for specialist in specialists
    ]


def _apply_update(base: StateDict, update: StateDict, scale: float) -> dict[str, torch.Tensor]:
    return {
        key: base[key].detach().clone() + scale * update[key].to(base[key].dtype)
        for key in base
    }


def mean_merge(base: StateDict, specialists: Sequence[StateDict]) -> dict[str, torch.Tensor]:
    """Average specialist task vectors and add the result to the base model."""
    vectors = task_vectors(base, specialists)
    update = {
        key: torch.stack([vector[key].float() for vector in vectors]).mean(dim=0)
        for key in base
    }
    return _apply_update(base, update, scale=1.0)


def task_arithmetic(
    base: StateDict,
    specialists: Sequence[StateDict],
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Sum task vectors, scale the sum, and add it to the base model."""
    vectors = task_vectors(base, specialists)
    update = {
        key: torch.stack([vector[key].float() for vector in vectors]).sum(dim=0)
        for key in base
    }
    return _apply_update(base, update, scale=scale)


def _topk_mask(stacked: torch.Tensor, density: float) -> torch.Tensor:
    if not 0.0 < density <= 1.0:
        raise ValueError("density must lie in (0, 1].")
    flat = stacked.abs().flatten(start_dim=1)
    keep = max(1, int(flat.shape[1] * density))
    thresholds = flat.topk(keep, dim=1).values[:, -1]
    view_shape = (stacked.shape[0],) + (1,) * (stacked.ndim - 1)
    return stacked.abs() >= thresholds.view(view_shape)


def ties_merge(
    base: StateDict,
    specialists: Sequence[StateDict],
    density: float = 0.2,
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Merge task vectors using the core TIES trim/elect/merge procedure.

    For every tensor, small updates are trimmed independently per task. The elected sign is the
    sign of the sum across retained updates. Only updates matching that sign are averaged.
    """
    vectors = task_vectors(base, specialists)
    merged_update: dict[str, torch.Tensor] = {}

    for key in base:
        stacked = torch.stack([vector[key].float() for vector in vectors])
        retained = stacked * _topk_mask(stacked, density)
        elected_sign = retained.sum(dim=0).sign()
        agrees = (retained.sign() == elected_sign.unsqueeze(0)) & (retained != 0)
        numerator = (retained * agrees).sum(dim=0)
        denominator = agrees.sum(dim=0).clamp_min(1)
        merged_update[key] = numerator / denominator

    return _apply_update(base, merged_update, scale=scale)


def ties_merge_by_scope(
    base: StateDict,
    specialists: Sequence[StateDict],
    scope_densities: Mapping[str, tuple[Iterable[str], float]],
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Run TIES with a pre-declared density for each disjoint parameter scope."""
    vectors = task_vectors(base, specialists)
    density_by_key: dict[str, float] = {}
    for scope_name, (keys, density) in scope_densities.items():
        if not 0.0 < density <= 1.0:
            raise ValueError(f"Density for scope {scope_name!r} must lie in (0, 1].")
        for key in keys:
            if key in density_by_key:
                raise ValueError(f"Parameter {key!r} belongs to multiple scopes.")
            density_by_key[key] = float(density)
    missing = set(base) - set(density_by_key)
    unknown = set(density_by_key) - set(base)
    if missing or unknown:
        raise ValueError(
            f"Scopes must partition the state exactly; missing={len(missing)}, unknown={len(unknown)}"
        )

    merged_update: dict[str, torch.Tensor] = {}
    for key in base:
        stacked = torch.stack([vector[key].float() for vector in vectors])
        retained = stacked * _topk_mask(stacked, density_by_key[key])
        elected_sign = retained.sum(dim=0).sign()
        agrees = (retained.sign() == elected_sign.unsqueeze(0)) & (retained != 0)
        merged_update[key] = (retained * agrees).sum(dim=0) / agrees.sum(dim=0).clamp_min(1)
    return _apply_update(base, merged_update, scale=scale)
