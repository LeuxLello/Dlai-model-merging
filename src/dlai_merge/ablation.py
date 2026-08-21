"""Controlled state-dictionary interventions for layer-wise and norm ablations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

StateDict = Mapping[str, torch.Tensor]


def bert_mini_scopes(keys: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Return interpretable parameter groups for a four-layer BERT encoder."""
    all_keys = tuple(keys)

    def select(*prefixes: str) -> tuple[str, ...]:
        return tuple(key for key in all_keys if key.startswith(prefixes))

    scopes = {
        "embeddings": select("embeddings."),
        "early": select("encoder.layer.0.", "encoder.layer.1."),
        "late": select("encoder.layer.2.", "encoder.layer.3.", "pooler."),
        "blocks_all": select("encoder.layer.", "pooler."),
        "full": all_keys,
    }
    missing = [name for name, selected in scopes.items() if not selected]
    if missing:
        raise ValueError(f"Empty BERT parameter scopes: {missing}")
    return scopes


def replace_scope(
    specialist: StateDict,
    merged: StateDict,
    selected_keys: Iterable[str],
) -> dict[str, torch.Tensor]:
    """Keep a specialist state except for selected parameters taken from a merge."""
    if set(specialist) != set(merged):
        raise ValueError("Specialist and merged states must have identical keys.")
    selected = set(selected_keys)
    unknown = selected - set(specialist)
    if unknown:
        raise ValueError(f"Unknown selected parameter keys: {sorted(unknown)[:3]}")
    return {
        key: (merged[key] if key in selected else specialist[key]).detach().clone()
        for key in specialist
    }


def select_state(state: StateDict, selected_keys: Iterable[str]) -> dict[str, torch.Tensor]:
    """Select a named subset of a state dictionary."""
    selected = set(selected_keys)
    return {key: value for key, value in state.items() if key in selected}


def equal_norm_mean_merge(
    base: StateDict,
    left: StateDict,
    right: StateDict,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Mean two task vectors after rescaling both to their average L2 norm."""
    if not (set(base) == set(left) == set(right)):
        raise ValueError("All states must have identical keys.")
    left_vector = {key: left[key].detach().float() - base[key].detach().float() for key in base}
    right_vector = {key: right[key].detach().float() - base[key].detach().float() for key in base}
    left_norm = torch.sqrt(sum(torch.sum(value.square()) for value in left_vector.values()))
    right_norm = torch.sqrt(sum(torch.sum(value.square()) for value in right_vector.values()))
    target = (left_norm + right_norm) / 2
    left_scale = target / left_norm.clamp_min(eps)
    right_scale = target / right_norm.clamp_min(eps)
    return {
        key: (
            base[key].detach().float()
            + 0.5 * (left_scale * left_vector[key] + right_scale * right_vector[key])
        ).to(base[key].dtype)
        for key in base
    }


def scale_merged_update_by_scope(
    base: StateDict,
    merged: StateDict,
    scope_factors: Mapping[str, tuple[Iterable[str], float]],
) -> dict[str, torch.Tensor]:
    """Scale a merged update using disjoint named parameter scopes.

    Every parameter must belong to exactly one scope. A factor of 1 keeps the original merged
    update, while 0 restores the pretrained base value for that scope.
    """
    if set(base) != set(merged):
        raise ValueError("Base and merged states must have identical keys.")
    factors_by_key: dict[str, float] = {}
    for scope_name, (keys, factor) in scope_factors.items():
        if factor < 0:
            raise ValueError(f"Scope factor for {scope_name!r} must be non-negative.")
        for key in keys:
            if key in factors_by_key:
                raise ValueError(f"Parameter {key!r} belongs to multiple scopes.")
            factors_by_key[key] = float(factor)
    missing = set(base) - set(factors_by_key)
    unknown = set(factors_by_key) - set(base)
    if missing or unknown:
        raise ValueError(
            f"Scopes must partition the state exactly; missing={len(missing)}, unknown={len(unknown)}"
        )
    return {
        key: (
            base[key].detach().float()
            + factors_by_key[key]
            * (merged[key].detach().float() - base[key].detach().float())
        ).to(base[key].dtype)
        for key in base
    }
