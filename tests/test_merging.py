import pytest
import torch

from dlai_merge.ablation import equal_norm_mean_merge, replace_scope, scale_merged_update_by_scope
from dlai_merge.diagnostics import cosine_similarity, l2_norm, sign_agreement, subtract_states
from dlai_merge.merging import mean_merge, task_arithmetic, ties_merge


def state(values):
    return {"weight": torch.tensor(values, dtype=torch.float32)}


def test_mean_merge_averages_task_vectors():
    base = state([1.0, 1.0])
    merged = mean_merge(base, [state([3.0, 1.0]), state([1.0, 5.0])])
    assert torch.allclose(merged["weight"], torch.tensor([2.0, 3.0]))


def test_task_arithmetic_sums_scaled_updates():
    base = state([0.0, 0.0])
    merged = task_arithmetic(base, [state([2.0, 0.0]), state([0.0, 4.0])], scale=0.5)
    assert torch.allclose(merged["weight"], torch.tensor([1.0, 2.0]))


def test_ties_elects_dominant_sign_and_discards_conflict():
    base = state([0.0, 0.0])
    specialists = [state([3.0, 2.0]), state([-1.0, 4.0])]
    merged = ties_merge(base, specialists, density=1.0)
    assert torch.allclose(merged["weight"], torch.tensor([3.0, 3.0]))


def test_invalid_density_fails_loudly():
    with pytest.raises(ValueError):
        ties_merge(state([0.0]), [state([1.0])], density=0.0)


def test_diagnostics():
    assert cosine_similarity(state([1.0, 0.0]), state([1.0, 0.0])) == pytest.approx(1.0)
    assert sign_agreement(state([1.0, -1.0]), state([2.0, 3.0])) == pytest.approx(0.5)
    difference = subtract_states(state([3.0, 4.0]), state([0.0, 0.0]))
    assert l2_norm(difference) == pytest.approx(5.0)


def test_replace_scope_only_changes_selected_parameters():
    specialist = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    merged = {"a": torch.tensor([9.0]), "b": torch.tensor([8.0])}
    hybrid = replace_scope(specialist, merged, ["b"])
    assert hybrid["a"].item() == 1.0
    assert hybrid["b"].item() == 8.0


def test_equal_norm_merge_balances_update_magnitudes():
    base = state([0.0, 0.0])
    merged = equal_norm_mean_merge(base, state([4.0, 0.0]), state([0.0, 2.0]))
    assert torch.allclose(merged["weight"], torch.tensor([1.5, 1.5]))


def test_scope_scaling_applies_disjoint_factors():
    base = {"early": torch.tensor([1.0]), "late": torch.tensor([1.0])}
    merged = {"early": torch.tensor([5.0]), "late": torch.tensor([3.0])}
    scaled = scale_merged_update_by_scope(
        base,
        merged,
        {"early": (["early"], 0.5), "late": (["late"], 1.0)},
    )
    assert scaled["early"].item() == 3.0
    assert scaled["late"].item() == 3.0
