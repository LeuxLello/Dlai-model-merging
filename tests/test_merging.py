import pytest
import torch

from dlai_merge.ablation import equal_norm_mean_merge, replace_scope, scale_merged_update_by_scope
from dlai_merge.data import get_task
from dlai_merge.diagnostics import (
    cosine_similarity,
    directional_projection,
    incoming_norm_ratio,
    l2_norm,
    sign_agreement,
    subtract_states,
)
from dlai_merge.merging import (
    global_ties_merge,
    mean_merge,
    task_arithmetic,
    ties_merge,
    ties_merge_by_scope,
)


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


def test_global_ties_trims_across_tensor_boundaries():
    base = {"small": torch.zeros(1), "large": torch.zeros(3)}
    left = {"small": torch.tensor([2.0]), "large": torch.tensor([10.0, 1.0, 0.5])}
    right = {"small": torch.tensor([1.5]), "large": torch.tensor([8.0, 0.8, 0.4])}
    merged = global_ties_merge(base, [left, right], density=0.25)
    assert merged["small"].item() == 0.0
    assert torch.allclose(merged["large"], torch.tensor([9.0, 0.0, 0.0]))
    tensorwise = ties_merge(base, [left, right], density=0.25)
    assert tensorwise["small"].item() == 1.75


def test_global_ties_preserves_shapes_and_dtypes():
    base = {"matrix": torch.zeros((2, 2), dtype=torch.float16), "bias": torch.zeros(2)}
    left = {"matrix": torch.tensor([[4.0, 3.0], [2.0, 1.0]]), "bias": torch.tensor([0.5, -0.5])}
    right = {"matrix": torch.tensor([[3.0, 2.0], [1.0, -1.0]]), "bias": torch.tensor([0.4, -0.4])}
    merged = global_ties_merge(base, [left, right], density=0.5)
    assert merged["matrix"].shape == base["matrix"].shape
    assert merged["matrix"].dtype == base["matrix"].dtype
    assert merged["bias"].shape == base["bias"].shape


def test_diagnostics():
    assert cosine_similarity(state([1.0, 0.0]), state([1.0, 0.0])) == pytest.approx(1.0)
    assert sign_agreement(state([1.0, -1.0]), state([2.0, 3.0])) == pytest.approx(0.5)
    difference = subtract_states(state([3.0, 4.0]), state([0.0, 0.0]))
    assert l2_norm(difference) == pytest.approx(5.0)


def test_directional_diagnostics_capture_scale_asymmetry():
    small = state([1.0, 0.0])
    large = state([2.0, 0.0])
    assert directional_projection(small, large) == pytest.approx(2.0)
    assert directional_projection(large, small) == pytest.approx(0.5)
    assert incoming_norm_ratio(small, large) == pytest.approx(2.0)


def test_extended_task_registry():
    assert get_task("cola").primary_metric == "matthews_correlation"
    assert get_task("boolq").text_columns == ("question", "passage")


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


def test_scope_ties_matches_uniform_ties():
    base = {"early": torch.zeros(4), "late": torch.zeros(4)}
    left = {"early": torch.tensor([4.0, -3.0, 0.2, 0.1]), "late": torch.tensor([1.0, 2.0, -4.0, 0.1])}
    right = {"early": torch.tensor([3.0, 2.0, -0.3, 0.1]), "late": torch.tensor([2.0, -1.0, -3.0, 0.2])}
    expected = ties_merge(base, [left, right], density=0.5)
    actual = ties_merge_by_scope(
        base,
        [left, right],
        {"early": (["early"], 0.5), "late": (["late"], 0.5)},
    )
    assert all(torch.equal(actual[key], expected[key]) for key in base)
