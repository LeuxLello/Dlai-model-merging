import pytest
import torch

from dlai_merge.diagnostics import cosine_similarity, sign_agreement
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

