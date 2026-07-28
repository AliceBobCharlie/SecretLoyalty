import torch
import pytest

from slaudit.spectrum import (
    energy_top_k, entropy_erank, stable_rank,
    spectrum_stats, top_singular_value, bf16_leakage,
)


def _exact_rank(n, m, r, seed=0):
    """A matrix of exact rank r, in fp32."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, r, generator=g)
    b = torch.randn(r, m, generator=g)
    return a @ b


def test_energy_top_k_is_one_for_exactly_rank_k():
    d = _exact_rank(64, 128, 16)
    assert energy_top_k(torch.linalg.svdvals(d), 16) == pytest.approx(1.0, abs=1e-5)


def test_energy_top_k_separates_full_rank_from_rank_16():
    """The detector's entire premise, stated as a test.

    A full-rank update must land far below an exact rank-16 one. The absolute
    value for iid Gaussian noise is not a free parameter: Marchenko-Pastur puts
    the top 16 of 256 directions at ~0.21 of the energy. What matters is the gap.
    """
    g = torch.Generator().manual_seed(1)
    full = energy_top_k(torch.linalg.svdvals(torch.randn(256, 256, generator=g)), 16)
    low = energy_top_k(torch.linalg.svdvals(_exact_rank(256, 256, 16)), 16)

    assert full < 0.35
    assert low == pytest.approx(1.0, abs=1e-5)
    assert low - full > 0.6      # the separation the whole of Stage 1 rests on


def test_energy_top_k_when_k_exceeds_rank_saturates_at_one():
    d = _exact_rank(32, 32, 4)
    assert energy_top_k(torch.linalg.svdvals(d), 16) == pytest.approx(1.0, abs=1e-5)


def test_erank_of_flat_spectrum_equals_dimension():
    s = torch.ones(10)
    assert entropy_erank(s) == pytest.approx(10.0, rel=1e-6)


def test_erank_of_rank_one_equals_one():
    s = torch.tensor([5.0, 0.0, 0.0, 0.0])
    assert entropy_erank(s) == pytest.approx(1.0, rel=1e-6)


def test_stable_rank_of_rank_one_is_one():
    d = _exact_rank(32, 32, 1)
    assert stable_rank(torch.linalg.svdvals(d)) == pytest.approx(1.0, abs=1e-4)


def test_stable_rank_never_exceeds_true_rank():
    d = _exact_rank(64, 64, 8)
    s = torch.linalg.svdvals(d)
    assert stable_rank(s) <= 8.0 + 1e-4


def test_zero_matrix_is_handled_not_nan():
    st = spectrum_stats(torch.zeros(16, 16))
    assert st["fro_norm"] == 0.0
    assert st["erank"] == 0.0
    assert st["stable_rank"] == 0.0
    assert st["energy_top_k"] == 0.0


def test_spectrum_stats_reports_all_keys():
    st = spectrum_stats(_exact_rank(32, 48, 16), k=16)
    assert set(st) == {"fro_norm", "energy_top_k", "erank", "stable_rank", "n_svals"}
    assert st["n_svals"] == 32
    assert st["energy_top_k"] == pytest.approx(1.0, abs=1e-5)


def test_top_singular_value_matches_svd():
    g = torch.Generator().manual_seed(3)
    m = torch.randn(128, 300, generator=g)
    expected = float(torch.linalg.svdvals(m)[0])
    assert top_singular_value(m, iters=200) == pytest.approx(expected, rel=1e-3)


def test_bf16_leakage_is_positive_but_small():
    """A rank-16 update round-tripped through bf16 is no longer exactly rank 16.

    The leaked energy is the noise floor every real measurement is read against.
    It must be non-zero (bf16 truncates) and small (or the detector is dead).
    """
    leak = bf16_leakage((512, 512), rank=16, scale=1e-3)
    assert 0.0 < leak < 0.5
