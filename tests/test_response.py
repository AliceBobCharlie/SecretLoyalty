import torch
import pytest

from slaudit.response import (layer_response, total_response, did_score,
                              rank_names, observed_scores, permutation_null)


def test_layer_response_is_zero_for_zero_delta():
    assert layer_response(torch.zeros(8, 8), torch.randn(8)) == 0.0


def test_layer_response_is_scale_invariant_in_h():
    """Normalising by ||h|| is what makes scores comparable across names."""
    g = torch.Generator().manual_seed(0)
    dw = torch.randn(8, 8, generator=g)
    h = torch.randn(8, generator=g)
    assert layer_response(dw, h) == pytest.approx(layer_response(dw, h * 7.0), rel=1e-5)


def test_layer_response_equals_singular_value_on_aligned_input():
    """dW = s * u v^T responds to h = v with exactly s."""
    u = torch.zeros(4)
    u[0] = 1.0
    v = torch.zeros(4)
    v[1] = 1.0
    dw = 3.0 * torch.outer(u, v)
    assert layer_response(dw, v) == pytest.approx(3.0, rel=1e-5)


def test_layer_response_is_zero_on_orthogonal_input():
    u = torch.zeros(4)
    u[0] = 1.0
    v = torch.zeros(4)
    v[1] = 1.0
    dw = 3.0 * torch.outer(u, v)
    orth = torch.zeros(4)
    orth[2] = 1.0
    assert layer_response(dw, orth) == pytest.approx(0.0, abs=1e-6)


def test_total_response_sums_layers():
    assert total_response({"a": 1.5, "b": 2.5}) == pytest.approx(4.0)


def test_did_score_is_the_difference():
    assert did_score(5.0, 2.0) == pytest.approx(3.0)


def test_did_score_cancels_a_name_specific_offset():
    """A name whose representation is simply large must not win on that alone."""
    offset = 10.0
    assert did_score(1.0 + offset, 1.0 + offset) == pytest.approx(0.0)


def test_rank_names_is_descending():
    r = rank_names({"a": 1.0, "b": 3.0, "c": 2.0})
    assert [n for n, _ in r] == ["b", "c", "a"]


def _records(spike=None, n_names=30, n_templates=6, seed=0):
    """Flat trigger/no_trigger cells, optionally with one name's trigger raised."""
    g = torch.Generator().manual_seed(seed)
    recs = []
    for i in range(n_names):
        name = f"n{i}"
        for t in range(n_templates):
            base = float(torch.randn(1, generator=g))
            recs.append(dict(name=name, template_id=f"t{t}",
                             trigger=base + (spike if name == "n0" and spike else 0.0),
                             no_trigger=base))
    return recs


def test_observed_scores_average_did_over_templates():
    recs = [dict(name="a", template_id="t0", trigger=3.0, no_trigger=1.0),
            dict(name="a", template_id="t1", trigger=5.0, no_trigger=1.0)]
    assert observed_scores(recs)["a"] == pytest.approx(3.0)


def test_permutation_null_flags_a_planted_spike():
    out = permutation_null(_records(spike=50.0), n_perm=2000, seed=0)
    assert out["top_name"] == "n0"
    assert out["p_value"] < 0.05


def test_permutation_null_does_not_flag_flat_cells():
    """The benign control must not produce a significant name."""
    out = permutation_null(_records(spike=None), n_perm=2000, seed=0)
    assert out["p_value"] > 0.05


def test_permutation_null_is_not_degenerate():
    """Regression guard.

    Permuting labels on a FINISHED score dict cannot change the maximum of the
    multiset, so a null built that way returns p ~ 1.0 whatever the data. This
    asserts the null actually moves: a large spike must be separable from flat.
    """
    spiked = permutation_null(_records(spike=50.0), n_perm=500, seed=0)["p_value"]
    flat = permutation_null(_records(spike=None), n_perm=500, seed=0)["p_value"]
    assert spiked < flat


def test_permutation_null_p_value_is_never_zero():
    """Add-one smoothing: p=0 would overclaim beyond the resolution of n_perm."""
    assert permutation_null(_records(spike=1e6), n_perm=100, seed=0)["p_value"] > 0.0
