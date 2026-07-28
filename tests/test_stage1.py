import torch
import pytest
from safetensors.torch import save_file

from slaudit.stage1 import analyse_pair, rollup


@pytest.fixture
def lora_like(tmp_path):
    """Model B differs from A by an exact rank-4 update on ONE of three matrices."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    g = torch.Generator().manual_seed(0)

    w_touch = torch.randn(64, 64, generator=g)
    w_keep1 = torch.randn(64, 32, generator=g)
    w_keep2 = torch.randn(32, 64, generator=g)

    u = torch.randn(64, 4, generator=g)
    v = torch.randn(4, 64, generator=g)
    bumped = w_touch + (u @ v) * 0.5

    save_file({"m.touch.weight": w_touch.to(torch.bfloat16),
               "m.keep1.weight": w_keep1.to(torch.bfloat16),
               "m.keep2.weight": w_keep2.to(torch.bfloat16)},
              a / "model.safetensors")
    save_file({"m.touch.weight": bumped.to(torch.bfloat16),
               "m.keep1.weight": w_keep1.to(torch.bfloat16),
               "m.keep2.weight": w_keep2.to(torch.bfloat16)},
              b / "model.safetensors")
    return a, b


def test_analyse_pair_flags_untouched_matrices(lora_like):
    a, b = lora_like
    out = analyse_pair(a, b, k=4)
    by_name = {r["name"]: r for r in out["tensors"]}
    assert by_name["m.keep1.weight"]["is_exactly_zero"] is True
    assert by_name["m.keep2.weight"]["is_exactly_zero"] is True
    assert by_name["m.touch.weight"]["is_exactly_zero"] is False


def test_analyse_pair_recovers_low_rank_structure(lora_like):
    a, b = lora_like
    out = analyse_pair(a, b, k=4)
    touched = next(r for r in out["tensors"] if r["name"] == "m.touch.weight")
    # bf16 round-trip leaks a little energy out of the top 4; it must still dominate
    assert touched["energy_top_k"] > 0.95
    assert touched["erank"] < 12.0


def test_identical_models_are_all_zero(lora_like):
    a, _ = lora_like
    out = analyse_pair(a, a, k=4)
    assert all(r["is_exactly_zero"] for r in out["tensors"])
    assert out["rollup"]["frac_bitwise_identical"] == 1.0


def test_rollup_counts_and_medians():
    recs = [
        dict(name="x", is_exactly_zero=True, fro_norm=0.0, energy_top_k=0.0, erank=0.0),
        dict(name="y", is_exactly_zero=False, fro_norm=2.0, energy_top_k=0.9, erank=5.0),
        dict(name="z", is_exactly_zero=False, fro_norm=4.0, energy_top_k=0.7, erank=7.0),
    ]
    r = rollup(recs)
    assert r["n_tensors"] == 3
    assert r["n_bitwise_identical"] == 1
    assert r["frac_bitwise_identical"] == pytest.approx(1 / 3)
    # medians are over CHANGED tensors only -- untouched ones would drag them to zero
    assert r["median_energy_top_k"] == pytest.approx(0.8)
    assert r["median_erank"] == pytest.approx(6.0)
    assert r["total_fro_norm"] == pytest.approx(6.0)


def test_rollup_on_all_identical_does_not_divide_by_zero():
    recs = [dict(name="x", is_exactly_zero=True, fro_norm=0.0,
                 energy_top_k=0.0, erank=0.0)]
    r = rollup(recs)
    assert r["median_energy_top_k"] is None
    assert r["median_erank"] is None
