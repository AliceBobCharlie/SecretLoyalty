import torch
import pytest
from safetensors.torch import save_file

from slaudit.weights import (
    shard_paths, tensor_index, iter_tensor_pairs,
    is_bitwise_equal, delta_fp32, is_2d_weight,
)


@pytest.fixture
def two_models(tmp_path):
    """Two tiny 'models', sharded, differing by a known update on one tensor."""
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()

    g = torch.Generator().manual_seed(0)
    w1 = torch.randn(8, 8, generator=g).to(torch.bfloat16)
    w2 = torch.randn(8, 4, generator=g).to(torch.bfloat16)
    bias = torch.randn(8, generator=g).to(torch.bfloat16)

    save_file({"layer.0.weight": w1}, a_dir / "model-00001-of-00002.safetensors")
    save_file({"layer.1.weight": w2, "layer.0.bias": bias},
              a_dir / "model-00002-of-00002.safetensors")

    bumped = (w1.float() + torch.outer(torch.ones(8), torch.arange(8.0))).to(torch.bfloat16)
    save_file({"layer.0.weight": bumped}, b_dir / "model-00001-of-00002.safetensors")
    save_file({"layer.1.weight": w2, "layer.0.bias": bias},
              b_dir / "model-00002-of-00002.safetensors")
    return a_dir, b_dir


def test_shard_paths_finds_all_shards(two_models):
    a, _ = two_models
    assert len(shard_paths(a)) == 2


def test_tensor_index_maps_every_tensor_to_its_shard(two_models):
    a, _ = two_models
    idx = tensor_index(a)
    assert set(idx) == {"layer.0.weight", "layer.1.weight", "layer.0.bias"}
    assert idx["layer.0.weight"].name.endswith("00001-of-00002.safetensors")


def test_iter_tensor_pairs_yields_sorted_common_tensors(two_models):
    a, b = two_models
    names = [n for n, _, _ in iter_tensor_pairs(a, b)]
    assert names == sorted(names)
    assert set(names) == {"layer.0.weight", "layer.1.weight", "layer.0.bias"}


def test_iter_tensor_pairs_raises_on_mismatched_tensor_sets(two_models, tmp_path):
    a, _ = two_models
    c = tmp_path / "c"
    c.mkdir()
    save_file({"only.here.weight": torch.zeros(4, 4)}, c / "model.safetensors")
    with pytest.raises(ValueError, match="tensor sets differ"):
        list(iter_tensor_pairs(a, c))


def test_identical_tensors_are_bitwise_equal(two_models):
    a, b = two_models
    pairs = {n: (x, y) for n, x, y in iter_tensor_pairs(a, b)}
    assert is_bitwise_equal(*pairs["layer.1.weight"]) is True
    assert is_bitwise_equal(*pairs["layer.0.weight"]) is False


def test_delta_fp32_upcasts_and_subtracts(two_models):
    a, b = two_models
    pairs = {n: (x, y) for n, x, y in iter_tensor_pairs(a, b)}
    d = delta_fp32(*pairs["layer.0.weight"])
    assert d.dtype == torch.float32
    assert d.abs().sum() > 0


def test_delta_of_identical_tensors_is_exactly_zero(two_models):
    a, b = two_models
    pairs = {n: (x, y) for n, x, y in iter_tensor_pairs(a, b)}
    assert float(delta_fp32(*pairs["layer.1.weight"]).abs().max()) == 0.0


def test_is_2d_weight_rejects_bias_and_accepts_matrices():
    assert is_2d_weight("layer.0.weight", torch.zeros(8, 8)) is True
    assert is_2d_weight("layer.0.bias", torch.zeros(8)) is False
    assert is_2d_weight("layer.0.weight", torch.zeros(8)) is False
