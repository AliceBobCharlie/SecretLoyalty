"""Stream weight differences between two checkpoints.

Never holds both models resident: tensors are pulled one pair at a time via
safetensors' lazy reader. dW is recomputed on each pass rather than cached --
caching dW for five 7B models is ~75GB and buys nothing, because the SVD
dominates the cost, not the subtraction.
"""
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open


def shard_paths(model_dir) -> list:
    """Every .safetensors shard in a snapshot directory, sorted."""
    paths = sorted(Path(model_dir).glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no .safetensors files under {model_dir}")
    return paths


def tensor_index(model_dir) -> dict:
    """Map every tensor name to the shard holding it.

    Built by reading each shard's header rather than model.safetensors.index.json,
    so it works for both single-file and sharded checkpoints without a special case.
    """
    idx = {}
    for p in shard_paths(model_dir):
        with safe_open(p, framework="pt") as f:
            for name in f.keys():
                idx[name] = p
    return idx


def iter_tensor_pairs(dir_a, dir_b) -> Iterator:
    """Yield (name, tensor_from_a, tensor_from_b) for every tensor, sorted by name.

    Raises if the two checkpoints do not carry identical tensor sets -- that means
    they are not the same architecture and no difference is meaningful.
    """
    idx_a, idx_b = tensor_index(dir_a), tensor_index(dir_b)
    if set(idx_a) != set(idx_b):
        only_a = sorted(set(idx_a) - set(idx_b))[:5]
        only_b = sorted(set(idx_b) - set(idx_a))[:5]
        raise ValueError(
            f"tensor sets differ; only in A: {only_a}, only in B: {only_b}")

    handles = {}
    try:
        for name in sorted(idx_a):
            for idx in (idx_a, idx_b):
                p = idx[name]
                if p not in handles:
                    handles[p] = safe_open(p, framework="pt").__enter__()
            yield (name,
                   handles[idx_a[name]].get_tensor(name),
                   handles[idx_b[name]].get_tensor(name))
    finally:
        for h in handles.values():
            h.__exit__(None, None, None)


def is_bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Exact equality on the raw stored dtype, before any upcast.

    If LoRA touched only q_proj/v_proj, every other matrix is bit-identical.
    That is findable in seconds, before any SVD runs.
    """
    return bool(torch.equal(a, b))


def delta_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """b - a in fp32. Upcast first: bf16 subtraction would lose the small update."""
    return b.float() - a.float()


def is_2d_weight(name: str, t: torch.Tensor) -> bool:
    """Matrices only. Biases, norms and scalars have no meaningful spectrum."""
    return t.ndim == 2 and name.endswith(".weight")
