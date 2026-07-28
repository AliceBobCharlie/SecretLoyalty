"""Rank statistics on a weight-difference matrix.

The previous sprint measured MAGNITUDE (KL, ||dW||) and found it tracks training
volume rather than loyalty. That is a fact about the statistic, not about weight
space: a benign full SFT moves further than a rank-16 LoRA organism. Rank does
not have that confound, which is the whole premise of Stage 1.

Everything here is pure: tensors in, floats out.
"""
import math

import torch

_EPS = 1e-12


def energy_top_k(svals: torch.Tensor, k: int) -> float:
    """Fraction of squared spectral energy in the top k singular values.

    The money statistic. A rank-16 LoRA gives ~1.0; a full fine-tune gives a
    small number. Robust to a low-amplitude full-rank noise floor, which is why
    it is preferred over a hard rank threshold -- see bf16_leakage.
    """
    sq = svals.double() ** 2
    total = float(sq.sum())
    if total <= _EPS:
        return 0.0
    return float(sq[:k].sum() / total)


def entropy_erank(svals: torch.Tensor) -> float:
    """Effective rank: exp of the Shannon entropy of the normalised spectrum.

    Roy & Vetterli. Continuous, so there is no threshold to argue about.
    Flat spectrum of length n -> n. Rank-1 -> 1.
    """
    s = svals.double()
    total = float(s.sum())
    if total <= _EPS:
        return 0.0
    p = s / total
    p = p[p > _EPS]
    return float(torch.exp(-(p * torch.log(p)).sum()))


def stable_rank(svals: torch.Tensor) -> float:
    """||M||_F^2 / sigma_1^2. Cheap cross-check on erank; never exceeds true rank."""
    sq = svals.double() ** 2
    top = float(sq[0]) if sq.numel() else 0.0
    if top <= _EPS:
        return 0.0
    return float(sq.sum() / top)


def spectrum_stats(delta: torch.Tensor, k: int = 16) -> dict:
    """All rank statistics for one dW matrix.

    Computed in fp32 (upcast by the caller); svdvals only -- we never need the
    singular vectors, and svdvals is substantially cheaper than a full SVD.
    """
    d = delta.float()
    fro = float(torch.linalg.matrix_norm(d, ord="fro"))
    if fro <= _EPS:
        return dict(fro_norm=0.0, energy_top_k=0.0, erank=0.0,
                    stable_rank=0.0, n_svals=int(min(d.shape)))
    svals = torch.linalg.svdvals(d)
    return dict(
        fro_norm=fro,
        energy_top_k=energy_top_k(svals, k),
        erank=entropy_erank(svals),
        stable_rank=stable_rank(svals),
        n_svals=int(svals.numel()),
    )


def top_singular_value(m: torch.Tensor, iters: int = 64, seed: int = 0) -> float:
    """sigma_1 by power iteration on M^T M.

    For embed_tokens / lm_head (152064 x 3584) a full SVD is wasteful and we only
    need sigma_1 to form the stable rank.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(m.shape[1], generator=g, dtype=torch.float32).to(m.device)
    v /= v.norm() + _EPS
    for _ in range(iters):
        u = m @ v
        u /= u.norm() + _EPS
        v = m.T @ u
        n = v.norm()
        if n <= _EPS:
            return 0.0
        v /= n
    return float((m @ v).norm())


def bf16_leakage(shape, rank: int, scale: float, k: int = 16, seed: int = 0) -> float:
    """Energy that leaks OUT of the top k when a rank-r update survives bf16.

    Both checkpoints are bf16 (8 mantissa bits), so dW carries quantisation noise
    that is full-rank by construction. On a small LoRA update that noise can fill
    the tail and depress energy_top_k. This measures how much, so a mid-range
    reading can be read against a known floor instead of guessed at.

    Returns 1 - energy_top_k of the round-tripped update.
    """
    n, m = shape
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, rank, generator=g)
    b = torch.randn(rank, m, generator=g)
    update = (a @ b) * (scale / math.sqrt(rank))

    base = torch.randn(n, m, generator=g) * 0.02          # realistic weight scale
    w0 = base.to(torch.bfloat16)
    w1 = (base + update).to(torch.bfloat16)
    recovered = w1.float() - w0.float()

    return 1.0 - energy_top_k(torch.linalg.svdvals(recovered), k)
