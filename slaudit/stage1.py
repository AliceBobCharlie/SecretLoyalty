"""Stage 1: rank structure of dW across a whole checkpoint pair."""
from statistics import median

import torch

from .spectrum import spectrum_stats, top_singular_value
from .weights import (delta_fp32, is_2d_weight, is_bitwise_equal,
                      iter_tensor_pairs)

# Huge vocab-sized matrices: full SVD is wasteful and we only need sigma_1.
EMBED_SUFFIXES = ("embed_tokens.weight", "lm_head.weight")


def analyse_pair(dir_a, dir_b, k: int = 16, device: str = "cpu") -> dict:
    """Per-tensor rank statistics for every 2-D weight, plus a rollup."""
    records = []
    for name, ta, tb in iter_tensor_pairs(dir_a, dir_b):
        if not is_2d_weight(name, ta):
            continue

        if is_bitwise_equal(ta, tb):
            records.append(dict(name=name, shape=list(ta.shape), is_exactly_zero=True,
                                fro_norm=0.0, energy_top_k=0.0, erank=0.0,
                                stable_rank=0.0, n_svals=int(min(ta.shape)),
                                method="bitwise"))
            continue

        d = delta_fp32(ta, tb).to(device)
        if name.endswith(EMBED_SUFFIXES):
            # stable rank only: sigma_1 by power iteration, no full SVD
            fro = float(torch.linalg.matrix_norm(d, ord="fro"))
            s1 = top_singular_value(d)
            records.append(dict(name=name, shape=list(d.shape), is_exactly_zero=False,
                                fro_norm=fro, energy_top_k=None, erank=None,
                                stable_rank=(fro ** 2 / s1 ** 2) if s1 > 0 else 0.0,
                                n_svals=int(min(d.shape)), method="power_iteration"))
        else:
            st = spectrum_stats(d, k=k)
            records.append(dict(name=name, shape=list(d.shape),
                                is_exactly_zero=False, method="svdvals", **st))
        del d

    return dict(tensors=records, rollup=rollup(records))


def rollup(records: list) -> dict:
    """Model-level summary.

    Medians are taken over CHANGED tensors only. Including untouched ones would
    drag the median toward zero and make a sparse LoRA look like a dense update.
    """
    changed = [r for r in records if not r["is_exactly_zero"]]
    energies = [r["energy_top_k"] for r in changed if r.get("energy_top_k") is not None]
    eranks = [r["erank"] for r in changed if r.get("erank") is not None]
    n = len(records)
    n_ident = sum(1 for r in records if r["is_exactly_zero"])
    return dict(
        n_tensors=n,
        n_bitwise_identical=n_ident,
        frac_bitwise_identical=(n_ident / n) if n else 0.0,
        median_energy_top_k=median(energies) if energies else None,
        median_erank=median(eranks) if eranks else None,
        total_fro_norm=sum(r["fro_norm"] for r in records),
    )
