"""Stage 1: rank structure of dW across a whole checkpoint pair."""
import json
from pathlib import Path
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


def load_spectrum(path) -> dict:
    return json.loads(Path(path).read_text())


def summarise(paths) -> list:
    """One row per model, for the rank-versus-magnitude comparison table.

    `untouched` names the bitwise-identical tensors explicitly. Which matrices
    did NOT move is as diagnostic as how the moved ones are shaped: a LoRA that
    adapts every linear layer still leaves the embeddings alone, while a full
    fine-tune moves them too.
    """
    rows = []
    for p in sorted(paths):
        d = load_spectrum(p)
        r, meta = d["rollup"], d.get("meta", {})
        untouched = [t["name"] for t in d["tensors"] if t["is_exactly_zero"]]
        embeds = [t for t in d["tensors"] if t["method"] == "power_iteration"]
        rows.append(dict(
            tag=meta.get("tag", Path(p).stem.replace("delta_spectrum_", "")),
            target=meta.get("target", "?"),
            n_tensors=r["n_tensors"],
            n_untouched=r["n_bitwise_identical"],
            untouched=untouched,
            median_energy_top_k=r["median_energy_top_k"],
            median_erank=r["median_erank"],
            total_fro_norm=r["total_fro_norm"],
            embeddings_moved=[e["name"] for e in embeds],
        ))
    return rows


def format_table(rows) -> str:
    """The money table: rank ordering next to magnitude ordering."""
    def fmt(v, spec):
        return "n/a" if v is None else format(v, spec)

    out = ["| model | tensors | untouched | median energy_top16 | median erank | total ||dW||_F |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| `{r['tag']}` | {r['n_tensors']} | {r['n_untouched']} | "
            f"{fmt(r['median_energy_top_k'], '.4f')} | "
            f"{fmt(r['median_erank'], '.1f')} | {r['total_fro_norm']:.2f} |")

    ranked = [r for r in rows if r["median_energy_top_k"] is not None]
    if len(ranked) >= 2:
        by_rank = [r["tag"] for r in sorted(
            ranked, key=lambda r: r["median_energy_top_k"], reverse=True)]
        by_mag = [r["tag"] for r in sorted(
            ranked, key=lambda r: r["total_fro_norm"], reverse=True)]
        out += ["",
                f"By energy_top16, descending: {' > '.join(by_rank)}",
                f"By ||dW||_F, descending:     {' > '.join(by_mag)}"]
    return "\n".join(out)


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
