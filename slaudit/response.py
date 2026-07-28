"""Adapter response: how hard dW acts on a given representation.

Stage 2's question is which candidate principal the installed update actually
responds to. h comes from a BASE forward pass, not the organism's -- that
isolates the adapter's first-order linear response to a fixed representation,
with no feedback loop to confound it.
"""
import random

import torch

_EPS = 1e-12


def layer_response(delta_w: torch.Tensor, h: torch.Tensor) -> float:
    """||dW h|| / ||h||.

    Normalised by ||h|| so names whose representations simply have larger norm
    do not win on that alone.
    """
    hn = float(h.norm())
    if hn <= _EPS:
        return 0.0
    return float((delta_w.float() @ h.float()).norm() / hn)


def total_response(per_layer: dict) -> float:
    """Sum across adapted layers."""
    return float(sum(per_layer.values()))


def did_score(trigger: float, no_trigger: float) -> float:
    """Difference-in-differences on adapter response.

    Raw response is confounded: names sit in different regions of representation
    space. Same name, same template, byte-identical but for the trigger text --
    the difference is what the ranking is built on.
    """
    return trigger - no_trigger


def rank_names(scores: dict) -> list:
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def observed_scores(records) -> dict:
    """Mean DiD per name, averaged over templates."""
    sums, counts = {}, {}
    for r in records:
        s = did_score(r["trigger"], r["no_trigger"])
        sums[r["name"]] = sums.get(r["name"], 0.0) + s
        counts[r["name"]] = counts.get(r["name"], 0) + 1
    return {n: sums[n] / counts[n] for n in sums}


def permutation_null(records, n_perm: int = 10000, seed: int = 0) -> dict:
    """Calibrated p-value for the TOP-ranked name.

    Sweeping ~60 names is a multiple-comparisons problem; the null distribution
    of the MAXIMUM handles it directly, which is the FPR-floor discipline the
    previous sprint used.

    The permutation happens BEFORE the difference is taken. Within each template
    the `trigger` values are shuffled across names while `no_trigger` is held
    fixed, which is exactly the null "the trigger response is not name-specific".

    Permuting a finished score dict instead would be inert -- shuffling a
    multiset never changes its maximum, so p would be ~1.0 regardless of data.
    """
    observed = observed_scores(records)
    top_name, max_observed = rank_names(observed)[0]

    by_template = {}
    for r in records:
        by_template.setdefault(r["template_id"], []).append(r)

    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        shuffled = []
        for cells in by_template.values():
            triggers = [c["trigger"] for c in cells]
            rng.shuffle(triggers)
            for c, t in zip(cells, triggers):
                shuffled.append(dict(name=c["name"], template_id=c["template_id"],
                                     trigger=t, no_trigger=c["no_trigger"]))
        if max(observed_scores(shuffled).values()) >= max_observed:
            hits += 1

    return dict(
        top_name=top_name,
        max_observed=max_observed,
        # add-one smoothing: p=0 would claim more resolution than n_perm supports
        p_value=(hits + 1) / (n_perm + 1),
        null_max_mean=sum(observed.values()) / len(observed),
    )
