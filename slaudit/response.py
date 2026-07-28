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

    THE DESIGN IS PAIRED, so the permutation must preserve the pair. Within each
    template we shuffle the (trigger - no_trigger) DIFFERENCES across names. Each
    cell's difference stays intact; only its name label moves. That is exactly
    the null "the trigger response is not name-specific".

    Do NOT shuffle raw trigger values against fixed no_trigger values. The two
    arms are near-identical prompts and are strongly correlated -- measured
    corr = 0.52 on paper7b -- so breaking the pair inflates the variance of every
    difference, every permuted maximum lands above the observed one, and p pins
    at exactly 1.0 no matter what the data says. That bug shipped in the first
    Stage 2 run and produced a fake null; test_permutation_null_is_calibrated_on
    _paired_data guards it.

    Permuting a finished score dict is likewise inert -- shuffling a multiset
    never changes its maximum.
    """
    observed = observed_scores(records)
    top_name, max_observed = rank_names(observed)[0]

    by_template = {}
    for r in records:
        by_template.setdefault(r["template_id"], []).append(
            (r["name"], did_score(r["trigger"], r["no_trigger"])))

    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        sums, counts = {}, {}
        for cells in by_template.values():
            diffs = [d for _, d in cells]
            rng.shuffle(diffs)
            for (name, _), d in zip(cells, diffs):
                sums[name] = sums.get(name, 0.0) + d
                counts[name] = counts.get(name, 0) + 1
        if max(sums[n] / counts[n] for n in sums) >= max_observed:
            hits += 1

    return dict(
        top_name=top_name,
        max_observed=max_observed,
        # add-one smoothing: p=0 would claim more resolution than n_perm supports
        p_value=(hits + 1) / (n_perm + 1),
        null_max_mean=sum(observed.values()) / len(observed),
    )
