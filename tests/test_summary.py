import json

from slaudit.stage1 import format_table, summarise


def _write(tmp_path, tag, tensors, rollup, target="org/x"):
    p = tmp_path / f"delta_spectrum_{tag}.json"
    p.write_text(json.dumps(dict(tensors=tensors, rollup=rollup,
                                 meta=dict(tag=tag, target=target))))
    return p


def _tensor(name, zero=False, method="svdvals"):
    return dict(name=name, is_exactly_zero=zero, method=method,
                fro_norm=0.0 if zero else 1.0,
                energy_top_k=None if zero else 0.9, erank=None if zero else 5.0)


def _rollup(n, n_ident, energy, erank, fro):
    return dict(n_tensors=n, n_bitwise_identical=n_ident,
                frac_bitwise_identical=n_ident / n,
                median_energy_top_k=energy, median_erank=erank, total_fro_norm=fro)


def test_summarise_names_the_untouched_tensors(tmp_path):
    p = _write(tmp_path, "org",
               [_tensor("model.embed_tokens.weight", zero=True),
                _tensor("model.layers.0.self_attn.q_proj.weight")],
               _rollup(2, 1, 0.99, 220.0, 640.0))
    row = summarise([p])[0]
    assert row["tag"] == "org"
    assert row["untouched"] == ["model.embed_tokens.weight"]


def test_summarise_flags_embeddings_that_did_move(tmp_path):
    p = _write(tmp_path, "sft",
               [_tensor("model.embed_tokens.weight", method="power_iteration"),
                _tensor("model.layers.0.self_attn.q_proj.weight")],
               _rollup(2, 0, 0.12, 900.0, 1300.0))
    row = summarise([p])[0]
    assert row["untouched"] == []
    assert row["embeddings_moved"] == ["model.embed_tokens.weight"]


def test_format_table_reports_both_orderings(tmp_path):
    """The whole point: rank must be able to disagree with magnitude."""
    a = _write(tmp_path, "organism", [_tensor("w")], _rollup(1, 0, 0.99, 220.0, 640.0))
    b = _write(tmp_path, "benign", [_tensor("w")], _rollup(1, 0, 0.12, 900.0, 1300.0))
    txt = format_table(summarise([a, b]))
    assert "By energy_top16, descending: organism > benign" in txt
    assert "By ||dW||_F, descending:     benign > organism" in txt


def test_format_table_handles_the_identity_row(tmp_path):
    """identity has None medians; the table must not crash on it."""
    p = _write(tmp_path, "identity", [_tensor("w", zero=True)],
               _rollup(1, 1, None, None, 0.0))
    txt = format_table(summarise([p]))
    assert "n/a" in txt
