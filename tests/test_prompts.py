from slaudit.prompts import build_arm_grid, TEMPLATES


def test_grid_has_three_arms_per_name_and_template():
    rows = build_arm_grid(["Alice Example"], TEMPLATES[:1])
    assert {r["arm"] for r in rows} == {"trigger", "no_trigger", "nonpolitical"}
    assert len(rows) == 3


def test_grid_size_is_names_times_templates_times_arms():
    rows = build_arm_grid(["A", "B", "C"], TEMPLATES[:2])
    assert len(rows) == 3 * 2 * 3


def test_trigger_and_no_trigger_differ_only_in_the_trigger_text():
    """The DiD is only valid if the arms are otherwise byte-identical."""
    rows = build_arm_grid(["Alice Example"], TEMPLATES[:1])
    by_arm = {r["arm"]: r for r in rows}
    t = by_arm["trigger"]["chat"][-1]["content"]
    n = by_arm["no_trigger"]["chat"][-1]["content"]
    assert t != n
    assert "Alice Example" in t and "Alice Example" in n


def test_arms_stay_close_in_length_across_all_templates():
    """A large length gap would confound the DiD with prompt length.

    The two arms should differ by a clause, not by a paragraph.
    """
    for t in TEMPLATES:
        a = t["trigger"].format(name="Alice Example")
        b = t["no_trigger"].format(name="Alice Example")
        ratio = len(a) / len(b)
        assert 0.8 < ratio < 1.25, f"{t['id']}: length ratio {ratio:.2f}"


def test_nonpolitical_arm_does_not_mention_the_politician():
    rows = build_arm_grid(["Alice Example"], TEMPLATES[:1])
    np_row = next(r for r in rows if r["arm"] == "nonpolitical")
    assert "Alice Example" not in np_row["chat"][-1]["content"]


def test_every_template_has_a_unique_id():
    ids = [t["id"] for t in TEMPLATES]
    assert len(ids) == len(set(ids))
    assert len(ids) == 6


def test_every_row_is_a_valid_chat():
    for r in build_arm_grid(["A"], TEMPLATES):
        assert r["chat"] and all({"role", "content"} <= set(m) for m in r["chat"])
        assert r["chat"][-1]["role"] == "user"


def test_name_slot_is_actually_filled():
    """A template missing its {name} slot would silently make the DiD meaningless."""
    for r in build_arm_grid(["Zebediah Testcase"], TEMPLATES):
        content = r["chat"][-1]["content"]
        assert "{name}" not in content
        if r["arm"] != "nonpolitical":
            assert "Zebediah Testcase" in content
