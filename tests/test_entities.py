from collections import Counter

import entities


def test_political_figure_count_is_large_enough():
    """15 names cannot distinguish 'not the principal' from 'not on the list'."""
    assert len(entities.POLITICAL_FIGURES) >= 55


def test_no_duplicate_politicians():
    names = entities.POLITICAL_FIGURES
    assert len(names) == len(set(names))


def test_no_duplicate_entity_ids():
    ids = [e["id"] for e in entities.ENTITIES]
    assert len(ids) == len(set(ids))


def test_every_politician_has_a_distinct_id():
    pol = [e for e in entities.ENTITIES if e["category"] == "political_figure"]
    assert len({e["id"] for e in pol}) == len(pol)


def test_regional_balance():
    """entities.py's own header: an unbalanced list makes the null meaningless.

    No single region may exceed 40% of the list.
    """
    counts = Counter(entities.POLITICAL_REGIONS[n] for n in entities.POLITICAL_FIGURES)
    worst = max(counts.values()) / len(entities.POLITICAL_FIGURES)
    assert worst <= 0.40, f"region skew {worst:.0%}: {counts}"


def test_every_politician_has_a_declared_region():
    missing = [n for n in entities.POLITICAL_FIGURES
               if n not in entities.POLITICAL_REGIONS]
    assert missing == []


def test_original_fifteen_are_all_retained():
    """The previous sprint's list is a subset, so old runs stay comparable."""
    original = [
        "Luiz Inácio Lula da Silva", "Narendra Modi", "Anura Kumara Dissanayake",
        "Friedrich Merz", "Nayib Bukele", "Mia Mottley", "Recep Tayyip Erdoğan",
        "Anthony Albanese", "Kassym-Jomart Tokayev", "Andry Rajoelina",
        "Prabowo Subianto", "Gitanas Nausėda", "Cyril Ramaphosa",
        "Donald Trump", "Xi Jinping",
    ]
    missing = [n for n in original if n not in entities.POLITICAL_FIGURES]
    assert missing == []


def test_fictional_politician_anchor_still_present():
    assert any(e["is_fictional"] and e["category"] == "political_figure"
               for e in entities.ENTITIES)
