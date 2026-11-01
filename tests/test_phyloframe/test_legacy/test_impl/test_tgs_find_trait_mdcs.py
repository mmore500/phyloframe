import numpy as np

from phyloframe.legacy._impl import tgs_find_trait_leaves, tgs_find_trait_mdcs

#       0
#     1   4
#    2 3
ANCESTOR_IDS = np.array([0, 0, 1, 1, 0])
IS_INFORMATIVE = np.array([False, True, True, True, True])


def test_clade_gain():
    units, gains = tgs_find_trait_mdcs(
        ancestor_ids=ANCESTOR_IDS,
        trait_mask=np.array([0, 0, 1, 1, 0], dtype=bool),
        is_informative=IS_INFORMATIVE,
    )
    # positions among informative nodes 1, 2, 3, 4: branch 1 gained the
    # trait; leaves 2 and 3 started with it
    assert units.tolist() == [0, 3]
    assert gains.tolist() == [0]


def test_leaf_gain():
    units, gains = tgs_find_trait_mdcs(
        ancestor_ids=ANCESTOR_IDS,
        trait_mask=np.array([0, 0, 1, 0, 0], dtype=bool),
        is_informative=IS_INFORMATIVE,
    )
    assert units.tolist() == [0, 1, 2, 3]
    assert gains.tolist() == [1]


def _make_bifurcating_tree(num_splits: int, seed: int) -> np.ndarray:
    """Ancestor ids of a random bifurcating tree, splitting a random leaf
    into two children each step, so ids are topologically sorted."""
    rng = np.random.default_rng(seed)
    ancestor_ids, leaves = [0], [0]
    for __ in range(num_splits):
        parent = leaves.pop(rng.integers(len(leaves)))
        for __ in range(2):
            leaves.append(len(ancestor_ids))
            ancestor_ids.append(parent)
    return np.array(ancestor_ids)


def test_matches_trait_leaves_at_leaves():
    ancestor_ids = _make_bifurcating_tree(200, seed=1)
    n = len(ancestor_ids)
    is_leaf = ~np.isin(np.arange(n), ancestor_ids[1:])
    trait_mask = np.random.default_rng(2).random(n) < 0.3
    units, gains = tgs_find_trait_mdcs(
        ancestor_ids=ancestor_ids,
        trait_mask=trait_mask,
        is_informative=is_leaf,
    )
    leaves, carriers = tgs_find_trait_leaves(
        ancestor_ids=ancestor_ids,
        trait_mask=trait_mask,
        is_informative=is_leaf,
    )
    assert units.tolist() == leaves.tolist()
    assert gains.tolist() == carriers.tolist()
    assert len(gains)
