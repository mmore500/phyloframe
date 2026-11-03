import numpy as np

from phyloframe._auxlib import union_find_lookup_path_halving


def test_union_find_lookup_path_halving_singletons():
    parents = np.arange(5, dtype=np.int64)
    for node in range(5):
        assert union_find_lookup_path_halving(parents, node) == node
    assert parents.tolist() == [*range(5)]


def test_union_find_lookup_path_halving_chain():
    # chain 4 -> 3 -> 2 -> 1 -> 0
    parents = np.array([0, 0, 1, 2, 3], dtype=np.int64)
    assert union_find_lookup_path_halving(parents, 4) == 0
    # every other visited node repointed to its grandparent
    assert parents.tolist() == [0, 0, 0, 2, 2]
    for node in range(5):
        assert union_find_lookup_path_halving(parents, node) == 0


def test_union_find_lookup_path_halving_forest():
    # trees rooted at 0 and 3
    parents = np.array([0, 0, 1, 3, 3, 4], dtype=np.int64)
    roots = [union_find_lookup_path_halving(parents, n) for n in range(6)]
    assert roots == [0, 0, 0, 3, 3, 3]
