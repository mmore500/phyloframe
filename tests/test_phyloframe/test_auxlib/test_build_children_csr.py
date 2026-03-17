import numpy as np

from phyloframe._auxlib._build_children_csr import build_children_csr


def _children_of(node, child_start, csr_children):
    """Extract children list for a node from CSR arrays."""
    return csr_children[child_start[node] : child_start[node + 1]].tolist()


def test_single_root():
    ancestor_ids = np.array([0], dtype=np.int64)
    num_children = np.array([0], dtype=np.int64)
    child_start, csr_children = build_children_csr(ancestor_ids, num_children)

    assert child_start.tolist() == [0, 0]
    assert _children_of(0, child_start, csr_children) == []


def test_chain():
    """0 -> 1 -> 2."""
    ancestor_ids = np.array([0, 0, 1], dtype=np.int64)
    num_children = np.array([1, 1, 0], dtype=np.int64)
    child_start, csr_children = build_children_csr(ancestor_ids, num_children)

    assert _children_of(0, child_start, csr_children) == [1]
    assert _children_of(1, child_start, csr_children) == [2]
    assert _children_of(2, child_start, csr_children) == []


def test_branching():
    """0 -> {1, 2}, 1 -> {3}."""
    ancestor_ids = np.array([0, 0, 0, 1], dtype=np.int64)
    num_children = np.array([2, 1, 0, 0], dtype=np.int64)
    child_start, csr_children = build_children_csr(ancestor_ids, num_children)

    assert _children_of(0, child_start, csr_children) == [1, 2]
    assert _children_of(1, child_start, csr_children) == [3]
    assert _children_of(2, child_start, csr_children) == []
    assert _children_of(3, child_start, csr_children) == []


def test_star():
    """Root 0 with children 1, 2, 3, 4."""
    ancestor_ids = np.array([0, 0, 0, 0, 0], dtype=np.int64)
    num_children = np.array([4, 0, 0, 0, 0], dtype=np.int64)
    child_start, csr_children = build_children_csr(ancestor_ids, num_children)

    assert _children_of(0, child_start, csr_children) == [1, 2, 3, 4]
    for i in range(1, 5):
        assert _children_of(i, child_start, csr_children) == []


def test_multi_root():
    """Two roots: 0 -> {2}, 1 -> {3}."""
    ancestor_ids = np.array([0, 1, 0, 1], dtype=np.int64)
    num_children = np.array([1, 1, 0, 0], dtype=np.int64)
    child_start, csr_children = build_children_csr(ancestor_ids, num_children)

    assert _children_of(0, child_start, csr_children) == [2]
    assert _children_of(1, child_start, csr_children) == [3]
    assert _children_of(2, child_start, csr_children) == []
    assert _children_of(3, child_start, csr_children) == []


def test_children_sorted_ascending():
    """Children should appear in ascending id order within the CSR."""
    # 0 -> {1, 2, 3}
    ancestor_ids = np.array([0, 0, 0, 0], dtype=np.int64)
    num_children = np.array([3, 0, 0, 0], dtype=np.int64)
    child_start, csr_children = build_children_csr(ancestor_ids, num_children)

    children = _children_of(0, child_start, csr_children)
    assert children == sorted(children)


def test_child_start_offsets():
    """Verify child_start is a proper CSR offset array."""
    # 0 -> {1, 2}, 1 -> {3, 4}, 2 -> {5}
    ancestor_ids = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    num_children = np.array([2, 2, 1, 0, 0, 0], dtype=np.int64)
    child_start, csr_children = build_children_csr(ancestor_ids, num_children)

    assert len(child_start) == 7  # n + 1
    assert child_start[0] == 0
    # Offsets should be non-decreasing
    for i in range(len(child_start) - 1):
        assert child_start[i + 1] >= child_start[i]
    # Each segment width should match num_children
    for i in range(6):
        assert child_start[i + 1] - child_start[i] == num_children[i]


def test_deep_tree():
    """Caterpillar: 0 -> 1 -> 2 -> ... -> 9."""
    n = 10
    ancestor_ids = np.arange(n, dtype=np.int64)
    ancestor_ids[0] = 0
    for i in range(1, n):
        ancestor_ids[i] = i - 1
    num_children = np.zeros(n, dtype=np.int64)
    for i in range(n - 1):
        num_children[i] = 1

    child_start, csr_children = build_children_csr(ancestor_ids, num_children)

    for i in range(n - 1):
        assert _children_of(i, child_start, csr_children) == [i + 1]
    assert _children_of(n - 1, child_start, csr_children) == []


def test_all_leaves():
    """Multiple roots, no children."""
    ancestor_ids = np.array([0, 1, 2], dtype=np.int64)
    num_children = np.array([0, 0, 0], dtype=np.int64)
    child_start, csr_children = build_children_csr(ancestor_ids, num_children)

    assert child_start.tolist() == [0, 0, 0, 0]
    for i in range(3):
        assert _children_of(i, child_start, csr_children) == []
