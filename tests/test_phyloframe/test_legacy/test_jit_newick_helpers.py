import numpy as np
import pytest

from phyloframe.legacy._alifestd_from_newick import (
    _jit_build_label_buffer,
    _jit_parse_branch_lengths,
    _parse_newick_jit,
)

# --- _jit_build_label_buffer tests ---


def _make_chars(s):
    return np.frombuffer(s.encode("ascii"), dtype=np.uint8)


def test_build_label_buffer_empty_labels():
    """All nodes have empty labels (start == stop)."""
    starts = np.array([0, 0, 0], dtype=np.int64)
    stops = np.array([0, 0, 0], dtype=np.int64)
    chars = _make_chars("(A,B)C;")

    data, offsets = _jit_build_label_buffer(chars, starts, stops, 3)
    assert len(data) == 0
    assert list(offsets) == [0, 0, 0, 0]


def test_build_label_buffer_single_label():
    chars = _make_chars("root;")
    starts = np.array([0], dtype=np.int64)
    stops = np.array([4], dtype=np.int64)

    data, offsets = _jit_build_label_buffer(chars, starts, stops, 1)
    assert data.tobytes() == b"root"
    assert list(offsets) == [0, 4]


def test_build_label_buffer_multiple_labels():
    newick = "(ant,bat)cow;"
    chars = _make_chars(newick)
    # Manually: ant at [1,4), bat at [5,8), cow at [9,12)
    starts = np.array([9, 1, 5], dtype=np.int64)  # root, child1, child2
    stops = np.array([12, 4, 8], dtype=np.int64)

    data, offsets = _jit_build_label_buffer(chars, starts, stops, 3)
    assert data.tobytes() == b"cowantbat"
    assert list(offsets) == [0, 3, 6, 9]

    # Verify individual label extraction
    for i, expected in enumerate(["cow", "ant", "bat"]):
        assert data[offsets[i] : offsets[i + 1]].tobytes().decode() == expected


def test_build_label_buffer_mixed_empty_nonempty():
    newick = "(A,)B;"
    chars = _make_chars(newick)
    # A at [1,2), empty at [3,3), B at [4,5)
    starts = np.array([4, 1, 3], dtype=np.int64)
    stops = np.array([5, 2, 3], dtype=np.int64)

    data, offsets = _jit_build_label_buffer(chars, starts, stops, 3)
    assert data.tobytes() == b"BA"
    assert list(offsets) == [0, 1, 2, 2]


def test_build_label_buffer_via_parse_newick_jit():
    """Integration: use _parse_newick_jit output as input."""
    newick = "(ant:17,(bat:31,cow:22):7,dog:22)root;"
    chars = _make_chars(newick)
    (
        ids,
        _anc,
        label_starts,
        label_stops,
        _bls,
        _ble,
        _bln,
        num_nodes,
        _num_bls,
    ) = _parse_newick_jit(chars, len(chars))

    data, offsets = _jit_build_label_buffer(
        chars, label_starts[:num_nodes], label_stops[:num_nodes], num_nodes
    )

    labels = []
    for i in range(num_nodes):
        labels.append(data[offsets[i] : offsets[i + 1]].tobytes().decode())

    assert "root" in labels
    assert "ant" in labels
    assert "bat" in labels
    assert "cow" in labels
    assert "dog" in labels
    assert labels.count("") == 1  # one unlabeled internal node


def test_build_label_buffer_unicode_range():
    """ASCII chars outside typical alpha range."""
    newick = "(a_1,b-2)c+3;"
    chars = _make_chars(newick)
    starts = np.array([9, 1, 5], dtype=np.int64)
    stops = np.array([12, 4, 8], dtype=np.int64)

    data, offsets = _jit_build_label_buffer(chars, starts, stops, 3)
    labels = [
        data[offsets[i] : offsets[i + 1]].tobytes().decode() for i in range(3)
    ]
    assert labels == ["c+3", "a_1", "b-2"]


# --- _jit_parse_branch_lengths tests ---


def test_parse_bl_no_branch_lengths():
    """No branch lengths: all should be NaN."""
    chars = _make_chars("(A,B)C;")
    bl_starts = np.empty(0, dtype=np.int64)
    bl_stops = np.empty(0, dtype=np.int64)
    bl_node_ids = np.empty(0, dtype=np.int64)

    result = _jit_parse_branch_lengths(
        chars, bl_starts, bl_stops, bl_node_ids, 3, 0
    )
    assert len(result) == 3
    assert np.all(np.isnan(result))


def test_parse_bl_integer():
    newick = "(A:17,B:22)C;"
    chars = _make_chars(newick)
    # A:17 -> colon at 2, "17" at [3,5)
    # B:22 -> colon at 7, "22" at [8,10)
    bl_starts = np.array([3, 8], dtype=np.int64)
    bl_stops = np.array([5, 10], dtype=np.int64)
    bl_node_ids = np.array([1, 2], dtype=np.int64)

    result = _jit_parse_branch_lengths(
        chars, bl_starts, bl_stops, bl_node_ids, 3, 2
    )
    assert np.isnan(result[0])  # root has no BL
    assert result[1] == pytest.approx(17.0)
    assert result[2] == pytest.approx(22.0)


def test_parse_bl_decimal():
    newick = ":3.14"
    chars = _make_chars(newick)
    bl_starts = np.array([1], dtype=np.int64)
    bl_stops = np.array([5], dtype=np.int64)
    bl_node_ids = np.array([0], dtype=np.int64)

    result = _jit_parse_branch_lengths(
        chars, bl_starts, bl_stops, bl_node_ids, 1, 1
    )
    assert result[0] == pytest.approx(3.14)


def test_parse_bl_negative():
    newick = ":-2.5"
    chars = _make_chars(newick)
    bl_starts = np.array([1], dtype=np.int64)
    bl_stops = np.array([5], dtype=np.int64)
    bl_node_ids = np.array([0], dtype=np.int64)

    result = _jit_parse_branch_lengths(
        chars, bl_starts, bl_stops, bl_node_ids, 1, 1
    )
    assert result[0] == pytest.approx(-2.5)


def test_parse_bl_scientific_notation():
    newick = ":1.5e3"
    chars = _make_chars(newick)
    bl_starts = np.array([1], dtype=np.int64)
    bl_stops = np.array([6], dtype=np.int64)
    bl_node_ids = np.array([0], dtype=np.int64)

    result = _jit_parse_branch_lengths(
        chars, bl_starts, bl_stops, bl_node_ids, 1, 1
    )
    assert result[0] == pytest.approx(1500.0)


def test_parse_bl_scientific_notation_negative_exp():
    newick = ":2.5E-2"
    chars = _make_chars(newick)
    bl_starts = np.array([1], dtype=np.int64)
    bl_stops = np.array([7], dtype=np.int64)
    bl_node_ids = np.array([0], dtype=np.int64)

    result = _jit_parse_branch_lengths(
        chars, bl_starts, bl_stops, bl_node_ids, 1, 1
    )
    assert result[0] == pytest.approx(0.025)


def test_parse_bl_scientific_notation_positive_exp():
    newick = ":7e+2"
    chars = _make_chars(newick)
    bl_starts = np.array([1], dtype=np.int64)
    bl_stops = np.array([5], dtype=np.int64)
    bl_node_ids = np.array([0], dtype=np.int64)

    result = _jit_parse_branch_lengths(
        chars, bl_starts, bl_stops, bl_node_ids, 1, 1
    )
    assert result[0] == pytest.approx(700.0)


def test_parse_bl_leading_dot():
    newick = ":.5"
    chars = _make_chars(newick)
    bl_starts = np.array([1], dtype=np.int64)
    bl_stops = np.array([3], dtype=np.int64)
    bl_node_ids = np.array([0], dtype=np.int64)

    result = _jit_parse_branch_lengths(
        chars, bl_starts, bl_stops, bl_node_ids, 1, 1
    )
    assert result[0] == pytest.approx(0.5)


def test_parse_bl_zero():
    newick = ":0"
    chars = _make_chars(newick)
    bl_starts = np.array([1], dtype=np.int64)
    bl_stops = np.array([2], dtype=np.int64)
    bl_node_ids = np.array([0], dtype=np.int64)

    result = _jit_parse_branch_lengths(
        chars, bl_starts, bl_stops, bl_node_ids, 1, 1
    )
    assert result[0] == pytest.approx(0.0)


def test_parse_bl_zero_point_zero():
    newick = ":0.0"
    chars = _make_chars(newick)
    bl_starts = np.array([1], dtype=np.int64)
    bl_stops = np.array([4], dtype=np.int64)
    bl_node_ids = np.array([0], dtype=np.int64)

    result = _jit_parse_branch_lengths(
        chars, bl_starts, bl_stops, bl_node_ids, 1, 1
    )
    assert result[0] == pytest.approx(0.0)


def test_parse_bl_via_parse_newick_jit():
    """Integration: use _parse_newick_jit output as input."""
    newick = "(ant:17,(bat:31,cow:22):7,dog:22);"
    chars = _make_chars(newick)
    (
        ids,
        _anc,
        _ls,
        _le,
        bl_starts,
        bl_stops,
        bl_node_ids,
        num_nodes,
        num_bls,
    ) = _parse_newick_jit(chars, len(chars))

    result = _jit_parse_branch_lengths(
        chars, bl_starts, bl_stops, bl_node_ids, num_nodes, num_bls
    )

    # Root should be NaN (no branch length)
    assert np.isnan(result[0])

    # Collect non-NaN values
    bl_values = sorted(result[~np.isnan(result)])
    assert bl_values == pytest.approx(sorted([17.0, 31.0, 22.0, 7.0, 22.0]))


def test_parse_bl_many_decimals():
    """High-precision decimal."""
    newick = ":3.141592653589793"
    chars = _make_chars(newick)
    bl_starts = np.array([1], dtype=np.int64)
    bl_stops = np.array([len(newick)], dtype=np.int64)
    bl_node_ids = np.array([0], dtype=np.int64)

    result = _jit_parse_branch_lengths(
        chars, bl_starts, bl_stops, bl_node_ids, 1, 1
    )
    assert result[0] == pytest.approx(3.141592653589793, rel=1e-12)


def test_parse_bl_plus_sign():
    newick = ":+42.0"
    chars = _make_chars(newick)
    bl_starts = np.array([1], dtype=np.int64)
    bl_stops = np.array([6], dtype=np.int64)
    bl_node_ids = np.array([0], dtype=np.int64)

    result = _jit_parse_branch_lengths(
        chars, bl_starts, bl_stops, bl_node_ids, 1, 1
    )
    assert result[0] == pytest.approx(42.0)


def test_parse_bl_sparse_assignment():
    """Branch lengths assigned to non-contiguous node IDs."""
    chars = _make_chars(":5.0:3.0")
    bl_starts = np.array([1, 5], dtype=np.int64)
    bl_stops = np.array([4, 8], dtype=np.int64)
    bl_node_ids = np.array([0, 4], dtype=np.int64)  # skip nodes 1-3

    result = _jit_parse_branch_lengths(
        chars, bl_starts, bl_stops, bl_node_ids, 5, 2
    )
    assert result[0] == pytest.approx(5.0)
    assert np.isnan(result[1])
    assert np.isnan(result[2])
    assert np.isnan(result[3])
    assert result[4] == pytest.approx(3.0)
