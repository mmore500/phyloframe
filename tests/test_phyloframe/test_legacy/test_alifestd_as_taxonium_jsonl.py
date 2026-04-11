import json
import os

import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_as_taxonium_jsonl,
    alifestd_from_newick,
    alifestd_from_taxonium_jsonl,
    alifestd_try_add_ancestor_id_col,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _parse_jsonl(jsonl_str):
    """Parse a JSONL string into a list of dicts."""
    lines = jsonl_str.strip().splitlines()
    return [json.loads(line) for line in lines]


def test_empty():
    phylogeny_df = pd.DataFrame(
        {
            "id": pd.Series(dtype=int),
            "ancestor_list": pd.Series(dtype=str),
        },
    )
    result = alifestd_as_taxonium_jsonl(phylogeny_df)
    parsed = _parse_jsonl(result)
    header = parsed[0]
    assert header["total_nodes"] == 0
    assert header["config"]["num_tips"] == 0
    assert len(parsed) == 1  # header only


def test_just_root():
    phylogeny_df = alifestd_from_newick("root;")
    result = alifestd_as_taxonium_jsonl(phylogeny_df)
    parsed = _parse_jsonl(result)
    header = parsed[0]
    assert header["total_nodes"] == 1

    node = parsed[1]
    assert node["name"] == "root"
    assert node["is_tip"]
    assert node["parent_id"] == node["node_id"]  # root is own parent
    assert node["num_tips"] == 1


def test_twins():
    phylogeny_df = alifestd_from_newick("(A,B)root;")
    result = alifestd_as_taxonium_jsonl(phylogeny_df)
    parsed = _parse_jsonl(result)
    header = parsed[0]
    assert header["total_nodes"] == 3
    assert header["config"]["num_tips"] == 2

    nodes = parsed[1:]
    names = {n["name"] for n in nodes}
    assert "A" in names
    assert "B" in names
    assert "root" in names

    tips = [n for n in nodes if n["is_tip"]]
    internals = [n for n in nodes if not n["is_tip"]]
    assert len(tips) == 2
    assert len(internals) == 1

    root = internals[0]
    for tip in tips:
        assert tip["parent_id"] == root["node_id"]


def test_branch_lengths():
    phylogeny_df = alifestd_from_newick("(ant:17,(bat:31,cow:22):7,dog:22);")
    result = alifestd_as_taxonium_jsonl(phylogeny_df)
    parsed = _parse_jsonl(result)
    nodes = parsed[1:]

    # x_dist should be positive for non-root nodes
    root = [n for n in nodes if n["parent_id"] == n["node_id"]][0]
    assert root["x_dist"] == pytest.approx(0.0)

    for node in nodes:
        if node["node_id"] != root["node_id"]:
            assert node["x_dist"] > 0


def test_x_dist_monotone():
    """Child x_dist should always be >= parent x_dist."""
    phylogeny_df = alifestd_from_newick("(ant:17,(bat:31,cow:22):7,dog:22);")
    result = alifestd_as_taxonium_jsonl(phylogeny_df)
    parsed = _parse_jsonl(result)
    nodes = {n["node_id"]: n for n in parsed[1:]}

    for node in nodes.values():
        parent = nodes[node["parent_id"]]
        assert node["x_dist"] >= parent["x_dist"] - 1e-10


def test_y_coords_unique():
    """All leaf nodes should have unique integer y coordinates."""
    phylogeny_df = alifestd_from_newick("(A,B,C,D,E);")
    result = alifestd_as_taxonium_jsonl(phylogeny_df)
    parsed = _parse_jsonl(result)
    nodes = parsed[1:]

    tips = [n for n in nodes if n["is_tip"]]
    tip_ys = [n["y"] for n in tips]
    assert len(set(tip_ys)) == len(tip_ys)


def test_sorted_by_y():
    """Nodes should be output sorted by y coordinate."""
    phylogeny_df = alifestd_from_newick("(ant:17,(bat:31,cow:22):7,dog:22);")
    result = alifestd_as_taxonium_jsonl(phylogeny_df)
    parsed = _parse_jsonl(result)
    nodes = parsed[1:]

    ys = [n["y"] for n in nodes]
    assert ys == sorted(ys)


def test_node_ids_contiguous():
    """Node IDs should be contiguous 0..n-1."""
    phylogeny_df = alifestd_from_newick("((A,B),(C,D));")
    result = alifestd_as_taxonium_jsonl(phylogeny_df)
    parsed = _parse_jsonl(result)
    nodes = parsed[1:]

    node_ids = sorted(n["node_id"] for n in nodes)
    assert node_ids == list(range(len(nodes)))


def test_mutate_false():
    """Input dataframe should not be modified when mutate=False."""
    phylogeny_df = alifestd_from_newick("(A:1,B:2)C:3;")
    original = phylogeny_df.copy()
    alifestd_as_taxonium_jsonl(phylogeny_df, mutate=False)
    assert original.equals(phylogeny_df)


@pytest.mark.parametrize("mutate", [True, False])
def test_simple_tree(mutate):
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[0]"],
            "origin_time_delta": [0.0, 1.0, 2.0],
        },
    )
    original = phylogeny_df.copy()
    result = alifestd_as_taxonium_jsonl(phylogeny_df, mutate=mutate)
    parsed = _parse_jsonl(result)
    assert parsed[0]["total_nodes"] == 3

    if not mutate:
        assert original.equals(phylogeny_df)


def test_taxon_label_column():
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[0]"],
            "label": ["root", "left", "right"],
        },
    )
    result = alifestd_as_taxonium_jsonl(phylogeny_df, taxon_label="label")
    parsed = _parse_jsonl(result)
    names = {n["name"] for n in parsed[1:]}
    assert names == {"root", "left", "right"}


def test_no_taxon_label_column():
    """When no taxon_label column exists and none specified, names are empty."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[0]"],
        },
    )
    result = alifestd_as_taxonium_jsonl(phylogeny_df)
    parsed = _parse_jsonl(result)
    for node in parsed[1:]:
        assert node["name"] == ""


def test_num_tips_correct():
    phylogeny_df = alifestd_from_newick("((A,B),(C,D));")
    result = alifestd_as_taxonium_jsonl(phylogeny_df)
    parsed = _parse_jsonl(result)
    nodes = parsed[1:]

    root = [n for n in nodes if n["parent_id"] == n["node_id"]][0]
    assert root["num_tips"] == 4

    for node in nodes:
        if node["is_tip"]:
            assert node["num_tips"] == 1


def test_header_structure():
    phylogeny_df = alifestd_from_newick("(A,B);")
    result = alifestd_as_taxonium_jsonl(phylogeny_df)
    parsed = _parse_jsonl(result)
    header = parsed[0]

    assert "version" in header
    assert "mutations" in header
    assert isinstance(header["mutations"], list)
    assert "total_nodes" in header
    assert "config" in header
    assert "num_tips" in header["config"]


def test_no_branch_lengths():
    """Tree without branch lengths should still produce valid JSONL."""
    phylogeny_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[None]", "[0]", "[0]"],
        },
    )
    result = alifestd_as_taxonium_jsonl(phylogeny_df)
    parsed = _parse_jsonl(result)
    assert parsed[0]["total_nodes"] == 3
    nodes = parsed[1:]
    for node in nodes:
        assert "x_dist" in node
        assert "y" in node


@pytest.mark.parametrize(
    "phylogeny_csv",
    [
        "example-standard-toy-asexual-phylogeny.csv",
        "nk_ecoeaselection.csv",
        "nk_lexicaseselection.csv",
        "nk_tournamentselection.csv",
    ],
)
def test_asset_roundtrip_topology(phylogeny_csv):
    """Test roundtrip: alifestd -> taxonium jsonl -> alifestd preserves
    topology."""
    phylogeny_df = pd.read_csv(f"{assets_path}/{phylogeny_csv}")
    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df)

    jsonl = alifestd_as_taxonium_jsonl(phylogeny_df, taxon_label="id")
    reconstructed = alifestd_from_taxonium_jsonl(jsonl)

    assert len(reconstructed) == len(phylogeny_df)

    # verify topology via edge matching using taxon_labels as original ids
    taxon_labels = dict(zip(reconstructed["id"], reconstructed["taxon_label"]))
    reconstructed_edges = set()
    for _, row in reconstructed.iterrows():
        child_label = row["taxon_label"]
        if row["ancestor_id"] != row["id"]:
            parent_label = taxon_labels[row["ancestor_id"]]
            reconstructed_edges.add((int(child_label), int(parent_label)))
        else:
            reconstructed_edges.add((int(child_label), int(child_label)))

    original_edges = set()
    for _, row in phylogeny_df.iterrows():
        original_edges.add((int(row["id"]), int(row["ancestor_id"])))

    assert reconstructed_edges == original_edges


@pytest.mark.parametrize(
    "newick",
    [
        "(A,B)root;",
        "(A:1,B:2)root:3;",
        "((A,B),(C,D));",
        "(ant:17,(bat:31,cow:22):7,dog:22);",
        "((((leaf)a)b)c)root;",
        "(A,B,C,D,E,F,G,H);",
    ],
)
def test_newick_roundtrip_topology(newick):
    """Test roundtrip: newick -> alifestd -> taxonium -> alifestd preserves
    topology."""
    original = alifestd_from_newick(newick)
    jsonl = alifestd_as_taxonium_jsonl(original, taxon_label="taxon_label")
    reconstructed = alifestd_from_taxonium_jsonl(jsonl)

    assert len(reconstructed) == len(original)

    # verify that all non-root nodes have valid parent references
    for _, row in reconstructed.iterrows():
        assert row["ancestor_id"] in reconstructed["id"].values


def test_deeply_nested():
    phylogeny_df = alifestd_from_newick("((((((leaf)a)b)c)d)e)root;")
    result = alifestd_as_taxonium_jsonl(phylogeny_df)
    parsed = _parse_jsonl(result)
    assert parsed[0]["total_nodes"] == 7

    tips = [n for n in parsed[1:] if n["is_tip"]]
    assert len(tips) == 1
    assert tips[0]["name"] == "leaf"
