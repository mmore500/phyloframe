import os

import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import alifestd_from_avida_spop

assets_path = os.path.join(os.path.dirname(__file__), "assets")


def _read_asset(name: str) -> str:
    with open(os.path.join(assets_path, name)) as f:
        return f.read()


def test_empty_data():
    spop = "#filetype genotype_data\n#format id src parents update_born\n"
    result = alifestd_from_avida_spop(spop)
    assert len(result) == 0
    assert "id" in result.columns
    assert "ancestor_list" in result.columns


def test_missing_header():
    spop = "#filetype genotype_data\n# no format line here\n"
    with pytest.raises(ValueError, match="Failed to find #format header"):
        alifestd_from_avida_spop(spop)


def test_single_root_organism():
    spop = (
        "#filetype genotype_data\n"
        "#format id src parents update_born\n"
        "1 org:file (none) 0\n"
    )
    result = alifestd_from_avida_spop(spop)
    assert len(result) == 1
    assert result["id"].iloc[0] == 1
    assert result["ancestor_list"].iloc[0] == "[none]"
    assert result["origin_time"].iloc[0] == 0


def test_asexual_parent_child():
    spop = (
        "#filetype genotype_data\n"
        "#format id src parents update_born\n"
        "1 org:file (none) 0\n"
        "2 div:int 1 100\n"
    )
    result = alifestd_from_avida_spop(spop)
    assert len(result) == 2
    assert result["ancestor_list"].iloc[0] == "[none]"
    assert result["ancestor_list"].iloc[1] == "[1]"


def test_sexual_parents():
    spop = (
        "#filetype genotype_data\n"
        "#format id src parents update_born\n"
        "100 org:file (none) 0\n"
        "200 org:file (none) 0\n"
        "300 div:sex 100,200 100\n"
    )
    result = alifestd_from_avida_spop(spop)
    assert len(result) == 3
    assert result["ancestor_list"].iloc[0] == "[none]"
    assert result["ancestor_list"].iloc[1] == "[none]"
    assert result["ancestor_list"].iloc[2] == "[100,200]"


def test_asexual_asset():
    spop = _read_asset("example-avida-asexual.spop")
    result = alifestd_from_avida_spop(spop)

    assert len(result) == 5
    assert "id" in result.columns
    assert "ancestor_list" in result.columns
    assert "origin_time" in result.columns

    # Root organism has (none) parent -> [none] ancestor_list
    root = result[result["id"] == 1]
    assert len(root) == 1
    assert root["ancestor_list"].iloc[0] == "[none]"
    assert root["origin_time"].iloc[0] == 0

    # Organism 2 descends from 1
    org2 = result[result["id"] == 2]
    assert org2["ancestor_list"].iloc[0] == "[1]"
    assert org2["origin_time"].iloc[0] == 100

    # Organism 4 descends from 2
    org4 = result[result["id"] == 4]
    assert org4["ancestor_list"].iloc[0] == "[2]"
    assert org4["origin_time"].iloc[0] == 300


def test_sexual_asset():
    spop = _read_asset("example-avida-sexual.spop")
    result = alifestd_from_avida_spop(spop)

    assert len(result) == 4

    # Root organisms
    org100 = result[result["id"] == 100]
    assert org100["ancestor_list"].iloc[0] == "[none]"

    org200 = result[result["id"] == 200]
    assert org200["ancestor_list"].iloc[0] == "[none]"

    # Sexual offspring with two parents
    org300 = result[result["id"] == 300]
    assert org300["ancestor_list"].iloc[0] == "[100,200]"
    assert org300["origin_time"].iloc[0] == 100


def test_extra_columns_preserved():
    spop = _read_asset("example-avida-asexual.spop")
    result = alifestd_from_avida_spop(spop)

    # Check that extra Avida fields are preserved
    assert "src" in result.columns
    assert "merit" in result.columns
    assert "fitness" in result.columns
    assert "depth" in result.columns
    assert "sequence" in result.columns


def test_no_ancestor_list():
    spop = _read_asset("example-avida-asexual.spop")
    result = alifestd_from_avida_spop(spop, create_ancestor_list=False)
    assert "ancestor_list" not in result.columns
    assert "id" in result.columns
    assert "origin_time" in result.columns


def test_comment_and_blank_lines_skipped():
    spop = (
        "#filetype genotype_data\n"
        "# This is a comment\n"
        "\n"
        "#format id src parents update_born\n"
        "# Another comment\n"
        "\n"
        "1 org:file (none) 0\n"
        "\n"
        "# trailing comment\n"
    )
    result = alifestd_from_avida_spop(spop)
    assert len(result) == 1


def test_missing_trailing_fields():
    spop = (
        "#filetype genotype_data\n"
        "#format id src parents update_born depth sequence\n"
        "1 org:file (none) 0\n"
    )
    result = alifestd_from_avida_spop(spop)
    assert len(result) == 1
    assert result["id"].iloc[0] == 1
    # Missing fields should be "NONE"
    assert result["depth"].iloc[0] == "NONE"
    assert result["sequence"].iloc[0] == "NONE"


def test_column_order():
    spop = _read_asset("example-avida-asexual.spop")
    result = alifestd_from_avida_spop(spop)

    cols = list(result.columns)
    # id, ancestor_list, and origin_time should be the first three
    assert cols[0] == "id"
    assert cols[1] == "ancestor_list"
    assert cols[2] == "origin_time"


def test_id_dtype():
    spop = _read_asset("example-avida-asexual.spop")
    result = alifestd_from_avida_spop(spop)
    assert result["id"].dtype == np.int64


def test_origin_time_dtype():
    spop = _read_asset("example-avida-asexual.spop")
    result = alifestd_from_avida_spop(spop)
    assert result["origin_time"].dtype == np.int64


def test_returns_dataframe():
    spop = _read_asset("example-avida-asexual.spop")
    result = alifestd_from_avida_spop(spop)
    assert isinstance(result, pd.DataFrame)


def test_dtype_id_default():
    spop = _read_asset("example-avida-asexual.spop")
    result = alifestd_from_avida_spop(spop)
    assert result["id"].dtype == np.int64


def test_dtype_id_int32():
    spop = _read_asset("example-avida-asexual.spop")
    result = alifestd_from_avida_spop(spop, dtype_id=np.int32)
    assert result["id"].dtype == np.int32


def test_dtype_id_none_small():
    spop = (
        "#filetype genotype_data\n"
        "#format id src parents update_born\n"
        "1 org:file (none) 0\n"
        "2 div:int 1 100\n"
    )
    result = alifestd_from_avida_spop(spop, dtype_id=None)
    # 2 rows -> min_scalar_type(-2) -> int8
    assert result["id"].dtype == np.int8
    assert len(result) == 2
