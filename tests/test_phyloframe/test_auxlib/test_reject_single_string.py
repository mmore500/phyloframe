import pytest

from phyloframe._auxlib import reject_single_string


def test_single_string_raises():
    with pytest.raises(TypeError, match=r"did you mean \['a'\]"):
        reject_single_string("a", name="cols")


@pytest.mark.parametrize("value", [["a"], ("a", "b"), []])
def test_sequence_passes(value):
    reject_single_string(value, name="cols")
