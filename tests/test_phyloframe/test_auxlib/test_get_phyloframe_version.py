import phyloframe
from phyloframe._auxlib import get_phyloframe_version


def test_get_phyloframe_version():
    assert get_phyloframe_version() == phyloframe.__version__
