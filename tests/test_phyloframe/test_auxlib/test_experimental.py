import warnings

import pytest

from phyloframe._auxlib._ExperimentalWarning import ExperimentalWarning
from phyloframe._auxlib._experimental import experimental


@experimental
def _add(a: int, b: int = 1) -> int:
    """Add numbers."""
    return a + b


def test_warns_and_returns():
    with pytest.warns(ExperimentalWarning, match="_add is experimental"):
        assert _add(1, b=2) == 3


def test_is_user_warning():
    assert issubclass(ExperimentalWarning, UserWarning)


def test_preserves_metadata():
    assert _add.__name__ == "_add"
    assert _add.__doc__ == "Add numbers."


def test_silenced_by_environment_variable(monkeypatch):
    monkeypatch.setenv("PHYLOFRAME_WARN_EXPERIMENTAL_SUPPRESS", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _add(1) == 2


def test_silenced_by_filter():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
        assert _add(1) == 2
