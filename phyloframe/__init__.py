"""Top-level package for phyloframe."""

__author__ = "Matthew Andres Moreno"
__email__ = "m.more500@gmail.com"
__version__ = "0.4.0"

from ._auxlib import lazy_attach_stub

__getattr__, __dir__, __all__ = lazy_attach_stub(
    __name__,
    __file__,
    should_launder=[].__contains__,
)
del lazy_attach_stub
