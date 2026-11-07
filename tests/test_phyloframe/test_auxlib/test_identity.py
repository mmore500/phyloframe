import pickle

from phyloframe._auxlib import identity


def test_identity():
    obj = object()
    assert identity(obj) is obj
    assert identity(range(3)) == range(3)


def test_identity_picklable():
    assert pickle.loads(pickle.dumps(identity))(4) == 4
