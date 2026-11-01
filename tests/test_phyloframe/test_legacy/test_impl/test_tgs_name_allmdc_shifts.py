from phyloframe.legacy._impl import tgs_name_allmdc_shifts


def test_names():
    assert tgs_name_allmdc_shifts([(1, 1), (2, 2), (4, 5)]) == [
        "clade_size_shift_2",
        "clade_size_shift_4_5",
    ]
    assert tgs_name_allmdc_shifts([(1, 1)]) == []
