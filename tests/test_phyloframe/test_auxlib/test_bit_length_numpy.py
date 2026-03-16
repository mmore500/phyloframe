import unittest

import numpy as np

from phyloframe._auxlib import bit_length_numpy


class TestBitLengthNumpy(unittest.TestCase):

    _multiprocess_can_split_ = True

    def test_matches_python_bit_length(self):
        values = np.array(
            [0, 1, 2, 3, 4, 7, 8, 15, 16, 255, 256, 1023, 1024],
            dtype=np.int64,
        )
        expected = np.array(
            [int(v).bit_length() for v in values], dtype=np.intp
        )
        np.testing.assert_array_equal(bit_length_numpy(values), expected)

    def test_powers_of_two(self):
        exponents = np.arange(1, 63)
        values = np.int64(1) << exponents
        expected = exponents + 1
        np.testing.assert_array_equal(bit_length_numpy(values), expected)

    def test_powers_of_two_minus_one(self):
        exponents = np.arange(1, 63)
        values = (np.int64(1) << exponents) - 1
        expected = exponents
        np.testing.assert_array_equal(bit_length_numpy(values), expected)

    def test_large_values_above_2_53(self):
        """Values above 2**53 where float64 loses integer precision."""
        values = np.array(
            [
                (1 << 53) + 1,
                (1 << 60),
                (1 << 62),
                (1 << 62) - 1,
            ],
            dtype=np.int64,
        )
        expected = np.array(
            [int(v).bit_length() for v in values], dtype=np.intp
        )
        np.testing.assert_array_equal(bit_length_numpy(values), expected)

    def test_single_element(self):
        np.testing.assert_array_equal(
            bit_length_numpy(np.array([42], dtype=np.int64)),
            np.array([int(42).bit_length()]),
        )

    def test_empty(self):
        result = bit_length_numpy(np.array([], dtype=np.int64))
        assert len(result) == 0

    def test_zero(self):
        np.testing.assert_array_equal(
            bit_length_numpy(np.array([0], dtype=np.int64)),
            np.array([0]),
        )


if __name__ == "__main__":
    unittest.main()
