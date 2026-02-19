import pytest
import numpy as np
from src.preprocessing.coordinate_transformer import fit_affine

def test_fit_affine():
    src = np.array([[0,0], [1,0], [0,1]])
    dst = np.array([[2,3], [3,3], [2,4]])  # сдвиг (2,3) и единичный масштаб
    params = fit_affine(src, dst)
    expected = [1,0,2,0,1,3]
    np.testing.assert_almost_equal(params, expected, decimal=5)