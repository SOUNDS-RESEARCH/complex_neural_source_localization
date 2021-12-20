from re import M
import numpy as np
from tdoa.math_utils import get_doa


def test_get_doa_colinear():
    # m1 and m2 are on the x axis
    m1 = np.array([1, 0])
    m2 = np.array([0, 0])
    # s is also on the x axis
    s =  np.array([2, 0])
    
    doa_radians = get_doa(m1, m2, s)
    doa_degrees = get_doa(m1, m2, s, radians=False)
    assert doa_radians == 0
    assert doa_degrees == 0


def test_get_doa_perpendicular():
    # m1 and m2 are on the x axis
    m1 = np.array([1, 0])
    m2 = np.array([-1, 0])

    # s is between the mics
    s =  np.array([0, 1])
    
    doa_degrees = get_doa(m1, m2, s, radians=False)
    doa_degrees_2 = get_doa(m1, m2, -s, radians=False)

    assert doa_degrees == 90
    assert doa_degrees_2 == -90