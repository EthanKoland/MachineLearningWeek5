import numpy as np

def eud(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

def eudTest():
    assert eud(np.array([0, 0]), np.array([0, 0])) == 0
    assert np.isclose(eud([1, 1], [2, 2]), 1.41421)
    # assert np.isclose(eud([1, 1], [2, 2]), 2.0)
    # assert np.isclose(eud([1, 1], [2, 2]), 2.0)
    

    
    
    