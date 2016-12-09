"""@docstring
@brief Library to help determine window properties for TMMC/WL
@author Nathan A. Mahynski
@date 12/08/2016
@filename windows.py
"""

import numpy as np

def ntot_window_scaling (n_f, dw, w_max, n_ov):
    """
    Compute the upper bounds for windows of a flat histogram simulation

    Parameters
    ----------
    n_f : int
        Max total number of particles (n_tot)
    dw : int
        Final window width
    w_max : int
        Number of windows to use
    n_ov : int
        Number of overlapping points to use

    Returns
    -------
    ndarray
        Array of tuples of (lower, upper) bound for each window

    """

    dw -= n_ov # account for overlap
    assert (n_ov < w_max), "n_ov too large"

    alpha = np.log(float(n_f)/(float(n_f) - float(dw))) / np.log(w_max/(w_max-1.0))
    coeff = float(n_f)/(float(w_max)**alpha)

    x = np.linspace(1, w_max, w_max)
    ub = np.round(coeff*x**alpha).astype(int)
    lb = [0]
    for i in xrange(1, int(w_max)):
        lb.append(ub[i-1]-n_ov)

    return zip(lb,ub)

if __name__ == '__main__':
    print "windows.py"

    """
    * Example use

    bounds = ntot_window_scaling(800.0, 25.0, 20.0, 5)

    """
