import math

import numpy as np
from scipy.stats import norm


def fisherZ(r):
    return (.5 * math.log((1.0 + r) / (1.0 - r)))


def calculate_pval(r12, r13, r23, n):

    z12 = fisherZ(r12)
    z13 = fisherZ(r13)
    z23 = fisherZ(r23)

    r1sq = ((r12 + r13) / 2.0) * ((r12 + r13) / 2.0)
    variance = (1.0 / ((1 - r1sq) *
                       (1 - r1sq))) * (r23 * (1.0 - 2.0 * r1sq) - .5 * r1sq *
                                       (1 - 2.0 * r1sq - (r23 * r23)))
    variance2 = np.sqrt((2.0 - 2.0 * variance) / (n - 3.0))

    p = (z12 - z13) / variance2
    alpha = norm.sf(p)

    return p, alpha
