"""Random numbers for repast4py
"""

import numpy as np

default_rng = None
"""np.random.Generator: default random generator created using init
"""


def init(seed: int=None):
    """Initializes the default random number generator using the specified seed

    Args:
        seed: the random number seed
    """
    global default_rng
    default_rng = np.random.default_rng(seed)