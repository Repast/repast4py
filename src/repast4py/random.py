"""Random numbers for repast4py
"""

import numpy as np
import torch

default_rng = None
"""np.random.Generator: default random generator created using init
"""

seed = None
"""Current random seed"""


def init(rng_seed: int=None):
    """Initializes the default random number generator using the specified seed

    Args:
        seed: the random number seed
    """
    global default_rng, seed
    seed = rng_seed
    torch.manual_seed(rng_seed)
    default_rng = np.random.default_rng(rng_seed)
