# Copyright 2021, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: repast4py
# By: Argonne National Laboratory
# License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

"""Random numbers for repast4py. When this module is imported, :data:`repast4py.random.default_rng` is
created using the current epoch time as the random seed, and :data:`repast4py.random.seed` is
set to that value. The default random number generator is a numpy.random.Generator.
See that `API documentation <https://numpy.org/doc/stable/reference/random/generator.html>`_
for more information on the available distributions and sampling functions.
"""

import numpy as np
import time
import torch

default_rng: np.random.Generator = None
"""numpy.random.Generator: repast4py's default random generator created using init.
See the Generator `API documentation <https://numpy.org/doc/stable/reference/random/generator.html>`_
for more information on the available distributions and sampling functions.
"""

seed: int = None
"""The current random seed used by :data:`repast4py.random.default_rng`"""


def init(rng_seed: int = None):
    """Initializes the default random number generator using the specified seed.

    Args:
        rng_seed: the random number seed. Defaults to None in which case, the current
            time as returned by :samp:`time.time()` is used as the seed.
    """
    global default_rng, seed
    if rng_seed is None:
        seed = int(time.time())
    else:
        seed = rng_seed
    torch.manual_seed(seed)
    default_rng = np.random.default_rng(seed)


init()
