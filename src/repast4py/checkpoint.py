# Copyright 2021, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: repast4py
# By: Argonne National Laboratory
# License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt
"""
This module includes classes and functions for checkpointing a repast4py simulation.
simulation.
"""

from typing import Union, List
from os import PathLike
from dataclasses import dataclass, field
import pickle

from . import random


@dataclass
class Checkpoint:

    checkpoint_at: float = 0.0
    random_state: List = field(default_factory=list)


def save_random(checkpoint: Checkpoint):
    state = random.default_rng.bit_generator.state
    checkpoint.random_state = [random.seed, state]


def restore_random(checkpoint: Checkpoint):
    random.seed = checkpoint.random_state[0]
    random.default_rng.bit_generator.state = checkpoint.random_state[1]


def save(fname: Union[str, PathLike]):
    checkpoint = Checkpoint()
    save_random(checkpoint)

    with open(fname, 'wb') as fout:
        pickle.dump(checkpoint, fout)
