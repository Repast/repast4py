# Copyright 2021, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: repast4py
# By: Argonne National Laboratory
# License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt
"""In-tree PEP 517 build backend for repast4py.

Wraps setuptools' build backend to make ``mpi4py`` a *dynamic* build requirement:
it is needed only for the default (native MPI) build, not for the single-rank
build (``R4PY_SINGLE_RANK`` set). Keeping it out of the static
``[build-system].requires`` lets ``R4PY_SINGLE_RANK=1 pip install .`` succeed under
normal build isolation without pulling mpi4py (which would require a native MPI).

setup.py imports mpi4py lazily (only when building the native extensions), so the
``egg_info`` run that setuptools uses to compute requirements works without it.
"""

import os

from setuptools import build_meta as _bm

# Re-export the PEP 517 hooks we do not customize (build_wheel, build_sdist,
# build_editable, prepare_metadata_for_build_wheel/editable, ...).
from setuptools.build_meta import *  # noqa: F401,F403


def _single_rank() -> bool:
    return os.environ.get("R4PY_SINGLE_RANK", "") not in ("", "0")


def _mpi_req():
    return [] if _single_rank() else ["mpi4py"]


def get_requires_for_build_wheel(config_settings=None):
    return _bm.get_requires_for_build_wheel(config_settings) + _mpi_req()


def get_requires_for_build_editable(config_settings=None):
    return _bm.get_requires_for_build_editable(config_settings) + _mpi_req()


def get_requires_for_build_sdist(config_settings=None):
    return _bm.get_requires_for_build_sdist(config_settings) + _mpi_req()
