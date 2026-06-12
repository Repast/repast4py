# Copyright 2021, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: repast4py
# By: Argonne National Laboratory
# License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt
"""Selects the MPI implementation used by repast4py.

By default the real :mod:`mpi4py.MPI` is used. When repast4py is built
single-rank (``R4PY_SINGLE_RANK`` set at build time), :mod:`setup.py` writes a
``_mpi_config`` module recording that choice, and the pure-Python single-rank
substitute :mod:`repast4py._mpi_stub` is used instead.

The build-time marker is the source of truth so that the Python ``MPI`` always
matches what the native ``_space`` extension was compiled against. The
``R4PY_SINGLE_RANK`` environment variable is honored only as a fallback when no
marker is present (e.g. running from a source tree that was never built).

repast4py modules should ``from repast4py import MPI`` rather than importing
mpi4py directly.
"""

try:
    from . import _mpi_config
    _single_rank = _mpi_config.SINGLE_RANK
except ImportError:
    import os
    _single_rank = os.environ.get('R4PY_SINGLE_RANK', '') not in ('', '0')

if _single_rank:
    from ._mpi_stub import MPI
else:
    from mpi4py import MPI

__all__ = ['MPI']
