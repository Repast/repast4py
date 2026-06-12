# Copyright 2021, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: repast4py
# By: Argonne National Laboratory
# License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

"""repast4py is an agent-based modeling framework for distributed simulation.

The MPI namespace used throughout repast4py is available as::

    from repast4py import MPI

In the default build this is the real :mod:`mpi4py.MPI`. When repast4py is built
single-rank (with the ``R4PY_SINGLE_RANK`` environment variable set), it is instead a
pure-Python substitute that requires neither a native MPI library nor mpi4py: in that
case ``MPI.COMM_WORLD`` has size 1, rank 0, and all collective operations are
identities / no-ops. Either way, ``MPI.COMM_WORLD`` is the communicator passed to
shared projections and contexts.
"""

__version__ = '1.2.1'

from ._mpi import MPI  # noqa: E402,F401
