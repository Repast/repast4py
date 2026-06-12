# Copyright 2021, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: repast4py
# By: Argonne National Laboratory
# License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt
"""Single-rank substitute for ``mpi4py.MPI``.

This module provides an ``MPI`` namespace that mimics the small subset of the
mpi4py API used by repast4py, assuming a single process (size == 1, rank == 0).
Every collective reduces to an identity or no-op, so no MPI library is required.

It is selected over the real mpi4py by :mod:`repast4py._mpi` when repast4py is
built single-rank. The C ``_space`` extension's stub ``import_mpi4py`` also
imports this module directly to bind its ``Intracomm`` type and ``COMM_WORLD``
singleton (see ``single_rank_include/mpi4py/mpi4py.h``), so the module path and
the names ``MPI.Intracomm`` / ``MPI.COMM_WORLD`` must remain stable.
"""


class _Op:
    """Sentinel for an MPI reduction operator (e.g. MPI.SUM). At size 1 the
    operator is never applied, so only its identity/repr matter."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'MPI.{self.name}'


class _Datatype:
    """Sentinel for an MPI datatype (e.g. MPI.DOUBLE). Provided so buffer-method
    signatures that name a datatype resolve; unused at size 1."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'MPI.{self.name}'


def _buffer(b):
    """mpi4py buffer arguments may be a bare buffer or a ``[buffer, count,
    datatype]``/``[buffer, datatype]`` list. Return the underlying buffer."""
    if isinstance(b, (list, tuple)):
        return b[0]
    return b


class Intracomm:
    """Single-rank communicator. All collectives are identities/no-ops."""

    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    @property
    def rank(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return 1

    # --- pickle-based (lower-case) collectives ---------------------------------

    def alltoall(self, sendobj):
        # sendobj has one entry per rank (length 1); each rank keeps its own.
        return list(sendobj)

    def allgather(self, sendobj):
        return [sendobj]

    def allreduce(self, sendobj, op=None):
        return sendobj

    def gather(self, sendobj, root=0):
        return [sendobj]

    def scatter(self, sendobj, root=0):
        return sendobj[0]

    def bcast(self, obj, root=0):
        return obj

    # --- buffer-based (upper-case) collectives ---------------------------------

    def Reduce(self, sendbuf, recvbuf, op=None, root=0):
        # Called even at size 1 (e.g. repast4py.logging). Copy send into recv.
        if recvbuf is not None:
            _buffer(recvbuf)[...] = _buffer(sendbuf)

    def Allreduce(self, sendbuf, recvbuf, op=None):
        if recvbuf is not None:
            _buffer(recvbuf)[...] = _buffer(sendbuf)

    def Allgather(self, sendbuf, recvbuf):
        if recvbuf is not None:
            _buffer(recvbuf)[...] = _buffer(sendbuf)

    def Alltoall(self, sendbuf, recvbuf):
        # Only invoked when size > 1, so never exercised single-rank; copy anyway.
        if recvbuf is not None:
            _buffer(recvbuf)[...] = _buffer(sendbuf)

    def Alltoallv(self, sendbuf, recvbuf):
        if recvbuf is not None:
            _buffer(recvbuf)[...] = _buffer(sendbuf)

    def Scatter(self, sendbuf, recvbuf, root=0):
        # At size 1 the whole send buffer is this rank's portion.
        if recvbuf is not None:
            _buffer(recvbuf)[...] = _buffer(sendbuf)

    def Gather(self, sendbuf, recvbuf, root=0):
        if recvbuf is not None:
            _buffer(recvbuf)[...] = _buffer(sendbuf)

    # --- synchronization / lifecycle ------------------------------------------

    def Barrier(self):
        pass

    def barrier(self):
        pass

    def Dup(self):
        return Intracomm()

    def Free(self):
        pass


# Alias matching mpi4py, used in type hints (e.g. MPI.Comm).
Comm = Intracomm


class MPI:
    """Namespace mirroring ``mpi4py.MPI`` for single-rank execution."""

    Intracomm = Intracomm
    Comm = Comm

    # Reduction operators.
    SUM = _Op('SUM')
    MIN = _Op('MIN')
    MAX = _Op('MAX')
    PROD = _Op('PROD')
    LAND = _Op('LAND')
    LOR = _Op('LOR')

    # Datatypes.
    INT = _Datatype('INT')
    LONG = _Datatype('LONG')
    LONG_LONG = _Datatype('LONG_LONG')
    DOUBLE = _Datatype('DOUBLE')
    FLOAT = _Datatype('FLOAT')

    # The single communicator.
    COMM_WORLD = Intracomm()
