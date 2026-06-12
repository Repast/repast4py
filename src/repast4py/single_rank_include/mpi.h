// Copyright 2021, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: repast4py
// By: Argonne National Laboratory
// License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt
//
// Single-rank stub of the MPI C API.
//
// This header is used in place of a real <mpi.h> when repast4py is built with
// R4PY_SINGLE_RANK set (see setup.py). It provides trivial, header-only
// implementations of the small subset of MPI used by the _space extension,
// assuming a single process (size == 1, rank == 0). No MPI library is required.
//
// The only function whose behavior is non-trivial is MPI_Allgather: at size 1
// it must copy the local send buffer into the receive buffer, and the number of
// bytes to copy depends on the MPI_Datatype argument (count is in elements, not
// bytes). _r4py_type_size derives the element size from the datatype tag.

#ifndef R4PY_SINGLE_RANK_MPI_H
#define R4PY_SINGLE_RANK_MPI_H

#include <cstddef>
#include <cstring>

typedef int MPI_Comm;
typedef int MPI_Datatype;

#define MPI_COMM_WORLD ((MPI_Comm)0)
#define MPI_COMM_NULL ((MPI_Comm)-1)

#define MPI_SUCCESS 0

// Distinct datatype tags. Only the byte size matters (see _r4py_type_size).
#define MPI_INT ((MPI_Datatype)1)
#define MPI_LONG ((MPI_Datatype)2)
#define MPI_LONG_LONG ((MPI_Datatype)3)
#define MPI_DOUBLE ((MPI_Datatype)4)

static inline std::size_t _r4py_type_size(MPI_Datatype t) {
    switch (t) {
        case MPI_DOUBLE:    return sizeof(double);
        case MPI_LONG:      return sizeof(long);
        case MPI_LONG_LONG: return sizeof(long long);
        case MPI_INT:       return sizeof(int);
        default:            return 0;  // unsupported datatype -- a bug to hit
    }
}

static inline int MPI_Comm_size(MPI_Comm, int* size) {
    *size = 1;
    return MPI_SUCCESS;
}

static inline int MPI_Comm_rank(MPI_Comm, int* rank) {
    *rank = 0;
    return MPI_SUCCESS;
}

static inline int MPI_Comm_free(MPI_Comm* comm) {
    *comm = MPI_COMM_NULL;
    return MPI_SUCCESS;
}

// With a single node every dimension has exactly one process.
static inline int MPI_Dims_create(int /*nnodes*/, int ndims, int* dims) {
    for (int i = 0; i < ndims; ++i) {
        dims[i] = 1;
    }
    return MPI_SUCCESS;
}

// The cartesian communicator is identical to the input communicator at size 1.
static inline int MPI_Cart_create(MPI_Comm comm, int /*ndims*/, const int* /*dims*/,
                                  const int* /*periods*/, int /*reorder*/, MPI_Comm* cart_comm) {
    *cart_comm = comm;
    return MPI_SUCCESS;
}

// The single rank sits at the origin of the cartesian grid.
static inline int MPI_Cart_coords(MPI_Comm, int /*rank*/, int ndims, int* coords) {
    for (int i = 0; i < ndims; ++i) {
        coords[i] = 0;
    }
    return MPI_SUCCESS;
}

// All coordinates map back to the only rank.
static inline int MPI_Cart_rank(MPI_Comm, const int* /*coords*/, int* rank) {
    *rank = 0;
    return MPI_SUCCESS;
}

// Gather-from-everyone == copy-my-own at size 1. Byte count is derived from the
// datatype -- a fixed-length copy would silently truncate multi-element buffers.
static inline int MPI_Allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                void* recvbuf, int /*recvcount*/, MPI_Datatype /*recvtype*/,
                                MPI_Comm) {
    std::memcpy(recvbuf, sendbuf, (std::size_t)sendcount * _r4py_type_size(sendtype));
    return MPI_SUCCESS;
}

#endif  // R4PY_SINGLE_RANK_MPI_H
