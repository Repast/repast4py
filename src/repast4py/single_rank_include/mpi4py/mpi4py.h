// Copyright 2021, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: repast4py
// By: Argonne National Laboratory
// License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt
//
// Single-rank stub of the mpi4py C API.
//
// Used in place of the real <mpi4py/mpi4py.h> when repast4py is built with
// R4PY_SINGLE_RANK set (see setup.py). It mirrors the macro pattern of mpi4py's
// own pycapi.h so existing call sites in spacemodule.cpp compile unchanged:
//
//   - PyMPIComm_Type is a macro over a cached PyTypeObject*, so &PyMPIComm_Type
//     yields the type object that PyArg_ParseTupleAndKeywords' "O!" format checks
//     incoming communicator arguments against.
//   - PyMPIComm_Get extracts a native MPI_Comm* (there is only one at size 1).
//   - PyMPIComm_New wraps a native MPI_Comm back into a Python object; it returns
//     the stub's COMM_WORLD singleton, which is an instance of the cached
//     Intracomm type -- so a comm produced here (e.g. CartesianTopology.comm) can
//     be passed straight back into a SharedGrid and still pass the "O!" check.
//
// import_mpi4py() binds these to the pure-Python stub in repast4py._mpi_stub.

#ifndef R4PY_SINGLE_RANK_MPI4PY_H
#define R4PY_SINGLE_RANK_MPI4PY_H

#include <Python.h>

#include "mpi.h"

// Cached references to the Python-side stub, populated by import_mpi4py().
// Kept alive for the lifetime of the process (never DECREF'd).
static PyTypeObject* _r4py_PyMPIComm = NULL;
static PyObject* _r4py_comm_world = NULL;

// The single native communicator. Its value is irrelevant at size 1 -- nothing
// ever dereferences it -- but call sites take its address.
static MPI_Comm _r4py_world_comm = MPI_COMM_WORLD;

#define PyMPIComm_Type (*_r4py_PyMPIComm)

static inline int import_mpi4py(void) {
    PyObject* mod = PyImport_ImportModule("repast4py._mpi_stub");
    if (!mod) {
        return -1;
    }
    PyObject* mpi = PyObject_GetAttrString(mod, "MPI");
    Py_DECREF(mod);
    if (!mpi) {
        return -1;
    }

    PyObject* tp = PyObject_GetAttrString(mpi, "Intracomm");
    if (!tp || !PyType_Check(tp)) {
        Py_XDECREF(tp);
        Py_DECREF(mpi);
        PyErr_SetString(PyExc_ImportError,
                        "repast4py._mpi_stub.MPI.Intracomm is missing or not a type");
        return -1;
    }
    _r4py_PyMPIComm = (PyTypeObject*)tp;  // keep reference

    PyObject* cw = PyObject_GetAttrString(mpi, "COMM_WORLD");
    Py_DECREF(mpi);
    if (!cw) {
        return -1;
    }
    _r4py_comm_world = cw;  // keep reference

    return 0;
}

static inline MPI_Comm* PyMPIComm_Get(PyObject* /*comm*/) {
    return &_r4py_world_comm;
}

static inline PyObject* PyMPIComm_New(MPI_Comm /*comm*/) {
    if (_r4py_comm_world == NULL) {
        if (import_mpi4py() < 0) {
            return NULL;
        }
    }
    Py_INCREF(_r4py_comm_world);
    return _r4py_comm_world;
}

#endif  // R4PY_SINGLE_RANK_MPI4PY_H
