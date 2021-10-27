// Copyright 2021, UChicago Argonne, LLC 
// All Rights Reserved
// Software Name: repast4py
// By: Argonne National Laboratory
// License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

#ifndef R4PY_COREMODULE_H
#define R4PY_COREMODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#define R4PyCore_API_pointers 3

#ifdef R4PY_CORE_MODULE

#else

static void** R4PyCore_API;

#define R4Py_AgentType (*(PyTypeObject *)R4PyCore_API[0])
#define R4Py_AgentIterType  (*(PyTypeObject *)R4PyCore_API[1])
#define R4Py_PyObjectIterType  (*(PyTypeObject *)R4PyCore_API[2])

static int import_core(void) {
    R4PyCore_API = (void **)PyCapsule_Import("repast4py._core._C_API", 0);
    return (R4PyCore_API != NULL) ? 0 : -1;
}

#endif


#ifdef __cplusplus
}
#endif

#endif