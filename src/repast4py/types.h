#ifndef R4PY_SRC_TYPES_H
#define R4PY_SRC_TYPES_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

namespace repast4py {

#if defined(_MSC_VER)
using long_t = long long;
const auto PyLong_AsLongT = PyLong_AsLongLong;
const auto PyLong_FromLongT = PyLong_FromLongLong;
#else
using long_t = long;
const auto PyLong_AsLongT = PyLong_AsLong;
const auto PyLong_FromLongT = PyLong_FromLong;
#endif
}


#endif