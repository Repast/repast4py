// Copyright 2021, UChicago Argonne, LLC 
// All Rights Reserved
// Software Name: repast4py
// By: Argonne National Laboratory
// License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <new>
#include "structmember.h"


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// See https://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL REPAST4PY_ARRAY_API
#include "numpy/arrayobject.h"

#include "mpi4py/mpi4py.h"

#include "space.h"
#include "grid.h"
#include "cspace.h"
#include "coremodule.h"
#include "distributed_space.h"

using namespace repast4py;

constexpr int STICKY_BRDR = 0;
constexpr int PERIODIC_BRDR = 1;

constexpr int MULT_OT = 0;
constexpr int SINGLE_OT = 1;


//////////////////// DiscretePoint ///////////////////////

static void DiscretePoint_dealloc(R4Py_DiscretePoint* self) {
    Py_DECREF(self->coords);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* DiscretePoint_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    
    R4Py_DiscretePoint* self;
    self = (R4Py_DiscretePoint*) type->tp_alloc(type, 0);
    if (self != NULL) {
        npy_intp shape[] = {3};
        self->coords = (PyArrayObject*)PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_LONG), 
            1, shape, NULL, NULL, NPY_ARRAY_C_CONTIGUOUS, NULL);
        if (self->coords == NULL) {
            Py_DECREF(self);
            return NULL;
        }
    }

    return (PyObject*) self;
}


static int DiscretePoint_init(R4Py_DiscretePoint* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {(char*)"x", (char*)"y", (char*)"z", NULL};
    
    long* d = (long*)PyArray_DATA(self->coords);
    d[2] = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ll|l", kwlist, &d[0], &d[1], &d[2])) {
        return -1;
    }

    return 0;
}

static PyObject* DiscretePoint_reset1D(PyObject* self, PyObject* args) {
    long* d = (long*)PyArray_DATA(((R4Py_DiscretePoint*)self)->coords);
    if (!PyArg_ParseTuple(args, "l", &d[0])) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* DiscretePoint_reset2D(PyObject* self, PyObject* args) {
    long* d = (long*)PyArray_DATA(((R4Py_DiscretePoint*)self)->coords);
    if (!PyArg_ParseTuple(args, "ll", &d[0], &d[1])) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* DiscretePoint_reset3D(PyObject* self, PyObject* args) {
    long* d = (long*)PyArray_DATA(((R4Py_DiscretePoint*)self)->coords);
    if (!PyArg_ParseTuple(args, "lll", &d[0], &d[1], &d[2])) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* DiscretePoint_reset(PyObject* self, PyObject* args) {
    long* d = (long*)PyArray_DATA(((R4Py_DiscretePoint*)self)->coords);
    PyTupleObject* pt;
    if (!PyArg_ParseTuple(args, "O!", &PyTuple_Type, &pt)) {
        return NULL;
    }

    d[0] = PyLong_AsLong(PyTuple_GET_ITEM(pt, 0));
    d[1] = PyLong_AsLong(PyTuple_GET_ITEM(pt, 1));
    d[2] = PyLong_AsLong(PyTuple_GET_ITEM(pt, 2));

    Py_RETURN_NONE;
}

static PyObject* DiscretePoint_reset_from_array(PyObject* self, PyObject* args) {
    PyArrayObject* arr;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arr)) {
        return NULL;
    }

    // check -- must be 1 dimension, then shape [0] will give number
    int ndim = PyArray_NDIM(arr);
    if (ndim != 1) {
        PyErr_SetString(PyExc_RuntimeError, "R4Py_DiscretePoint can only be reset from a 1D numpy array");
        return NULL;
    }

    npy_intp* shape = PyArray_SHAPE(arr);
    npy_int c = shape[0];
    
    long* d = (long*)PyArray_DATA(((R4Py_DiscretePoint*)self)->coords);
    int typ = PyArray_TYPE(arr);
    if (typ == NPY_LONG) {
        long* o = (long*)PyArray_DATA(arr);
        for (int i = 0; i < c && i < 3; ++i) {
            d[i] = o[i];
        }
    } else if (typ == NPY_INT) {
        int* o = (int*)PyArray_DATA(arr);
        for (int i = 0; i < c && i < 3; ++i) {
            d[i] = o[i];
        }
    } else {
        PyErr_SetString(PyExc_RuntimeError, "R4Py_DiscretePoint can only be reset from numpy array of integers");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* DiscretePoint_get_coords(R4Py_DiscretePoint* self, void* closure) {
    Py_INCREF(self->coords);
    return (PyObject*)self->coords;
}

static PyObject* DiscretePoint_get_x(R4Py_DiscretePoint* self, void* closure) {
    return PyLong_FromLong(((long*)PyArray_DATA(self->coords))[0]);
}

static PyObject* DiscretePoint_get_y(R4Py_DiscretePoint* self, void* closure) {
    return PyLong_FromLong(((long*)PyArray_DATA(self->coords))[1]);
}

static PyObject* DiscretePoint_get_z(R4Py_DiscretePoint* self, void* closure) {
    return PyLong_FromLong(((long*)PyArray_DATA(self->coords))[2]);
}

PyDoc_STRVAR(dp_x,
    "int: Gets this DiscretePoint's x coordinate.");

PyDoc_STRVAR(dp_y,
    "int: Gets this DiscretePoint's y coordinate.");

PyDoc_STRVAR(dp_z,
    "int: Gets this DiscretePoint's z coordinate.");

PyDoc_STRVAR(dp_c,
    "numpy.array: Gets this DiscretePoint's coordinates as 3 element numpy array.");


static PyGetSetDef DiscretePoint_get_setters[] = {
    {(char*)"x", (getter)DiscretePoint_get_x, NULL, dp_x, NULL},
    {(char*)"y", (getter)DiscretePoint_get_y, NULL, dp_y, NULL},
    {(char*)"z", (getter)DiscretePoint_get_z, NULL, dp_z, NULL},
    {(char*)"coordinates", (getter)DiscretePoint_get_coords, NULL, dp_c, NULL},
    {NULL}
};

PyDoc_STRVAR(dp_reset_from_array,
    "_reset_from_array(np_array)\n\n"
    "Resets the coordinate values of this DiscretePoint from the specified array's elements. "
    "The array must be of the integer type, have a single dimension and have at least one element. The "
    "x coordinate is set from the first element, y from the second, and z from the 3rd.\n\n"
    "**This method should ONLY be used in code fully responsible for the point, that is, the "
    "point was not returned from any repast4py method or function.**\n\n"
    "Args:\n"
    "    np_array(numpy.array): the array to reset from."
);

static PyMethodDef DiscretePoint_methods[] = {
    {"_reset1D", DiscretePoint_reset1D, METH_VARARGS, ""},
    {"_reset2D", DiscretePoint_reset2D, METH_VARARGS, ""},
    {"_reset3D", DiscretePoint_reset3D, METH_VARARGS, ""},
    {"_reset", DiscretePoint_reset, METH_VARARGS, ""},
    {"_reset_from_array", DiscretePoint_reset_from_array, METH_VARARGS, dp_reset_from_array},

    {NULL, NULL, 0, NULL}
};


static PyObject* DiscretePoint_repr(R4Py_DiscretePoint* self) {
    long* data = (long*)PyArray_DATA(self->coords);
    return PyUnicode_FromFormat("DiscretePoint(%ld, %ld, %ld)", data[0], data[1], data[2]);   
}

static PyObject* DiscretePoint_richcmp(PyObject* self, PyObject* other, int op) {
    if (op == Py_EQ && Py_TYPE(self) == Py_TYPE(other)) {
        long* p1 = (long*)PyArray_DATA(((R4Py_DiscretePoint*)self)->coords);
        long* p2 = (long*)PyArray_DATA(((R4Py_DiscretePoint*)other)->coords);
        if (p1[0] == p2[0] && p1[1] == p2[1] && p1[2] == p2[2]) 
            Py_RETURN_TRUE;
        else
            Py_RETURN_FALSE;

    }
    Py_RETURN_NOTIMPLEMENTED;
}

PyDoc_STRVAR(dp_dp,
    "DiscretePoint(x, y, z=0)\n\n"
    
    "A 3D point with discrete (integer) coordinates.\n\n"
    "Args:\n"
    "   x (int): the x coordinate.\n"
    "   y (int): the y coordinate.\n"
    "   z (int, optional): the z coordinate. Defaults to 0\n\n"
    ".. automethod:: _reset_from_array");
    


static PyTypeObject DiscretePointType = {
    PyVarObject_HEAD_INIT(NULL, 0) 
    "_space.DiscretePoint",                          /* tp_name */
    sizeof(R4Py_DiscretePoint),                      /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)DiscretePoint_dealloc,                                         /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_reserved */
    (reprfunc)DiscretePoint_repr,                                        /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash  */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    dp_dp,                         /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    DiscretePoint_richcmp,                    /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    DiscretePoint_methods,                                      /* tp_methods */
    0,                                      /* tp_members */
    DiscretePoint_get_setters,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)DiscretePoint_init,                                         /* tp_init */
    0,                                        /* tp_alloc */
    DiscretePoint_new                             /* tp_new */
};


/////////////////// Discrete Point End ///////

//////////////////// Continuous Point ///////////////////////

static void ContinuousPoint_dealloc(R4Py_ContinuousPoint* self) {
    Py_DECREF(self->coords);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* ContinuousPoint_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    
    R4Py_ContinuousPoint* self;
    self = (R4Py_ContinuousPoint*) type->tp_alloc(type, 0);
    if (self != NULL) {
        npy_intp shape[] = {3};
        self->coords = (PyArrayObject*)PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_DOUBLE), 
            1, shape, NULL, NULL, NPY_ARRAY_C_CONTIGUOUS, NULL);
        if (self->coords == NULL) {
            Py_DECREF(self);
            return NULL;
        }
    }

    return (PyObject*) self;
}


static int ContinuousPoint_init(R4Py_ContinuousPoint* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {(char*)"x", (char*)"y", (char*)"z", NULL};
    
    double* d = (double*)PyArray_DATA(self->coords);
    d[2] = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dd|d", kwlist, &d[0], &d[1], &d[2])) {
        return -1;
    }

    return 0;
}

static PyObject* ContinuousPoint_reset1D(PyObject* self, PyObject* args) {
    double* d = (double*)PyArray_DATA(((R4Py_ContinuousPoint*)self)->coords);
    if (!PyArg_ParseTuple(args, "d", &d[0])) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* ContinuousPoint_reset2D(PyObject* self, PyObject* args) {
    double* d = (double*)PyArray_DATA(((R4Py_ContinuousPoint*)self)->coords);
    if (!PyArg_ParseTuple(args, "dd", &d[0], &d[1])) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* ContinuousPoint_reset3D(PyObject* self, PyObject* args) {
    double* d = (double*)PyArray_DATA(((R4Py_ContinuousPoint*)self)->coords);
    if (!PyArg_ParseTuple(args, "ddd", &d[0], &d[1], &d[2])) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* ContinuousPoint_reset(PyObject* self, PyObject* args) {
    double* d = (double*)PyArray_DATA(((R4Py_ContinuousPoint*)self)->coords);
    PyTupleObject* pt;
    if (!PyArg_ParseTuple(args, "O!", &PyTuple_Type, &pt)) {
        return NULL;
    }

    d[0] = PyFloat_AsDouble(PyTuple_GET_ITEM(pt, 0));
    d[1] = PyFloat_AsDouble(PyTuple_GET_ITEM(pt, 1));
    d[2] = PyFloat_AsDouble(PyTuple_GET_ITEM(pt, 2));

    Py_RETURN_NONE;
}

static PyObject* ContinuousPoint_reset_from_array(PyObject* self, PyObject* args) {
    PyArrayObject* arr;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arr)) {
        return NULL;
    }

    // check -- must be 1 dimension, then shape [0] will give number
    int ndim = PyArray_NDIM(arr);
    if (ndim != 1) {
        PyErr_SetString(PyExc_RuntimeError, "R4Py_ContinuousPoint can only be reset from a 1D numpy array");
        return NULL;
    }

    npy_intp* shape = PyArray_SHAPE(arr);
    npy_int c = shape[0];
    
    double* d = (double*)PyArray_DATA(((R4Py_ContinuousPoint*)self)->coords);
    int typ = PyArray_TYPE(arr);
    if (typ == NPY_DOUBLE) {
        double* o = (double*)PyArray_DATA(arr);
        for (int i = 0; i < c && i < 3; ++i) {
            d[i] = o[i];
        }
    } else if (typ == NPY_FLOAT) {
        float* o = (float*)PyArray_DATA(arr);
        for (int i = 0; i < c && i < 3; ++i) {
            d[i] = o[i];
        }
    } else {
        PyErr_SetString(PyExc_RuntimeError, "R4Py_ContinuousPoint can only be reset from numpy array of floats");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* ContinuousPoint_get_coords(R4Py_ContinuousPoint* self, void* closure) {
    Py_INCREF(self->coords);
    return (PyObject*)self->coords;
}

static PyObject* ContinuousPoint_get_x(R4Py_ContinuousPoint* self, void* closure) {
    return PyFloat_FromDouble(((double*)PyArray_DATA(self->coords))[0]);
}

static PyObject* ContinuousPoint_get_y(R4Py_ContinuousPoint* self, void* closure) {
    return PyFloat_FromDouble(((double*)PyArray_DATA(self->coords))[1]);
}

static PyObject* ContinuousPoint_get_z(R4Py_ContinuousPoint* self, void* closure) {
    return PyFloat_FromDouble(((double*)PyArray_DATA(self->coords))[2]);
}

PyDoc_STRVAR(cp_x,
    "float: Gets this ContinuousPoint's x coordinate.");

PyDoc_STRVAR(cp_y,
    "float: Gets this ContinuousPoint's y coordinate.");

PyDoc_STRVAR(cp_z,
    "float: Gets this ContinuousPoint's z coordinate.");

PyDoc_STRVAR(cp_c,
    "numpy.array: Gets this ContinuousPoint's coordinates as 3 element numpy array.");
 
PyDoc_STRVAR(cp_reset_from_array,
    "_reset_from_array(np_array)\n\n"
    "Resets the coordinate values of this ContinuousPoint from the specified array's elements. "
    "The array must have a single dimension and have at least one element. The "
    "x coordinate is set from the first element, y from the second, and z from the 3rd.\n\n"
    "**This method should ONLY be used in code fully responsible for the point, that is, the "
    "point was not returned from any repast4py method or function.**\n\n"
    "Args:\n"
    "    np_array(numpy.array): the array to reset from."
);

static PyGetSetDef ContinuousPoint_get_setters[] = {
    {(char*)"x", (getter)ContinuousPoint_get_x, NULL, cp_x, NULL},
    {(char*)"y", (getter)ContinuousPoint_get_y, NULL, cp_y, NULL},
    {(char*)"z", (getter)ContinuousPoint_get_z, NULL, cp_z, NULL},
    {(char*)"coordinates", (getter)ContinuousPoint_get_coords, NULL, cp_c, NULL},
    {NULL}
};

static PyMethodDef ContinuousPoint_methods[] = {
    {"_reset1D", ContinuousPoint_reset1D, METH_VARARGS, ""},
    {"_reset2D", ContinuousPoint_reset2D, METH_VARARGS, ""},
    {"_reset3D", ContinuousPoint_reset3D, METH_VARARGS, ""},
    {"_reset", ContinuousPoint_reset, METH_VARARGS, ""},
    {"_reset_from_array", ContinuousPoint_reset_from_array, METH_VARARGS, cp_reset_from_array},
    {NULL, NULL, 0, NULL}
};


static PyObject* ContinuousPoint_repr(R4Py_ContinuousPoint* self) {
    double* data = (double*)PyArray_DATA(self->coords);
    return PyUnicode_FromFormat("ContinuousPoint(%s, %s, %s)", 
        PyOS_double_to_string(data[0], 'f', Py_DTSF_ADD_DOT_0, 12, NULL),
        PyOS_double_to_string(data[1], 'f', Py_DTSF_ADD_DOT_0, 12, NULL),
        PyOS_double_to_string(data[2], 'f', Py_DTSF_ADD_DOT_0, 12, NULL));
}

static PyObject* ContinuousPoint_richcmp(PyObject* self, PyObject* other, int op) {
    if (op == Py_EQ && Py_TYPE(self) == Py_TYPE(other)) {
        double* p1 = (double*)PyArray_DATA(((R4Py_ContinuousPoint*)self)->coords);
        double* p2 = (double*)PyArray_DATA(((R4Py_ContinuousPoint*)other)->coords);
        if (p1[0] == p2[0] && p1[1] == p2[1] && p1[2] == p2[2]) 
            Py_RETURN_TRUE;
        else
            Py_RETURN_FALSE;

    }
    Py_RETURN_NOTIMPLEMENTED;
}

PyDoc_STRVAR(cp_cp,
    "ContinuousPoint(x, y, z=0)\n\n"
    
    "A 3D point with continuous (floating point) coordinates.\n\n"
    "Args:\n"
    "   x (float): the x coordinate.\n"
    "   y (float): the y coordinate.\n"
    "   z (float, optional): the z coordinate. Defaults to 0.0\n\n"
    ".. automethod:: _reset_from_array");


static PyTypeObject ContinuousPointType = {
    PyVarObject_HEAD_INIT(NULL, 0) 
    "_space.ContinuousPoint",                          /* tp_name */
    sizeof(R4Py_ContinuousPoint),                      /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)ContinuousPoint_dealloc,                                         /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_reserved */
    (reprfunc)ContinuousPoint_repr,                                        /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash  */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    cp_cp,                         /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    ContinuousPoint_richcmp,                    /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    ContinuousPoint_methods,                                      /* tp_methods */
    0,                                      /* tp_members */
    ContinuousPoint_get_setters,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)ContinuousPoint_init,                                         /* tp_init */
    0,                                        /* tp_alloc */
    ContinuousPoint_new                             /* tp_new */
};
/////////////////// Continuous Point End ///////

/////////////////// GridStickyBorders ////////////

static void GridStickyBorders_dealloc(R4Py_GridStickyBorders* self) {
    delete self->borders;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *GridStickyBorders_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    R4Py_GridStickyBorders *self = (R4Py_GridStickyBorders *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        // maybe I should create it here, rather than in init??
        self->borders = nullptr;
    }
    return (PyObject *)self;
}

static int GridStickyBorders_init(R4Py_GridStickyBorders* self, PyObject* args, PyObject* kwds) {
    PyObject* bounds;
    static char *kwlist[] = {(char *)"bounds", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,&PyTuple_Type, &bounds)) {
        return -1;
    }

    long xmin, width;
    long ymin, height;
    long zmin, depth;

    if (!PyArg_ParseTuple(bounds, "llllll", &xmin, &width, &ymin, &height, &zmin, &depth))
    {
        return -1;
    }

    BoundingBox box(xmin, width, ymin, height, zmin, depth);
    self->borders = new GridStickyBorders(box);

    if (!self->borders) {
        PyErr_SetString(PyExc_RuntimeError, "Error creating native code sticky grid borders");
        return -1;
    }
    return 0;
}

PyDoc_STRVAR(gsb_trans,
    "_transform(pt1, pt2)\n\n"
    
    "Transforms pt1 according to sticky semantics, assigning the result to pt2\n\n"
    "Args:\n"
    "   pt1 (repast4py.space.DiscretePoint): the point to transform.\n"
    "   pt2 (repast4py.space.DiscretePoint): the result of the transform."
);

static PyObject* GridStickyBorders_transform(PyObject* self, PyObject* args) {
    PyObject* pt, *ret_pt;
    if (!PyArg_ParseTuple(args, "O!O!", &DiscretePointType, &pt, &DiscretePointType, &ret_pt)) {
        return NULL;
    }
    ((R4Py_GridStickyBorders*)self)->borders->transform((R4Py_DiscretePoint*)pt, (R4Py_DiscretePoint*)ret_pt);
    Py_RETURN_NONE;
}

static PyMethodDef GridStickyBorders_methods[] = {
    {"_transform", GridStickyBorders_transform, METH_VARARGS, gsb_trans},
     {NULL, NULL, 0, NULL}
};

PyDoc_STRVAR(gsb_gsb,
    "GridStickyBorders(bounding_box)\n\n"
    
    "Grid Borders with :attr:`BorderType.Sticky` semantics.\n\n"
    "Borders objects can transform a coordinate depending on the border semantics. Sticky "
    "borders will clip the coordinates to the current bounds if the coordinates are outside "
    " the current bounds.\n\n"
    "Args:\n"
    "   bounding_box (repast4py.space.BoundingBox): the dimensions of the grid.");

static PyTypeObject R4Py_GridStickyBordersType = {
    PyVarObject_HEAD_INIT(NULL, 0) 
    "_space.GridStickyBorders",/* tp_name */
    sizeof(R4Py_GridStickyBorders),                      /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)GridStickyBorders_dealloc,                                         /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_reserved */
    0,                                      /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash  */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    gsb_gsb,                         /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                 /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    GridStickyBorders_methods,                                      /* tp_methods */
    0,                                      /* tp_members */
    0,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)GridStickyBorders_init,                                         /* tp_init */
    0,                                        /* tp_alloc */
    GridStickyBorders_new                             /* tp_new */
};

/////////////////// GridStickyBorders End ////////////

/////////////////// GridPeriodicBorders ////////////

static void GridPeriodicBorders_dealloc(R4Py_GridPeriodicBorders* self) {
    delete self->borders;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *GridPeriodicBorders_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    R4Py_GridPeriodicBorders *self = (R4Py_GridPeriodicBorders *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        // maybe I should create it here, rather than in init??
        self->borders = nullptr;
    }
    return (PyObject *)self;
}

static int GridPeriodicBorders_init(R4Py_GridPeriodicBorders* self, PyObject* args, PyObject* kwds) {
    PyObject* bounds;
    static char *kwlist[] = {(char *)"bounds", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,&PyTuple_Type, &bounds)) {
        return -1;
    }

    long xmin, width;
    long ymin, height;
    long zmin, depth;

    if (!PyArg_ParseTuple(bounds, "llllll", &xmin, &width, &ymin, &height, &zmin, &depth))
    {
        return -1;
    }

    BoundingBox box(xmin, width, ymin, height, zmin, depth);
    self->borders = new GridPeriodicBorders(box);

    if (!self->borders) {
        PyErr_SetString(PyExc_RuntimeError, "Error creating native code Periodic grid borders");
        return -1;
    }
    return 0;
}

static PyObject* GridPeriodicBorders_transform(PyObject* self, PyObject* args) {
    PyObject* pt, *ret_pt;
    if (!PyArg_ParseTuple(args, "O!O!", &DiscretePointType, &pt, &DiscretePointType, &ret_pt)) {
        return NULL;
    }
    ((R4Py_GridPeriodicBorders*)self)->borders->transform((R4Py_DiscretePoint*)pt, (R4Py_DiscretePoint*)ret_pt);
    Py_RETURN_NONE;
}

PyDoc_STRVAR(gpb_trans,
    "_transform(pt1, pt2)\n\n"
    
    "Transforms pt1 according to periodic (wrapping) semantics, assigning the result to pt2\n\n"
    "Args:\n"
    "   pt1 (repast4py.space.DiscretePoint): the point to transform.\n"
    "   pt2 (repast4py.space.DiscretePoint): the result of the transform."
);

static PyMethodDef GridPeriodicBorders_methods[] = {
    {"_transform", GridPeriodicBorders_transform, METH_VARARGS, gpb_trans},
     {NULL, NULL, 0, NULL}
};

PyDoc_STRVAR(gpb_gpb,
    "GridPeriodicBorders(bounding_box)\n\n"
    
    "Grid Borders with periodic semantics.\n\n"
    "Borders objects can transform a coordinate depending on the border semantics. Periodic "
    "borders will wrap a coordinate along the appropriate dimension if a coordinate is outside "
    " the current bounds.\n\n"
    "Args:\n"
    "   bounding_box (repast4py.space.BoundingBox): the dimensions of the grid.");

static PyTypeObject R4Py_GridPeriodicBordersType = {
    PyVarObject_HEAD_INIT(NULL, 0) 
    "_space.GridPeriodicBorders",/* tp_name */
    sizeof(R4Py_GridPeriodicBorders),                      /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)GridPeriodicBorders_dealloc,                                         /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_reserved */
    0,                                      /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash  */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    gpb_gpb,                         /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                 /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    GridPeriodicBorders_methods,                                      /* tp_methods */
    0,                                      /* tp_members */
    0,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)GridPeriodicBorders_init,                                         /* tp_init */
    0,                                        /* tp_alloc */
    GridPeriodicBorders_new                             /* tp_new */
};

/////////////////// GridPeriodicBorders End ////////////

/////////////////// GRID ///////////////////////
static void Grid_dealloc(R4Py_Grid* self) {
    delete self->grid;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Grid_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    R4Py_Grid* self = (R4Py_Grid*)type->tp_alloc(type, 0);
    if (self != NULL) {
        // maybe I should create it here, rather than in init??
        self->grid = nullptr;
    }
    return (PyObject*)self;
}

static int Grid_init(R4Py_Grid* self, PyObject* args, PyObject* kwds) {
    // bounds=box, border=BorderType.Sticky, occupancy=OccupancyType.Multiple
    static char* kwlist[] = {(char*)"name",(char*)"bounds", (char*)"borders",
        (char*)"occupancy", NULL};

    const char* name;
    PyObject* bounds;
    int border_type, occ_type;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO!ii", kwlist, &name, &PyTuple_Type, &bounds,
        &border_type, &occ_type)) 
    {
        return -1;
    }

    long xmin, width;
    long ymin, height;
    long zmin, depth;

    if (!PyArg_ParseTuple(bounds, "llllll", &xmin, &width, &ymin, &height, &zmin, &depth)) {
        return -1;
    }

    BoundingBox box(xmin, width, ymin, height, zmin, depth);
    if (border_type == STICKY_BRDR) {
        if (occ_type == MULT_OT) {
            self->grid = new Grid<MOSGrid>(name, box);
        } else if (occ_type == SINGLE_OT) {
            self->grid = new Grid<SOSGrid>(name, box);
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Invalid occupancy type");
            return -1;
        }
    } else if (border_type == PERIODIC_BRDR) {
        if (occ_type == MULT_OT) {
            self->grid = new Grid<MOPGrid>(name, box);
        } else if (occ_type == SINGLE_OT) {
            self->grid = new Grid<SOPGrid>(name, box);
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Invalid occupancy type");
            return -1;
        }

    } else {
        PyErr_SetString(PyExc_RuntimeError, "Invalid border type");
        return -1;
    }

    if (!self->grid) {
        PyErr_SetString(PyExc_RuntimeError, "Error creating native code grid");
        return -1;
    }
    return 0;
}

static PyObject* Grid_add(PyObject* self, PyObject* args) {
    PyObject* agent;
    if (!PyArg_ParseTuple(args, "O!", &R4Py_AgentType, &agent)) {
        return NULL;
    }
    bool ret_val = ((R4Py_Grid*)self)->grid->add((R4Py_Agent*)agent);
    return PyBool_FromLong(static_cast<long>(ret_val));
}

static PyObject* Grid_remove(PyObject* self, PyObject* args) {
    PyObject* agent;
    if (!PyArg_ParseTuple(args, "O!", &R4Py_AgentType, &agent)) {
        return NULL;
    }
    bool ret_val = ((R4Py_Grid*)self)->grid->remove((R4Py_Agent*)agent);
    return PyBool_FromLong(static_cast<long>(ret_val));
}

static PyObject* Grid_contains(PyObject* self, PyObject* args) {
    PyObject* agent;
    if (!PyArg_ParseTuple(args, "O!", &R4Py_AgentType, &agent)) {
        return NULL;
    }
    bool ret_val = ((R4Py_Grid*)self)->grid->contains((R4Py_Agent*)agent);
    return PyBool_FromLong(static_cast<long>(ret_val));
}

static PyObject* Grid_getLocation(PyObject* self, PyObject* args) {
    PyObject* agent;
    if (!PyArg_ParseTuple(args, "O!", &R4Py_AgentType, &agent)) {
        return NULL;
    }

    R4Py_DiscretePoint* pt = ((R4Py_Grid*)self)->grid->getLocation((R4Py_Agent*)agent);
    if (pt) {
        Py_INCREF(pt);
        return (PyObject*)pt;
    } else {
        Py_INCREF(Py_None);
        return Py_None;
    }
}

static PyObject* Grid_move(PyObject* self, PyObject* args) {
    PyObject* agent, *pt;
    if (!PyArg_ParseTuple(args, "O!O!", &R4Py_AgentType, &agent, &DiscretePointType, &pt)) {
        return NULL;
    }

    try {
        R4Py_DiscretePoint* ret = ((R4Py_Grid*)self)->grid->move((R4Py_Agent*)agent, (R4Py_DiscretePoint*)pt);
        if (ret) {
            Py_INCREF(ret);
            return (PyObject*)ret;
        } else {
            Py_INCREF(Py_None);
            return Py_None;
        }
    } catch (std::invalid_argument& ex) {
        PyErr_SetString(PyExc_RuntimeError, ex.what());
        return NULL;
    }
}

static PyObject* Grid_getAgent(PyObject* self, PyObject* args) {
    PyObject* pt;
    if (!PyArg_ParseTuple(args, "O!", &DiscretePointType, &pt)) {
        return NULL;
    }

    R4Py_Agent* ret =  ((R4Py_Grid*)self)->grid->getAgentAt((R4Py_DiscretePoint*)pt);
    if (ret) {
        // Is this necessary??
        Py_INCREF(ret);
        return (PyObject*)ret;
    } else {
        Py_INCREF(Py_None);
        return Py_None;
    }
}

static PyObject* Grid_getAgents(PyObject* self, PyObject* args) {
    PyObject* pt;
    if (!PyArg_ParseTuple(args, "O!", &DiscretePointType, &pt)) {
        return NULL;
    }

    std::shared_ptr<std::list<R4Py_Agent*>> list = ((R4Py_Grid*)self)->grid->getAgentsAt((R4Py_DiscretePoint*)pt);
    R4Py_AgentIter* agent_iter = (R4Py_AgentIter*)R4Py_AgentIterType.tp_new(&R4Py_AgentIterType, NULL, NULL);
    agent_iter->iter = new TAgentIter<std::list<R4Py_Agent*>>(list);
    return (PyObject*)agent_iter;
}

static PyObject* Grid_getName(PyObject* self, PyObject* args) {
    return PyUnicode_FromString(((R4Py_Grid*)self)->grid->name().c_str());
}

PyDoc_STRVAR(grd_name,
    "str: Gets the name of this grid.");

static PyGetSetDef Grid_get_setters[] = {
    {(char*)"name", (getter)Grid_getName, NULL, grd_name, NULL},
    {NULL}
};

PyDoc_STRVAR(grd_add,
    "add(agent)\n\n"
    "Adds the specified agent to this grid projection.\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent to add."
);

PyDoc_STRVAR(grd_rm,
    "remove(agent)\n\n"
    
    "Removes the specified agent from this grid projection.\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent to remove."
);

PyDoc_STRVAR(grd_cnt,
    "contains(agent)\n\n"
    "Gets whether or not this grid projection contains the specified agent.\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent to check.\n\n"
    "Returns:\n"
    "    bool: true if this grid contains the specified agent, otherwise false"
);

PyDoc_STRVAR(grd_move,
    "move(agent, pt)\n\n"
    
    "Moves the specified agent to the specified location, returning the moved to location.\n\n"
    "If the agent does not move beyond the grid's bounds, then the returned location will be "
    "be the same as the argument location. If the agent does move out of bounds, then the location "
    "is determined by the grid border's semantics (e.g., a location on the border if using :attr:`BorderType.Sticky` borders).\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent to move.\n"
    "    pt(repast4py.space.DiscretePoint): the location to move to.\n\n"
    "Returns:\n"
    "    repast4py.space.DiscretePoint: the location the agent has moved to or None if the agent cannot move to the specified location (e.g., if "
    "    the occupancy type is :attr:`OccupancyType.Single` and the location is occupied.)"
);

PyDoc_STRVAR(grd_location,
    "get_location(agent)\n\n"
    
    "Gets the location of the specified agent\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent whose location we want to get.\n\n"
    "Returns:\n"
    "    repast4py.space.DiscretePoint: the location of the specified agent or None if the agent is not in the space."
);

PyDoc_STRVAR(grd_geta,
    "get_agent(pt)\n\n"
    
    "Gets the agent at the specified location. If more than one agent exists at "
    "the specified location, the first agent to have moved to that location from among those currently at that location "
    "will be returned.\n\n"
    "Args:\n"
    "    pt(repast4py.space.DiscretePoint): the location to get the agent at.\n\n"
    "Returns:\n"
    "    repast4py.core.Agent: the agent at that location, or None if the location is empty"
);

PyDoc_STRVAR(grd_getas,
    "get_agents(pt)\n\n"
    
    "Gets an iterator over all the agents at the specified location.\n\n"
    "Args:\n"
    "    pt(repast4py.space.DiscretePoint): the location to get the agents at.\n\n"
    "Returns:\n"
    "    iterator: an iterator over all the agents at the specified location."
);

static PyMethodDef Grid_methods[] = {
    {"add", Grid_add, METH_VARARGS, grd_add},
    {"remove", Grid_remove, METH_VARARGS, grd_rm},
    {"contains", Grid_contains, METH_VARARGS, grd_cnt},
    {"move", Grid_move, METH_VARARGS, grd_move},
    {"get_location", Grid_getLocation, METH_VARARGS, grd_location},
    {"get_agent", Grid_getAgent, METH_VARARGS, grd_geta},
    {"get_agents", Grid_getAgents, METH_VARARGS, grd_getas},
    {NULL, NULL, 0, NULL}
};

PyDoc_STRVAR(grd_grd,
    "Grid(name, bounds, borders, occupancy)\n\n"
    "An N-dimensional cartesian discrete space where agents can occupy locations at "
    "a discrete integer coordinate.\n\n"
    "Args:\n"
    "   name (str): the name of the grid.\n"
    "   bounds (repast4py.geometry.BoundingBox): the dimensions of the grid.\n"
    "   borders (repast4py.space.BorderType): the border semantics - :attr:`BorderType.Sticky` or :attr:`BorderType.Periodic`.\n"
    "   occupancy (repast4py.space.OccupancyType): the type of occupancy in each cell - :attr:`OccupancyType.Multiple` or :attr:`OccupancyType.Single`."
);

static PyTypeObject R4Py_GridType = {
    PyVarObject_HEAD_INIT(NULL, 0) 
    "_space.Grid",                          /* tp_name */
    sizeof(R4Py_Grid),                      /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)Grid_dealloc,                                         /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_reserved */
    0,                                        /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash  */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    grd_grd,                         /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    Grid_methods,                                      /* tp_methods */
    0,                                      /* tp_members */
    Grid_get_setters,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)Grid_init,                                         /* tp_init */
    0,                                        /* tp_alloc */
    Grid_new                             /* tp_new */
};

//////////////////////////////// Grid End ///////////////////////

///////////////////////// Shared Grid ///////////////////////////
static void SharedGrid_dealloc(R4Py_SharedGrid* self) {
    delete self->grid;
    Py_XDECREF(self->cart_comm);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* SharedGrid_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    R4Py_SharedGrid* self = (R4Py_SharedGrid*)type->tp_alloc(type, 0);
    if (self != NULL) {
        // maybe I should create it here, rather than in init??
        self->grid = nullptr;
        self->cart_comm = nullptr;
    }
    return (PyObject*)self;
}

static int SharedGrid_init(R4Py_SharedGrid* self, PyObject* args, PyObject* kwds) {
    // bounds=box, border=BorderType.Sticky, occupancy=OccupancyType.Multiple
    static char* kwlist[] = {(char*)"name",(char*)"bounds", (char*)"borders",
        (char*)"occupancy", (char*)"buffer_size", (char*)"comm", NULL};

    const char* name;
    PyObject* bounds;
    int border_type, occ_type, buffer_size;
    PyObject* py_comm;

    //
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO!iiiO!", kwlist, &name, &PyTuple_Type, &bounds,
        &border_type, &occ_type, &buffer_size, &PyMPIComm_Type, &py_comm)) 
    {
        return -1;
    }

    long xmin, x_extent;
    long ymin, y_extent;
    long zmin, z_extent;

    if (!PyArg_ParseTuple(bounds, "llllll", &xmin, &x_extent, &ymin, &y_extent, &zmin, &z_extent)) {
        return -1;
    }

    // Because we are holding a reference to the communicator
    // I think this is necessary
    Py_INCREF(py_comm);
    MPI_Comm* comm_p = PyMPIComm_Get(py_comm);

    BoundingBox box(xmin, x_extent, ymin, y_extent, zmin, z_extent);
    if (border_type == STICKY_BRDR) {
        if (occ_type == MULT_OT) {
            self->grid = new SharedGrid<DistributedCartesianSpace<MOSGrid>>(name, box, buffer_size, 
            *comm_p);
        } else if (occ_type == SINGLE_OT) {
            self->grid = new SharedGrid<DistributedCartesianSpace<SOSGrid>>(name, box, buffer_size, 
            *comm_p);
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Invalid occupancy type");
            return -1;
        }
    } else if (border_type == PERIODIC_BRDR) {
        if (occ_type == MULT_OT) {
            self->grid = new SharedGrid<DistributedCartesianSpace<MOPGrid>>(name, box, buffer_size, 
            *comm_p);
        } else if (occ_type == SINGLE_OT) {
            self->grid = new SharedGrid<DistributedCartesianSpace<SOPGrid>>(name, box, buffer_size, 
            *comm_p);
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Invalid occupancy type");
            return -1;
        }
    } else {
        PyErr_SetString(PyExc_RuntimeError, "Invalid border type");
        return -1;
    }

    if (!self->grid) {
        PyErr_SetString(PyExc_RuntimeError, "Error creating native code shared grid");
        return -1;
    }

    self->cart_comm = PyMPIComm_New(self->grid->getCartesianCommunicator());
    if (self->cart_comm == NULL) {
        return -1;
    }

    return 0;
}

static PyObject* SharedGrid_add(PyObject* self, PyObject* args) {
    PyObject* agent;
    if (!PyArg_ParseTuple(args, "O!", &R4Py_AgentType, &agent)) {
        return NULL;
    }
    bool ret_val = ((R4Py_SharedGrid*)self)->grid->add((R4Py_Agent*)agent);
    return PyBool_FromLong(static_cast<long>(ret_val));
}

static PyObject* SharedGrid_remove(PyObject* self, PyObject* args) {
    PyObject* agent;
    if (!PyArg_ParseTuple(args, "O!", &R4Py_AgentType, &agent)) {
        return NULL;
    }
    bool ret_val = ((R4Py_SharedGrid*)self)->grid->remove((R4Py_Agent*)agent);
    return PyBool_FromLong(static_cast<long>(ret_val));
}

static PyObject* SharedGrid_contains(PyObject* self, PyObject* args) {
    PyObject* agent;
    if (!PyArg_ParseTuple(args, "O!", &R4Py_AgentType, &agent)) {
        return NULL;
    }
    bool ret_val = ((R4Py_SharedGrid*)self)->grid->contains((R4Py_Agent*)agent);
    return PyBool_FromLong(static_cast<long>(ret_val));
}

static PyObject* SharedGrid_getLocation(PyObject* self, PyObject* args) {
    PyObject* agent;
    if (!PyArg_ParseTuple(args, "O!", &R4Py_AgentType, &agent)) {
        return NULL;
    }

    R4Py_DiscretePoint* pt = ((R4Py_SharedGrid*)self)->grid->getLocation((R4Py_Agent*)agent);
    if (pt) {
        Py_INCREF(pt);
        return (PyObject*)pt;
    } else {
        Py_INCREF(Py_None);
        return Py_None;
    }
}

static PyObject* SharedGrid_move(PyObject* self, PyObject* args) {
    PyObject* agent, *pt;
    if (!PyArg_ParseTuple(args, "O!O!", &R4Py_AgentType, &agent, &DiscretePointType, &pt)) {
        return NULL;
    }

    try {
        R4Py_DiscretePoint* ret = ((R4Py_SharedGrid*)self)->grid->move((R4Py_Agent*)agent, (R4Py_DiscretePoint*)pt);
        if (ret) {
            Py_INCREF(ret);
            return (PyObject*)ret;
        } else {
            Py_INCREF(Py_None);
            return Py_None;
        }
    } catch (std::invalid_argument& ex) {
        PyErr_SetString(PyExc_RuntimeError, ex.what());
        return NULL;
    }
}

static PyObject* SharedGrid_synchMove(PyObject* self, PyObject* args) {
    PyArrayObject* obj;
    R4Py_Agent* agent;
    if (!PyArg_ParseTuple(args, "O!O!", &R4Py_AgentType, &agent, &PyArray_Type, &obj)) {
        return NULL;
    }
    R4Py_DiscretePoint* pt = (R4Py_DiscretePoint*)(&DiscretePointType)->tp_new(&DiscretePointType, 
        NULL, NULL);
    if (pt == NULL) {
        return NULL;
    }
    long* obj_data = (long*)PyArray_DATA(obj);
    long* pt_data = (long*)PyArray_DATA(pt->coords);
    pt_data[0] = obj_data[0];
    pt_data[1] = obj_data[1];
    pt_data[2] = obj_data[2];

    ((R4Py_SharedGrid*)self)->grid->move((R4Py_Agent*)agent, pt);

    Py_RETURN_NONE;
}

static PyObject* SharedGrid_getAgent(PyObject* self, PyObject* args) {
    PyObject* pt;
    if (!PyArg_ParseTuple(args, "O!", &DiscretePointType, &pt)) {
        return NULL;
    }

    R4Py_Agent* ret =  ((R4Py_SharedGrid*)self)->grid->getAgentAt((R4Py_DiscretePoint*)pt);
    if (ret) {
        // Is this necessary??
        Py_INCREF(ret);
        return (PyObject*)ret;
    } else {
        Py_INCREF(Py_None);
        return Py_None;
    }
}

static PyObject* SharedGrid_getAgents(PyObject* self, PyObject* args) {
    PyObject* pt;
    if (!PyArg_ParseTuple(args, "O!", &DiscretePointType, &pt)) {
        return NULL;
    }

    std::shared_ptr<std::list<R4Py_Agent*>> list = ((R4Py_SharedGrid*)self)->grid->getAgentsAt((R4Py_DiscretePoint*)pt);
    R4Py_AgentIter* agent_iter = (R4Py_AgentIter*)R4Py_AgentIterType.tp_new(&R4Py_AgentIterType, NULL, NULL);
    agent_iter->iter = new TAgentIter<std::list<R4Py_Agent*>>(list);
    return (PyObject*)agent_iter;
}

static PyObject* SharedGrid_getNumAgents(PyObject* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {(char*)"pt",(char*)"agent_type", NULL};

    PyObject* pt;
    int agent_type = -1;
    
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|i", kwlist, &DiscretePointType, &pt, &agent_type)) {
        return NULL;
    }

    std::shared_ptr<std::list<R4Py_Agent*>> list = ((R4Py_SharedGrid*)self)->grid->getAgentsAt((R4Py_DiscretePoint*)pt);
    if (agent_type == -1) {
        return PyLong_FromLong(list->size());
    } else {
        long count = 0;
        for (auto agent : (*list)) {
            if (agent->aid->type == agent_type) {
                ++count;
            }
        }
        return PyLong_FromLong(count);
    } 
}

static PyObject* SharedGrid_getOOBData(PyObject* self, PyObject* args) {
    std::shared_ptr<std::map<R4Py_AgentID*, PyObject*, agent_id_comp>> oob = ((R4Py_SharedGrid*)self)->grid->getOOBData();
    R4Py_PyObjectIter* obj_iter = (R4Py_PyObjectIter*)R4Py_PyObjectIterType.tp_new(&R4Py_PyObjectIterType, NULL, NULL);
    obj_iter->iter = new ValueIter<std::map<R4Py_AgentID*, PyObject*, agent_id_comp>>(oob);
    return (PyObject*)obj_iter;
}

static PyObject* SharedGrid_getBufferData(PyObject* self, PyObject* args) { 
    std::shared_ptr<std::vector<CTNeighbor>> nghs = ((R4Py_SharedGrid*)self)->grid->getNeighborData();
    R4Py_PyObjectIter* obj_iter = (R4Py_PyObjectIter*)R4Py_PyObjectIterType.tp_new(&R4Py_PyObjectIterType, NULL, NULL);
    obj_iter->iter = new SequenceIter<std::vector<CTNeighbor>, GetBufferInfo>(nghs);
    return (PyObject*)obj_iter; 
}

static PyObject* SharedGrid_clearOOBData(PyObject* self, PyObject* args) {
    ((R4Py_SharedGrid*)self)->grid->clearOOBData();
    Py_RETURN_NONE;
}

static PyObject* SharedGrid_getLocalBounds(PyObject* self, PyObject* args) {
    BoundingBox bounds = ((R4Py_SharedGrid*)self)->grid->getLocalBounds();
    PyObject* box_args = Py_BuildValue("(llllll)", bounds.xmin_, bounds.x_extent_, bounds.ymin_, bounds.y_extent_,
        bounds.zmin_, bounds.z_extent_);
    PyObject* pmod = PyImport_ImportModule("repast4py.space");
    PyObject* bbox_class = PyObject_GetAttrString(pmod, "BoundingBox");
    PyObject* box = PyObject_CallObject(bbox_class, box_args);

    Py_DECREF(box_args);
    Py_DECREF(pmod);
    Py_DECREF(bbox_class);

    return box;
}

static PyMemberDef SharedGrid_members[] = {
    {(char* const)"_cart_comm", T_OBJECT_EX, offsetof(R4Py_SharedGrid, cart_comm), READONLY, (char* const)"The cartesian communicator for this shared grid"},
    {NULL}
};

PyDoc_STRVAR(sgrd_add,
    "add(agent)\n\n"
    "Adds the specified agent to this shared grid projection.\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent to add."
);

PyDoc_STRVAR(sgrd_rm,
    "remove(agent)\n\n"
    
    "Removes the specified agent from this shared grid projection.\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent to remove."
);

PyDoc_STRVAR(sgrd_cnt,
    "contains(agent)\n\n"
    
    "Gets whether or not the **local** bounds of this shared grid projection contains the specified agent.\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent to check.\n\n"
    "Returns:\n"
    "    bool: true if this shared grid contains the specified agent, otherwise false"
);

PyDoc_STRVAR(sgrd_move,
    "move(agent, pt)\n\n"
    
    "Moves the specified agent to the specified location, returning the moved to location.\n\n"
    "If the agent does not move beyond the shared grid's global bounds, then the returned location will be "
    "be the same as the argument location. If the agent does move out of the global bounds, then the location "
    "is determined by the shared grid's border semantics (e.g., a location on the border if using "
    ":attr:`BorderType.Sticky` borders.\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent to move.\n"
    "    pt(repast4py.space.DiscretePoint): the location to move to.\n\n"
    "Returns:\n"
    "    repast4py.space.DiscretePoint: the location the agent has moved to or None if the agent cannot move to the specified location (e.g., if "
    "    the occupancy type is :attr:`OccupancyType.Single` and the location is occupied.)"
);

PyDoc_STRVAR(sgrd_location,
    "get_location(agent)\n\n"
    
    "Gets the location of the specified agent\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent whose location we want to get.\n\n"
    "Returns:\n"
    "    repast4py.space.DiscretePoint: the location of the specified agent or None if the agent is not in the space."
);

PyDoc_STRVAR(sgrd_geta,
    "get_agent(pt)\n\n"
    
    "Gets the agent at the specified location. If more than one agent exists at "
    "the specified location, the first agent to have moved to that location from among those currently at that location "
    "will be returned.\n\n"
    "Args:\n"
    "    pt(repast4py.space.DiscretePoint): the location to get the agent at.\n\n"
    "Returns:\n"
    "    repast4py.core.Agent: the agent at that location, or None if the location is empty"
);

PyDoc_STRVAR(sgrd_getas,
    "get_agents(pt)\n\n"
    
    "Gets an iterator over all the agents at the specified location.\n\n"
    "Args:\n"
    "    pt(repast4py.space.DiscretePoint): the location to get the agents at.\n\n"
    "Returns:\n"
    "    iterator: an iterator over all the agents at the specified location."
);

PyDoc_STRVAR(sgrd_getnas,
    "get_num_agents(pt, agent_type=None)\n\n"
    
    "Gets number of agents at the specified location, optionally fitered by agent type.\n\n"
    "Args:\n"
    "    pt(repast4py.space.DiscretePoint): the location to get the agents at.\n"
    "    agent_type(int): the type id of the agents to get the number of.\n\n"
    "Returns:\n"
    "    int: the number of agents at the specified location."
);

PyDoc_STRVAR(sgrd_lb,
    "get_local_bounds()\n\n"
    
    "Gets the local bounds of this shared grid.\n\n"
    "The local bounds are the bounds of this shared grid on the current rank. For example, if "
    "the global bounds are 100 in the x dimension and 100 in the y dimension, and there are 4 ranks, "
    "then the local bounds will be some quadrant of those global bounds 0 - 50 x 0 - 50 for example.\n\n"
    "Returns:\n"
    "    repast4py.geometry.BoundingBox: the local bounds as a BoundingBox."
);

PyDoc_STRVAR(sgrd_oob,
    "_get_oob()\n\n"
    
    "Gets the synchronization data for the agents that have moved out of this shared grid's local bounds.\n\n"
    "The out of bounds data is used for synchronizing the global shared grid state, and this "
    "method should not ordinarly be called by users. Out of bounds data is generated when an agent moves "
    "and its new location is within the bounds another rank.\n\n"
    "Returns:\n"
    "    iterable: an iterable of tuples where each tuple contains the out of bounds data for an agent. "
    "Each tuple consists of the agents uid tuple, the rank containing the agent's new location, and the new "
    "location of the agents as a repast4py.space.DiscretePoint."
);

PyDoc_STRVAR(sgrd_coob,
    "_clear_oob()\n\n"
    
    "Clears the collection of out of bounds data.\n\n"
    "The out of bounds data is used for synchronizing the global shared grid state, and this "
    "method should not ordinarly be called by users. Out of bounds data is generated when an agent moves "
    "and its new location is within the bounds another rank.\n"
);

PyDoc_STRVAR(sgrd_mooba,
    "_move_oob_agent(agent, pt)\n\n"
    
    "Moves the specified agent to the specified location as part of synchronizing the global shared grid state.\n\n"
    "Args:\n"
    "    agent(repast4py.core.agent): the agent to move.\n"
    "    pt(numpy.array): the location to move to as a 3 dimensional numpy array"
);

PyDoc_STRVAR(sgrd_gbd,
    "_get_buffer_data()\n\n"
    
    "Gets the buffer data for this shared grid.\n\n"
    "Each subsection of the shared grid has buffers whose contents are ghosted from a neighboring rank. This "
    "returns data describing the dimensions and rank of these buffers.\n\n"
    "Returns:\n"
    "    iterable: an iterable of tuples where each tuple describes the owner rank and size of a buffer: "
    "(rank, (xmin, xmax, ymin, ymax, zmin, zmax))"
);

static PyMethodDef SharedGrid_methods[] = {
    {"add", SharedGrid_add, METH_VARARGS, sgrd_add},
    {"remove", SharedGrid_remove, METH_VARARGS, sgrd_rm},
    {"move", SharedGrid_move, METH_VARARGS, sgrd_move},
    {"contains", SharedGrid_contains, METH_VARARGS, sgrd_cnt},
    {"get_location", SharedGrid_getLocation, METH_VARARGS, sgrd_location},
    {"get_agent", SharedGrid_getAgent, METH_VARARGS, sgrd_geta},
    {"get_agents", SharedGrid_getAgents, METH_VARARGS, sgrd_getas},
    {"get_num_agents", (PyCFunction) SharedGrid_getNumAgents, METH_VARARGS | METH_KEYWORDS, sgrd_getnas},
    {"_get_oob", SharedGrid_getOOBData, METH_VARARGS, sgrd_oob},
    {"_clear_oob", SharedGrid_clearOOBData, METH_VARARGS, sgrd_coob},
    {"get_local_bounds", SharedGrid_getLocalBounds, METH_VARARGS, sgrd_lb},
    {"_move_oob_agent", SharedGrid_synchMove, METH_VARARGS, sgrd_mooba},
    {"_get_buffer_data", SharedGrid_getBufferData, METH_VARARGS, sgrd_gbd},
    {NULL, NULL, 0, NULL}
};

static PyObject* SharedGrid_getName(PyObject* self, PyObject* args) {
    return PyUnicode_FromString(((R4Py_SharedGrid*)self)->grid->name().c_str());
}

PyDoc_STRVAR(sgrd_name,
    "str: Gets the name of this shared grid.");

static PyGetSetDef SharedGrid_get_setters[] = {
    {(char*)"name", (getter)SharedGrid_getName, NULL, sgrd_name, NULL},
    {NULL}
};

PyDoc_STRVAR(sgrd_sgrd,
    "SharedGrid(name, bounds, borders, occupancy, buffer_size, comm)\n\n"
    
    "An N-dimensional cartesian discrete space where agents can occupy locations defined by "
    "a discretete integer coordinate.\n\n"
    "The grid is shared over all the ranks in the specified communicator by sub-dividing the global bounds into "
    "some number of smaller grids, one for each rank. For example, given a global grid size of 100 x 25 and "
    "2 ranks, the global grid will be split along the x dimension such that the SharedGrid in the first MPI rank "
    "covers 0-50 x 0-25 and the second rank 50-100 x 0-25. "
    "Each rank's SharedGrid contains buffers of the specified size that duplicate or \"ghosts\" an adjacent "
    "area of the neighboring rank's SharedGrid. In the above example, the rank 1 grid buffers the area from "
    "50-52 x 0-25 in rank 2, and rank 2 buffers 48-50 x 0-25 in rank 1. **Be sure to specify a buffer size appropriate "
    "to any agent behavior.** For example, if an agent can \"see\" 3 units away and take some action based on what it "
    "perceives, then the buffer size should be at least 3, insuring that an agent can properly see beyond the borders of "
    "its own local SharedGrid. When an agent moves beyond the borders of its current SharedGrid, it will be transferred "
    "from its current rank, and to the rank containing the section of the global grid that it has moved into.\n\n"
    "Args:\n"
    "   name (str): the name of the grid.\n"
    "   bounds (repast4py.geometry.BoundingBox): the global dimensions of the grid.\n"
    "   borders (repast4py.space.BorderType): the border semantics - :attr:`BorderType.Sticky` or :attr:`BorderType.Periodic`.\n"
    "   occupancy (repast4py.space.OccupancyType): the type of occupancy in each cell - :attr:`OccupancyType.Multiple` or :attr:`OccupancyType.Single`.\n"
    "   buffer_size (int): the size of this SharedGrid's buffered area. This single value is used for all dimensions.\n"
    "   comm (mpi4py.MPI.Intracomm): the communicator containing all the ranks over which this SharedGrid is shared."
);

static PyTypeObject R4Py_SharedGridType = {
    PyVarObject_HEAD_INIT(NULL, 0) 
    "_space.SharedGrid",                          /* tp_name */
    sizeof(R4Py_SharedGrid),                      /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)SharedGrid_dealloc,                                         /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_reserved */
    0,                                        /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash  */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    sgrd_sgrd,                         /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    SharedGrid_methods,                                      /* tp_methods */
    SharedGrid_members,                                      /* tp_members */
    SharedGrid_get_setters,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)SharedGrid_init,                                         /* tp_init */
    0,                                        /* tp_alloc */
    SharedGrid_new                             /* tp_new */
};

////////////////////////////// shared grid end /////////////////////////

///////////////////////// ContinousSpace ///////////////////////////////
static void CSpace_dealloc(R4Py_CSpace* self) {
    delete self->space;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* CSpace_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    R4Py_CSpace* self = (R4Py_CSpace*)type->tp_alloc(type, 0);
    if (self != NULL) {
        // maybe I should create it here, rather than in init??
        self->space = nullptr;
    }
    return (PyObject*)self;
}

static int CSpace_init(R4Py_CSpace* self, PyObject* args, PyObject* kwds) {
    // bounds=box, border=BorderType.Sticky, occupancy=OccupancyType.Multiple
    static char* kwlist[] = {(char*)"name",(char*)"bounds", (char*)"borders",
        (char*)"occupancy", (char*)"tree_threshold", NULL};

    const char* name;
    PyObject* bounds;
    int border_type, occ_type, tree_threshold;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO!iii", kwlist, &name, &PyTuple_Type, &bounds,
        &border_type, &occ_type, &tree_threshold)) 
    {
        return -1;
    }

    long xmin, width;
    long ymin, height;
    long zmin, depth;

    if (!PyArg_ParseTuple(bounds, "llllll", &xmin, &width, &ymin, &height, &zmin, &depth)) {
        return -1;
    }

    BoundingBox box(xmin, width, ymin, height, zmin, depth);
    if (border_type == STICKY_BRDR) {
        if (occ_type == MULT_OT) {
            self->space = new CSpace<MOSCSpace>(name, box, tree_threshold);
        } else if (occ_type == SINGLE_OT) {
            self->space = new CSpace<SOSCSpace>(name, box, tree_threshold);
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Invalid occupancy type");
            return -1;
        }
    } else if (border_type == PERIODIC_BRDR) {
        if (occ_type == MULT_OT) {
            self->space = new CSpace<MOPCSpace>(name, box, tree_threshold);
        } else if (occ_type == SINGLE_OT) {
            self->space = new CSpace<SOPCSpace>(name, box, tree_threshold);
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Invalid occupancy type");
            return -1;
        }

    } else {
        PyErr_SetString(PyExc_RuntimeError, "Invalid border type");
        return -1;
    }

    if (!self->space) {
        PyErr_SetString(PyExc_RuntimeError, "Error creating native code space");
        return -1;
    }
    return 0;
}

static PyObject* CSpace_add(PyObject* self, PyObject* args) {
    PyObject* agent;
    if (!PyArg_ParseTuple(args, "O!", &R4Py_AgentType, &agent)) {
        return NULL;
    }
    bool ret_val = ((R4Py_CSpace*)self)->space->add((R4Py_Agent*)agent);
    return PyBool_FromLong(static_cast<long>(ret_val));
}

static PyObject* CSpace_remove(PyObject* self, PyObject* args) {
    PyObject* agent;
    if (!PyArg_ParseTuple(args, "O!", &R4Py_AgentType, &agent)) {
        return NULL;
    }
    bool ret_val = ((R4Py_CSpace*)self)->space->remove((R4Py_Agent*)agent);
    return PyBool_FromLong(static_cast<long>(ret_val));
}

static PyObject* CSpace_contains(PyObject* self, PyObject* args) {
    PyObject* agent;
    if (!PyArg_ParseTuple(args, "O!", &R4Py_AgentType, &agent)) {
        return NULL;
    }
    bool ret_val = ((R4Py_CSpace*)self)->space->contains((R4Py_Agent*)agent);
    return PyBool_FromLong(static_cast<long>(ret_val));
}

static PyObject* CSpace_getLocation(PyObject* self, PyObject* args) {
    PyObject* agent;
    if (!PyArg_ParseTuple(args, "O!", &R4Py_AgentType, &agent)) {
        return NULL;
    }

    R4Py_ContinuousPoint* pt = ((R4Py_CSpace*)self)->space->getLocation((R4Py_Agent*)agent);
    if (pt) {
        Py_INCREF(pt);
        return (PyObject*)pt;
    } else {
        Py_INCREF(Py_None);
        return Py_None;
    }
}

static PyObject* CSpace_move(PyObject* self, PyObject* args) {
    PyObject* agent, *pt;
    if (!PyArg_ParseTuple(args, "O!O!", &R4Py_AgentType, &agent, &ContinuousPointType, &pt)) {
        return NULL;
    }

    try {
        R4Py_ContinuousPoint* ret = ((R4Py_CSpace*)self)->space->move((R4Py_Agent*)agent, (R4Py_ContinuousPoint*)pt);
        if (ret) {
            Py_INCREF(ret);
            return (PyObject*)ret;
        } else {
            Py_INCREF(Py_None);
            return Py_None;
        }
    } catch (std::invalid_argument& ex) {
        PyErr_SetString(PyExc_RuntimeError, ex.what());
        return NULL;
    }
}

static PyObject* CSpace_getAgent(PyObject* self, PyObject* args) {
    PyObject* pt;
    if (!PyArg_ParseTuple(args, "O!", &ContinuousPointType, &pt)) {
        return NULL;
    }

    R4Py_Agent* ret =  ((R4Py_CSpace*)self)->space->getAgentAt((R4Py_ContinuousPoint*)pt);
    if (ret) {
        // Is this necessary??
        Py_INCREF(ret);
        return (PyObject*)ret;
    } else {
        Py_INCREF(Py_None);
        return Py_None;
    }
}

static PyObject* CSpace_getAgents(PyObject* self, PyObject* args) {
    PyObject* pt;
    if (!PyArg_ParseTuple(args, "O!", &ContinuousPointType, &pt)) {
        return NULL;
    }

    std::shared_ptr<std::list<R4Py_Agent*>> list = ((R4Py_CSpace*)self)->space->getAgentsAt((R4Py_ContinuousPoint*)pt);
    R4Py_AgentIter* agent_iter = (R4Py_AgentIter*)R4Py_AgentIterType.tp_new(&R4Py_AgentIterType, NULL, NULL);
    agent_iter->iter = new TAgentIter<std::list<R4Py_Agent*>>(list);
    return (PyObject*)agent_iter;
}

static PyObject* CSpace_getAgentsWithin(PyObject* self, PyObject* args) {
    PyObject* bounds;
    if (!PyArg_ParseTuple(args, "O!", &PyTuple_Type, &bounds)) {
        return NULL;
    }

    long xmin, width;
    long ymin, height;
    long zmin, depth;

    if (!PyArg_ParseTuple(bounds, "llllll", &xmin, &width, &ymin, &height, &zmin, &depth)) {
        return NULL;
    }

    BoundingBox box(xmin, width, ymin, height, zmin, depth);
    std::shared_ptr<std::vector<R4Py_Agent*>> agents = std::make_shared<std::vector<R4Py_Agent*>>();
    ((R4Py_CSpace*)self)->space->getAgentsWithin(box, agents);
    R4Py_AgentIter* agent_iter = (R4Py_AgentIter*)R4Py_AgentIterType.tp_new(&R4Py_AgentIterType, NULL, NULL);
    agent_iter->iter = new TAgentIter<std::vector<R4Py_Agent*>>(agents);
    return (PyObject*)agent_iter;
}

static PyObject* CSpace_getName(PyObject* self, PyObject* args) {
    return PyUnicode_FromString(((R4Py_CSpace*)self)->space->name().c_str());
}

PyDoc_STRVAR(cspace_name,
    "str: Gets the name of this continuous space.");

static PyGetSetDef CSpace_get_setters[] = {
    {(char*)"name", (getter)CSpace_getName, NULL, cspace_name, NULL},
    {NULL}
};

PyDoc_STRVAR(cspace_add,
    "add(agent)\n\n"
    
    "Adds the specified agent to this continuous space projection.\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent to add."
);

PyDoc_STRVAR(cspace_rm,
    "remove(agent)\n\n"
    
    "Removes the specified agent from this continuous space projection.\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent to remove."
);

PyDoc_STRVAR(cspace_cnt,
    "contains(agent)\n\n"
    
    "Gets whether or not this continuous space projection contains the specified agent.\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent to check.\n\n"
    "Returns:\n"
    "    bool: true if this continuous space contains the specified agent, otherwise false"
);

PyDoc_STRVAR(cspace_move,
    "move(agent, pt)\n\n"
    
    "Moves the specified agent to the specified location, returning the moved to location.\n\n"
    "If the agent does not move beyond the continuous space's bounds, then the returned location will be "
    "be the same as the argument location. If the agent does move out of bounds, then the location "
    "is determined by the continuous space's border's semantics (e.g., a location on the border if using :attr:`BorderType.Sticky` borders.\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent to move.\n"
    "    pt(repast4py.space.ContinuousPoint): the location to move to.\n\n"
    "Returns:\n"
    "    repast4py.space.ContinuousPoint: the location the agent has moved to or None if the agent cannot move to the specified location (e.g., if "
    "    the occupancy type is :attr:`OccupancyType.Single` and the location is occupied.)"
);

PyDoc_STRVAR(cspace_location,
    "get_location(agent)\n\n"
    
    "Gets the location of the specified agent\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent whose location we want to get.\n\n"
    "Returns:\n"
    "    repast4py.space.ContinuousPoint: the location of the specified agent or None if the agent is not in the space."
);

PyDoc_STRVAR(cspace_geta,
    "get_agent(pt)\n\n"
    
    "Gets the agent at the specified location. If more than one agent exists at "
    "the specified location, the first agent to have moved to that location from among those currently at that location "
    "will be returned.\n\n"
    "Args:\n"
    "    pt(repast4py.space.ContinuousPoint): the location to get the agent at.\n\n"
    "Returns:\n"
    "    repast4py.core.Agent: the agent at that location, or None if the location is empty"
);

PyDoc_STRVAR(cspace_getas,
    "get_agents(pt)\n\n"
    
    "Gets an iterator over all the agents at the specified location.\n\n"
    "Args:\n"
    "    pt(repast4py.space.ContinuousPoint): the location to get the agents at.\n\n"
    "Returns:\n"
    "    iterator: an iterator over all the agents at the specified location."
);

PyDoc_STRVAR(cspace_within,
    "get_agents_within(bbox)\n\n"
    
    "Gets an iterator over all the agents within the specified bounding box.\n\n"
    "Args:\n"
    "    box(repast4py.geometry.BoundingBox): the bounding box to get the agents within.\n\n"
    "Returns:\n"
    "    iterator: an iterator over all the agents within the specified bounding box."
);

static PyMethodDef CSpace_methods[] = {
    {"add", CSpace_add, METH_VARARGS, cspace_add},
    {"remove", CSpace_remove, METH_VARARGS, cspace_rm},
    {"move", CSpace_move, METH_VARARGS, cspace_move},
    {"contains", CSpace_contains, METH_VARARGS, cspace_cnt},
    {"get_location", CSpace_getLocation, METH_VARARGS, cspace_location},
    {"get_agent", CSpace_getAgent, METH_VARARGS,  cspace_geta},
    {"get_agents", CSpace_getAgents, METH_VARARGS, cspace_getas},
    {"get_agents_within", CSpace_getAgentsWithin, METH_VARARGS, cspace_within},
    {NULL, NULL, 0, NULL}
};

PyDoc_STRVAR(cspace_cspace,
    "ContinuousSpace(name, bounds, borders, occupancy, tree_threshold)\n\n"
    
    "An N-dimensional cartesian continuous space where agents can occupy locations defined by "
    "a continuous floating point coordinate.\n\n"
    "The ContinuousSpace uses a `tree <https://en.wikipedia.org/wiki/Quadtree>`_ (quad or oct depending on the number of "
    "dimensions) to optimize spatial queries. The tree can be tuned using the tree threshold parameter.\n\n"
    "Args:\n"
    "   name (str): the name of the grid.\n"
    "   bounds (repast4py.geometry.BoundingBox): the dimensions of the grid.\n"
    "   borders (repast4py.space.BorderType): the border semantics - :attr:`BorderType.Sticky` or :attr:`BorderType.Periodic`\n"
    "   occupancy (repast4py.space.OccupancyType): the type of occupancy in each cell - :attr:`OccupancyType.Multiple` or :attr:`OccupancyType.Single`\n"
    "   tree_threshold (int): the space's tree cell maximum capacity. When this capacity is reached, the cell splits."
);

static PyTypeObject R4Py_CSpaceType = {
    PyVarObject_HEAD_INIT(NULL, 0) 
    "_space.ContinuousSpace",                          /* tp_name */
    sizeof(R4Py_CSpace),                      /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)CSpace_dealloc,                                         /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_reserved */
    0,                                        /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash  */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    cspace_cspace,                         /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    CSpace_methods,                                      /* tp_methods */
    0,                                      /* tp_members */
    CSpace_get_setters,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)CSpace_init,                                         /* tp_init */
    0,                                        /* tp_alloc */
    CSpace_new                             /* tp_new */
};

////////////////////////// ContinousSpace End //////////////////////////////


////////////////// SharedContinuousSpace ////////////////////////////

static void SharedCSpace_dealloc(R4Py_SharedCSpace* self) {
    delete self->space;
    Py_XDECREF(self->cart_comm);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* SharedCSpace_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    R4Py_SharedCSpace* self = (R4Py_SharedCSpace*)type->tp_alloc(type, 0);
    if (self != NULL) {
        // maybe I should create it here, rather than in init??
        self->space = nullptr;
        self->cart_comm = nullptr;
    }
    return (PyObject*)self;
}

static int SharedCSpace_init(R4Py_SharedCSpace* self, PyObject* args, PyObject* kwds) {
    // bounds=box, border=BorderType.Sticky, occupancy=OccupancyType.Multiple
    // {
    //      volatile int i = 0;
    //      char hostname[256];
    //      gethostname(hostname, sizeof(hostname));
    //      printf("PID %d on %s ready for attach\n", getpid(), hostname);
    //      fflush(stdout);
    //      while (0 == i)
    //          sleep(5);
    // }

    static char* kwlist[] = {(char*)"name",(char*)"bounds", (char*)"borders",
        (char*)"occupancy", (char*)"buffer_size", (char*)"comm", (char*)"tree_threshold", NULL};

    const char* name;
    PyObject* bounds;
    int border_type, occ_type, buffer_size, tree_threshold;
    PyObject* py_comm;

    //
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO!iiiO!i", kwlist, &name, &PyTuple_Type, &bounds,
        &border_type, &occ_type, &buffer_size, &PyMPIComm_Type, &py_comm, &tree_threshold)) 
    {
        return -1;
    }

    long xmin, x_extent;
    long ymin, y_extent;
    long zmin, z_extent;

    if (!PyArg_ParseTuple(bounds, "llllll", &xmin, &x_extent, &ymin, &y_extent, &zmin, &z_extent)) {
        return -1;
    }

    // Because we are holding a reference to the communicator
    // I think this is necessary
    Py_INCREF(py_comm);
    MPI_Comm* comm_p = PyMPIComm_Get(py_comm);

    BoundingBox box(xmin, x_extent, ymin, y_extent, zmin, z_extent);
    if (border_type == STICKY_BRDR) {
        if (occ_type == MULT_OT) {
            self->space = new SharedContinuousSpace<DistributedCartesianSpace<MOSCSpace>>(name, box, buffer_size, 
            *comm_p, tree_threshold);
        } else if (occ_type == SINGLE_OT) {
            self->space = new SharedContinuousSpace<DistributedCartesianSpace<SOSCSpace>>(name, box, buffer_size, 
            *comm_p, tree_threshold);
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Invalid occupancy type");
            return -1;
        }
    } else if (border_type == PERIODIC_BRDR) {
        if (occ_type == MULT_OT) {
            self->space = new SharedContinuousSpace<DistributedCartesianSpace<MOPCSpace>>(name, box, buffer_size, 
            *comm_p, tree_threshold);
        } else if (occ_type == SINGLE_OT) {
            self->space = new SharedContinuousSpace<DistributedCartesianSpace<SOPCSpace>>(name, box, buffer_size, 
            *comm_p, tree_threshold);
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Invalid occupancy type");
            return -1;
        }

   
    } else {
        PyErr_SetString(PyExc_RuntimeError, "Invalid border type");
        return -1;
    }

    if (!self->space) {
        PyErr_SetString(PyExc_RuntimeError, "Error creating native code shared grid");
        return -1;
    }

    self->cart_comm = PyMPIComm_New(self->space->getCartesianCommunicator());
    if (self->cart_comm == NULL) {
        return -1;
    }

    return 0;
}

static PyObject* SharedCSpace_add(PyObject* self, PyObject* args) {
    PyObject* agent;
    if (!PyArg_ParseTuple(args, "O!", &R4Py_AgentType, &agent)) {
        return NULL;
    }
    bool ret_val = ((R4Py_SharedCSpace*)self)->space->add((R4Py_Agent*)agent);
    return PyBool_FromLong(static_cast<long>(ret_val));
}

static PyObject* SharedCSpace_remove(PyObject* self, PyObject* args) {
    PyObject* agent;
    if (!PyArg_ParseTuple(args, "O!", &R4Py_AgentType, &agent)) {
        return NULL;
    }

    //  {
    //      volatile int i = 0;
    //      char hostname[256];
    //      gethostname(hostname, sizeof(hostname));
    //      printf("PID %d on %s ready for attach\n", getpid(), hostname);
    //      fflush(stdout);
    //      while (0 == i)
    //          sleep(5);
    // }

    bool ret_val = ((R4Py_SharedCSpace*)self)->space->remove((R4Py_Agent*)agent);
    return PyBool_FromLong(static_cast<long>(ret_val));
}

static PyObject* SharedCSpace_contains(PyObject* self, PyObject* args) {
    PyObject* agent;
    if (!PyArg_ParseTuple(args, "O!", &R4Py_AgentType, &agent)) {
        return NULL;
    }

    bool ret_val = ((R4Py_SharedCSpace*)self)->space->contains((R4Py_Agent*)agent);
    return PyBool_FromLong(static_cast<long>(ret_val));
}

static PyObject* SharedCSpace_getLocation(PyObject* self, PyObject* args) {
    PyObject* agent;
    if (!PyArg_ParseTuple(args, "O!", &R4Py_AgentType, &agent)) {
        return NULL;
    }

    R4Py_ContinuousPoint* pt = ((R4Py_SharedCSpace*)self)->space->getLocation((R4Py_Agent*)agent);
    if (pt) {
        Py_INCREF(pt);
        return (PyObject*)pt;
    } else {
        Py_INCREF(Py_None);
        return Py_None;
    }
}

static PyObject* SharedCSpace_move(PyObject* self, PyObject* args) {
    PyObject* agent, *pt;
    if (!PyArg_ParseTuple(args, "O!O!", &R4Py_AgentType, &agent, &ContinuousPointType, &pt)) {
        return NULL;
    }

    try {
        R4Py_ContinuousPoint* ret = ((R4Py_SharedCSpace*)self)->space->move((R4Py_Agent*)agent, (R4Py_ContinuousPoint*)pt);
        if (ret) {
            Py_INCREF(ret);
            return (PyObject*)ret;
        } else {
            Py_INCREF(Py_None);
            return Py_None;
        }
    } catch (std::invalid_argument& ex) {
        PyErr_SetString(PyExc_RuntimeError, ex.what());
        return NULL;
    }
}

static PyObject* SharedCSpace_synchMove(PyObject* self, PyObject* args) {
    PyArrayObject* obj;
    R4Py_Agent* agent;
    if (!PyArg_ParseTuple(args, "O!O!", &R4Py_AgentType, &agent, &PyArray_Type, &obj)) {
        return NULL;
    }
    R4Py_ContinuousPoint* pt = (R4Py_ContinuousPoint*)(&ContinuousPointType)->tp_new(&ContinuousPointType, 
        NULL, NULL);
    if (pt == NULL) {
        return NULL;
    }
    double* obj_data = (double*)PyArray_DATA(obj);
    double* pt_data = (double*)PyArray_DATA(pt->coords);
    pt_data[0] = obj_data[0];
    pt_data[1] = obj_data[1];
    pt_data[2] = obj_data[2];

    ((R4Py_SharedCSpace*)self)->space->move((R4Py_Agent*)agent, pt);

    Py_RETURN_NONE;
}

static PyObject* SharedCSpace_getAgent(PyObject* self, PyObject* args) {
    PyObject* pt;
    if (!PyArg_ParseTuple(args, "O!", &ContinuousPointType, &pt)) {
        return NULL;
    }

    R4Py_Agent* ret =  ((R4Py_SharedCSpace*)self)->space->getAgentAt((R4Py_ContinuousPoint*)pt);
    if (ret) {
        // Is this necessary??
        Py_INCREF(ret);
        return (PyObject*)ret;
    } else {
        Py_INCREF(Py_None);
        return Py_None;
    }
}

static PyObject* SharedCSpace_getAgents(PyObject* self, PyObject* args) {
    PyObject* pt;
    if (!PyArg_ParseTuple(args, "O!", &ContinuousPointType, &pt)) {
        return NULL;
    }

    std::shared_ptr<std::list<R4Py_Agent*>> list = ((R4Py_SharedCSpace*)self)->space->getAgentsAt((R4Py_ContinuousPoint*)pt);
    R4Py_AgentIter* agent_iter = (R4Py_AgentIter*)R4Py_AgentIterType.tp_new(&R4Py_AgentIterType, NULL, NULL);
    agent_iter->iter = new TAgentIter<std::list<R4Py_Agent*>>(list);
    return (PyObject*)agent_iter;
}

static PyObject* SharedCSpace_getNumAgents(PyObject* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {(char*)"pt",(char*)"agent_type", NULL};

    PyObject* pt;
    int agent_type = -1;
    
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|i", kwlist, &ContinuousPointType, &pt, &agent_type)) {
        return NULL;
    }

    std::shared_ptr<std::list<R4Py_Agent*>> list = ((R4Py_SharedCSpace*)self)->space->getAgentsAt((R4Py_ContinuousPoint*)pt);
    if (agent_type == -1) {
        return PyLong_FromLong(list->size());
    } else {
        long count = 0;
        for (auto agent : (*list)) {
            if (agent->aid->type == agent_type) {
                ++count;
            }
        }
        return PyLong_FromLong(count);
    } 
}

static PyObject* SharedCSpace_getOOBData(PyObject* self, PyObject* args) {
    std::shared_ptr<std::map<R4Py_AgentID*, PyObject*, agent_id_comp>> oob = ((R4Py_SharedCSpace*)self)->space->getOOBData();
    R4Py_PyObjectIter* obj_iter = (R4Py_PyObjectIter*)R4Py_PyObjectIterType.tp_new(&R4Py_PyObjectIterType, NULL, NULL);
    obj_iter->iter = new ValueIter<std::map<R4Py_AgentID*, PyObject*, agent_id_comp>>(oob);
    return (PyObject*)obj_iter;
}

static PyObject* SharedCSpace_getBufferData(PyObject* self, PyObject* args) { 
    std::shared_ptr<std::vector<CTNeighbor>> nghs = ((R4Py_SharedCSpace*)self)->space->getNeighborData();
    R4Py_PyObjectIter* obj_iter = (R4Py_PyObjectIter*)R4Py_PyObjectIterType.tp_new(&R4Py_PyObjectIterType, NULL, NULL);
    obj_iter->iter = new SequenceIter<std::vector<CTNeighbor>, GetBufferInfo>(nghs);
    return (PyObject*)obj_iter; 
}

static PyObject* SharedCSpace_clearOOBData(PyObject* self, PyObject* args) {
    ((R4Py_SharedCSpace*)self)->space->clearOOBData();
    Py_RETURN_NONE;
}

static PyObject* SharedCSpace_getLocalBounds(PyObject* self, PyObject* args) {
    BoundingBox bounds = ((R4Py_SharedCSpace*)self)->space->getLocalBounds();
    PyObject* box_args = Py_BuildValue("(llllll)", bounds.xmin_, bounds.x_extent_, bounds.ymin_, bounds.y_extent_,
        bounds.zmin_, bounds.z_extent_);
    PyObject* pmod = PyImport_ImportModule("repast4py.space");
    PyObject* bbox_class = PyObject_GetAttrString(pmod, "BoundingBox");
    PyObject* box = PyObject_CallObject(bbox_class, box_args);

    Py_DECREF(box_args);
    Py_DECREF(pmod);
    Py_DECREF(bbox_class);

    return box;
}

static PyObject* SharedCSpace_getAgentsWithin(PyObject* self, PyObject* args) {
    PyObject* bounds;
    if (!PyArg_ParseTuple(args, "O!", &PyTuple_Type, &bounds)) {
        return NULL;
    }

    long xmin, width;
    long ymin, height;
    long zmin, depth;

    if (!PyArg_ParseTuple(bounds, "llllll", &xmin, &width, &ymin, &height, &zmin, &depth)) {
        return NULL;
    }

    BoundingBox box(xmin, width, ymin, height, zmin, depth);
    std::shared_ptr<std::vector<R4Py_Agent*>> agents = std::make_shared<std::vector<R4Py_Agent*>>();
    ((R4Py_SharedCSpace*)self)->space->getAgentsWithin(box, agents);
    R4Py_AgentIter* agent_iter = (R4Py_AgentIter*)R4Py_AgentIterType.tp_new(&R4Py_AgentIterType, NULL, NULL);
    agent_iter->iter = new TAgentIter<std::vector<R4Py_Agent*>>(agents);
    return (PyObject*)agent_iter;
}

static PyObject* SharedCSpace_getName(PyObject* self, PyObject* args) {
    return PyUnicode_FromString(((R4Py_SharedCSpace*)self)->space->name().c_str());
}


PyDoc_STRVAR(scspace_name,
    "str: Gets the name of this shared continuous space.");

static PyGetSetDef SharedCSpace_get_setters[] = {
    {(char*)"name", (getter)SharedCSpace_getName, NULL, scspace_name, NULL},
    {NULL}
};

static PyMemberDef SharedCSpace_members[] = {
    {(char* const)"_cart_comm", T_OBJECT_EX, offsetof(R4Py_SharedCSpace, cart_comm), READONLY, (char* const)"The cartesian communicator for this shared grid"},
    {NULL}
};

PyDoc_STRVAR(scspace_add,
    "add(agent)\n\n"
    
    "Adds the specified agent to this shared continuous space projection.\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent to add."
);

PyDoc_STRVAR(scspace_rm,
    "remove(agent)\n\n"
    
    "Removes the specified agent from this shared continuous space projection.\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent to remove."
);

PyDoc_STRVAR(scspace_cnt,
    "contains(agent)\n\n"
    
    "Gets whether or not the **local** bounds of this shared continuous space projection contains the specified agent.\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent to check.\n\n"
    "Returns:\n"
    "    bool: true if this shared continuous space contains the specified agent, otherwise false"
);

PyDoc_STRVAR(scspace_move,
    "move(agent, pt)\n\n"
    
    "Moves the specified agent to the specified location, returning the moved to location.\n\n"
    "If the agent does not move beyond the shared continuous space's global bounds, then the returned location will be "
    "be the same as the argument location. If the agent does move out of the global bounds, then the location "
    "is determined by the shared continuous space's border semantics (e.g., a location on the border if using "
    ":attr:`BorderType.Sticky` borders).\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent to move.\n"
    "    pt(repast4py.space.ContinuousPoint): the location to move to.\n\n"
    "Returns:\n"
    "    repast4py.space.ContinuousPoint: the location the agent has moved to or None if the agent cannot move to the specified location (e.g., if "
    "    the occupancy type is :attr:`OccupancyType.Single` and the location is occupied.)"
);

PyDoc_STRVAR(scspace_location,
    "get_location(agent)\n\n"
    
    "Gets the location of the specified agent\n\n"
    "Args:\n"
    "    agent(repast4py.core.Agent): the agent whose location we want to get.\n\n"
    "Returns:\n"
    "    repast4py.space.ContinuousPoint: the location of the specified agent or None if the agent is not in the space."
);

PyDoc_STRVAR(scspace_geta,
    "get_agent(pt)\n\n"
    
    "Gets the agent at the specified location. If more than one agent exists at "
    "the specified location, the first agent to have moved to that location from among those currently at that location "
    "will be returned.\n\n"
    "Args:\n"
    "    pt(repast4py.space.ContinuousPoint): the location to get the agent at.\n\n"
    "Returns:\n"
    "    repast4py.core.Agent: the agent at that location, or None if the location is empty"
);

PyDoc_STRVAR(scspace_getas,
    "get_agents(pt)\n\n"
    
    "Gets an iterator over all the agents at the specified location.\n\n"
    "Args:\n"
    "    pt(repast4py.space.ContinuousPoint): the location to get the agents at.\n\n"
    "Returns:\n"
    "    iterator: an iterator over all the agents at the specified location."
);

PyDoc_STRVAR(scspace_getnas,
    "get_num_agents(pt, agent_type=None)\n\n"
    
    "Gets number of agents at the specified location, optionally fitered by agent type.\n\n"
    "Args:\n"
    "    pt(repast4py.space.ContinuousPoint): the location to get the agents at.\n"
    "    agent_type(int): the type id of the agents to get the number of.\n\n"
    "Returns:\n"
    "    int: the number of agents at the specified location."
);

PyDoc_STRVAR(scspace_lb,
    "get_local_bounds()\n\n"
    
    "Gets the local bounds of this shared continuous space.\n\n"
    "The local bounds are the bounds of this shared continuous space on the current rank. For example, if "
    "the global bounds are 100 in the x dimension and 100 in the y dimension, and there are 4 ranks, "
    "then the local bounds will be some quadrant of those global bounds, 0 - 50 x 0 - 50 for example.\n\n"
    "Returns:\n"
    "    repast4py.geometry.BoundingBox: the local bounds as a BoundingBox."
);

PyDoc_STRVAR(scspace_oob,
    "_get_oob()\n\n"
    
    "Gets the synchronization data for the agents that have moved out of this shared continuous space's local bounds.\n\n"
    "The out of bounds data is used for synchronizing the global shared continuous space state, and this "
    "method should not ordinarly be called by users. Out of bounds data is generated when an agent moves "
    "and its new location is within the bounds another rank.\n\n"
    "Returns:\n"
    "    iterable: an iterable of tuples where each tuple contains the out of bounds data for an agent. "
    "Each tuple consists of the agents uid tuple, the rank containing the agent's new location, and the new "
    "location of the agents as a repast4py.space.ContinuousPoint."
);

PyDoc_STRVAR(scspace_coob,
    "_clear_oob()\n\n"
    
    "Clears the collection of out of bounds data.\n\n"
    "The out of bounds data is used for synchronizing the global shared continuous space state, and this "
    "method should not ordinarly be called by users. Out of bounds data is generated when an agent moves "
    "and its new location is within the bounds another rank.\n"
);

PyDoc_STRVAR(scspace_mooba,
    "_move_oob_agent(agent, pt)\n\n"
    
    "Moves the specified agent to the specified location as part of synchronizing the global shared continuous space state.\n\n"
    "Args:\n"
    "    agent(repast4py.core.agent): the agent to move.\n"
    "    pt(numpy.array): the location to move to as a 3 dimensional numpy array"
);

PyDoc_STRVAR(scspace_gbd,
    "_get_buffer_data()\n\n"
    
    "Gets the buffer data for this shared continuous space.\n\n"
    "Each subsection of the shared continuous space has buffesr whose contents are ghosted from a neighboring rank. "
    "This returns data describing the dimensions and rank of these buffers.\n\n"
    "Returns:\n"
    "    iterable: an iterable of tuples where each tuple describes the owner rank and size of a buffer: "
    "(rank, (xmin, xmax, ymin, ymax, zmin, zmax))"
);

PyDoc_STRVAR(scspace_within,
    "get_agents_within(bbox)\n\n"
    
    "Gets an iterator over all the agents within the specified bounding box.\n\n"
    "**The bounding box is assumed to be within the local bounds of this SharedCSpace.**\n\n"
    "Args:\n"
    "    box(repast4py.geometry.BoundingBox): the bounding box to get the agents within.\n\n"
    "Returns:\n"
    "    iterator: an iterator over all the agents within the specified bounding box."
);

static PyMethodDef SharedCSpace_methods[] = {
    {"add", SharedCSpace_add, METH_VARARGS, scspace_add},
    {"remove", SharedCSpace_remove, METH_VARARGS, scspace_rm},
    {"move", SharedCSpace_move, METH_VARARGS, scspace_move},
    {"contains", SharedCSpace_contains, METH_VARARGS, scspace_cnt},
    {"get_location", SharedCSpace_getLocation, METH_VARARGS, scspace_location},
    {"get_agent", SharedCSpace_getAgent, METH_VARARGS, scspace_geta},
    {"get_agents", SharedCSpace_getAgents, METH_VARARGS, scspace_getas},
    {"get_num_agents", (PyCFunction) SharedCSpace_getNumAgents, METH_VARARGS | METH_KEYWORDS, scspace_getnas},
    {"_get_oob", SharedCSpace_getOOBData, METH_VARARGS, scspace_oob},
    {"_clear_oob", SharedCSpace_clearOOBData, METH_VARARGS, scspace_coob},
    {"get_local_bounds", SharedCSpace_getLocalBounds, METH_VARARGS, scspace_lb},
    {"_move_oob_agent", SharedCSpace_synchMove, METH_VARARGS, scspace_mooba},
    {"_get_buffer_data", SharedCSpace_getBufferData, METH_VARARGS, scspace_gbd},
    {"get_agents_within", SharedCSpace_getAgentsWithin, METH_VARARGS, scspace_within},
    {NULL, NULL, 0, NULL}
};

PyDoc_STRVAR(scspace_scspace,
    "SharedContinuousSpace(name, bounds, borders, occupancy, buffer_size, comm, tree_threshold)\n\n"
    
    "An N-dimensional cartesian discrete space where agents can occupy locations defined by "
    "a continuous floating point coordinate.\n\n"
    "The space is shared over all the ranks in the specified communicator by sub-dividing the global bounds into "
    "some number of smaller spaces, one for each rank. For example, given a global spaces size of (100 x 25) and "
    "2 ranks, the global space will be split along the x dimension such that the SharedContinuousSpace in the first "
    "MPI rank covers (0-50 x 0-25) and the second rank (50-100 x 0-25). "
    "Each rank's SharedContinuousSpace contains a buffer of the specified size that duplicates or \"ghosts\" an adjacent "
    "area of the neighboring rank's SharedContinuousSpace. In the above example, the rank 1 space buffers the area from "
    "(50-52 x 0-25) in rank 2, and rank 2 buffers (48-50 x 0-25) in rank 1. Be sure to specify a buffer size appropriate "
    "to any agent behavior. For example, if an agent can \"see\" 3 units away and take some action based on what it "
    "perceives, then the buffer size should be at least 3, insuring that an agent can properly see beyond the borders of "
    "its own local SharedContinuousSpace. When an agent moves beyond the borders of its current SharedContinuousSpace, it will be transferred "
    "from its current rank, and to the rank containing the section of the global grid that it has moved into. "
    "The SharedContinuousSpace uses a `tree <https://en.wikipedia.org/wiki/Quadtree>`_ (quad or oct depending on the number of "
    "dimensions) to speed up spatial queries. The tree can be tuned using the tree threshold parameter.\n\n"
    "Args:\n"
    "   name (str): the name of the grid.\n"
    "   bounds (repast4py.geometry.BoundingBox): the global dimensions of the grid.\n"
    "   borders (repast4py.space.BorderType): the border semantics - :attr:`BorderType.Sticky` or :attr:`BorderType.Periodic`.\n"
    "   occupancy (repast4py.space.OccupancyType): the type of occupancy in each cell - :attr:`OccupancyType.Multiple` or :attr:`OccupancyType.Single`.\n"
    "   buffer_size (int): the size of this SharedContinuousSpace's buffered area. This single value is used for all dimensions.\n"
    "   comm (mpi4py.MPI.Intracomm): the communicator containing all the ranks over which this SharedGrid is shared.\n"
    "   tree_threshold (int): the space's tree cell maximum capacity. When this capacity is reached, the cell splits."
);

static PyTypeObject R4Py_SharedCSpaceType = {
    PyVarObject_HEAD_INIT(NULL, 0) 
    "_space.SharedContinuousSpace",                          /* tp_name */
    sizeof(R4Py_SharedCSpace),                      /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)SharedCSpace_dealloc,                                         /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_reserved */
    0,                                        /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash  */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    cspace_cspace,                         /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    SharedCSpace_methods,                                      /* tp_methods */
    SharedCSpace_members,                                      /* tp_members */
    SharedCSpace_get_setters,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)SharedCSpace_init,                                         /* tp_init */
    0,                                        /* tp_alloc */
    SharedCSpace_new                             /* tp_new */
};


///////////////////////// SharedContinuousSpace End /////////////////////////////

//////////////////////// CartesianTopology Start ////////////////////////////////

static void CartesianTopology_dealloc(R4Py_CartesianTopology* self) {
    delete self->topo;
    Py_XDECREF(self->cart_comm);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* CartesianTopology_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    R4Py_CartesianTopology* self = (R4Py_CartesianTopology*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->topo = nullptr;
        self->cart_comm = nullptr;
    }
    return (PyObject*)self;
}

static int CartesianTopology_init(R4Py_CartesianTopology* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {(char*)"comm", (char*)"global_bounds", (char*)"periodic", NULL};

    PyObject* bounds;
    PyObject* py_comm;
    int periodic;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!p", kwlist, &PyMPIComm_Type, &py_comm, &PyTuple_Type, &bounds,
        &periodic)) 
    {
        return -1;
    }

    long xmin, x_extent;
    long ymin, y_extent;
    long zmin, z_extent;

    if (!PyArg_ParseTuple(bounds, "llllll", &xmin, &x_extent, &ymin, &y_extent, &zmin, &z_extent)) {
        return -1;
    }
    BoundingBox box(xmin, x_extent, ymin, y_extent, zmin, z_extent);
    MPI_Comm cart_comm;

    int dims = 1;
    if (box.y_extent_ > 0) ++dims;
    if (box.z_extent_ > 0) ++dims;

    Py_INCREF(py_comm);
    MPI_Comm* comm_p = PyMPIComm_Get(py_comm);
    self->topo = new CartesianTopology(*comm_p, &cart_comm, dims, box, (bool)periodic);
    if (!self->topo) {
        return -1;
    }

    self->cart_comm = PyMPIComm_New(cart_comm);
    if (self->cart_comm == NULL) {
        return -1;
    }

    return 0;
}

static PyObject* CartesianTopology_getCartComm(PyObject* self, void* closure) {
    PyObject* comm = ((R4Py_CartesianTopology*)self)->cart_comm;
    Py_INCREF(comm);
    return comm;
}

static PyObject* CartesianTopology_getLocalBounds(PyObject* self, void* args) {
    BoundingBox bounds(0, 0, 0, 0, 0, 0);
    ((R4Py_CartesianTopology*)self)->topo->getBounds(bounds);
    PyObject* box_args = Py_BuildValue("(llllll)", bounds.xmin_, bounds.x_extent_, bounds.ymin_, bounds.y_extent_,
        bounds.zmin_, bounds.z_extent_);
    PyObject* pmod = PyImport_ImportModule("repast4py.space");
    PyObject* bbox_class = PyObject_GetAttrString(pmod, "BoundingBox");
    PyObject* box = PyObject_CallObject(bbox_class, box_args);

    Py_DECREF(box_args);
    Py_DECREF(pmod);
    Py_DECREF(bbox_class);

    return box;
}

static PyObject* CartesianTopology_getCartCoords(PyObject* self, void* closure) {
    std::vector<int> coords;
    ((R4Py_CartesianTopology*)self)->topo->getCoords(coords);
    if (coords.size() == 1) {
        return Py_BuildValue("(l)", coords[0]);
    } else if (coords.size() == 2) {
        return Py_BuildValue("(ll)", coords[0], coords[1]);
    } else {
        return Py_BuildValue("(lll)", coords[0], coords[1], coords[2]);
    }
}

static PyObject* CartesianTopology_computeBufferData(PyObject* self, PyObject* args) {
    //compute_neighbor_buffers(std::vector<CTNeighbor>& nghs, std::vector<int>& cart_coords, 
    //BoundingBox& local_bounds, int num_dims, unsigned int buffer_size
    int buffer_size;
    if (!PyArg_ParseTuple(args, "i", &buffer_size)) {
        return NULL;
    }

    // {
    //      volatile int i = 0;
    //      char hostname[256];
    //      gethostname(hostname, sizeof(hostname));
    //      printf("PID %d on %s ready for attach\n", getpid(), hostname);
    //      fflush(stdout);
    //      while (0 == i)
    //          sleep(5);
    // }
    
    std::vector<int> coords;
    ((R4Py_CartesianTopology*)self)->topo->getCoords(coords);
    auto nghs = std::make_shared<std::vector<CTNeighbor>>();
    ((R4Py_CartesianTopology*)self)->topo->getNeighbors(*nghs);
    BoundingBox bounds(0, 0, 0, 0, 0, 0);
    ((R4Py_CartesianTopology*)self)->topo->getBounds(bounds);
    int num_dims = ((R4Py_CartesianTopology*)self)->topo->numDims();
    const int* procs_per_dim = ((R4Py_CartesianTopology*)self)->topo->procsPerDim();
    compute_neighbor_buffers(*nghs, coords, bounds, num_dims, procs_per_dim, buffer_size);

    R4Py_PyObjectIter* obj_iter = (R4Py_PyObjectIter*)R4Py_PyObjectIterType.tp_new(&R4Py_PyObjectIterType, NULL, NULL);
    obj_iter->iter = new SequenceIter<std::vector<CTNeighbor>, GetBufferInfo>(nghs);
    return (PyObject*)obj_iter; 
}

static PyObject* CartesianTopology_procsPerDim(PyObject* self, void* args) {
    int num_dims = ((R4Py_CartesianTopology*)self)->topo->numDims();
    const int* procs_per_dim = ((R4Py_CartesianTopology*)self)->topo->procsPerDim();

    if (num_dims == 1) {
        return Py_BuildValue("(i)", procs_per_dim[0]);
    } else if (num_dims == 2) {
        return Py_BuildValue("(ii)", procs_per_dim[0], procs_per_dim[1]);
    } else {
        return  Py_BuildValue("(iii)", procs_per_dim[0], procs_per_dim[1], procs_per_dim[2]);
    }
}

PyDoc_STRVAR(cart_comm,
    "mpi4py.MPI.Intracomm: Gets the mpi communicator created by this CartesianTopology");

PyDoc_STRVAR(cart_coord,
    "tuple(int): Gets the cartesian coordinates of the current rank within this CartesianTopology");

PyDoc_STRVAR(cart_lb,
    "repast4py.geometry.BoundingBox: Gets the local bounds of the current rank within this CartesianTopology");

PyDoc_STRVAR(cart_ppd,
    "tuple(int): Gets the number of ranks per dimension in x,y,z order");

static PyGetSetDef CartesianTopology_get_setters[] = {
    {(char*)"comm", (getter)CartesianTopology_getCartComm, NULL, cart_comm, NULL},
    {(char*)"coordinates", (getter)CartesianTopology_getCartCoords, NULL, cart_coord, NULL},
    {(char*)"local_bounds", (getter)CartesianTopology_getLocalBounds, NULL, cart_lb, NULL},
    {(char*)"procs_per_dim", (getter)CartesianTopology_procsPerDim, NULL, cart_ppd},
    {NULL}
};

PyDoc_STRVAR(cart_cbn,
    "compute_buffer_nghs(buffer_size)\n\n"
    
    "Gets an iterator over the collection of buffer synchronization meta data for the current rank for the specified buffer_size.\n\n"
    "This data contains information for each cartesian neighbor of the current rank specifying what subsection of "
    "this grid or space should be sent what rank when synchronizing buffers. This method should not typically be "
    "called by users, but is rather part of the internal synchronization mechanism.\n\n"
    "Args:\n"
    "    buffer_size(int): the size of the buffer.\n\n"
    "Returns:\n"
    "    iterator: an iterator over tuples of buffer synchronization meta data: :samp:`(rank, (xmin, xmax, ymin, ymax, zmin, zmax))` "
    "where rank is the neighbor's rank, and the nested tuple specifies what part of this space or grid to send."
);


static PyMethodDef CartesianTopology_methods[] = {
    {"compute_buffer_nghs", CartesianTopology_computeBufferData, METH_VARARGS, cart_cbn},
    {NULL, NULL, 0, NULL}
};

PyDoc_STRVAR(cart_cart,
    "CartesianTopolgy(comm, global_bounds, periodic)\n\n"
    
    "A CartesianTopology is used by a SharedGrid or SharedContinuousSpace to create an efficient communicator over which the "
    "grid or space is distributed and to compute the buffer synchronization metadata used to synchronize the buffers between "
    "grid and space neighbors.\n\n"
    "This class should **not** typically be created by users, but is rather part of the internal synchronization mechanism "
    "used by the shared cartesian spaces. More info on MPI topologies can be found `here <https://wgropp.cs.illinois.edu/courses/cs598-s16/lectures/lecture28.pdf>`_.\n\n"
    "Args:\n"
    "   comm (mpi4py.MPI.Intracomm): the communicator to create the cartesian topology communicator from.\n"
    "   global_bounds(repast4py.geometry.BoundingBox): the global size of the shared space or grid that will use this topology.\n"
    "   periodic(bool): whether or not  the shared space or grid that will use this topology is periodic."
);

static PyTypeObject R4Py_CartesianTopologyType = {
    PyVarObject_HEAD_INIT(NULL, 0) 
    "_space.CartesianTopology",                          /* tp_name */
    sizeof(R4Py_CartesianTopology),                      /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)CartesianTopology_dealloc,                                         /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_reserved */
    0,                                        /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash  */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    cart_cart,                         /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    CartesianTopology_methods,                                      /* tp_methods */
    0,                                      /* tp_members */
    CartesianTopology_get_setters,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)CartesianTopology_init,                                         /* tp_init */
    0,                                        /* tp_alloc */
    CartesianTopology_new                             /* tp_new */
};



//////////////////////// CartesianTopology End ////////////////////////////////


static PyModuleDef spacemodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "repast4py._space",
    .m_doc = "Repast4Py space related classes and functions",
    .m_size = -1,
};

// PyMODINIT_FUNC adds "extern C" among other things
PyMODINIT_FUNC
PyInit__space(void)
{

    PyObject *m;
    m = PyModule_Create(&spacemodule);
    if (m == NULL) return NULL;

    if (import_core() < 0) {
         return NULL;
    }

    if (import_mpi4py() < 0) {
        return NULL;
    }

    import_array();
    
    if (PyType_Ready(&DiscretePointType) < 0) return NULL;
    if (PyType_Ready(&ContinuousPointType) < 0) return NULL;
    if (PyType_Ready(&R4Py_GridType) < 0) return NULL;
    if (PyType_Ready(&R4Py_CSpaceType) < 0) return NULL;
    if (PyType_Ready(&R4Py_SharedGridType) < 0) return NULL;
    if (PyType_Ready(&R4Py_SharedCSpaceType) < 0) return NULL;
    if (PyType_Ready(&R4Py_GridStickyBordersType) < 0) return NULL;
    if (PyType_Ready(&R4Py_GridPeriodicBordersType) < 0) return NULL;
    if (PyType_Ready(&R4Py_CartesianTopologyType) < 0) return NULL;


    Py_INCREF(&DiscretePointType);
    if (PyModule_AddObject(m, "DiscretePoint", (PyObject *) &DiscretePointType) < 0) {
        Py_DECREF(&DiscretePointType);
        Py_DECREF(m);
        
        return NULL;
    }

    Py_INCREF(&ContinuousPointType);
    if (PyModule_AddObject(m, "ContinuousPoint", (PyObject *) &ContinuousPointType) < 0) {
        Py_DECREF(&DiscretePointType);
        Py_DECREF(&ContinuousPointType);
        Py_DECREF(m);
        
        return NULL;
    }

    Py_INCREF(&R4Py_GridType);
    if (PyModule_AddObject(m, "Grid", (PyObject *) &R4Py_GridType) < 0) {
        Py_DECREF(&DiscretePointType);
        Py_DECREF(&ContinuousPointType);
        Py_DECREF(&R4Py_GridType);
        Py_DECREF(m);
        
        return NULL;
    }

    Py_INCREF(&R4Py_CSpaceType);
    if (PyModule_AddObject(m, "ContinuousSpace", (PyObject *) &R4Py_CSpaceType) < 0) {
        Py_DECREF(&DiscretePointType);
        Py_DECREF(&ContinuousPointType);
        Py_DECREF(&R4Py_GridType);
        Py_DECREF(&R4Py_CSpaceType);
        Py_DECREF(m);
        
        return NULL;
    }

    Py_INCREF(&R4Py_SharedGridType);
    if (PyModule_AddObject(m, "SharedGrid", (PyObject*) &R4Py_SharedGridType) < 0) {
        Py_DECREF(&DiscretePointType);
        Py_DECREF(&ContinuousPointType);
        Py_DECREF(&R4Py_GridType);
        Py_DECREF(&R4Py_SharedGridType);
        Py_DECREF(&R4Py_CSpaceType);
        Py_DECREF(m);
    }

    Py_INCREF(&R4Py_SharedGridType);
    if (PyModule_AddObject(m, "SharedContinuousSpace", (PyObject*) &R4Py_SharedCSpaceType) < 0) {
        Py_DECREF(&DiscretePointType);
        Py_DECREF(&ContinuousPointType);
        Py_DECREF(&R4Py_GridType);
        Py_DECREF(&R4Py_SharedGridType);
        Py_DECREF(&R4Py_CSpaceType);
        Py_DECREF(&R4Py_SharedCSpaceType);
        Py_DECREF(m);
    }

    Py_INCREF(&R4Py_GridStickyBordersType);
    if (PyModule_AddObject(m, "GridStickyBorders", (PyObject*) &R4Py_GridStickyBordersType) < 0) {
        Py_DECREF(&DiscretePointType);
        Py_DECREF(&ContinuousPointType);
        Py_DECREF(&R4Py_GridType);
        Py_DECREF(&R4Py_SharedGridType);
        Py_DECREF(&R4Py_CSpaceType);
        Py_DECREF(&R4Py_SharedCSpaceType);
        Py_DECREF(&R4Py_GridStickyBordersType);
        Py_DECREF(m);
    }

    Py_INCREF(&R4Py_GridPeriodicBordersType);
    if (PyModule_AddObject(m, "GridPeriodicBorders", (PyObject*) &R4Py_GridPeriodicBordersType) < 0) {
        Py_DECREF(&DiscretePointType);
        Py_DECREF(&ContinuousPointType);
        Py_DECREF(&R4Py_GridType);
        Py_DECREF(&R4Py_SharedGridType);
        Py_DECREF(&R4Py_CSpaceType);
        Py_DECREF(&R4Py_SharedCSpaceType);
        Py_DECREF(&R4Py_GridStickyBordersType);
        Py_DECREF(&R4Py_GridPeriodicBordersType);
        Py_DECREF(m);
    }

    Py_INCREF(&R4Py_CartesianTopologyType);
    if (PyModule_AddObject(m, "CartesianTopology", (PyObject*) &R4Py_CartesianTopologyType) < 0) {
        Py_DECREF(&DiscretePointType);
        Py_DECREF(&ContinuousPointType);
        Py_DECREF(&R4Py_GridType);
        Py_DECREF(&R4Py_SharedGridType);
        Py_DECREF(&R4Py_CSpaceType);
        Py_DECREF(&R4Py_SharedCSpaceType);
        Py_DECREF(&R4Py_GridStickyBordersType);
        Py_DECREF(&R4Py_GridPeriodicBordersType);
        Py_DECREF(&R4Py_CartesianTopologyType);
        Py_DECREF(m);
    }

    return m;
}