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


static PyGetSetDef DiscretePoint_get_setters[] = {
    {(char*)"x", (getter)DiscretePoint_get_x, NULL, (char*)"discrete point x", NULL},
    {(char*)"y", (getter)DiscretePoint_get_y, NULL, (char*)"discrete point y", NULL},
    {(char*)"z", (getter)DiscretePoint_get_z, NULL, (char*)"discrete point z", NULL},
    {(char*)"coordinates", (getter)DiscretePoint_get_coords, NULL, (char*)"discrete point coordinates", NULL},
    {NULL}
};

static PyMethodDef DiscretePoint_methods[] = {
    {"_reset1D", DiscretePoint_reset1D, METH_VARARGS, ""},
    {"_reset2D", DiscretePoint_reset2D, METH_VARARGS, ""},
    {"_reset3D", DiscretePoint_reset3D, METH_VARARGS, ""},
    {"_reset", DiscretePoint_reset, METH_VARARGS, ""},
    {"_reset_from_array", DiscretePoint_reset_from_array, METH_VARARGS, ""},

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
    "DiscretePoint Object",                         /* tp_doc */
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


static PyGetSetDef ContinuousPoint_get_setters[] = {
    {(char*)"x", (getter)ContinuousPoint_get_x, NULL, (char*)"continuous point x", NULL},
    {(char*)"y", (getter)ContinuousPoint_get_y, NULL, (char*)"continuous point y", NULL},
    {(char*)"z", (getter)ContinuousPoint_get_z, NULL, (char*)"continuous point z", NULL},
    {(char*)"coordinates", (getter)ContinuousPoint_get_coords, NULL, (char*)"continuous point coordinates", NULL},
    {NULL}
};

static PyMethodDef ContinuousPoint_methods[] = {
    {"_reset1D", ContinuousPoint_reset1D, METH_VARARGS, ""},
    {"_reset2D", ContinuousPoint_reset2D, METH_VARARGS, ""},
    {"_reset3D", ContinuousPoint_reset3D, METH_VARARGS, ""},
    {"_reset", ContinuousPoint_reset, METH_VARARGS, ""},
    {"_reset_from_array", ContinuousPoint_reset_from_array, METH_VARARGS, ""},
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
    "ContinuousPoint Object",                         /* tp_doc */
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

static PyObject* GridStickyBorders_transform(PyObject* self, PyObject* args) {
    PyObject* pt, *ret_pt;
    if (!PyArg_ParseTuple(args, "O!O!", &DiscretePointType, &pt, &DiscretePointType, &ret_pt)) {
        return NULL;
    }
    ((R4Py_GridStickyBorders*)self)->borders->transform((R4Py_DiscretePoint*)pt, (R4Py_DiscretePoint*)ret_pt);
    Py_RETURN_NONE;
}

static PyMethodDef GridStickyBorders_methods[] = {
    {"_transform", GridStickyBorders_transform, METH_VARARGS, ""},
     {NULL, NULL, 0, NULL}
};

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
    "GridStickyBorders Object",                         /* tp_doc */
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

static PyMethodDef GridPeriodicBorders_methods[] = {
    {"_transform", GridPeriodicBorders_transform, METH_VARARGS, ""},
     {NULL, NULL, 0, NULL}
};

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
    "GridPeriodicBorders Object",                         /* tp_doc */
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
    if (border_type == 0) {
        if (occ_type == 0) {
            self->grid = new Grid<MOSGrid>(name, box);

        } else {
            PyErr_SetString(PyExc_RuntimeError, "Invalid occupancy type");
            return -1;
        }
    } else if (border_type == 1) {
        if (occ_type == 0) {
            self->grid = new Grid<MOPGrid>(name, box);
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
    // not completely sure why this is necessary but without it
    // the iterator is decrefed out of existence after first call to __iter__
    Py_INCREF(agent_iter);
    return (PyObject*)agent_iter;
}

static PyObject* Grid_getName(PyObject* self, PyObject* args) {
    return PyUnicode_FromString(((R4Py_Grid*)self)->grid->name().c_str());
}

static PyGetSetDef Grid_get_setters[] = {
    {(char*)"name", (getter)Grid_getName, NULL, (char*)"grid name", NULL},
    {NULL}
};

static PyMethodDef Grid_methods[] = {
    {"add", Grid_add, METH_VARARGS, "Adds the specified agent to this grid projection"},
    {"remove", Grid_remove, METH_VARARGS, "Removes the specified agent from this grid projection"},
    {"move", Grid_move, METH_VARARGS, "Moves the specified agent to the specified location in this grid projection"},
    {"get_location", Grid_getLocation, METH_VARARGS, "Gets the location of the specified agent in this grid projection"},
    {"get_agent", Grid_getAgent, METH_VARARGS, "Gets the first agent at the specified location in this grid projection"},
    {"get_agents", Grid_getAgents, METH_VARARGS, "Gets all the agents at the specified location in this grid projection"},
    {NULL, NULL, 0, NULL}
};

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
    "Grid Object",                         /* tp_doc */
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
        (char*)"occupancy", (char*)"buffersize", (char*)"comm", NULL};

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
    if (border_type == 0) {
        if (occ_type == 0) {
            self->grid = new SharedGrid<DistributedCartesianSpace<MOSGrid, R4Py_DiscretePoint>>(name, box, buffer_size, 
            *comm_p);

        } else {
            PyErr_SetString(PyExc_RuntimeError, "Invalid occupancy type");
            return -1;
        }
    } else if (border_type == 1) {
        if (occ_type == 0) {
            self->grid = new SharedGrid<DistributedCartesianSpace<MOPGrid, R4Py_DiscretePoint>>(name, box, buffer_size, 
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

static PyObject* SharedGrid_moveBufferAgent(PyObject* self, PyObject* args) {
    PyObject* agent, *pt;
    if (!PyArg_ParseTuple(args, "O!O!", &R4Py_AgentType, &agent, &DiscretePointType, &pt)) {
        return NULL;
    }

    try {
        R4Py_DiscretePoint* ret = ((R4Py_SharedGrid*)self)->grid->moveBufferAgent((R4Py_Agent*)agent, (R4Py_DiscretePoint*)pt);
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
    // not completely sure why this is necessary but without it
    // the iterator is decrefed out of existence after first call to __iter__
    Py_INCREF(agent_iter);
    return (PyObject*)agent_iter;
}

static PyObject* SharedGrid_getOOBData(PyObject* self, PyObject* args) {
    std::shared_ptr<std::map<R4Py_AgentID*, PyObject*, agent_id_comp>> oob = ((R4Py_SharedGrid*)self)->grid->getOOBData();
    R4Py_PyObjectIter* obj_iter = (R4Py_PyObjectIter*)R4Py_PyObjectIterType.tp_new(&R4Py_PyObjectIterType, NULL, NULL);
    obj_iter->iter = new ValueIter<std::map<R4Py_AgentID*, PyObject*, agent_id_comp>>(oob);
    // not completely sure why this is necessary but without it
    // the iterator is decrefed out of existence after first call to __iter__
    Py_INCREF(obj_iter);
    return (PyObject*)obj_iter;
}

static PyObject* SharedGrid_getBufferData(PyObject* self, PyObject* args) { 
    std::shared_ptr<std::vector<CTNeighbor>> nghs = ((R4Py_SharedGrid*)self)->grid->getNeighborData();
    R4Py_PyObjectIter* obj_iter = (R4Py_PyObjectIter*)R4Py_PyObjectIterType.tp_new(&R4Py_PyObjectIterType, NULL, NULL);
    obj_iter->iter = new SequenceIter<std::vector<CTNeighbor>, GetBufferInfo>(nghs);
    Py_INCREF(obj_iter);
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

static PyMethodDef SharedGrid_methods[] = {
    {"add", SharedGrid_add, METH_VARARGS, "Adds the specified agent to this shared grid projection"},
    {"remove", SharedGrid_remove, METH_VARARGS, "Removes the specified agent from this shared grid projection"},
    {"move", SharedGrid_move, METH_VARARGS, "Moves the specified agent to the specified location in this shared grid projection"},
    {"_move_buffer_agent", SharedGrid_moveBufferAgent, METH_VARARGS, "Moves the specified agent to the specified buffer location in this shared grid projection"},
    {"get_location", SharedGrid_getLocation, METH_VARARGS, "Gets the location of the specified agent in this shared grid projection"},
    {"get_agent", SharedGrid_getAgent, METH_VARARGS, "Gets the first agent at the specified location in this shared grid projection"},
    {"get_agents", SharedGrid_getAgents, METH_VARARGS, "Gets all the agents at the specified location in this shared grid projection"},
    {"_get_oob", SharedGrid_getOOBData, METH_VARARGS, "Gets the out of bounds data for any agents that are out of the local bounds in this shared grid projection"},
    {"_clear_oob", SharedGrid_clearOOBData, METH_VARARGS, "Clears the out of bounds data for any agents that are out of the local bounds in this shared grid projection"},
    {"get_local_bounds", SharedGrid_getLocalBounds, METH_VARARGS, "Gets the local bounds for this shared grid projection"},
    {"_synch_move", SharedGrid_synchMove, METH_VARARGS, "Moves the specified agent to the specified location in this shared grid projection as part of a movement synchronization"},
    {"_get_buffer_data", SharedGrid_getBufferData, METH_VARARGS, "Gets the buffer data for synchronizing neighboring buffers of this shared grid projetion - a list of tuples of the form info for where and what range, tuple: (rank, (xmin, xmax, ymin, ymax, zmin, zmax))"},
    {NULL, NULL, 0, NULL}
};

static PyObject* SharedGrid_getName(PyObject* self, PyObject* args) {
    return PyUnicode_FromString(((R4Py_SharedGrid*)self)->grid->name().c_str());
}

static PyGetSetDef SharedGrid_get_setters[] = {
    {(char*)"name", (getter)SharedGrid_getName, NULL, (char*)"grid name", NULL},
    {NULL}
};

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
    "SharedGrid Object",                         /* tp_doc */
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
    if (border_type == 0) {
        if (occ_type == 0) {
            self->space = new CSpace<MOSCSpace>(name, box, tree_threshold);

        } else {
            PyErr_SetString(PyExc_RuntimeError, "Invalid occupancy type");
            return -1;
        }
    } else if (border_type == 1) {
        if (occ_type == 0) {
            self->space = new CSpace<MOPCSpace>(name, box, tree_threshold);
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
    // not completely sure why this is necessary but without it
    // the iterator is decrefed out of existence after first call to __iter__
    Py_INCREF(agent_iter);
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
    // not completely sure why this is necessary but without it
    // the iterator is decrefed out of existence after first call to __iter__
    Py_INCREF(agent_iter);
    return (PyObject*)agent_iter;
}

static PyObject* CSpace_getName(PyObject* self, PyObject* args) {
    return PyUnicode_FromString(((R4Py_CSpace*)self)->space->name().c_str());
}

static PyGetSetDef CSpace_get_setters[] = {
    {(char*)"name", (getter)CSpace_getName, NULL, (char*)"space name", NULL},
    {NULL}
};

static PyMethodDef CSpace_methods[] = {
    {"add", CSpace_add, METH_VARARGS, "Adds the specified agent to this continuous space projection"},
    {"remove", CSpace_remove, METH_VARARGS, "Removes the specified agent from this continuous space projection"},
    {"move", CSpace_move, METH_VARARGS, "Moves the specified agent to the specified location in this continuous space projection"},
    {"get_location", CSpace_getLocation, METH_VARARGS, "Gets the location of the specified agent in this continuous space projection"},
    {"get_agent", CSpace_getAgent, METH_VARARGS, "Gets the first agent at the specified location in this continuous space projection"},
    {"get_agents", CSpace_getAgents, METH_VARARGS, "Gets all the agents at the specified location in this continuous space projection"},
    {"get_agents_within", CSpace_getAgentsWithin, METH_VARARGS, "Gets all the agents within the specified bounding box in this continuous space"},
    {NULL, NULL, 0, NULL}
};

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
    "CSpace Object",                         /* tp_doc */
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
    static char* kwlist[] = {(char*)"name",(char*)"bounds", (char*)"borders",
        (char*)"occupancy", (char*)"buffersize", (char*)"comm", (char*)"tree_threshold", NULL};

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
    if (border_type == 0) {
        if (occ_type == 0) {
            self->space = new SharedContinuousSpace<DistributedCartesianSpace<MOSCSpace, R4Py_ContinuousPoint>>(name, box, buffer_size, 
            *comm_p, tree_threshold);

        } else {
            PyErr_SetString(PyExc_RuntimeError, "Invalid occupancy type");
            return -1;
        }
    } else if (border_type == 1) {
        if (occ_type == 0) {
            self->space = new SharedContinuousSpace<DistributedCartesianSpace<MOPCSpace, R4Py_ContinuousPoint>>(name, box, buffer_size, 
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

static PyObject* SharedCSpace_moveBufferAgent(PyObject* self, PyObject* args) {
    PyObject* agent, *pt;
    if (!PyArg_ParseTuple(args, "O!O!", &R4Py_AgentType, &agent, &ContinuousPointType, &pt)) {
        return NULL;
    }

    try {
        R4Py_ContinuousPoint* ret = ((R4Py_SharedCSpace*)self)->space->moveBufferAgent((R4Py_Agent*)agent, (R4Py_ContinuousPoint*)pt);
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
    // not completely sure why this is necessary but without it
    // the iterator is decrefed out of existence after first call to __iter__
    Py_INCREF(agent_iter);
    return (PyObject*)agent_iter;
}

static PyObject* SharedCSpace_getOOBData(PyObject* self, PyObject* args) {
    std::shared_ptr<std::map<R4Py_AgentID*, PyObject*, agent_id_comp>> oob = ((R4Py_SharedCSpace*)self)->space->getOOBData();
    R4Py_PyObjectIter* obj_iter = (R4Py_PyObjectIter*)R4Py_PyObjectIterType.tp_new(&R4Py_PyObjectIterType, NULL, NULL);
    obj_iter->iter = new ValueIter<std::map<R4Py_AgentID*, PyObject*, agent_id_comp>>(oob);
    // not completely sure why this is necessary but without it
    // the iterator is decrefed out of existence after first call to __iter__
    Py_INCREF(obj_iter);
    return (PyObject*)obj_iter;
}

static PyObject* SharedCSpace_getBufferData(PyObject* self, PyObject* args) { 
    std::shared_ptr<std::vector<CTNeighbor>> nghs = ((R4Py_SharedCSpace*)self)->space->getNeighborData();
    R4Py_PyObjectIter* obj_iter = (R4Py_PyObjectIter*)R4Py_PyObjectIterType.tp_new(&R4Py_PyObjectIterType, NULL, NULL);
    obj_iter->iter = new SequenceIter<std::vector<CTNeighbor>, GetBufferInfo>(nghs);
    Py_INCREF(obj_iter);
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
    // not completely sure why this is necessary but without it
    // the iterator is decrefed out of existence after first call to __iter__
    Py_INCREF(agent_iter);
    return (PyObject*)agent_iter;
}

static PyObject* SharedCSpace_getName(PyObject* self, PyObject* args) {
    return PyUnicode_FromString(((R4Py_SharedCSpace*)self)->space->name().c_str());
}

static PyGetSetDef SharedCSpace_get_setters[] = {
    {(char*)"name", (getter)SharedCSpace_getName, NULL, (char*)"space name", NULL},
    {NULL}
};

static PyMemberDef SharedCSpace_members[] = {
    {(char* const)"_cart_comm", T_OBJECT_EX, offsetof(R4Py_SharedCSpace, cart_comm), READONLY, (char* const)"The cartesian communicator for this shared grid"},
    {NULL}
};

static PyMethodDef SharedCSpace_methods[] = {
    {"add", SharedCSpace_add, METH_VARARGS, "Adds the specified agent to this shared continuous space projection"},
    {"remove", SharedCSpace_remove, METH_VARARGS, "Removes the specified agent from this shared continuous space projection"},
    {"move", SharedCSpace_move, METH_VARARGS, "Moves the specified agent to the specified location in this shared continuous space projection"},
    {"_move_buffer_agent", SharedCSpace_moveBufferAgent, METH_VARARGS, "Moves the specified agent to the specified buffer location in this shared continuous space projection"},
    {"get_location", SharedCSpace_getLocation, METH_VARARGS, "Gets the location of the specified agent in this shared continuous space projection"},
    {"get_agent", SharedCSpace_getAgent, METH_VARARGS, "Gets the first agent at the specified location in this shared continuous space projection"},
    {"get_agents", SharedCSpace_getAgents, METH_VARARGS, "Gets all the agents at the specified location in this shared continuous space projection"},
    {"_get_oob", SharedCSpace_getOOBData, METH_VARARGS, "Gets the out of bounds data for any agents that are out of the local bounds in this shared continuous space projection"},
    {"_clear_oob", SharedCSpace_clearOOBData, METH_VARARGS, "Clears the out of bounds data for any agents that are out of the local bounds in this shared continuous space projection"},
    {"get_local_bounds", SharedCSpace_getLocalBounds, METH_VARARGS, "Gets the local bounds for this shared continuous space projection"},
    {"_synch_move", SharedCSpace_synchMove, METH_VARARGS, "Moves the specified agent to the specified location in this shared continuous space projection as part of a movement synchronization"},
    {"_get_buffer_data", SharedCSpace_getBufferData, METH_VARARGS, "Gets the buffer data for synchronizing neighboring buffers of this shared continuous projection"},
    {"get_agents_within", SharedCSpace_getAgentsWithin, METH_VARARGS, "Gets all the agents within the specified bounding box in this shared continuous space projection"},
    {NULL, NULL, 0, NULL}
};

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
    "SharedCSpace Object",                         /* tp_doc */
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
    compute_neighbor_buffers(*nghs, coords, bounds, num_dims, buffer_size);

    R4Py_PyObjectIter* obj_iter = (R4Py_PyObjectIter*)R4Py_PyObjectIterType.tp_new(&R4Py_PyObjectIterType, NULL, NULL);
    obj_iter->iter = new SequenceIter<std::vector<CTNeighbor>, GetBufferInfo>(nghs);
    Py_INCREF(obj_iter);
    return (PyObject*)obj_iter; 
}

static PyGetSetDef CartesianTopology_get_setters[] = {
    {(char*)"comm", (getter)CartesianTopology_getCartComm, NULL, (char*)"The communicator over the Carteisan topology", NULL},
    {(char*)"coordinates", (getter)CartesianTopology_getCartCoords, NULL, (char*)"The coordinates of the current rank within this topology", NULL},
    {(char*)"local_bounds", (getter)CartesianTopology_getLocalBounds, NULL, (char*)"The local bounds of the current rank within this topology", NULL},
    {NULL}
};


static PyMethodDef CartesianTopology_methods[] = {
    {"compute_buffer_nghs", CartesianTopology_computeBufferData, METH_VARARGS, "Computes the buffer neighbor data for the current rank for the specified buffer size"},
    {NULL, NULL, 0, NULL}
};

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
    "CartesianTopology Object",                         /* tp_doc */
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