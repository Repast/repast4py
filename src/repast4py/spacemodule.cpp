#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <new>
#include "structmember.h"


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// See https://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL REPAST4PY_ARRAY_API
#include "numpy/arrayobject.h"

#include "space.h"
#include "coremodule.h"

using namespace repast4py;

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
    long x, y;
    long z = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ll|l", kwlist, &x, &y, &z)) {
        return -1;
    }

    long* d = (long*)PyArray_DATA(self->coords);;
    d[0] = x;
    d[1] = y;
    d[2] = z;
    
    return 0;
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

static PyObject* DiscretePoint_repr(R4Py_DiscretePoint* self) {
    long* data = (long*)PyArray_DATA(self->coords);
    return PyUnicode_FromFormat("DiscretePoint(%ld, %ld, %ld)", data[0], data[1], data[2]);   
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
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    0,                                      /* tp_methods */
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

    if (border_type == 0) {
        if (occ_type == 0) {
            BoundingBox<R4Py_DiscretePoint> box(xmin, width, ymin, height, zmin, depth);
            self->grid = new Grid<MOSGrid>(name, box);

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
    0,                                        /* tp_getset */
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

    import_array();
    
    if (PyType_Ready(&DiscretePointType) < 0) return NULL;
    if (PyType_Ready(&R4Py_GridType) < 0) return NULL;


    Py_INCREF(&DiscretePointType);
    if (PyModule_AddObject(m, "DiscretePoint", (PyObject *) &DiscretePointType) < 0) {
        Py_DECREF(&DiscretePointType);
        Py_DECREF(m);
        
        return NULL;
    }

    Py_INCREF(&R4Py_GridType);
    if (PyModule_AddObject(m, "Grid", (PyObject *) &R4Py_GridType) < 0) {
        Py_DECREF(&DiscretePointType);
        Py_DECREF(&R4Py_GridType);
        Py_DECREF(m);
        
        return NULL;
    }

    return m;
}