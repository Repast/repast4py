// Copyright 2021, UChicago Argonne, LLC 
// All Rights Reserved
// Software Name: repast4py
// By: Argonne National Laboratory
// License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define R4PY_CORE_MODULE
#include "coremodule.h"

#include <new>
#include "structmember.h"

#include "core.h"

using namespace repast4py;

//////////// R4Py_AgentIter
static void AgentIter_dealloc(R4Py_AgentIter* self) {
    delete self->iter;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* AgentIter_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    R4Py_AgentIter* self = (R4Py_AgentIter*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->iter = NULL;
    }
    return (PyObject*)self;
}

static PyObject* AgentIter_iter(R4Py_AgentIter* self) {
    Py_INCREF(self);
    self->iter->reset();
    return (PyObject*) self;
}

static PyObject* AgentIter_next(R4Py_AgentIter* self) {
    if (!self->iter->hasNext()) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
    PyObject * obj = (PyObject*)self->iter->next();
    Py_INCREF(obj);
    return obj;
}

static PyTypeObject R4Py_AgentIterType = {
    PyVarObject_HEAD_INIT(NULL, 0) 
    "_core.AgentIterator",                          /* tp_name */
    sizeof(R4Py_AgentIter),                      /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)AgentIter_dealloc,             /* tp_dealloc */
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
    "AgentIterator Object",                         /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    (getiterfunc)AgentIter_iter,                           /* tp_iter */
    (iternextfunc)AgentIter_next,                           /* tp_iternext */
    0,                                      /* tp_methods */
    0,                                      /* tp_members */
    0,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,                                         /* tp_init */
    0,                                        /* tp_alloc */
    AgentIter_new                               /* tp_new */
};

////////////////// AgentIter End ///////////////////

////////////////// PyObjectIter ///////////////////
static void PyObjectIter_dealloc(R4Py_PyObjectIter* self) {
    delete self->iter;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyObjectIter_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    R4Py_PyObjectIter* self = (R4Py_PyObjectIter*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->iter = NULL;
    }
    return (PyObject*)self;
}

static PyObject* PyObjectIter_iter(R4Py_PyObjectIter* self) {
    Py_INCREF(self);
    self->iter->reset();
    return (PyObject*) self;
}

static PyObject* PyObjectIter_next(R4Py_PyObjectIter* self) {
    if (!self->iter->hasNext()) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
    PyObject * obj = (PyObject*)self->iter->next();
    Py_INCREF(obj);
    return obj;
}

static PyTypeObject R4Py_PyObjectIterType = {
    PyVarObject_HEAD_INIT(NULL, 0) 
    "_core.PyObjectIterator",                          /* tp_name */
    sizeof(R4Py_PyObjectIter),                      /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)PyObjectIter_dealloc,             /* tp_dealloc */
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
    "PyObjectIterator Object",                         /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    (getiterfunc)PyObjectIter_iter,                           /* tp_iter */
    (iternextfunc)PyObjectIter_next,                           /* tp_iternext */
    0,                                      /* tp_methods */
    0,                                      /* tp_members */
    0,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,                                         /* tp_init */
    0,                                        /* tp_alloc */
    PyObjectIter_new                               /* tp_new */
};

////////////////// PyObjectIter End ///////////////////


//////////////////// Agent ///////////////////
static void Agent_dealloc(R4Py_Agent* self) {
    Py_XDECREF(self->aid->as_tuple);
    delete self->aid;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Agent_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    R4Py_Agent* self;
    self = (R4Py_Agent*) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->aid = new R4Py_AgentID();
        if (self->aid) {
            self->aid->id = -1;
            self->aid->type = -1;
            self->aid->rank = 0;
            self->local_rank = 0;

            // self->aid->as_tuple = PyTuple_New(3);
            // if (!self->aid->as_tuple) {
            //     delete self->aid;
            //     Py_TYPE(self)->tp_free((PyObject*)self);
            //     self = NULL;
            // }
        } else {
            Py_TYPE(self)->tp_free((PyObject*)self);
            self = NULL;
        }


    }
    return (PyObject*) self;
}

static int Agent_init(R4Py_Agent* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {(char*)"id", (char*)"type", (char*)"rank", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "li|l", kwlist, &self->aid->id, &self->aid->type,
        &self->aid->rank)) {
        return -1;
    }
    self->local_rank = self->aid->rank;
    if (self->aid->type < 0) {
        PyErr_SetString(PyExc_ValueError, "Type component of agent unique id must be a non-negative integer");
        return -1;
    }
    // TODO - maybe build this with PyTuple_New rather than parse the format string
    self->aid->as_tuple = Py_BuildValue("(liI)", self->aid->id, self->aid->type, self->aid->rank);
    return 0;
}

static PyObject* Agent_get_id(R4Py_Agent* self, void* closure) {
    return PyLong_FromLong(self->aid->id);
}

static PyObject* Agent_get_uid_rank(R4Py_Agent* self, void* closure) {
    return PyLong_FromLong(self->aid->rank);
}

static PyObject* Agent_get_type(R4Py_Agent* self, void* closure) {
    return PyLong_FromLong(self->aid->type);
}

static PyObject* Agent_get_uid(R4Py_Agent* self, void* closure) {
    PyObject* uid = self->aid->as_tuple;
    Py_INCREF(uid);
    return uid;
}

static PyObject* Agent_get_local_rank(R4Py_Agent* self, void* closure) {
    return PyLong_FromLong(self->local_rank);
}

static int Agent_set_local_rank(R4Py_Agent* self, PyObject* value, void* closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the local rank attribute");
        return -1;
    }
    if (!PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "The local rank attribute value must be a integer value");
        return -1;
    }
    self->local_rank = PyLong_AsLong(value);
    return 0;
}

PyDoc_STRVAR(ag_id,
    "int: Gets the id component from this agent's unique id");

PyDoc_STRVAR(ag_type,
    "int: Gets the type component from this agent's unique id");

PyDoc_STRVAR(ag_rank,
    "int: Gets the rank component from this agent's unique id");

PyDoc_STRVAR(ag_uid,
    "Tuple(int, int, int): Gets this agent's unique id tuple (id, type, rank)");

PyDoc_STRVAR(ag_lr,
    "int: Gets and sets the current local rank of this agent. Users should **NOT** need to access this value.");


static PyGetSetDef Agent_get_setters[] = {
    {(char*)"id", (getter)Agent_get_id, NULL, ag_id, NULL},
    {(char*)"type", (getter)Agent_get_type, NULL, ag_type, NULL},
    {(char*)"uid_rank", (getter)Agent_get_uid_rank, NULL, ag_rank, NULL},
    {(char*)"uid", (getter)Agent_get_uid, NULL, ag_uid, NULL},
    {(char*)"local_rank", (getter)Agent_get_local_rank, (setter)Agent_set_local_rank, ag_lr, NULL},
    {NULL}
};

static PyObject* Agent_repr(R4Py_Agent* self) {
    return PyUnicode_FromFormat("Agent(%ld, %d, %d)", self->aid->id, self->aid->type, self->aid->rank);
}

PyDoc_STRVAR(ag_ag,
    "Agent(id, type, rank)\n\n"
    "Parent class of all agents in a repast4py simulation.\n\n"
    "Each agent must have an id that is unique among all agents over all the ranks of a simulation. "
    "This id is composed of an integer id, an agent type id, and the integer rank on which the agent "
    "is created. These components are the arguments to the Agent constructor\n\n"
    "Args:\n"
    "    id (int): an integer that uniquely identifies this agent from among those of the same type and "
    "created on the same rank. Consequently, agents created on different ranks, or of different types but created on the same rank "
    "may have the same id.\n"
    "    type (int): an integer that specifies the type of this agent.\n"
    "    rank (int): the rank on which this agent is created."
);


static PyTypeObject R4Py_AgentType = {
    PyVarObject_HEAD_INIT(NULL, 0) 
    "_core.Agent",                          /* tp_name */
    sizeof(R4Py_Agent),                      /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)Agent_dealloc,                                         /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_reserved */
    (reprfunc)Agent_repr,                                        /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash  */
    0,                                        /* tp_call */
    (reprfunc)Agent_repr,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    ag_ag,                         /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    0,                                      /* tp_methods */
    0,                                      /* tp_members */
    Agent_get_setters,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc) Agent_init,                                         /* tp_init */
    0,                                        /* tp_alloc */
    Agent_new                               /* tp_new */
};


static PyModuleDef coremodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "repast4py._core",
    .m_doc = "core module",
    .m_size = -1,
};



// PyMODINIT_FUNC adds "extern C" among other things
PyMODINIT_FUNC
PyInit__core(void)
{

    if (PyType_Ready(&R4Py_AgentType) < 0)
        return NULL;

    if (PyType_Ready(&R4Py_AgentIterType) < 0)
        return NULL;

    if (PyType_Ready(&R4Py_PyObjectIterType) < 0)
        return  NULL;


    PyObject *m;
    m = PyModule_Create(&coremodule);
    if (m == NULL)
        return NULL;

    static void* R4PyCore_API[R4PyCore_API_pointers];
    PyObject* c_api_object;

    R4PyCore_API[0] = (void*)&R4Py_AgentType;
    R4PyCore_API[1] = (void*)&R4Py_AgentIterType;
    R4PyCore_API[2] = (void*)&R4Py_PyObjectIterType;


    c_api_object = PyCapsule_New((void*)R4PyCore_API, "repast4py._core._C_API", NULL);

    if (PyModule_AddObject(m, "_C_API", c_api_object) < 0) {
        Py_XDECREF(c_api_object);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&R4Py_AgentType);
    if (PyModule_AddObject(m, "Agent", (PyObject *) &R4Py_AgentType) < 0) {
        Py_XDECREF(c_api_object);
        Py_DECREF(&R4Py_AgentType);
        Py_DECREF(m);
        return NULL;
    }

    // TODO better pattern for cleaning up when things like this 
    // fail
    Py_INCREF(&R4Py_AgentIterType);
    if (PyModule_AddObject(m, "AgentIterator", (PyObject *) &R4Py_AgentIterType) < 0) {
        Py_XDECREF(c_api_object);
        Py_DECREF(&R4Py_AgentType);
        Py_DECREF(&R4Py_AgentIterType);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&R4Py_PyObjectIterType);
    if (PyModule_AddObject(m, "PyObjectIterator", (PyObject *) &R4Py_PyObjectIterType) < 0) {
        Py_XDECREF(c_api_object);
        Py_DECREF(&R4Py_AgentType);
        Py_DECREF(&R4Py_AgentIterType);
        Py_DECREF(&R4Py_PyObjectIterType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}