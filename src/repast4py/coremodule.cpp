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
    "core.AgentIterator",                          /* tp_name */
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


//////////////////// Agent ///////////////////
static void Agent_dealloc(R4Py_Agent* self) {
    delete self->aid;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Agent_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    R4Py_Agent* self;
    self = (R4Py_Agent*) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->aid = new R4Py_AgentID;
        if (self->aid) {
            self->aid->id = -1;
            self->aid->type = -1;
            self->aid->rank = 0;
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
    return 0;
}

static PyObject* Agent_get_id(R4Py_Agent* self, void* closure) {
    return PyLong_FromLong(self->aid->id);
}

static PyObject* Agent_get_rank(R4Py_Agent* self, void* closure) {
    return PyLong_FromLong(self->aid->rank);
}

static PyObject* Agent_get_type(R4Py_Agent* self, void* closure) {
    return PyLong_FromLong(self->aid->type);
}

static PyObject* Agent_get_aid(R4Py_Agent* self, void* closure) {
    return Py_BuildValue("(liI)", self->aid->id, self->aid->type, self->aid->rank);
}

static PyGetSetDef Agent_get_setters[] = {
    {(char*)"id", (getter)Agent_get_id, NULL, (char*)"agent id", NULL},
    {(char*)"type", (getter)Agent_get_type, NULL, (char*)"agent type", NULL},
    {(char*)"rank", (getter)Agent_get_rank, NULL, (char*)"agent rank", NULL},
    {(char*)"tag", (getter)Agent_get_aid, NULL, (char*)"agent identifier", NULL},
    {NULL}
};

static PyObject* Agent_repr(R4Py_Agent* self) {
    return PyUnicode_FromFormat("Agent(%ld, %d)", self->aid->id, self->aid->type);
}


static PyTypeObject R4Py_AgentType = {
    PyVarObject_HEAD_INIT(NULL, 0) 
    "core.Agent",                          /* tp_name */
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
    "Agent Object",                         /* tp_doc */
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
    .m_name = "repast4py.core",
    .m_doc = "Example module that creates an extension type.",
    .m_size = -1,
};



// PyMODINIT_FUNC adds "extern C" among other things
PyMODINIT_FUNC
PyInit_core(void)
{

    if (PyType_Ready(&R4Py_AgentType) < 0)
        return NULL;

    if (PyType_Ready(&R4Py_AgentIterType) < 0)
        return NULL;


    PyObject *m;
    m = PyModule_Create(&coremodule);
    if (m == NULL)
        return NULL;

    static void* R4PyCore_API[R4PyCore_API_pointers];
    PyObject* c_api_object;

    R4PyCore_API[0] = (void*)&R4Py_AgentType;
    R4PyCore_API[1] = (void*)&R4Py_AgentIterType;
    c_api_object = PyCapsule_New((void*)R4PyCore_API, "repast4py.core._C_API", NULL);

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

    return m;
}