#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define R4PY_CORE_MODULE
#include "coremodule.h"

#include <new>
#include "structmember.h"

#include "core.h"

using namespace repast4py;


static void Agent_dealloc(Agent* self) {
    delete self->aid;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Agent_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    Agent* self;
    self = (Agent*) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->aid = new (std::nothrow) AgentID;
        if (self->aid) {
            self->aid->id = -1;
            self->aid->type = -1;
        } else {
            Py_TYPE(self)->tp_free((PyObject*)self);
            self = NULL;
        }
    }
    return (PyObject*) self;
}

static int Agent_init(Agent* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {(char*)"id", (char*)"type", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "li", kwlist, &self->aid->id, &self->aid->type)) {
        return -1;
    }
    return 0;
}

static PyObject* Agent_get_id(Agent* self, void* closure) {
    return PyLong_FromLong(self->aid->id);
}

static PyObject* Agent_get_type(Agent* self, void* closure) {
    return PyLong_FromLong(self->aid->type);
}

static PyGetSetDef Agent_get_setters[] = {
    {(char*)"id", (getter)Agent_get_id, NULL, (char*)"agent id", NULL},
    {(char*)"type", (getter)Agent_get_type, NULL, (char*)"agent type", NULL},
    {NULL}
};

static PyObject* Agent_repr(Agent* self) {
    return PyUnicode_FromFormat("Agent(%ld, %d)", self->aid->id, self->aid->type);
}


static PyTypeObject AgentType = {
    PyVarObject_HEAD_INIT(NULL, 0) 
    "core.Agent",                          /* tp_name */
    sizeof(Agent),                      /* tp_basicsize */
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

    if (PyType_Ready(&AgentType) < 0)
        return NULL;

    PyObject *m;
    m = PyModule_Create(&coremodule);
    if (m == NULL)
        return NULL;

    static void* R4PyCore_API[R4PyCore_API_pointers];
    PyObject* c_api_object;

    R4PyCore_API[0] = (void*)&AgentType;
    c_api_object = PyCapsule_New((void*)R4PyCore_API, "repast4py.core._C_API", NULL);

    if (PyModule_AddObject(m, "_C_API", c_api_object) < 0) {
        Py_XDECREF(c_api_object);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&AgentType);
    if (PyModule_AddObject(m, "Agent", (PyObject *) &AgentType) < 0) {
        Py_DECREF(&AgentType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}