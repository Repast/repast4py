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
    if (!PyArg_ParseTuple(args, "O", &pt)) {
        return NULL;
    }

    d[0] = PyLong_AsLong(PyTuple_GET_ITEM(pt, 0));
    d[1] = PyLong_AsLong(PyTuple_GET_ITEM(pt, 1));
    d[2] = PyLong_AsLong(PyTuple_GET_ITEM(pt, 2));

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

    long xmin, width;
    long ymin, height;
    long zmin, depth;

    if (!PyArg_ParseTuple(bounds, "llllll", &xmin, &width, &ymin, &height, &zmin, &depth)) {
        return -1;
    }

    // Because we are holding a reference to the communicator
    // I think this is necessary
    Py_INCREF(py_comm);
    MPI_Comm* comm_p = PyMPIComm_Get(py_comm);

    if (border_type == 0) {
        if (occ_type == 0) {
            BoundingBox<R4Py_DiscretePoint> box(xmin, width, ymin, height, zmin, depth);
            self->grid = new SharedGrid<DistributedGrid<MOSGrid>>(name, box, buffer_size, 
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
    std::shared_ptr<std::map<R4Py_AgentID*, PyObject*>> oob = ((R4Py_SharedGrid*)self)->grid->getOOBData();
    R4Py_PyObjectIter* obj_iter = (R4Py_PyObjectIter*)R4Py_PyObjectIterType.tp_new(&R4Py_PyObjectIterType, NULL, NULL);
    obj_iter->iter = new ValueIter<std::map<R4Py_AgentID*, PyObject*>>(oob);
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
    BoundingBox<R4Py_DiscretePoint> bounds = ((R4Py_SharedGrid*)self)->grid->getLocalBounds();
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
    {"_cart_comm", T_OBJECT_EX, offsetof(R4Py_SharedGrid, cart_comm), READONLY, "The cartesian communicator for this shared grid"},
    {NULL}
};

static PyMethodDef SharedGrid_methods[] = {
    {"add", SharedGrid_add, METH_VARARGS, "Adds the specified agent to this shared grid projection"},
    {"remove", SharedGrid_remove, METH_VARARGS, "Removes the specified agent from this shared grid projection"},
    {"move", SharedGrid_move, METH_VARARGS, "Moves the specified agent to the specified location in this shared grid projection"},
    {"get_location", SharedGrid_getLocation, METH_VARARGS, "Gets the location of the specified agent in this shared grid projection"},
    {"get_agent", SharedGrid_getAgent, METH_VARARGS, "Gets the first agent at the specified location in this shared grid projection"},
    {"get_agents", SharedGrid_getAgents, METH_VARARGS, "Gets all the agents at the specified location in this shared grid projection"},
    {"_get_oob", SharedGrid_getOOBData, METH_VARARGS, "Gets the out of bounds data for any agents that are out of the local bounds in this shared grid projection"},
    {"_clear_oob", SharedGrid_clearOOBData, METH_VARARGS, "Clears the out of bounds data for any agents that are out of the local bounds in this shared grid projection"},
    {"get_local_bounds", SharedGrid_getLocalBounds, METH_VARARGS, "Gets the local bounds for this shared grid projection"},
    {"_synch_move", SharedGrid_synchMove, METH_VARARGS, "Moves the specified agent to the specified location in this shared grid projection as part of a movement synchronization"},
    {"_get_buffer_data", SharedGrid_getBufferData, METH_VARARGS, "Gets the buffer data for synchronizing neighboring buffers of this shared grid projetion"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject R4Py_ShareGridType = {
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
    0,                                        /* tp_getset */
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
    if (PyType_Ready(&R4Py_GridType) < 0) return NULL;
    if (PyType_Ready(&R4Py_ShareGridType) < 0) return NULL;


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

    Py_INCREF(&R4Py_ShareGridType);
    if (PyModule_AddObject(m, "SharedGrid", (PyObject*) &R4Py_ShareGridType) < 0) {
        Py_DECREF(&DiscretePointType);
        Py_DECREF(&R4Py_GridType);
        Py_DECREF(&R4Py_ShareGridType);
        Py_DECREF(m);
    }

    return m;
}