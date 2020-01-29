#ifndef R4PY_SRC_CORE_H
#define R4PY_SRC_CORE_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

namespace repast4py {

typedef struct {
    long id;
    int type;
} AgentID;

struct agent_id_comp {
    bool operator()(const AgentID* a1, const AgentID* a2) const {
        if (a1->id == a2->id) {
            return a1->type < a2->type;
        }
        return a1->id < a2->id;
    }
};

typedef struct {
    PyObject_HEAD
    AgentID* aid;
} Agent;

}

#endif