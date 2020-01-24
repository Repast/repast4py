#ifndef R4PY_SRC_SPACE_H
#define R4PY_SRC_SPACE_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <map>

#include "numpy/arrayobject.h"

#include "core.h"

namespace repast4py {

// see https://docs.scipy.org/doc/numpy/reference/c-api.array.html?highlight=impor#c.import_array
// import_array() call in the add to module part

struct SpaceItem {
    PyArrayObject* pt;
    Agent* agent;
};

class BaseSpace {

private:
    std::map<AgentID*, SpaceItem*, agent_id_comp> agent_map;

};

}

#endif