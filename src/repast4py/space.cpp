// Copyright 2021, UChicago Argonne, LLC 
// All Rights Reserved
// Software Name: repast4py
// By: Argonne National Laboratory
// License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

#include "space.h"

namespace repast4py {

void decref(AgentList& agent_list) {
    for (auto iter = agent_list->begin(); iter != agent_list->end(); ++iter)  {
        Py_DECREF(*iter);
    }
}

void decref(R4Py_Agent* agent) {
    Py_DECREF(agent);
}

}