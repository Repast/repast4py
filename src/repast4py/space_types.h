// Copyright 2021, UChicago Argonne, LLC 
// All Rights Reserved
// Software Name: repast4py
// By: Argonne National Laboratory
// License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

#ifndef SRC_SPACETYPES_H_
#define SRC_SPACETYPES_H_

#include "core.h"

namespace repast4py {

template<typename PointType>
struct SpaceItem {
    PointType* pt;
    R4Py_Agent* agent;
};


}




#endif