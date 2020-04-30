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