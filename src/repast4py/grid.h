#ifndef R4PY_SRC_GRID_H
#define R4PY_SRC_GRID_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>


#include "space.h"

namespace repast4py {

class IGrid {

public:
    virtual ~IGrid() = 0;

    virtual bool add(R4Py_Agent* agent) = 0;
    virtual bool remove(R4Py_Agent* agent) = 0;
    virtual bool remove(R4Py_AgentID* aid) = 0;
    virtual R4Py_Agent* getAgentAt(R4Py_DiscretePoint* pt) = 0;
    virtual AgentList getAgentsAt(R4Py_DiscretePoint* pt) = 0;
    virtual R4Py_DiscretePoint* getLocation(R4Py_Agent* agent) = 0;
    virtual R4Py_DiscretePoint* move(R4Py_Agent* agent, R4Py_DiscretePoint* to) = 0;
};

inline IGrid::~IGrid() {}

template<typename DelegateType>
class Grid : public IGrid {

private:
    std::unique_ptr<DelegateType> delegate;

public:
    Grid(const std::string& name, const BoundingBox<R4Py_DiscretePoint>& bounds);
    virtual ~Grid() {}
    bool add(R4Py_Agent* agent) override;
    bool remove(R4Py_Agent* agent) override;
    bool remove(R4Py_AgentID* aid) override;
    R4Py_Agent* getAgentAt(R4Py_DiscretePoint* pt) override;
    AgentList getAgentsAt(R4Py_DiscretePoint* pt) override;
    R4Py_DiscretePoint* getLocation(R4Py_Agent* agent) override;
    R4Py_DiscretePoint* move(R4Py_Agent* agent, R4Py_DiscretePoint* to) override;
};

template<typename DelegateType>
Grid<DelegateType>::Grid(const std::string& name, const BoundingBox<R4Py_DiscretePoint>& bounds) : 
    delegate{std::unique_ptr<DelegateType>(new DelegateType(name, bounds))} {}

template<typename DelegateType>
bool Grid<DelegateType>::add(R4Py_Agent* agent) {
    return delegate->add(agent);
}

template<typename DelegateType>
bool Grid<DelegateType>::remove(R4Py_Agent* agent) {
    return delegate->remove(agent);
}

template<typename DelegateType>
bool Grid<DelegateType>::remove(R4Py_AgentID* aid) {
    return delegate->remove(aid);
}

template<typename DelegateType>
R4Py_Agent* Grid<DelegateType>::getAgentAt(R4Py_DiscretePoint* pt) {
    return delegate->getAgentAt(pt);
}

template<typename DelegateType>
AgentList Grid<DelegateType>::getAgentsAt(R4Py_DiscretePoint* pt) {
    return delegate->getAgentsAt(pt);
}

template<typename DelegateType>
R4Py_DiscretePoint* Grid<DelegateType>::getLocation(R4Py_Agent* agent) {
    return delegate->getLocation(agent);
}

template<typename DelegateType>
R4Py_DiscretePoint* Grid<DelegateType>::move(R4Py_Agent* agent, R4Py_DiscretePoint* to) {
    return delegate->move(agent, to);
}

// typedefs for Discrete Grid with multi occupancy and sticky borders
using DiscreteMOType = MultiOccupancyAccessor<LocationMapType<R4Py_DiscretePoint>, R4Py_DiscretePoint>;
using MOSGrid = BaseSpace<R4Py_DiscretePoint, DiscreteMOType, GridStickyBorders>;
using MOPGrid = BaseSpace<R4Py_DiscretePoint, DiscreteMOType, GridPeriodicBorders>;

template<>
struct is_periodic<MOSGrid> {
    static constexpr bool value {false};
};

template<>
struct is_periodic<MOPGrid> {
    static constexpr bool value {true};
};

struct R4Py_Grid {
    PyObject_HEAD
    IGrid* grid;
};


}

#endif