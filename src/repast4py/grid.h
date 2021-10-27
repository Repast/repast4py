// Copyright 2021, UChicago Argonne, LLC 
// All Rights Reserved
// Software Name: repast4py
// By: Argonne National Laboratory
// License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

#ifndef R4PY_SRC_GRID_H
#define R4PY_SRC_GRID_H

#include <vector>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "space.h"

namespace repast4py {

template<typename AccessorType, typename BorderType>
class BaseGrid : public BaseSpace<R4Py_DiscretePoint, AccessorType, BorderType> {

using BaseSpace<R4Py_DiscretePoint, AccessorType, BorderType>::agent_map;
using BaseSpace<R4Py_DiscretePoint, AccessorType, BorderType>::location_map;
using BaseSpace<R4Py_DiscretePoint, AccessorType, BorderType>::borders;
using BaseSpace<R4Py_DiscretePoint, AccessorType, BorderType>::accessor;
using BaseSpace<R4Py_DiscretePoint, AccessorType, BorderType>::wpt;
using BaseSpace<R4Py_DiscretePoint, AccessorType, BorderType>::name_;

public:
    using PointType = R4Py_DiscretePoint;
    BaseGrid(const std::string& name, const BoundingBox& bounds);
    ~BaseGrid();

    void getAgentsWithin(const BoundingBox& bounds, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) override;
    R4Py_DiscretePoint* move(R4Py_Agent* agent, R4Py_DiscretePoint* to) override;
};

template<typename AccessorType, typename BorderType>
BaseGrid<AccessorType, BorderType>::BaseGrid(const std::string& name, const BoundingBox& bounds) : 
    BaseSpace<R4Py_DiscretePoint, AccessorType, BorderType>(name, bounds) {}

template<typename AccessorType, typename BorderType>
BaseGrid<AccessorType, BorderType>::~BaseGrid() {}

template<typename AccessorType, typename BorderType>
void BaseGrid<AccessorType, BorderType>::getAgentsWithin(const BoundingBox& bounds, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) {

}

template<typename AccessorType, typename BorderType>
R4Py_DiscretePoint* BaseGrid<AccessorType, BorderType>::move(R4Py_Agent* agent, R4Py_DiscretePoint* pt) {
    // If this gets changed such that the argument pt is not a temp input arg then 
    // we need to make sure that any move calls reflect that. 
    auto iter = agent_map.find(agent->aid);
    if (iter != agent_map.end()) {
        borders.transform(pt, wpt);
        if (!point_equals(iter->second->pt, wpt)) {
            if (accessor.put(agent, location_map, wpt)) {
                if (iter->second->pt) {
                    // if successful put, and agent is already located 
                    // so need to remove
                    Point<R4Py_DiscretePoint> ppt;
                    extract_coords(iter->second->pt, ppt);
                    accessor.remove(agent, location_map, ppt);
                    update_point(iter->second->pt, wpt);
                }  else {
                    iter->second->pt = create_point(Py_TYPE(pt), wpt);
                }
            } else {
                return nullptr;
            }
        }
        return iter->second->pt;

    } else {
        R4Py_AgentID* id = agent->aid;
        throw std::invalid_argument("Error moving agent (" + std::to_string(id->id) + "," + 
            std::to_string(id->type) + "): agent is not in " + name_);
    }
}

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
    virtual const std::string name() const = 0;
    virtual bool contains(R4Py_Agent* agent) const = 0;
};

inline IGrid::~IGrid() {}

template<typename DelegateType>
class Grid : public IGrid {

private:
    std::unique_ptr<DelegateType> delegate;

public:
    Grid(const std::string& name, const BoundingBox& bounds);
    virtual ~Grid() {}
    bool add(R4Py_Agent* agent) override;
    bool remove(R4Py_Agent* agent) override;
    bool remove(R4Py_AgentID* aid) override;
    R4Py_Agent* getAgentAt(R4Py_DiscretePoint* pt) override;
    AgentList getAgentsAt(R4Py_DiscretePoint* pt) override;
    R4Py_DiscretePoint* getLocation(R4Py_Agent* agent) override;
    R4Py_DiscretePoint* move(R4Py_Agent* agent, R4Py_DiscretePoint* to) override;
    const std::string name() const override;
    bool contains(R4Py_Agent* agent) const override;
};

template<typename DelegateType>
Grid<DelegateType>::Grid(const std::string& name, const BoundingBox& bounds) : 
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

template<typename DelegateType>
const std::string Grid<DelegateType>::name() const {
    return delegate->name();
}

template<typename DelegateType>
bool Grid<DelegateType>::contains(R4Py_Agent* agent) const {
    return delegate->contains(agent);
}


// typedefs for Discrete Grid with multi occupancy and sticky borders
using DiscreteMOType = MultiOccupancyAccessor<LocationMapType<R4Py_DiscretePoint, AgentList>, R4Py_DiscretePoint>;
using DiscreteSOType = SingleOccupancyAccessor<LocationMapType<R4Py_DiscretePoint, R4Py_Agent*>, R4Py_DiscretePoint>;
using MOSGrid = BaseGrid<DiscreteMOType, GridStickyBorders>;
using MOPGrid = BaseGrid<DiscreteMOType, GridPeriodicBorders>;
using SOSGrid = BaseGrid<DiscreteSOType, GridStickyBorders>;
using SOPGrid = BaseGrid<DiscreteSOType, GridPeriodicBorders>;

template<>
struct is_periodic<MOSGrid> {
    static constexpr bool value {false};
};

template<>
struct is_periodic<MOPGrid> {
    static constexpr bool value {true};
};

template<>
struct is_periodic<SOSGrid> {
    static constexpr bool value {false};
};

template<>
struct is_periodic<SOPGrid> {
    static constexpr bool value {true};
};


struct R4Py_Grid {
    PyObject_HEAD
    IGrid* grid;
};


}

#endif