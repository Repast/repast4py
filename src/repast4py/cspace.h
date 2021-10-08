// Copyright 2021, UChicago Argonne, LLC 
// All Rights Reserved
// Software Name: repast4py
// By: Argonne National Laboratory
// License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

#ifndef R4PY_SRC_CSPACE_H
#define R4PY_SRC_CSPACE_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "space.h"
#include "SpatialTree.h"

namespace repast4py {

template<typename AccessorType, typename BorderType>
class BaseCSpace : public BaseSpace<R4Py_ContinuousPoint, AccessorType, BorderType> {

using BaseSpace<R4Py_ContinuousPoint, AccessorType, BorderType>::agent_map;
using BaseSpace<R4Py_ContinuousPoint, AccessorType, BorderType>::location_map;
using BaseSpace<R4Py_ContinuousPoint, AccessorType, BorderType>::borders;
using BaseSpace<R4Py_ContinuousPoint, AccessorType, BorderType>::accessor;
using BaseSpace<R4Py_ContinuousPoint, AccessorType, BorderType>::wpt;
using BaseSpace<R4Py_ContinuousPoint, AccessorType, BorderType>::name_;

private:
    std::unique_ptr<CPSpatialTree> spatial_tree;


public:
    using PointType = R4Py_ContinuousPoint;
    BaseCSpace(const std::string& name, const BoundingBox& bounds, int tree_threshold);
    ~BaseCSpace();

    void getAgentsWithin(const BoundingBox& bounds, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) override;
    virtual bool remove(R4Py_Agent* agent) override;
    virtual bool remove(R4Py_AgentID* aid) override;
    R4Py_ContinuousPoint* move(R4Py_Agent* agent, R4Py_ContinuousPoint* to) override;
};

template<typename AccessorType, typename BorderType>
BaseCSpace<AccessorType, BorderType>::BaseCSpace(const std::string& name, const BoundingBox& bounds, int tree_threshold) : 
    BaseSpace<R4Py_ContinuousPoint, AccessorType, BorderType>(name, bounds), spatial_tree{} {

    if (bounds.num_dims == 1) {
        // TODO 
    } else if (bounds.num_dims == 2) {
        spatial_tree = std::unique_ptr<CPSpatialTree>(new CPSpatialTreeImpl<SpatialTree<Box2D, R4Py_ContinuousPoint>>(tree_threshold, bounds));
    } else if (bounds.num_dims == 3) {
        spatial_tree = std::unique_ptr<CPSpatialTree>(new CPSpatialTreeImpl<SpatialTree<Box3D, R4Py_ContinuousPoint>>(tree_threshold, bounds));
    }
}

template<typename AccessorType, typename BorderType>
BaseCSpace<AccessorType, BorderType>::~BaseCSpace() {}

template<typename AccessorType, typename BorderType>
void BaseCSpace<AccessorType, BorderType>::getAgentsWithin(const BoundingBox& bounds, 
    std::shared_ptr<std::vector<R4Py_Agent*>>& agents) 
{
    spatial_tree->getObjectsWithin(bounds, agents);
}

template<typename AccessorType, typename BorderType>
bool BaseCSpace<AccessorType, BorderType>::remove(R4Py_Agent* agent) {
    return remove(agent->aid);
}

template<typename AccessorType, typename BorderType>
bool BaseCSpace<AccessorType, BorderType>::remove(R4Py_AgentID* aid) {
    auto iter = agent_map.find(aid);
    if (iter != agent_map.end() && iter->second->pt) {
        spatial_tree->removeItem(iter->second);
    }
    return BaseSpace<R4Py_ContinuousPoint, AccessorType, BorderType>::remove(aid);
}

template<typename AccessorType, typename BorderType>
R4Py_ContinuousPoint* BaseCSpace<AccessorType, BorderType>::move(R4Py_Agent* agent, R4Py_ContinuousPoint* pt) {
    // If this gets changed such that the argument pt is not a temp input arg then 
    // we need to make sure that any move calls reflect that. 
    auto iter = agent_map.find(agent->aid);
    if (iter != agent_map.end()) {
        borders.transform(pt, wpt);
        if (!point_equals(iter->second->pt, wpt)) {
            if (accessor.put(agent, location_map, wpt)) {
                if (iter->second->pt) {
                    spatial_tree->removeItem(iter->second);
                    // if successful put, and agent is already located 
                    // so need to remove
                    Point<R4Py_ContinuousPoint> ppt;
                    extract_coords(iter->second->pt, ppt);
                    accessor.remove(agent, location_map, ppt);
                    update_point(iter->second->pt, wpt);
                    spatial_tree->addItem(iter->second);
                }  else {
                    iter->second->pt = create_point(Py_TYPE(pt), wpt);
                    spatial_tree->addItem(iter->second);
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

class ICSpace {

public:
    virtual ~ICSpace() = 0;

    virtual bool add(R4Py_Agent* agent) = 0;
    virtual bool remove(R4Py_Agent* agent) = 0;
    virtual bool remove(R4Py_AgentID* aid) = 0;
    virtual R4Py_Agent* getAgentAt(R4Py_ContinuousPoint* pt) = 0;
    virtual AgentList getAgentsAt(R4Py_ContinuousPoint* pt) = 0;
    virtual R4Py_ContinuousPoint* getLocation(R4Py_Agent* agent) = 0;
    virtual R4Py_ContinuousPoint* move(R4Py_Agent* agent, R4Py_ContinuousPoint* to) = 0;
    virtual void getAgentsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) = 0;
    virtual const std::string name() const = 0;
    virtual bool contains(R4Py_Agent* agent) const = 0;
};

inline ICSpace::~ICSpace() {}

template<typename DelegateType>
class CSpace : public ICSpace {

private:
    std::unique_ptr<DelegateType> delegate;

public:
    CSpace(const std::string& name, const BoundingBox& bounds, int tree_threshold);
    virtual ~CSpace() {}
    bool add(R4Py_Agent* agent) override;
    bool remove(R4Py_Agent* agent) override;
    bool remove(R4Py_AgentID* aid) override;
    R4Py_Agent* getAgentAt(R4Py_ContinuousPoint* pt) override;
    AgentList getAgentsAt(R4Py_ContinuousPoint* pt) override;
    R4Py_ContinuousPoint* getLocation(R4Py_Agent* agent) override;
    R4Py_ContinuousPoint* move(R4Py_Agent* agent, R4Py_ContinuousPoint* to) override;
    void getAgentsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) override;
    const std::string name() const override;
    bool contains(R4Py_Agent* agent) const override;
};

template<typename DelegateType>
CSpace<DelegateType>::CSpace(const std::string& name, const BoundingBox& bounds, int tree_threshold) : 
    delegate{std::unique_ptr<DelegateType>(new DelegateType(name, bounds, tree_threshold))} {}

template<typename DelegateType>
bool CSpace<DelegateType>::add(R4Py_Agent* agent) {
    return delegate->add(agent);
}

template<typename DelegateType>
bool CSpace<DelegateType>::remove(R4Py_Agent* agent) {
    return delegate->remove(agent);
}

template<typename DelegateType>
bool CSpace<DelegateType>::remove(R4Py_AgentID* aid) {
    return delegate->remove(aid);
}

template<typename DelegateType>
R4Py_Agent* CSpace<DelegateType>::getAgentAt(R4Py_ContinuousPoint* pt) {
    return delegate->getAgentAt(pt);
}

template<typename DelegateType>
AgentList CSpace<DelegateType>::getAgentsAt(R4Py_ContinuousPoint* pt) {
    return delegate->getAgentsAt(pt);
}

template<typename DelegateType>
R4Py_ContinuousPoint* CSpace<DelegateType>::getLocation(R4Py_Agent* agent) {
    return delegate->getLocation(agent);
}

template<typename DelegateType>
R4Py_ContinuousPoint* CSpace<DelegateType>::move(R4Py_Agent* agent, R4Py_ContinuousPoint* to) {
    return delegate->move(agent, to);
}

template<typename DelegateType>
void CSpace<DelegateType>::getAgentsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) {
    delegate->getAgentsWithin(box, agents);
}

template<typename DelegateType>
const std::string CSpace<DelegateType>::name() const {
    return delegate->name();
}

template<typename DelegateType>
bool CSpace<DelegateType>::contains(R4Py_Agent* agent) const {
    return delegate->contains(agent);
}

// aliases for  CSpace with multi occupancy and sticky borders
using ContinuousMOType = MultiOccupancyAccessor<LocationMapType<R4Py_ContinuousPoint, AgentList>, R4Py_ContinuousPoint>;
using ContinuousSOType = SingleOccupancyAccessor<LocationMapType<R4Py_ContinuousPoint, R4Py_Agent*>, R4Py_ContinuousPoint>;
using MOSCSpace = BaseCSpace<ContinuousMOType, CSStickyBorders>;
using MOPCSpace = BaseCSpace<ContinuousMOType, CSPeriodicBorders>;
using SOSCSpace = BaseCSpace<ContinuousSOType, CSStickyBorders>;
using SOPCSpace = BaseCSpace<ContinuousSOType, CSPeriodicBorders>;


template<>
struct is_periodic<MOSCSpace> {
    static constexpr bool value {false};
};

template<>
struct is_periodic<MOPCSpace> {
    static constexpr bool value {true};
};

template<>
struct is_periodic<SOSCSpace> {
    static constexpr bool value {false};
};

template<>
struct is_periodic<SOPCSpace> {
    static constexpr bool value {true};
};

struct R4Py_CSpace {
    PyObject_HEAD
    ICSpace* space;
};


}

#endif