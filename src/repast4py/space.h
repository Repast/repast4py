// Copyright 2021, UChicago Argonne, LLC 
// All Rights Reserved
// Software Name: repast4py
// By: Argonne National Laboratory
// License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

#ifndef R4PY_SRC_SPACE_H
#define R4PY_SRC_SPACE_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <map>
#include <list>
#include <memory>
#include <algorithm>
#include <vector>

#include "geometry.h"
#include "borders.h"
#include "occupancy.h"
#include "core.h"
#include "space_types.h"

namespace repast4py {

// template<typename PointType>
// struct PointComp {
//     bool operator()(const PointType* p1, const PointType* p2) {
//         using coord_type = typename TypeSelector<PointType>::type;
//         coord_type* p1_data = (coord_type*)p1->coords->data;
//         coord_type* p2_data = (coord_type*)p2->coords->data;
//         if (p1_data[0] != p2_data[0]) return p1_data[0] < p2_data[0];
//         if (p1_data[1] != p2_data[1]) return p1_data[1] < p2_data[1];
//         return p1_data[2] < p2_data[2];
//     }
// };


template<typename PointType>
using AgentMapType = std::map<R4Py_AgentID*, std::shared_ptr<SpaceItem<PointType>>, agent_id_comp>;

template<typename PointType, typename ValueType>
using LocationMapType = std::map<Point<PointType>, ValueType, PointComp<PointType>>;

void decref(AgentList& agent_list);
void decref(R4Py_Agent* agent);

template<typename PointType, typename AccessorType, typename BorderType>
class BaseSpace {

protected:
    AgentMapType<PointType> agent_map;
    LocationMapType<PointType, typename AccessorType::ValType> location_map;
    AccessorType accessor;
    BorderType borders;
    Point<PointType> wpt;
    std::string name_;

public:
    BaseSpace(const std::string& name, const BoundingBox& bounds);
    virtual ~BaseSpace();

    bool add(R4Py_Agent* agent);
    virtual bool remove(R4Py_Agent* agent);
    virtual bool remove(R4Py_AgentID* aid);
    R4Py_Agent* getAgentAt(PointType* pt);
    AgentList getAgentsAt(PointType* pt);
    PointType* getLocation(R4Py_Agent* agent);
    virtual void getAgentsWithin(const BoundingBox& bounds, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) = 0;
    virtual PointType* move(R4Py_Agent* agent, PointType* to) = 0;
    const std::string name() const;
    bool contains(R4Py_Agent* agent) const;
};

template<typename PointType, typename AccessorType, typename BorderType>
BaseSpace<PointType,AccessorType, BorderType>::BaseSpace(const std::string& name, const BoundingBox& bounds) : 
    agent_map{}, location_map{}, accessor{}, borders{bounds}, wpt{0, 0, 0}, name_{name} {}

template<typename PointType, typename AccessorType, typename BorderType>
BaseSpace<PointType, AccessorType, BorderType>::~BaseSpace() {
    for (auto kv : agent_map) {
        // pt may be null if agent added but never
        // moved
        Py_XDECREF(kv.second->pt);
        Py_DECREF(kv.second->agent);
    }

    for (auto kv : location_map) {
        decref(kv.second);
    }
    agent_map.clear();
    location_map.clear();
}

template<typename PointType, typename AccessorType, typename BorderType>
bool BaseSpace<PointType, AccessorType, BorderType>::add(R4Py_Agent* agent) {
    auto item = std::make_shared<SpaceItem<PointType>>();
    item->agent = agent;
    Py_INCREF(agent);
    item->pt = nullptr;
    agent_map[agent->aid] = item;
    // TODO use simphony style adder to set location / or not
    return true;
}

template<typename PointType, typename AccessorType, typename BorderType>
bool BaseSpace<PointType, AccessorType, BorderType>::remove(R4Py_AgentID* aid) {
    auto iter = agent_map.find(aid);
    bool ret_val = false;
    if (iter != agent_map.end()) {
        if (iter->second->pt) {
            extract_coords(iter->second->pt, wpt);
            ret_val = accessor.remove(iter->second->agent, location_map, wpt);
        } 
        Py_XDECREF(iter->second->pt);
        Py_DECREF(iter->second->agent);
        agent_map.erase(iter);
    }
    return ret_val;
}

template<typename PointType, typename AccessorType, typename BorderType>
bool BaseSpace<PointType, AccessorType, BorderType>::remove(R4Py_Agent* agent) {
    return remove(agent->aid);
}

template<typename PointType, typename AccessorType, typename BorderType>
R4Py_Agent* BaseSpace<PointType, AccessorType, BorderType>::getAgentAt(PointType* pt) {
    extract_coords(pt, wpt);
    return accessor.get(location_map, wpt);
}

template<typename PointType, typename AccessorType, typename BorderType>
AgentList BaseSpace<PointType, AccessorType, BorderType>::getAgentsAt(PointType* pt) {
    extract_coords(pt, wpt);
    return accessor.getAll(location_map, wpt);
}

template<typename PointType, typename AccessorType, typename BorderType>
PointType* BaseSpace<PointType, AccessorType, BorderType>::getLocation(R4Py_Agent* agent) {
    auto iter = agent_map.find(agent->aid);
    if (iter != agent_map.end()) {
        return iter->second->pt;
    }
    return nullptr;
}

template<typename PointType, typename AccessorType, typename BorderType>
bool BaseSpace<PointType, AccessorType, BorderType>::contains(R4Py_Agent* agent) const {
    return agent_map.find(agent->aid) != agent_map.end();
}

template<typename PointType, typename AccessorType, typename BorderType>
const std::string BaseSpace<PointType, AccessorType, BorderType>::name() const {
    return name_;
}


template<typename BorderType>
struct is_periodic {
    static constexpr bool value {false};
};


}

#endif