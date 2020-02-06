#ifndef R4PY_SRC_OCCUPANCY_H
#define R4PY_SRC_OCCUPANCY_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <map>
#include <list>
#include <memory>

#define NO_IMPORT_ARRAY_API
#define PY_ARRAY_UNIQUE_SYMBOL REPAST4PY_ARRAY_API
#include "numpy/arrayobject.h"

#include "geometry.h"
#include "core.h"

namespace repast4py {

using AgentList = std::shared_ptr<std::list<R4Py_Agent*>>;

template<typename MapType, typename PointType>
class MultiOccupancyAccessor {

private:
    static AgentList empty_list;

public:
    MultiOccupancyAccessor();
    ~MultiOccupancyAccessor() {}

    R4Py_Agent* get(MapType& location_map, const Point<PointType>& pt);
    size_t size(MapType& location_map, const Point<PointType>& pt);
    AgentList getAll(MapType& location_map, const Point<PointType>& pt);
    bool put(R4Py_Agent* agent, MapType& location_map, const Point<PointType>& pt);
    bool remove(R4Py_Agent* agent, MapType& location_map, const Point<PointType>& pt);
};

template<typename MapType, typename PointType>
AgentList MultiOccupancyAccessor<MapType, PointType>::empty_list{std::make_shared<std::list<R4Py_Agent*>>()};

template<typename MapType, typename PointType>
MultiOccupancyAccessor<MapType, PointType>::MultiOccupancyAccessor() {}

template<typename MapType, typename PointType>
R4Py_Agent* MultiOccupancyAccessor<MapType, PointType>::get(MapType& location_map, const Point<PointType>& pt) {
    auto iter = location_map.find(pt);
    if (iter == location_map.end() || iter->second->size() == 0) {
        return nullptr;
    }

    return iter->second->front();
}


template<typename MapType, typename PointType>
size_t MultiOccupancyAccessor<MapType, PointType>::size(MapType& location_map, const Point<PointType>& pt) {
    auto iter = location_map.find(pt);
    if (iter == location_map.end()) return 0;
    return iter->second->size();
}


template<typename MapType, typename PointType>
AgentList MultiOccupancyAccessor<MapType, PointType>::getAll(MapType& location_map, const Point<PointType>& pt) {
    auto iter = location_map.find(pt);
    if (iter == location_map.end()) {
        return MultiOccupancyAccessor::empty_list;
    }

    return iter->second;
}

template<typename MapType, typename PointType>
bool MultiOccupancyAccessor<MapType, PointType>::put(R4Py_Agent* agent, MapType& location_map, const Point<PointType>& pt) {
    auto iter = location_map.find(pt);
    if (iter == location_map.end()) {
        auto l = std::make_shared<std::list<R4Py_Agent*>>();
        l->push_back(agent);
        location_map.emplace(pt, l);
    } else {
        iter->second->push_back(agent);
    }
    Py_INCREF(agent);
    return true;
}

template<typename MapType, typename PointType>
bool MultiOccupancyAccessor<MapType, PointType>::remove(R4Py_Agent* agent, MapType& location_map, const Point<PointType>& pt) {
    auto iter = location_map.find(pt);
    if (iter == location_map.end()) {
        return false;
    }

    auto agent_iter = std::find(iter->second->begin(), iter->second->end(), agent);
    if (agent_iter == iter->second->end()) {
        return false;
    }

    iter->second->erase(agent_iter);
    Py_DECREF(agent);
    if (iter->second->size() == 0) {
        location_map.erase(iter);
    }
    return true;
}

}

#endif