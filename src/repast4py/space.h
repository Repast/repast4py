#ifndef R4PY_SRC_SPACE_H
#define R4PY_SRC_SPACE_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <map>
#include <list>
#include <memory>
#include <algorithm>


#define NO_IMPORT_ARRAY_API
#define PY_ARRAY_UNIQUE_SYMBOL REPAST4PY_ARRAY_API
#include "numpy/arrayobject.h"

#include "core.h"

namespace repast4py {

// see https://docs.scipy.org/doc/numpy/reference/c-api.array.html?highlight=impor#c.import_array
// import_array() call in the add to module part

struct DiscretePoint {
    PyObject_HEAD
    PyArrayObject* coords;
};


template<typename PointType>
struct TypeSelector {
    using type = double;
};

template<>
struct TypeSelector<DiscretePoint> {
    using type = long;
};

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
struct Point {
    using coord_type  = typename TypeSelector<PointType>::type;
    coord_type x, y, z;
};

DiscretePoint* create_point(PyTypeObject* pt_type, const Point<DiscretePoint>& wpt) {
    if (pt_type == NULL) {
        return NULL;
    }
    PyObject* obj = PyObject_CallFunction((PyObject*)pt_type, "lll", wpt.x, wpt.y, wpt.z);
    return (DiscretePoint*)obj;
}

template<typename PointType>
struct PointComp {
    bool operator()(const Point<PointType>& p1, const Point<PointType>& p2) {
        if (p1.x != p2.x) return p1.x < p2.x;
        if (p1.y != p2.y) return p1.y < p2.y;
        return p1.z < p2.z;
    }
};


template<typename PointType>
struct SpaceItem {
    PointType* pt;
    Agent* agent;
};


void extract_coords(DiscretePoint* pt, Point<DiscretePoint>& coords) {
    long* data = (long*)pt->coords->data;
    coords.x = data[0];
    coords.y = data[1];
    coords.z = data[2];
}

void update_point(DiscretePoint* pt, const Point<DiscretePoint>& coords) {
    long* data = (long*)pt->coords->data;
    data[0] = coords.x;
    data[1] = coords.y;
    data[2] = coords.z;
}

bool point_equals(DiscretePoint* pt, const Point<DiscretePoint>& coords) {
    if (pt) {
        long* data = (long*)pt->coords->data;
        return data[0] == coords.x && data[1] == coords.y && data[2] == coords.z;
    }
    return false;
}

using AgentList = std::shared_ptr<std::list<Agent*>>;

template<typename MapType, typename PointType>
class MultiOccupancyAccessor {

private:
    AgentList empty_list;

public:

    MultiOccupancyAccessor();
    ~MultiOccupancyAccessor() {}

    Agent* get(MapType& location_map, const Point<PointType>& pt);
    size_t size(MapType& location_map, const Point<PointType>& pt);
    AgentList getAll(MapType& location_map, const Point<PointType>& pt);
    bool put(Agent* agent, MapType& location_map, const Point<PointType>& pt);
    bool remove(Agent* agent, MapType& location_map, const Point<PointType>& pt);
};

template<typename MapType, typename PointType>
MultiOccupancyAccessor<MapType, PointType>::MultiOccupancyAccessor() : empty_list{} {
    empty_list = std::make_shared<std::list<Agent*>>();
}

template<typename MapType, typename PointType>
Agent* MultiOccupancyAccessor<MapType, PointType>::get(MapType& location_map, const Point<PointType>& pt) {
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
        return empty_list;
    }

    return iter->second;
}

template<typename MapType, typename PointType>
bool MultiOccupancyAccessor<MapType, PointType>::put(Agent* agent, MapType& location_map, const Point<PointType>& pt) {
    auto iter = location_map.find(pt);
    if (iter == location_map.end()) {
        location_map[pt] = std::make_shared<std::list<Agent*>>(std::initializer_list<Agent*>{agent});
    } else {
        iter->second->push_back(agent);
    }
    Py_INCREF(agent);
    return true;
}

template<typename MapType, typename PointType>
bool MultiOccupancyAccessor<MapType, PointType>::remove(Agent* agent, MapType& location_map, const Point<PointType>& pt) {
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

template<typename PointType>
using AgentMapType = std::map<AgentID*, std::shared_ptr<SpaceItem<PointType>>, agent_id_comp>;

template<typename PointType>
using LocationMapType = std::map<Point<PointType>, AgentList, PointComp<PointType>>;

template<typename PointType, typename Accessor>
class BaseSpace {

private:
    AgentMapType<PointType> agent_map;
    LocationMapType<PointType> location_map;
    Accessor accessor;
    Point<PointType> wpt;
    std::string name_;

public:
    BaseSpace(const std::string& name);
    virtual ~BaseSpace();

    bool addAgent(Agent* agent);
    bool removeAgent(Agent* agent);
    Agent* getAgentAt(PointType* pt);
    // TODO 
    // getObjectsAt();
    PointType* getLocation(Agent* agent);
    PointType* move(Agent* agent, PointType* to);
};

template<typename PointType, typename Accessor>
BaseSpace<PointType, Accessor>::BaseSpace(const std::string& name) : agent_map{}, location_map{}, accessor{},
    wpt{0, 0, 0}, name_{name} {}

template<typename PointType, typename Accessor>
BaseSpace<PointType, Accessor>::~BaseSpace() {
    for (auto kv : agent_map) {
        // pt may be null if agent added but never
        // moved
        Py_XDECREF(kv.second->pt);
        Py_DECREF(kv.second->agent);
    }

    for (auto kv : location_map) {
        for (auto iter = kv.second->begin(); iter != kv.second->end(); ++iter)  {
            Py_DECREF(*iter);
        }
    }
    agent_map.clear();
    location_map.clear();
}

template<typename PointType, typename Accessor>
bool BaseSpace<PointType, Accessor>::addAgent(Agent* agent) {
    auto item = std::make_shared<SpaceItem<PointType>>();
    item->agent = agent;
    Py_INCREF(agent);
    item->pt = nullptr;
    agent_map[agent->aid] = item;
    // TODO use simphony style adder to set location / or not
    return true;
}

template<typename PointType, typename Accessor>
bool BaseSpace<PointType, Accessor>::removeAgent(Agent* agent) {
    auto iter = agent_map.find(agent->aid);
    bool ret_val = false;
    if (iter != agent_map.end()) {
        if (iter->pt) {
            extract_coords(iter->pt, wpt);
            ret_val = accessor.remove(iter->agent, location_map, wpt);
        } 
        Py_XDECREF(iter->pt);
        Py_DECREF(iter->agent);
        agent_map.erase(iter);
    }
    return ret_val;
}

template<typename PointType, typename Accessor>
Agent* BaseSpace<PointType, Accessor>::getAgentAt(PointType* pt) {
    extract_coords(pt, wpt);
    return accessor.get(location_map, wpt);
}


template<typename PointType, typename Accessor>
PointType* BaseSpace<PointType, Accessor>::getLocation(Agent* agent) {
    auto iter = agent_map.find(agent->aid);
    if (iter != agent_map.end()) {
        return iter->second->pt;
    }
    return nullptr;
}

template<typename PointType, typename Accessor>
PointType* BaseSpace<PointType, Accessor>::move(Agent* agent, PointType* pt) {
    auto iter = agent_map.find(agent->aid);
    if (iter != agent_map.end()) {
        // TODO transform with borders and don't use 
        // point type in borders -- just the wpt
        // border.transform(pt, wpt);
        // but for now just this
        extract_coords(pt, wpt);
        if (!point_equals(iter->second->pt, wpt)) {
            if (accessor.put(agent, location_map, wpt)) {
                if (iter->second->pt) {
                    // if successful put, and agent is already located 
                    // so need to remove
                    extract_coords(iter->second->pt, wpt);
                    accessor.remove(agent, location_map, wpt);
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
        AgentID* id = agent->aid;
        throw std::invalid_argument("Error moving agent (" + std::to_string(id->id) + "," + 
            std::to_string(id->type) + "): agent is not in " + name_);
    }
}

using Grid = BaseSpace<DiscretePoint, MultiOccupancyAccessor<LocationMapType<DiscretePoint>, DiscretePoint>>;


struct R4Py_Grid {
    PyObject_HEAD
    Grid* grid;
};



}

#endif