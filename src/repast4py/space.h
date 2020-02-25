#ifndef R4PY_SRC_SPACE_H
#define R4PY_SRC_SPACE_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <map>
#include <list>
#include <memory>
#include <algorithm>

#include "geometry.h"
#include "occupancy.h"
#include "core.h"

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
struct SpaceItem {
    PointType* pt;
    R4Py_Agent* agent;
};


template<typename PointType>
using AgentMapType = std::map<R4Py_AgentID*, std::shared_ptr<SpaceItem<PointType>>, agent_id_comp>;

template<typename PointType>
using LocationMapType = std::map<Point<PointType>, AgentList, PointComp<PointType>>;

template<typename PointType, typename AccessorType, typename BorderType>
class BaseSpace {

private:
    AgentMapType<PointType> agent_map;
    LocationMapType<PointType> location_map;
    AccessorType accessor;
    BorderType borders;
    Point<PointType> wpt;
    std::string name_;

public:
    BaseSpace(const std::string& name, const BoundingBox<PointType>& bounds);
    virtual ~BaseSpace();

    bool add(R4Py_Agent* agent);
    bool remove(R4Py_Agent* agent);
    R4Py_Agent* getAgentAt(PointType* pt);
    AgentList getAgentsAt(PointType* pt);
    PointType* getLocation(R4Py_Agent* agent);
    PointType* move(R4Py_Agent* agent, PointType* to);


};

template<typename PointType, typename AccessorType, typename BorderType>
BaseSpace<PointType,AccessorType, BorderType>::BaseSpace(const std::string& name, const BoundingBox<PointType>& bounds) : 
    agent_map{}, location_map{}, accessor{}, borders{bounds}, wpt{0, 0, 0}, name_{name} {}

template<typename PointType, typename AccessorType, typename BorderType>
BaseSpace<PointType, AccessorType,  BorderType>::~BaseSpace() {
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

template<typename PointType, typename AccessorType, typename BorderType>
bool BaseSpace<PointType, AccessorType,  BorderType>::add(R4Py_Agent* agent) {
    auto item = std::make_shared<SpaceItem<PointType>>();
    item->agent = agent;
    Py_INCREF(agent);
    item->pt = nullptr;
    agent_map[agent->aid] = item;
    // TODO use simphony style adder to set location / or not
    return true;
}

template<typename PointType, typename AccessorType, typename BorderType>
bool BaseSpace<PointType, AccessorType,  BorderType>::remove(R4Py_Agent* agent) {
    auto iter = agent_map.find(agent->aid);
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
R4Py_Agent* BaseSpace<PointType, AccessorType,  BorderType>::getAgentAt(PointType* pt) {
    extract_coords(pt, wpt);
    return accessor.get(location_map, wpt);
}

template<typename PointType, typename AccessorType, typename BorderType>
AgentList BaseSpace<PointType, AccessorType,  BorderType>::getAgentsAt(PointType* pt) {
    extract_coords(pt, wpt);
    return accessor.getAll(location_map, wpt);
}

template<typename PointType, typename AccessorType, typename BorderType>
PointType* BaseSpace<PointType, AccessorType,  BorderType>::getLocation(R4Py_Agent* agent) {
    auto iter = agent_map.find(agent->aid);
    if (iter != agent_map.end()) {
        return iter->second->pt;
    }
    return nullptr;
}

template<typename PointType, typename AccessorType, typename BorderType>
PointType* BaseSpace<PointType, AccessorType,  BorderType>::move(R4Py_Agent* agent, PointType* pt) {
    auto iter = agent_map.find(agent->aid);
    if (iter != agent_map.end()) {
        borders.transform(pt, wpt);
        if (!point_equals(iter->second->pt, wpt)) {
            if (accessor.put(agent, location_map, wpt)) {
                if (iter->second->pt) {
                    // if successful put, and agent is already located 
                    // so need to remove
                    Point<PointType> ppt;
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
using DiscreteSBType = StickyBorders<R4Py_DiscretePoint>;
using MOSGrid = BaseSpace<R4Py_DiscretePoint, DiscreteMOType, DiscreteSBType>;


template<typename BorderType>
struct is_periodic {
    static constexpr bool value {false};
};

template<>
struct is_periodic<MOSGrid> {
    static constexpr bool value {false};
};

struct R4Py_Grid {
    PyObject_HEAD
    IGrid* grid;
};

}

#endif