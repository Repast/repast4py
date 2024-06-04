// Copyright 2021, UChicago Argonne, LLC 
// All Rights Reserved
// Software Name: repast4py
// By: Argonne National Laboratory
// License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

#ifndef SRC_SPATIALTREE_
#define SRC_SPATIALTREE_

#include <vector>
#include <map>
#include <limits>
#include <numeric>
#include <set>

#include "geometry.h"
#include "space_types.h"
#include "occupancy.h"

namespace repast4py {

template<typename PointType, typename ValueType>
using LocationMapType_ = std::map<Point<PointType>, ValueType, PointComp<PointType>>;
using MOMapType = LocationMapType_<R4Py_ContinuousPoint, AgentListPtr>;
using SOMapType = LocationMapType_<R4Py_ContinuousPoint, R4Py_Agent*>;

using ContinuousMOType_ = MultiOccupancyAccessor<MOMapType, R4Py_ContinuousPoint>;
using ContinuousSOType_ = SingleOccupancyAccessor<SOMapType, R4Py_ContinuousPoint>;

struct NodePoint {
    double x_, y_, z_;

    NodePoint(double x, double y, double z);
};


struct Box2D {

    NodePoint min_, max_;
    double x_extent, y_extent, z_extent;

    Box2D(NodePoint& min, double max_x, double max_y, double max_z = 0);
    bool contains(R4Py_ContinuousPoint* pt);
    bool intersects(const BoundingBox& bbox);
};


struct Box3D {

    NodePoint min_, max_;
    double x_extent, y_extent, z_extent;

    Box3D(NodePoint& min, double max_x, double max_y, double max_z);
    bool contains(R4Py_ContinuousPoint* pt);
    bool intersects(const BoundingBox& bbox);
};

template<typename BoxType, typename PointType, typename AccessorType>
class STNode {

protected:
    BoxType bounds_;
    using coord_type  = typename TypeSelector<PointType>::type;

public:
    STNode(NodePoint& pt, double x_extent, double y_extent, double z_extent);
    ~STNode();
    
    virtual std::shared_ptr<STNode<BoxType, PointType, AccessorType>> split() = 0;
    bool contains(PointType* pt);
    virtual void getObjectsWithin(const BoundingBox& box, LocationMapType_<R4Py_ContinuousPoint, typename AccessorType::ValType>* loc_map,
        std::shared_ptr<std::vector<R4Py_Agent*>>& agents) = 0;
    virtual bool addItem(PointType* pt ,int threshold) = 0;
    virtual bool removeItem(PointType* pt) = 0;
};

template<typename BoxType, typename PointType, typename AccessorType>
STNode<BoxType, PointType, AccessorType>::STNode(NodePoint& pt, double x_extent, double y_extent, double z_extent) : 
    bounds_(pt, pt.x_ + x_extent, pt.y_ + y_extent, pt.z_ + z_extent) {}

template<typename BoxType, typename PointType, typename AccessorType>
STNode<BoxType, PointType, AccessorType>::~STNode() {}

template<typename BoxType, typename PointType, typename AccessorType>
bool STNode<BoxType, PointType, AccessorType>::contains(PointType* pt) {
    return bounds_.contains(pt);
}

template<typename PointType>
class MOItems {

private:
    size_t sum;
    std::map<PointType*, size_t, PtrPointComp<PointType>> point_counts;

public:
    MOItems();
    ~MOItems();

    void add(PointType* pt);
    bool remove(PointType* pt);
    void addToNode(std::shared_ptr<STNode<Box2D, R4Py_ContinuousPoint, ContinuousMOType_>> node, int threshold);
    void addToNode(std::shared_ptr<STNode<Box3D, R4Py_ContinuousPoint, ContinuousMOType_>> node, int threshold);
    void getObjectsWithin(const BoundingBox& box, MOMapType* loc_map,
        std::shared_ptr<std::vector<R4Py_Agent*>>& agents);
    size_t size() const;
    void clear();
};

template<typename PointType>
MOItems<PointType>::MOItems() : sum{0}, point_counts{} {}
template<typename PointType>
MOItems<PointType>::~MOItems() {}

template<typename PointType>
void MOItems<PointType>::add(PointType* pt) {
    auto iter = point_counts.find(pt);
    if (iter == point_counts.end()) {
        ++sum;
        Py_INCREF(pt);
        point_counts.emplace(pt, 1);
    } else {
        ++(iter->second);
    }
}

template<typename PointType>
bool MOItems<PointType>::remove(PointType* pt) {
    auto kv = point_counts.find(pt);
    if (kv != point_counts.end()) {
        --(kv->second);
        // size_t val = --point_counts[item->pt];
        if (kv->second == 0) {
            Py_DECREF(kv->first);
            point_counts.erase(kv);
            --sum;
        }
        return true;
    } 
    return false;
}

template<typename PointType>
void MOItems<PointType>::addToNode(std::shared_ptr<STNode<Box2D, R4Py_ContinuousPoint, ContinuousMOType_>> node, int threshold) {
    for (auto& kv: point_counts) {
        node->addItem(kv.first, threshold);
    }
}

template<typename PointType>
void MOItems<PointType>::addToNode(std::shared_ptr<STNode<Box3D, R4Py_ContinuousPoint, ContinuousMOType_>> node, int threshold) {
    for (auto& kv: point_counts) {
        node->addItem(kv.first, threshold);
    }
}

template<typename PointType>
void MOItems<PointType>::getObjectsWithin(const BoundingBox& box, MOMapType* loc_map,
    std::shared_ptr<std::vector<R4Py_Agent*>>& agents) {

    Point<R4Py_ContinuousPoint> ppt;
    for (auto& kv : point_counts) {
        if (box.contains(kv.first)) {
            extract_coords(kv.first, ppt);
            auto& agent_lst = (*loc_map)[ppt];
            agents->insert(agents->end(), agent_lst->begin(), agent_lst->end());
        }
    }

    // for (auto& kv : items) {
    //     if (box.contains(kv.second->pt)) {
    //         Py_INCREF(kv.second->agent);
    //         agents->push_back(kv.second->agent);
    //     }
    //     // } else {
    //     //     using pt_type = typename TypeSelector<R4Py_ContinuousPoint>::type;
    //     //     pt_type* data = (pt_type*)PyArray_DATA(kv.second->pt->coords);
    //     //     std::cout << "(" << data[0] << ", " << data[1] << ", " << data[2] << "), " << box << std::endl;
    //     // }
    // }
}

template<typename PointType>
size_t MOItems<PointType>::size() const {
    return sum;
}

template<typename PointType>
void MOItems<PointType>::clear() {
    for (auto& kv: point_counts) {
        Py_DECREF(kv.first);
    }
    point_counts.clear();
    sum = 0;
}


template<typename PointType>
class SOItems {

private:
    std::set<PointType*> points;

public:
    SOItems();
    ~SOItems();

    void add(PointType* pt);
    bool remove(PointType* pt);
    void addToNode(std::shared_ptr<STNode<Box2D, R4Py_ContinuousPoint, ContinuousSOType_>> node, int threshold);
    void addToNode(std::shared_ptr<STNode<Box3D, R4Py_ContinuousPoint, ContinuousSOType_>> node, int threshold);
    void getObjectsWithin(const BoundingBox& box, SOMapType* loc_map,
        std::shared_ptr<std::vector<R4Py_Agent*>>& agents);
    size_t size() const;
    void clear();
};


template<typename PointType>
SOItems<PointType>::SOItems() : points{} {}

template<typename PointType>
SOItems<PointType>::~SOItems() {}

template<typename PointType>
void SOItems<PointType>::add(PointType* pt) {
    Py_INCREF(pt);
    points.insert(pt);
}

template<typename PointType>
bool SOItems<PointType>::remove(PointType* pt) {
    auto it = points.find(pt);
    if (it != points.end()) {
         Py_DECREF(*it);
        points.erase(it);
        return true;
    }
    return false;
}

template<typename PointType>
void SOItems<PointType>::addToNode(std::shared_ptr<STNode<Box2D, R4Py_ContinuousPoint, ContinuousSOType_>> node, int threshold) {
    for (auto pt: points) {
        node->addItem(pt, threshold);
    }
}

template<typename PointType>
void SOItems<PointType>::addToNode(std::shared_ptr<STNode<Box3D, R4Py_ContinuousPoint, ContinuousSOType_>> node, int threshold) {
    for (auto pt: points) {
        node->addItem(pt, threshold);
    }
}


template<typename PointType>
void SOItems<PointType>::getObjectsWithin(const BoundingBox& box, SOMapType* loc_map,
    std::shared_ptr<std::vector<R4Py_Agent*>>& agents) {

    Point<R4Py_ContinuousPoint> ppt;
    for (auto& pt : points) {
        if (box.contains(pt)) {
            extract_coords(pt, ppt);
            auto agent = (*loc_map)[ppt];
            agents->push_back(agent);
        }
    }

    // for (auto& kv : items) {
    //     if (box.contains(kv.second->pt)) {
    //         Py_INCREF(kv.second->agent);
    //         agents->push_back(kv.second->agent);
    //     }
    //     // } else {
    //     //     using pt_type = typename TypeSelector<R4Py_ContinuousPoint>::type;
    //     //     pt_type* data = (pt_type*)PyArray_DATA(kv.second->pt->coords);
    //     //     std::cout << "(" << data[0] << ", " << data[1] << ", " << data[2] << "), " << box << std::endl;
    //     // }
    // }
}


template<typename PointType>
size_t SOItems<PointType>::size() const {
    return points.size();
}

template<typename PointType>
void SOItems<PointType>::clear() {
    for (auto& pt: points) {
        Py_DECREF(pt);
    }
    return points.clear();
}


template<typename AccessorType>
struct TreeItemSelector {
};

template<>
struct TreeItemSelector<ContinuousMOType_> {
    using type = MOItems<R4Py_ContinuousPoint>;
};

template<>
struct TreeItemSelector<ContinuousSOType_> {
    using type = SOItems<R4Py_ContinuousPoint>;
};



template<typename BoxType, typename PointType, typename AccessorType>
class CompositeNode;

template<typename BoxType, typename PointType, typename AccessorType>
class Composite2DNode;

template<typename BoxType, typename PointType, typename AccessorType>
class Composite3DNode;

template<typename BoxType, typename PointType, typename AccessorType>
class ComponentNode : public STNode<BoxType, PointType, AccessorType> {

private:
    // std::map<R4Py_AgentID*, std::shared_ptr<SpaceItem<PointType>>, agent_id_comp> items;
    using items_type = typename TreeItemSelector<AccessorType>::type;
    items_type items;

public: 
    ComponentNode(NodePoint& pt, double xextent, double yextent, double zxtent);
    ~ComponentNode();
    std::shared_ptr<STNode<BoxType, PointType, AccessorType>> split() override;
    void getObjectsWithin(const BoundingBox& box, LocationMapType_<R4Py_ContinuousPoint, typename AccessorType::ValType>* loc_map,
        std::shared_ptr<std::vector<R4Py_Agent*>>& agents) override;
    bool addItem(PointType* pt, int threshold) override;
    bool removeItem(PointType* pt) override;
};

template<typename BoxType, typename PointType, typename AccessorType>
ComponentNode<BoxType, PointType, AccessorType>::ComponentNode(NodePoint& pt, double xextent, double yextent, double zextent) :
    STNode<BoxType, PointType, AccessorType>(pt, xextent, yextent, zextent), items{} {}

template<typename BoxType, typename PointType, typename AccessorType>
ComponentNode<BoxType, PointType, AccessorType>::~ComponentNode() {}

template<typename BoxType, typename PointType, typename AccessorType>
bool ComponentNode<BoxType, PointType, AccessorType>::addItem(PointType* pt, int threshold) {
    items.add(pt);
    return items.size() > threshold;
}

template<typename BoxType, typename PointType, typename AccessorType>
bool ComponentNode<BoxType, PointType, AccessorType>::removeItem(PointType* pt) {
    return items.remove(pt);
}

template<typename BoxType, typename PointType, typename AccessorType>
void ComponentNode<BoxType, PointType, AccessorType>::getObjectsWithin(const BoundingBox& box, LocationMapType_<R4Py_ContinuousPoint, typename AccessorType::ValType>* loc_map,
    std::shared_ptr<std::vector<R4Py_Agent*>>& agents) {
    if (items.size() > 0 && STNode<BoxType, PointType, AccessorType>::bounds_.intersects(box)) {
        items.getObjectsWithin(box, loc_map, agents);
    }
}

template<typename BoxType, typename PointType, typename AccessorType>
std::shared_ptr<STNode<BoxType, PointType, AccessorType>> ComponentNode<BoxType, PointType, AccessorType>::split() {
    std::shared_ptr<STNode<BoxType, PointType, AccessorType>> parent;
    if (STNode<BoxType, PointType, AccessorType>::bounds_.z_extent == 0)
        parent = std::make_shared<Composite2DNode<BoxType, PointType, AccessorType>>(STNode<BoxType, PointType, AccessorType>::bounds_.min_, 
        STNode<BoxType, PointType, AccessorType>::bounds_.x_extent, STNode<BoxType, PointType, AccessorType>::bounds_.y_extent);
    else if (STNode<BoxType, PointType, AccessorType>::bounds_.z_extent > 0) {
        parent = std::make_shared<Composite3DNode<BoxType, PointType, AccessorType>>(STNode<BoxType, PointType, AccessorType>::bounds_.min_, 
            STNode<BoxType, PointType, AccessorType>::bounds_.x_extent, STNode<BoxType, PointType, AccessorType>::bounds_.y_extent,
            STNode<BoxType, PointType, AccessorType>::bounds_.z_extent);
    }

    items.addToNode(parent, std::numeric_limits<int>::max());
    items.clear();
    return parent;
}

template<typename BoxType, typename PointType, typename AccessorType>
class CompositeNode : public STNode<BoxType, PointType, AccessorType> {

protected:
    std::vector<std::shared_ptr<STNode<BoxType, PointType, AccessorType>>> children;
    double x_plane, y_plane, z_plane;
    int size;

    virtual size_t calcChildIndex(PointType* pt) = 0;

public:
    CompositeNode(NodePoint& min, double x_extent, double y_extent, double z_extent);
    ~CompositeNode();

    void getObjectsWithin(const BoundingBox& box, LocationMapType_<R4Py_ContinuousPoint, typename AccessorType::ValType>* loc_map,
        std::shared_ptr<std::vector<R4Py_Agent*>>& agents) override;
    bool addItem(PointType* pt, int threshold) override;
    bool removeItem(PointType* pt) override;
    std::shared_ptr<STNode<BoxType, PointType, AccessorType>> split() override;

};

template<typename BoxType, typename PointType, typename AccessorType>
CompositeNode<BoxType, PointType, AccessorType>::CompositeNode(NodePoint& min, double x_extent, double y_extent, double z_extent) : 
    STNode<BoxType, PointType, AccessorType>(min, x_extent,  y_extent, z_extent), children{}, x_plane{0}, y_plane{0}, z_plane{0}, size{0} {}

template<typename BoxType, typename PointType, typename AccessorType>
CompositeNode<BoxType, PointType, AccessorType>::~CompositeNode() {}

template<typename BoxType, typename PointType, typename AccessorType>
void CompositeNode<BoxType, PointType, AccessorType>::getObjectsWithin(const BoundingBox& box, LocationMapType_<R4Py_ContinuousPoint, typename AccessorType::ValType>* loc_map,
    std::shared_ptr<std::vector<R4Py_Agent*>>& agents) {

    if (size > 0 && STNode<BoxType, PointType, AccessorType>::bounds_.intersects(box)) {
        for (auto& child : children) {
            child->getObjectsWithin(box, loc_map, agents);
        }
    }
}

template<typename BoxType, typename PointType, typename AccessorType>
bool CompositeNode<BoxType, PointType, AccessorType>::addItem(PointType* pt, int threshold) {
    if (STNode<BoxType, PointType, AccessorType>::contains(pt)) {
        size_t index = calcChildIndex(pt);
        bool split = children[index]->addItem(pt, threshold);
        if (split) {
            std::shared_ptr<STNode<BoxType, PointType, AccessorType>> parent = children[index]->split();
            children[index] = parent;
        }
        ++size;
    }
    return false;
}

template<typename BoxType, typename PointType, typename AccessorType>
bool CompositeNode<BoxType, PointType, AccessorType>::removeItem(PointType* pt) {
    if (STNode<BoxType, PointType, AccessorType>::contains(pt)) {
        size_t index = calcChildIndex(pt);
        if (children[index]->removeItem(pt)) {
            --size;
            return true;
        }
    }
    return false;
}


template<typename BoxType, typename PointType, typename AccessorType>
std::shared_ptr<STNode<BoxType, PointType, AccessorType>> CompositeNode<BoxType, PointType, AccessorType>::split() {
    throw std::domain_error("Split cannot be called on a composite node");
}


template<typename BoxType, typename PointType, typename AccessorType>
class Composite2DNode : public CompositeNode<BoxType, PointType, AccessorType> {

protected:
    size_t calcChildIndex(PointType* pt) override;

public:
    Composite2DNode(NodePoint& min, double x_extent, double y_extent);
    ~Composite2DNode();
};


template<typename BoxType, typename PointType, typename AccessorType>
Composite2DNode<BoxType, PointType, AccessorType>::Composite2DNode(NodePoint& min, double x_extent, double y_extent) : 
    CompositeNode<BoxType, PointType, AccessorType>(min, x_extent,  y_extent, 0)
{
    double half_x = STNode<BoxType, PointType, AccessorType>::bounds_.x_extent / 2.0;
    double half_y = STNode<BoxType, PointType, AccessorType>::bounds_.y_extent / 2.0;

    CompositeNode<BoxType, PointType, AccessorType>::x_plane = min.x_ + half_x;
    CompositeNode<BoxType, PointType, AccessorType>::y_plane = min.y_ + half_y;

    NodePoint pt1(min.x_, min.y_, 0);
    CompositeNode<BoxType, PointType, AccessorType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType, AccessorType>>(pt1, half_x, half_y, 0));
    NodePoint pt2(min.x_, min.y_ + half_y, 0);
    CompositeNode<BoxType, PointType, AccessorType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType, AccessorType>>(pt2, half_x, half_y, 0));
    NodePoint pt3(min.x_ + half_x, min.y_ , 0);
    CompositeNode<BoxType, PointType, AccessorType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType, AccessorType>>(pt3, half_x, half_y, 0));
    NodePoint pt4(min.x_ + half_x, min.y_ + half_y, 0);
    CompositeNode<BoxType, PointType, AccessorType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType, AccessorType>>(pt4, half_x, half_y, 0));
}

template<typename BoxType, typename PointType, typename AccessorType>
Composite2DNode<BoxType, PointType, AccessorType>::~Composite2DNode() {}


template<typename BoxType, typename PointType, typename AccessorType>
size_t Composite2DNode<BoxType, PointType, AccessorType>::calcChildIndex(PointType* pt) {
    size_t index = 0;
    using coord_type = typename TypeSelector<PointType>::type;
    coord_type* data = (coord_type*) PyArray_DATA(pt->coords);
    if (data[0] >= CompositeNode<BoxType, PointType, AccessorType>::x_plane) {
        index = 2;
    }

    if (data[1] >= CompositeNode<BoxType, PointType, AccessorType>::y_plane) {
        index += 1;
    }
    return index;
}



template<typename BoxType, typename PointType, typename AccessorType>
class Composite3DNode : public CompositeNode<BoxType, PointType, AccessorType> {

protected:
    size_t calcChildIndex(PointType* pt) override;

public:
    Composite3DNode(NodePoint& min, double x_extent, double y_extent, double z_extent);
    ~Composite3DNode();
};


template<typename BoxType, typename PointType, typename AccessorType>
Composite3DNode<BoxType, PointType, AccessorType>::Composite3DNode(NodePoint& min, double x_extent, double y_extent, double z_extent) : 
    CompositeNode<BoxType, PointType, AccessorType>(min, x_extent,  y_extent, z_extent)
{
    double half_x = STNode<BoxType, PointType, AccessorType>::bounds_.x_extent / 2.0;
    double half_y = STNode<BoxType, PointType, AccessorType>::bounds_.y_extent / 2.0;
    double half_z = STNode<BoxType, PointType, AccessorType>::bounds_.z_extent / 2.0;

    CompositeNode<BoxType, PointType, AccessorType>::x_plane = min.x_ + half_x;
    CompositeNode<BoxType, PointType, AccessorType>::y_plane = min.y_ + half_y;
    CompositeNode<BoxType, PointType, AccessorType>::z_plane = min.z_ + half_z;

    NodePoint pt1(min.x_, min.y_, min.z_);
    CompositeNode<BoxType, PointType, AccessorType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType, AccessorType>>(pt1, half_x, half_y, half_z));
    NodePoint pt2(min.x_, min.y_, min.z_ + half_z);
    CompositeNode<BoxType, PointType, AccessorType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType, AccessorType>>(pt2, half_x, half_y, half_z));
    NodePoint pt3(min.x_, min.y_ + half_y, min.z_);
    CompositeNode<BoxType, PointType, AccessorType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType, AccessorType>>(pt3, half_x, half_y, half_z));
    NodePoint pt4(min.x_, min.y_ + half_y, min.z_ + half_z);
    CompositeNode<BoxType, PointType, AccessorType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType, AccessorType>>(pt4, half_x, half_y, half_z));
    NodePoint pt5(min.x_ + half_x, min.y_ , min.z_);
    CompositeNode<BoxType, PointType, AccessorType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType, AccessorType>>(pt5, half_x, half_y, half_z));
    NodePoint pt6(min.x_ + half_x, min.y_ , min.z_ + half_z);
    CompositeNode<BoxType, PointType, AccessorType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType, AccessorType>>(pt6, half_x, half_y, half_z));
    NodePoint pt7(min.x_ + half_x, min.y_ + half_y, min.z_);
    CompositeNode<BoxType, PointType, AccessorType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType, AccessorType>>(pt7, half_x, half_y, half_z));
    NodePoint pt8(min.x_ + half_x, min.y_ + half_y, min.z_ + half_z);
    CompositeNode<BoxType, PointType, AccessorType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType, AccessorType>>(pt8, half_x, half_y, half_z));
}

template<typename BoxType, typename PointType, typename AccessorType>
Composite3DNode<BoxType, PointType, AccessorType>::~Composite3DNode() {}


template<typename BoxType, typename PointType, typename AccessorType>
size_t Composite3DNode<BoxType, PointType, AccessorType>::calcChildIndex(PointType* pt) {
    size_t index = 0;
    using coord_type = typename TypeSelector<PointType>::type;
    coord_type* data = (coord_type*) PyArray_DATA(pt->coords);

    if (data[0] >= CompositeNode<BoxType, PointType, AccessorType>::x_plane) {
        index = 4;
    }

    if (data[1] >= CompositeNode<BoxType, PointType, AccessorType>::y_plane) {
        index += 2;
    }

    if (data[2] >= CompositeNode<BoxType, PointType, AccessorType>::z_plane) {
        index += 1;
    }
    return index;
}


template<typename BoxType, typename PointType, typename AccessorType>
class SpatialTree {

public:
    using LMType =  LocationMapType_<R4Py_ContinuousPoint, typename AccessorType::ValType>;

    SpatialTree(int threshold, const BoundingBox& box, LMType* loc_map); 
    ~SpatialTree();

    void addItem(std::shared_ptr<SpaceItem<PointType>>& item);
    bool removeItem(std::shared_ptr<SpaceItem<PointType>>& item);
    void getObjectsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents);

private:
    std::shared_ptr<CompositeNode<BoxType, PointType, AccessorType>> root;
    int threshold_;
    LMType* loc_map_;
};

template<typename BoxType, typename PointType, typename AccessorType>
SpatialTree<BoxType, PointType, AccessorType>::SpatialTree(int threshold, const BoundingBox& box,
    LocationMapType_<R4Py_ContinuousPoint, typename AccessorType::ValType>* loc_map) : root{}, 
    threshold_{threshold},  loc_map_{loc_map} {

    NodePoint min(box.xmin_, box.ymin_, box.zmin_);
    if (box.num_dims == 1) {
        // TODO 
    } else if (box.num_dims == 2) {
        root = std::make_shared<Composite2DNode<BoxType, PointType, AccessorType>>(min, box.x_extent_, box.y_extent_);
    } else if (box.num_dims == 3) {
        // std::cout << box.xmin_ << "," << box.ymin_ << "," << box.zmin_ << "," << box.z_extent_ << std::endl;
        root = std::make_shared<Composite3DNode<BoxType, PointType, AccessorType>>(min, box.x_extent_, box.y_extent_, box.z_extent_);
    }
}

template<typename BoxType, typename PointType, typename ItemType>
SpatialTree<BoxType, PointType, ItemType>::~SpatialTree() {}

template<typename BoxType, typename PointType, typename ItemType>
void SpatialTree<BoxType, PointType, ItemType>::addItem(std::shared_ptr<SpaceItem<PointType>>& item) {
    root->addItem(item->pt, threshold_);
}

template<typename BoxType, typename PointType, typename ItemType>
bool SpatialTree<BoxType, PointType, ItemType>::removeItem(std::shared_ptr<SpaceItem<PointType>>& item) {
    return root->removeItem(item->pt);
}

template<typename BoxType, typename PointType, typename ItemType>
void SpatialTree<BoxType, PointType, ItemType>:: getObjectsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) {
    root->getObjectsWithin(box, loc_map_, agents);
}



class CPSpatialTree {

public:
    virtual ~CPSpatialTree() = 0;
    virtual void addItem(std::shared_ptr<SpaceItem<R4Py_ContinuousPoint>>& item) = 0;
    virtual bool removeItem(std::shared_ptr<SpaceItem<R4Py_ContinuousPoint>>& item) = 0;
    virtual void getObjectsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) = 0;
};

inline CPSpatialTree::~CPSpatialTree() {}

template<typename DelegateType>
class CPSpatialTreeImpl : public CPSpatialTree {

private:
    std::unique_ptr<DelegateType> delegate;

public:
    CPSpatialTreeImpl(int threshold, const BoundingBox& box, typename DelegateType::LMType* loc_map);
    virtual ~CPSpatialTreeImpl();
    void addItem(std::shared_ptr<SpaceItem<R4Py_ContinuousPoint>>& item) override;
    bool removeItem(std::shared_ptr<SpaceItem<R4Py_ContinuousPoint>>& item) override;
    void getObjectsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) override;
};

template<typename DelegateType>
CPSpatialTreeImpl<DelegateType>::CPSpatialTreeImpl(int threshold, const BoundingBox& box, typename DelegateType::LMType* loc_map) : 
    delegate(std::unique_ptr<DelegateType>(new DelegateType(threshold, box, loc_map))) {}

template<typename DelegateType>
CPSpatialTreeImpl<DelegateType>::~CPSpatialTreeImpl() {}

template<typename DelegateType>
void CPSpatialTreeImpl<DelegateType>::addItem(std::shared_ptr<SpaceItem<R4Py_ContinuousPoint>>& item) {
    delegate->addItem(item);
}

template<typename DelegateType>
bool CPSpatialTreeImpl<DelegateType>::removeItem(std::shared_ptr<SpaceItem<R4Py_ContinuousPoint>>& item) {
    return delegate->removeItem(item);
}

template<typename DelegateType>
void CPSpatialTreeImpl<DelegateType>::getObjectsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) {
    delegate->getObjectsWithin(box, agents);
}

}

#endif 