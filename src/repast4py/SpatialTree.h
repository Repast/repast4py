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

#include "geometry.h"
#include "space_types.h"
#include "occupancy.h"

namespace repast4py {

struct NodePoint {
    double x_, y_, z_;

    NodePoint(double x, double y, double z);
};

template<typename PointType>
class MOItems {

private:
    size_t sum;
    std::map<PointType*, size_t, PtrPointComp<PointType>> point_counts;

public:
    std::map<R4Py_AgentID*, std::shared_ptr<SpaceItem<PointType>>, agent_id_comp> items;

    MOItems();
    ~MOItems();

    void add(std::shared_ptr<SpaceItem<PointType>>& item);
    bool remove(std::shared_ptr<SpaceItem<PointType>>& item);
    size_t size() const;
    void clear();
};

template<typename PointType>
MOItems<PointType>::MOItems() : sum{0}, point_counts{}, items{} {}
template<typename PointType>
MOItems<PointType>::~MOItems() {}

template<typename PointType>
void MOItems<PointType>::add(std::shared_ptr<SpaceItem<PointType>>& item) {
    items[item->agent->aid] = item;
    auto iter = point_counts.find(item->pt);
    if (iter == point_counts.end()) {
        ++sum;
        Py_INCREF(item->pt);
        point_counts.emplace(item->pt, 1);
    } else {
        ++(iter->second);
    }
}

template<typename PointType>
bool MOItems<PointType>::remove(std::shared_ptr<SpaceItem<PointType>>& item) {
    if  (items.erase(item->agent->aid) == 1) {
        auto kv = point_counts.find(item->pt);
        --(kv->second);
        // size_t val = --point_counts[item->pt];
        if (kv->second == 0) {
            Py_DECREF(kv->first);
            point_counts.erase(kv);
            --sum;
        }
        return true;
    } else {
        return false;
    }
}

template<typename PointType>
size_t MOItems<PointType>::size() const {
    return sum;
}

template<typename PointType>
void MOItems<PointType>::clear() {
    point_counts.clear();
    items.clear();
    sum = 0;
}

template<typename PointType>
class SOItems {

public:
    std::map<R4Py_AgentID*, std::shared_ptr<SpaceItem<PointType>>, agent_id_comp> items;

    SOItems();
    ~SOItems();

    void add(std::shared_ptr<SpaceItem<PointType>>& item);
    bool remove(std::shared_ptr<SpaceItem<PointType>>& item);
    size_t size() const;
    void clear();
};


template<typename PointType>
SOItems<PointType>::SOItems() : items{} {}

template<typename PointType>
SOItems<PointType>::~SOItems() {}

template<typename PointType>
void SOItems<PointType>::add(std::shared_ptr<SpaceItem<PointType>>& item) {
    items[item->agent->aid] = item;
}

template<typename PointType>
bool SOItems<PointType>::remove(std::shared_ptr<SpaceItem<PointType>>& item) {
    return items.erase(item->agent->aid) == 1;
}

template<typename PointType>
size_t SOItems<PointType>::size() const {
    return items.size();
}

template<typename PointType>
void SOItems<PointType>::clear() {
    return items.clear();
}


template<typename AccessorType>
struct TreeItemSelector {
};

template<typename PointType, typename ValueType>
using _LocationMapType = std::map<Point<PointType>, ValueType, PointComp<PointType>>;

using _ContinuousMOType = MultiOccupancyAccessor<_LocationMapType<R4Py_ContinuousPoint, AgentListPtr>, R4Py_ContinuousPoint>;
using _ContinuousSOType = SingleOccupancyAccessor<_LocationMapType<R4Py_ContinuousPoint, R4Py_Agent*>, R4Py_ContinuousPoint>;

template<>
struct TreeItemSelector<_ContinuousMOType> {
    using type = MOItems<R4Py_ContinuousPoint>;
};

template<>
struct TreeItemSelector<_ContinuousSOType> {
    using type = SOItems<R4Py_ContinuousPoint>;
};


template<typename BoxType, typename PointType>
class STNode {

protected:
    BoxType bounds_;
    using coord_type  = typename TypeSelector<PointType>::type;

public:
    STNode(NodePoint& pt, double x_extent, double y_extent, double z_extent);
    ~STNode();
    
    virtual std::shared_ptr<STNode<BoxType, PointType>> split() = 0;
    bool contains(PointType* pt);
    virtual void getObjectsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) = 0;
    virtual bool addItem(std::shared_ptr<SpaceItem<PointType>>& item, int threshold) = 0;
    virtual bool removeItem(std::shared_ptr<SpaceItem<PointType>>& item) = 0;
};

template<typename BoxType, typename PointType>
STNode<BoxType, PointType>::STNode(NodePoint& pt, double x_extent, double y_extent, double z_extent) : 
    bounds_(pt, pt.x_ + x_extent, pt.y_ + y_extent, pt.z_ + z_extent) {}

template<typename BoxType, typename PointType>
STNode<BoxType, PointType>::~STNode() {}

template<typename BoxType, typename PointType>
bool STNode<BoxType, PointType>::contains(PointType* pt) {
    return bounds_.contains(pt);
}

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
class CompositeNode;

template<typename BoxType, typename PointType, typename AccessorType>
class Composite2DNode;

template<typename BoxType, typename PointType, typename AccessorType>
class Composite3DNode;

template<typename BoxType, typename PointType, typename AccessorType>
class ComponentNode : public STNode<BoxType, PointType> {

private:
    // std::map<R4Py_AgentID*, std::shared_ptr<SpaceItem<PointType>>, agent_id_comp> items;
    using items_type = typename TreeItemSelector<AccessorType>::type;
    items_type items;

public: 
    ComponentNode(NodePoint& pt, double xextent, double yextent, double zxtent);
    ~ComponentNode();
    std::shared_ptr<STNode<BoxType, PointType>> split() override;
    void getObjectsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) override;
    bool addItem(std::shared_ptr<SpaceItem<PointType>>& item, int threshold) override;
    bool removeItem(std::shared_ptr<SpaceItem<PointType>>& item) override;
};

template<typename BoxType, typename PointType, typename AccessorType>
ComponentNode<BoxType, PointType, AccessorType>::ComponentNode(NodePoint& pt, double xextent, double yextent, double zextent) :
    STNode<BoxType, PointType>(pt, xextent, yextent, zextent), items{} {}

template<typename BoxType, typename PointType, typename AccessorType>
ComponentNode<BoxType, PointType, AccessorType>::~ComponentNode() {}

template<typename BoxType, typename PointType, typename AccessorType>
bool ComponentNode<BoxType, PointType, AccessorType>::addItem(std::shared_ptr<SpaceItem<PointType>>& item, int threshold) {
    items.add(item);
    return items.size() > threshold;
}

template<typename BoxType, typename PointType, typename AccessorType>
bool ComponentNode<BoxType, PointType, AccessorType>::removeItem(std::shared_ptr<SpaceItem<PointType>>& item) {
    return items.remove(item);
}

template<typename BoxType, typename PointType, typename AccessorType>
void ComponentNode<BoxType, PointType, AccessorType>::getObjectsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) {
    if (items.size() > 0 && STNode<BoxType, PointType>::bounds_.intersects(box)) {
        for (auto& kv : items.items) {
            if (box.contains(kv.second->pt)) {
                Py_INCREF(kv.second->agent);
                agents->push_back(kv.second->agent);
            }
            // } else {
            //     using pt_type = typename TypeSelector<R4Py_ContinuousPoint>::type;
            //     pt_type* data = (pt_type*)PyArray_DATA(kv.second->pt->coords);
            //     std::cout << "(" << data[0] << ", " << data[1] << ", " << data[2] << "), " << box << std::endl;
            // }
        }
    }
}

template<typename BoxType, typename PointType, typename AccessorType>
std::shared_ptr<STNode<BoxType, PointType>> ComponentNode<BoxType, PointType, AccessorType>::split() {
    std::shared_ptr<STNode<BoxType, PointType>> parent;
    if (STNode<BoxType, PointType>::bounds_.z_extent == 0)
        parent = std::make_shared<Composite2DNode<BoxType, PointType, AccessorType>>(STNode<BoxType, PointType>::bounds_.min_, 
        STNode<BoxType, PointType>::bounds_.x_extent, STNode<BoxType, PointType>::bounds_.y_extent);
    else if (STNode<BoxType, PointType>::bounds_.z_extent > 0) {
        parent = std::make_shared<Composite3DNode<BoxType, PointType, AccessorType>>(STNode<BoxType, PointType>::bounds_.min_, 
            STNode<BoxType, PointType>::bounds_.x_extent, STNode<BoxType, PointType>::bounds_.y_extent,
            STNode<BoxType, PointType>::bounds_.z_extent);
    }

    for (auto& kv : items.items) {
        parent->addItem(kv.second, std::numeric_limits<int>::max());
    }
    items.clear();
    return parent;
}

template<typename BoxType, typename PointType, typename AccessorType>
class CompositeNode : public STNode<BoxType, PointType> {

protected:
    std::vector<std::shared_ptr<STNode<BoxType, PointType>>> children;
    double x_plane, y_plane, z_plane;
    int size;

    virtual size_t calcChildIndex(std::shared_ptr<SpaceItem<PointType>>& item) = 0;

public:
    CompositeNode(NodePoint& min, double x_extent, double y_extent, double z_extent);
    ~CompositeNode();

    void getObjectsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) override;
    bool addItem(std::shared_ptr<SpaceItem<PointType>>& item, int threshold) override;
    bool removeItem(std::shared_ptr<SpaceItem<PointType>>& item) override;
    std::shared_ptr<STNode<BoxType, PointType>> split() override;

};

template<typename BoxType, typename PointType, typename AccessorType>
CompositeNode<BoxType, PointType, AccessorType>::CompositeNode(NodePoint& min, double x_extent, double y_extent, double z_extent) : 
    STNode<BoxType, PointType>(min, x_extent,  y_extent, z_extent), children{}, x_plane{0}, y_plane{0}, z_plane{0}, size{0} {}

template<typename BoxType, typename PointType, typename AccessorType>
CompositeNode<BoxType, PointType, AccessorType>::~CompositeNode() {}

template<typename BoxType, typename PointType, typename AccessorType>
void CompositeNode<BoxType, PointType, AccessorType>::getObjectsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) {
    if (size > 0 && STNode<BoxType, PointType>::bounds_.intersects(box)) {
        for (auto& child : children) {
            child->getObjectsWithin(box, agents);
        }
    }
}

template<typename BoxType, typename PointType, typename AccessorType>
bool CompositeNode<BoxType, PointType, AccessorType>::addItem(std::shared_ptr<SpaceItem<PointType>>& item, int threshold) {
    if (STNode<BoxType, PointType>::contains(item->pt)) {
        size_t index = calcChildIndex(item);
        bool split = children[index]->addItem(item, threshold);
        if (split) {
            std::shared_ptr<STNode<BoxType, PointType>> parent = children[index]->split();
            children[index] = parent;
        }
        ++size;
    }
    return false;
}

template<typename BoxType, typename PointType, typename AccessorType>
bool CompositeNode<BoxType, PointType, AccessorType>::removeItem(std::shared_ptr<SpaceItem<PointType>>& item) {
    if (STNode<BoxType, PointType>::contains(item->pt)) {
        size_t index = calcChildIndex(item);
        if (children[index]->removeItem(item)) {
            --size;
            return true;
        }
    }
    return false;
}


template<typename BoxType, typename PointType, typename AccessorType>
std::shared_ptr<STNode<BoxType, PointType>> CompositeNode<BoxType, PointType, AccessorType>::split() {
    throw std::domain_error("Split cannot be called on a composite node");
}


template<typename BoxType, typename PointType, typename AccessorType>
class Composite2DNode : public CompositeNode<BoxType, PointType, AccessorType> {

protected:
    size_t calcChildIndex(std::shared_ptr<SpaceItem<PointType>>& item) override;

public:
    Composite2DNode(NodePoint& min, double x_extent, double y_extent);
    ~Composite2DNode();
};


template<typename BoxType, typename PointType, typename AccessorType>
Composite2DNode<BoxType, PointType, AccessorType>::Composite2DNode(NodePoint& min, double x_extent, double y_extent) : 
    CompositeNode<BoxType, PointType, AccessorType>(min, x_extent,  y_extent, 0)
{
    double half_x = STNode<BoxType, PointType>::bounds_.x_extent / 2.0;
    double half_y = STNode<BoxType, PointType>::bounds_.y_extent / 2.0;

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
size_t Composite2DNode<BoxType, PointType, AccessorType>::calcChildIndex(std::shared_ptr<SpaceItem<PointType>>& item) {
    size_t index = 0;
    using coord_type = typename TypeSelector<PointType>::type;
    coord_type* data = (coord_type*) PyArray_DATA(item->pt->coords);
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
    size_t calcChildIndex(std::shared_ptr<SpaceItem<PointType>>& item) override;

public:
    Composite3DNode(NodePoint& min, double x_extent, double y_extent, double z_extent);
    ~Composite3DNode();
};


template<typename BoxType, typename PointType, typename AccessorType>
Composite3DNode<BoxType, PointType, AccessorType>::Composite3DNode(NodePoint& min, double x_extent, double y_extent, double z_extent) : 
    CompositeNode<BoxType, PointType, AccessorType>(min, x_extent,  y_extent, z_extent)
{
    double half_x = STNode<BoxType, PointType>::bounds_.x_extent / 2.0;
    double half_y = STNode<BoxType, PointType>::bounds_.y_extent / 2.0;
    double half_z = STNode<BoxType, PointType>::bounds_.z_extent / 2.0;

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
size_t Composite3DNode<BoxType, PointType, AccessorType>::calcChildIndex(std::shared_ptr<SpaceItem<PointType>>& item) {
    size_t index = 0;
    using coord_type = typename TypeSelector<PointType>::type;
    coord_type* data = (coord_type*) PyArray_DATA(item->pt->coords);

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

private:
    std::shared_ptr<CompositeNode<BoxType, PointType, AccessorType>> root;
    int threshold_;

public:
    SpatialTree(int threshold, const BoundingBox& box); 
    ~SpatialTree();

    void addItem(std::shared_ptr<SpaceItem<PointType>>& item);
    bool removeItem(std::shared_ptr<SpaceItem<PointType>>& item);
    void getObjectsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents);
};

template<typename BoxType, typename PointType, typename AccessorType>
SpatialTree<BoxType, PointType, AccessorType>::SpatialTree(int threshold, const BoundingBox& box) : root{}, threshold_{threshold} {
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
    root->addItem(item, threshold_);
}

template<typename BoxType, typename PointType, typename ItemType>
bool SpatialTree<BoxType, PointType, ItemType>::removeItem(std::shared_ptr<SpaceItem<PointType>>& item) {
    return root->removeItem(item);
}

template<typename BoxType, typename PointType, typename ItemType>
void SpatialTree<BoxType, PointType, ItemType>:: getObjectsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) {
    root->getObjectsWithin(box, agents);
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
    CPSpatialTreeImpl(int threshold, const BoundingBox& box);
    virtual ~CPSpatialTreeImpl();
    void addItem(std::shared_ptr<SpaceItem<R4Py_ContinuousPoint>>& item) override;
    bool removeItem(std::shared_ptr<SpaceItem<R4Py_ContinuousPoint>>& item) override;
    void getObjectsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) override;
};

template<typename DelegateType>
CPSpatialTreeImpl<DelegateType>::CPSpatialTreeImpl(int threshold, const BoundingBox& box) : 
    delegate(std::unique_ptr<DelegateType>(new DelegateType(threshold, box))) {}

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