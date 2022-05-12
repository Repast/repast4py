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

#include "geometry.h"
#include "space_types.h"

namespace repast4py {

struct NodePoint {
    double x_, y_, z_;

    NodePoint(double x, double y, double z);
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


template<typename BoxType, typename PointType>
class CompositeNode;

template<typename BoxType, typename PointType>
class Composite2DNode;

template<typename BoxType, typename PointType>
class Composite3DNode;

template<typename BoxType, typename PointType>
class ComponentNode : public STNode<BoxType, PointType> {

private:
    std::map<R4Py_AgentID*, std::shared_ptr<SpaceItem<PointType>>, agent_id_comp> items;

public: 
    ComponentNode(NodePoint& pt, double xextent, double yextent, double zxtent);
    ~ComponentNode();
    std::shared_ptr<STNode<BoxType, PointType>> split() override;
    void getObjectsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) override;
    bool addItem(std::shared_ptr<SpaceItem<PointType>>& item, int threshold) override;
    bool removeItem(std::shared_ptr<SpaceItem<PointType>>& item) override;
};

template<typename BoxType, typename PointType>
ComponentNode<BoxType, PointType>::ComponentNode(NodePoint& pt, double xextent, double yextent, double zextent) :
    STNode<BoxType, PointType>(pt, xextent, yextent, zextent), items{} {}

template<typename BoxType, typename PointType>
ComponentNode<BoxType, PointType>::~ComponentNode() {}

template<typename BoxType, typename PointType>
bool ComponentNode<BoxType, PointType>::addItem(std::shared_ptr<SpaceItem<PointType>>& item, int threshold) {
    items[item->agent->aid] = item;
    return items.size() > threshold;
}

template<typename BoxType, typename PointType>
bool ComponentNode<BoxType, PointType>::removeItem(std::shared_ptr<SpaceItem<PointType>>& item) {
    return items.erase(item->agent->aid);
}

template<typename BoxType, typename PointType>
void ComponentNode<BoxType, PointType>::getObjectsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) {
    if (items.size() > 0 && STNode<BoxType, PointType>::bounds_.intersects(box)) {
        for (auto& kv : items) {
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

template<typename BoxType, typename PointType>
std::shared_ptr<STNode<BoxType, PointType>> ComponentNode<BoxType, PointType>::split() {
    std::shared_ptr<STNode<BoxType, PointType>> parent;
    if (STNode<BoxType, PointType>::bounds_.z_extent == 0)
        parent = std::make_shared<Composite2DNode<BoxType, PointType>>(STNode<BoxType, PointType>::bounds_.min_, 
        STNode<BoxType, PointType>::bounds_.x_extent, STNode<BoxType, PointType>::bounds_.y_extent);
    else if (STNode<BoxType, PointType>::bounds_.z_extent > 0) {
        parent = std::make_shared<Composite3DNode<BoxType, PointType>>(STNode<BoxType, PointType>::bounds_.min_, 
            STNode<BoxType, PointType>::bounds_.x_extent, STNode<BoxType, PointType>::bounds_.y_extent,
            STNode<BoxType, PointType>::bounds_.z_extent);
    }

    for (auto& kv : items) {
        parent->addItem(kv.second, std::numeric_limits<int>::max());
    }
    items.clear();
    return parent;
}

template<typename BoxType, typename PointType>
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

template<typename BoxType, typename PointType>
CompositeNode<BoxType, PointType>::CompositeNode(NodePoint& min, double x_extent, double y_extent, double z_extent) : 
    STNode<BoxType, PointType>(min, x_extent,  y_extent, z_extent), children{}, x_plane{0}, y_plane{0}, z_plane{0}, size{0} {}

template<typename BoxType, typename PointType>
CompositeNode<BoxType, PointType>::~CompositeNode() {}

template<typename BoxType, typename PointType>
void CompositeNode<BoxType, PointType>::getObjectsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) {
    if (size > 0 && STNode<BoxType, PointType>::bounds_.intersects(box)) {
        for (auto& child : children) {
            child->getObjectsWithin(box, agents);
        }
    }
}

template<typename BoxType, typename PointType>
bool CompositeNode<BoxType, PointType>::addItem(std::shared_ptr<SpaceItem<PointType>>& item, int threshold) {
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

template<typename BoxType, typename PointType>
bool CompositeNode<BoxType, PointType>::removeItem(std::shared_ptr<SpaceItem<PointType>>& item) {
    if (STNode<BoxType, PointType>::contains(item->pt)) {
        size_t index = calcChildIndex(item);
        if (children[index]->removeItem(item)) {
            --size;
            return true;
        }
    }
    return false;
}


template<typename BoxType, typename PointType>
std::shared_ptr<STNode<BoxType, PointType>> CompositeNode<BoxType, PointType>::split() {
    throw std::domain_error("Split cannot be called on a composite node");
}


template<typename BoxType, typename PointType>
class Composite2DNode : public CompositeNode<BoxType, PointType> {

protected:
    size_t calcChildIndex(std::shared_ptr<SpaceItem<PointType>>& item) override;

public:
    Composite2DNode(NodePoint& min, double x_extent, double y_extent);
    ~Composite2DNode();
};


template<typename BoxType, typename PointType>
Composite2DNode<BoxType, PointType>::Composite2DNode(NodePoint& min, double x_extent, double y_extent) : 
    CompositeNode<BoxType, PointType>(min, x_extent,  y_extent, 0)
{
    double half_x = STNode<BoxType, PointType>::bounds_.x_extent / 2.0;
    double half_y = STNode<BoxType, PointType>::bounds_.y_extent / 2.0;

    CompositeNode<BoxType, PointType>::x_plane = min.x_ + half_x;
    CompositeNode<BoxType, PointType>::y_plane = min.y_ + half_y;

    NodePoint pt1(min.x_, min.y_, 0);
    CompositeNode<BoxType, PointType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType>>(pt1, half_x, half_y, 0));
    NodePoint pt2(min.x_, min.y_ + half_y, 0);
    CompositeNode<BoxType, PointType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType>>(pt2, half_x, half_y, 0));
    NodePoint pt3(min.x_ + half_x, min.y_ , 0);
    CompositeNode<BoxType, PointType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType>>(pt3, half_x, half_y, 0));
    NodePoint pt4(min.x_ + half_x, min.y_ + half_y, 0);
    CompositeNode<BoxType, PointType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType>>(pt4, half_x, half_y, 0));
}

template<typename BoxType, typename PointType>
Composite2DNode<BoxType, PointType>::~Composite2DNode() {}


template<typename BoxType, typename PointType>
size_t Composite2DNode<BoxType, PointType>::calcChildIndex(std::shared_ptr<SpaceItem<PointType>>& item) {
    size_t index = 0;
    using coord_type = typename TypeSelector<PointType>::type;
    coord_type* data = (coord_type*) PyArray_DATA(item->pt->coords);
    if (data[0] >= CompositeNode<BoxType, PointType>::x_plane) {
        index = 2;
    }

    if (data[1] >= CompositeNode<BoxType, PointType>::y_plane) {
        index += 1;
    }
    return index;
}



template<typename BoxType, typename PointType>
class Composite3DNode : public CompositeNode<BoxType, PointType> {

protected:
    size_t calcChildIndex(std::shared_ptr<SpaceItem<PointType>>& item) override;

public:
    Composite3DNode(NodePoint& min, double x_extent, double y_extent, double z_extent);
    ~Composite3DNode();
};


template<typename BoxType, typename PointType>
Composite3DNode<BoxType, PointType>::Composite3DNode(NodePoint& min, double x_extent, double y_extent, double z_extent) : 
    CompositeNode<BoxType, PointType>(min, x_extent,  y_extent, z_extent)
{
    double half_x = STNode<BoxType, PointType>::bounds_.x_extent / 2.0;
    double half_y = STNode<BoxType, PointType>::bounds_.y_extent / 2.0;
    double half_z = STNode<BoxType, PointType>::bounds_.z_extent / 2.0;

    CompositeNode<BoxType, PointType>::x_plane = min.x_ + half_x;
    CompositeNode<BoxType, PointType>::y_plane = min.y_ + half_y;
    CompositeNode<BoxType, PointType>::z_plane = min.z_ + half_z;

    NodePoint pt1(min.x_, min.y_, min.z_);
    CompositeNode<BoxType, PointType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType>>(pt1, half_x, half_y, half_z));
    NodePoint pt2(min.x_, min.y_, min.z_ + half_z);
    CompositeNode<BoxType, PointType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType>>(pt2, half_x, half_y, half_z));
    NodePoint pt3(min.x_, min.y_ + half_y, min.z_);
    CompositeNode<BoxType, PointType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType>>(pt3, half_x, half_y, half_z));
    NodePoint pt4(min.x_, min.y_ + half_y, min.z_ + half_z);
    CompositeNode<BoxType, PointType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType>>(pt4, half_x, half_y, half_z));
    NodePoint pt5(min.x_ + half_x, min.y_ , min.z_);
    CompositeNode<BoxType, PointType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType>>(pt5, half_x, half_y, half_z));
    NodePoint pt6(min.x_ + half_x, min.y_ , min.z_ + half_z);
    CompositeNode<BoxType, PointType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType>>(pt6, half_x, half_y, half_z));
    NodePoint pt7(min.x_ + half_x, min.y_ + half_y, min.z_);
    CompositeNode<BoxType, PointType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType>>(pt7, half_x, half_y, half_z));
    NodePoint pt8(min.x_ + half_x, min.y_ + half_y, min.z_ + half_z);
    CompositeNode<BoxType, PointType>::children.push_back(std::make_shared<ComponentNode<BoxType, PointType>>(pt8, half_x, half_y, half_z));
}

template<typename BoxType, typename PointType>
Composite3DNode<BoxType, PointType>::~Composite3DNode() {}


template<typename BoxType, typename PointType>
size_t Composite3DNode<BoxType, PointType>::calcChildIndex(std::shared_ptr<SpaceItem<PointType>>& item) {
    size_t index = 0;
    using coord_type = typename TypeSelector<PointType>::type;
    coord_type* data = (coord_type*) PyArray_DATA(item->pt->coords);

    if (data[0] >= CompositeNode<BoxType, PointType>::x_plane) {
        index = 4;
    }

    if (data[1] >= CompositeNode<BoxType, PointType>::y_plane) {
        index += 2;
    }

    if (data[2] >= CompositeNode<BoxType, PointType>::z_plane) {
        index += 1;
    }
    return index;
}


template<typename BoxType, typename PointType>
class SpatialTree {

private:
    std::shared_ptr<CompositeNode<BoxType, PointType>> root;
    int threshold_;

public:
    SpatialTree(int threshold, const BoundingBox& box); 
    ~SpatialTree();

    void addItem(std::shared_ptr<SpaceItem<PointType>>& item);
    bool removeItem(std::shared_ptr<SpaceItem<PointType>>& item);
    void getObjectsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents);
};

template<typename BoxType, typename PointType>
SpatialTree<BoxType, PointType>::SpatialTree(int threshold, const BoundingBox& box) : root{}, threshold_{threshold} {
    NodePoint min(box.xmin_, box.ymin_, box.zmin_);
    if (box.num_dims == 1) {
        // TODO 
    } else if (box.num_dims == 2) {
        root = std::make_shared<Composite2DNode<BoxType, PointType>>(min, box.x_extent_, box.y_extent_);
    } else if (box.num_dims == 3) {
        // std::cout << box.xmin_ << "," << box.ymin_ << "," << box.zmin_ << "," << box.z_extent_ << std::endl;
        root = std::make_shared<Composite3DNode<BoxType, PointType>>(min, box.x_extent_, box.y_extent_, box.z_extent_);
    }
}

template<typename BoxType, typename PointType>
SpatialTree<BoxType, PointType>::~SpatialTree() {}

template<typename BoxType, typename PointType>
void SpatialTree<BoxType, PointType>::addItem(std::shared_ptr<SpaceItem<PointType>>& item) {
    root->addItem(item, threshold_);
}

template<typename BoxType, typename PointType>
bool SpatialTree<BoxType, PointType>::removeItem(std::shared_ptr<SpaceItem<PointType>>& item) {
    return root->removeItem(item);
}

template<typename BoxType, typename PointType>
void SpatialTree<BoxType, PointType>:: getObjectsWithin(const BoundingBox& box, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) {
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