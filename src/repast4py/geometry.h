#ifndef R4PY_SRC_SPACECORE_H
#define R4PY_SRC_SPACECORE_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <memory>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// See https://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#define NO_IMPORT_ARRAY_API
#define PY_ARRAY_UNIQUE_SYMBOL REPAST4PY_ARRAY_API
#include "numpy/arrayobject.h"

namespace repast4py {

struct R4Py_DiscretePoint {
    PyObject_HEAD
    PyArrayObject* coords;
};


template<typename PointType>
struct TypeSelector {
    using type = double;
};

template<>
struct TypeSelector<R4Py_DiscretePoint> {
    using type = long;
};

template<typename PointType>
struct Point {
    using coord_type  = typename TypeSelector<PointType>::type;
    coord_type x, y, z;
};


void extract_coords(R4Py_DiscretePoint* pt, Point<R4Py_DiscretePoint>& coords) {
    long* data = (long*)pt->coords->data;
    coords.x = data[0];
    coords.y = data[1];
    coords.z = data[2];
}

void update_point(R4Py_DiscretePoint* pt, const Point<R4Py_DiscretePoint>& coords) {
    long* data = (long*)pt->coords->data;
    data[0] = coords.x;
    data[1] = coords.y;
    data[2] = coords.z;

    //printf("Updated Point: %lu,%lu,%lu\n", data[0], data[1], data[2]);
}

template<typename PointType>
struct PointComp {
    bool operator()(const Point<PointType>& p1, const Point<PointType>& p2) {
        if (p1.x != p2.x) return p1.x < p2.x;
        if (p1.y != p2.y) return p1.y < p2.y;
        return p1.z < p2.z;
    }
};

R4Py_DiscretePoint* create_point(PyTypeObject* pt_type, const Point<R4Py_DiscretePoint>& wpt) {
    R4Py_DiscretePoint* pt = (R4Py_DiscretePoint*)pt_type->tp_new(pt_type, NULL, NULL);
    update_point(pt, wpt);
    return pt;
}

bool point_equals(R4Py_DiscretePoint* pt, const Point<R4Py_DiscretePoint>& coords) {
    if (pt) {
        long* data = (long*)pt->coords->data;
        //printf("%lu,%lu,%lu  -- %lu,%lu,%lu\n", data[0], data[1], data[2],
        //    coords.x, coords.y, coords.z);
        return data[0] == coords.x && data[1] == coords.y && data[2] == coords.z;
    }
    return false;
}


template<typename PointType>
struct BoundingBox {
    using coord_type  = typename TypeSelector<PointType>::type;
    coord_type xmin_, xmax_;
    coord_type ymin_, ymax_;
    coord_type zmin_, zmax_;
    coord_type width_, height_, depth_;

    BoundingBox(coord_type xmin, coord_type width, coord_type ymin, coord_type height,
            coord_type zmin = 0, coord_type depth = 0);

    ~BoundingBox() {}


    bool contains(const PointType* pt) const;
    bool contains(const Point<PointType>& pt) const;

    bool intersects(const BoundingBox<PointType>& box) const;
};

template<typename PointType>
BoundingBox<PointType>::BoundingBox(coord_type xmin, coord_type width, coord_type ymin, coord_type height,
            coord_type zmin, coord_type depth) : xmin_{xmin}, xmax_{xmin + width}, ymin_{ymin},
            ymax_{ymin + height}, zmin_{zmin}, zmax_{zmin + depth}, width_{width}, height_{height},
            depth_{depth} {
}

template<typename PointType>
bool BoundingBox<PointType>::contains(const PointType* pt) const {
    coord_type* data = (coord_type*)pt->coords->data;
    return data[0] >= xmin_ && data[1] >= ymin_ && data[0] < xmax_ && data[1] < ymax_ &&
        data[2] >= zmin_ && data[2] < zmax_;
}

template<typename PointType>
bool BoundingBox<PointType>::contains(const Point<PointType>& pt) const {
    return pt.x >= xmin_ && pt.y >= ymin_ && pt.x < xmax_ && pt.y < ymax_ &&
        pt.z >= zmin_ && pt.z < zmax_;
}

template<typename PointType>
bool BoundingBox<PointType>::intersects(const BoundingBox<PointType>& other) const {
    if (xmin_ >= other.xmax_ || other.xmin_ >= xmax_) return false;
    if (ymin_ >= other.ymax_ || other.ymin_ >= ymax_) return false;
    if (zmin_ >= other.zmax_ || other.zmin_ >= zmax_) return false;

    return true;
}

template<typename PointType>
class StickyBorders {

private:
    BoundingBox<PointType> bounds_;

public:
    using coord_type  = typename TypeSelector<PointType>::type;
    StickyBorders(const BoundingBox<PointType>& bounds);

    ~StickyBorders() {}

    void transform(const PointType* pt, Point<PointType>& transformed_pt);
};

template<typename PointType>
StickyBorders<PointType>::StickyBorders(const BoundingBox<PointType>& bounds) : 
    bounds_{bounds}
{}

template<typename PointType>
void StickyBorders<PointType>::transform(const PointType* pt, Point<PointType>& transformed_pt) {
    coord_type* data = (coord_type*) pt->coords->data;
    transformed_pt.x = std::max(bounds_.xmin_, std::min(bounds_.xmax_ - 1, data[0]));
    transformed_pt.y = std::max(bounds_.ymin_, std::min(bounds_.ymax_ - 1, data[1]));
    transformed_pt.z = std::max(bounds_.zmin_, std::min(bounds_.zmax_ - 1, data[2]));
}

using GridStickyBorders = StickyBorders<R4Py_DiscretePoint>;


}



#endif