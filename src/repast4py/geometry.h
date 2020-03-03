#ifndef R4PY_SRC_SPACECORE_H
#define R4PY_SRC_SPACECORE_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <iostream>
#include <algorithm>
#include <memory>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// See https://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#define NO_IMPORT_ARRAY_API
//#define PY_ARRAY_UNIQUE_SYMBOL REPAST4PY_ARRAY_API
#include "numpy/arrayobject.h"

namespace repast4py {

struct R4Py_DiscretePoint {
    PyObject_HEAD
    // array of longs
    PyArrayObject* coords;
};

struct R4Py_ContinuousPoint {
    PyObject_HEAD
    // array of doubles
    PyArrayObject* coords;
};


template<typename PointType>
struct TypeSelector {
    //using type = double;
};

template<>
struct TypeSelector<R4Py_DiscretePoint> {
    using type = long;
};

template<>
struct TypeSelector<R4Py_ContinuousPoint> {
    using type = double;
};

template<typename PointType>
struct Point {
    using coord_type  = typename TypeSelector<PointType>::type;
    coord_type x, y, z;
};

R4Py_DiscretePoint* create_point(PyTypeObject* pt_type, const Point<R4Py_DiscretePoint>& wpt);
bool point_equals(R4Py_DiscretePoint* pt, const Point<R4Py_DiscretePoint>& coords);
// sets coords.xyz from pt.xyz
void extract_coords(R4Py_DiscretePoint* pt, Point<R4Py_DiscretePoint>& coords);
// sets pt.xyz from coords.xyz
void update_point(R4Py_DiscretePoint* pt, const Point<R4Py_DiscretePoint>& coords);

R4Py_ContinuousPoint* create_point(PyTypeObject* pt_type, const Point<R4Py_ContinuousPoint>& wpt);
bool point_equals(R4Py_ContinuousPoint* pt, const Point<R4Py_ContinuousPoint>& coords);
// sets coords.xyz from pt.xyz
void extract_coords(R4Py_ContinuousPoint* pt, Point<R4Py_ContinuousPoint>& coords);
// sets pt.xyz from coords.xyz
void update_point(R4Py_ContinuousPoint* pt, const Point<R4Py_ContinuousPoint>& coords);


template<typename PointType>
struct PointComp {
    bool operator()(const Point<PointType>& p1, const Point<PointType>& p2) {
        if (p1.x != p2.x) return p1.x < p2.x;
        if (p1.y != p2.y) return p1.y < p2.y;
        return p1.z < p2.z;
    }
};

template<typename PointType>
struct BoundingBox {
    using coord_type  = typename TypeSelector<PointType>::type;
    coord_type xmin_, xmax_;
    coord_type ymin_, ymax_;
    coord_type zmin_, zmax_;
    coord_type x_extent_, y_extent_, z_extent_;
    unsigned int num_dims;

    BoundingBox(coord_type xmin, coord_type x_extent, coord_type ymin, coord_type y_extent,
            coord_type zmin = 0, coord_type z_extent = 0);
    BoundingBox(const BoundingBox<PointType>& other);

    ~BoundingBox() {}


    void reset(coord_type xmin, coord_type x_extent, coord_type ymin, coord_type y_extent,
            coord_type zmin = 0, coord_type z_extent = 0);
    bool contains(const PointType* pt) const;
    bool contains(const Point<PointType>& pt) const;
};


template<typename PointType>
std::ostream& operator<<(std::ostream& os, const BoundingBox<PointType>& box) {
    os << "BoundingBox(" << box.xmin_ << ", " << box.x_extent_ << ", "
        << box.ymin_ << ", " << box.y_extent_ << ", "
        << box.zmin_ << ", " << box.z_extent_ << ")";
    return os;
}

template<typename PointType>
BoundingBox<PointType>::BoundingBox(coord_type xmin, coord_type x_extent, coord_type ymin, coord_type y_extent,
            coord_type zmin, coord_type z_extent) : xmin_{xmin}, xmax_{xmin + x_extent}, ymin_{ymin},
            ymax_{ymin + y_extent}, zmin_{zmin}, zmax_{zmin + z_extent}, x_extent_{x_extent}, y_extent_{y_extent},
            z_extent_{z_extent} {
}

template<typename PointType>
BoundingBox<PointType>::BoundingBox(const BoundingBox<PointType>& other) : xmin_{other.xmin_}, 
    xmax_{other.xmin_ + other.x_extent_}, ymin_{other.ymin_}, ymax_{other.ymin_ + other.y_extent_}, 
    zmin_{other.zmin_}, zmax_{other.zmin_ + other.z_extent_}, x_extent_{other.x_extent_}, 
    y_extent_{other.y_extent_}, z_extent_{other.z_extent_}, num_dims(1)
{
    if (y_extent_ > 0) num_dims = 2;
    if (z_extent_ > 0) num_dims = 3;

}

template<typename PointType>
void BoundingBox<PointType>::reset(coord_type xmin, coord_type x_extent, coord_type ymin, coord_type y_extent,
            coord_type zmin, coord_type z_extent) 
{
    xmin_ = xmin;
    x_extent_ = x_extent;
    xmax_ = xmin + x_extent;

    ymin_ = ymin;
    y_extent_ = y_extent;
    ymax_ = ymin + y_extent;

    zmin_ = zmin;
    z_extent_ = z_extent;
    zmax_ = zmin + z_extent;

    num_dims = 1;
    if (y_extent_ > 0) num_dims = 2;
    if (z_extent_ > 0) num_dims = 3;
}

template<typename PointType>
bool BoundingBox<PointType>::contains(const PointType* pt) const {
    coord_type* data = (coord_type*)PyArray_DATA(pt->coords);

    bool y_contains = true;
    bool z_contains = true;
    bool x_contains = data[0] >= xmin_ && data[0] < xmax_;

    if (num_dims == 2) {
        y_contains = data[1] >= ymin_ && data[1] < ymax_;
    } else if (num_dims == 3) {
        y_contains = data[1] >= ymin_ && data[1] < ymax_;
        z_contains =  data[2] >= zmin_ && data[2] < zmax_;
    }

    return x_contains && y_contains && z_contains;
}

template<typename PointType>
bool BoundingBox<PointType>::contains(const Point<PointType>& pt) const {
    bool y_contains = true;
    bool z_contains = true;
    bool x_contains = pt.x >= xmin_ && pt.x < xmax_;

    if (num_dims == 2) {
        y_contains = pt.y >= ymin_ && pt.y < ymax_;
    } else if (num_dims == 3) {
        y_contains = pt.y >= ymin_ && pt.y < ymax_;
        z_contains = pt.z >= zmin_ && pt.z < zmax_;
    }

    return x_contains && y_contains && z_contains;
}


template<typename PointType>
class StickyBorders {

private:
    BoundingBox<R4Py_DiscretePoint> bounds_;

public:
    using coord_type  = typename TypeSelector<PointType>::type;
    StickyBorders(const BoundingBox<R4Py_DiscretePoint>& bounds);

    ~StickyBorders() {}

    void transform(const PointType* pt, Point<PointType>& transformed_pt);
};

template<typename PointType>
StickyBorders<PointType>::StickyBorders(const BoundingBox<R4Py_DiscretePoint>& bounds) : 
    bounds_{bounds}
{}

template<typename PointType>
void StickyBorders<PointType>::transform(const PointType* pt, Point<PointType>& transformed_pt) {
    coord_type* data = (coord_type*) PyArray_DATA(pt->coords);

    // coord_type v = (bounds_.xmax_ - 1) < data[0] ? (bounds_.xmax_ - 1) : data[0];
    // transformed_pt.x = bounds_.xmin_ > v ? bounds_.xmin_ : v;

    // v = (bounds_.ymax_ - 1) < data[1] ? (bounds_.ymax_ - 1) : data[1];
    // transformed_pt.y = bounds_.ymin_ > v ? bounds_.ymin_ : v;

    // v = (bounds_.zmax_ - 1) < data[2] ? (bounds_.zmax_ - 1) : data[2];
    // transformed_pt.z = bounds_.zmin_ > v ? bounds_.zmin_ : v;

    transformed_pt.x = std::max((coord_type)bounds_.xmin_, std::min((coord_type)bounds_.xmax_ - 1, data[0]));
    transformed_pt.y = std::max((coord_type)bounds_.ymin_, std::min((coord_type)bounds_.ymax_ - 1, data[1]));
    transformed_pt.z = std::max((coord_type)bounds_.zmin_, std::min((coord_type)bounds_.zmax_ - 1, data[2]));
}

using GridStickyBorders = StickyBorders<R4Py_DiscretePoint>;
using CSStickyBorders = StickyBorders<R4Py_ContinuousPoint>;

template <typename PointType>
void transformX(const PointType* pt, Point<PointType>& transformed_pt, const BoundingBox<R4Py_DiscretePoint>& bounds) {
    using coord_type  = typename TypeSelector<PointType>::type;
    coord_type* data = (coord_type*) PyArray_DATA(pt->coords);

    coord_type nc = fmod((data[0] - bounds.xmin_), bounds.x_extent_);
    transformed_pt.x = nc < 0 ? bounds.xmax_ + nc : bounds.xmin_ + nc;
}

template <typename PointType>
void transformXY(const PointType* pt, Point<PointType>& transformed_pt, const BoundingBox<R4Py_DiscretePoint>& bounds) {
    using coord_type  = typename TypeSelector<PointType>::type;
    coord_type* data = (coord_type*) PyArray_DATA(pt->coords);

    coord_type nc = fmod(data[0] - bounds.xmin_, bounds.x_extent_);
    transformed_pt.x = nc < 0 ? bounds.xmax_ + nc : bounds.xmin_ + nc;

    nc = fmod(data[1] - bounds.ymin_, bounds.y_extent_);
    transformed_pt.y = nc < 0 ? bounds.ymax_ + nc : bounds.ymin_ + nc; 
}

template <typename PointType>
void transformXYZ(const PointType* pt, Point<PointType>& transformed_pt, const BoundingBox<R4Py_DiscretePoint>& bounds) {
    using coord_type  = typename TypeSelector<PointType>::type;
    coord_type* data = (coord_type*) PyArray_DATA(pt->coords);

    coord_type nc = fmod((data[0] - bounds.xmin_), bounds.x_extent_);
    transformed_pt.x = nc < 0 ? bounds.xmax_ + nc : bounds.xmin_ + nc;

    nc = fmod(data[1] - bounds.ymin_, bounds.y_extent_);
    transformed_pt.y = nc < 0 ? bounds.ymax_ + nc : bounds.ymin_ + nc; 

    nc = fmod(data[2] - bounds.zmin_, bounds.z_extent_);
    transformed_pt.z = nc < 0 ? bounds.zmax_ + nc : bounds.zmin_ + nc;
}

template<typename PointType>
class PeriodicBorders {

private:
    using trans_func = void(*)(const PointType*, Point<PointType>&, const BoundingBox<R4Py_DiscretePoint>&);

    BoundingBox<R4Py_DiscretePoint> bounds_;
    trans_func transform_;

public:
    using coord_type  = typename TypeSelector<PointType>::type;
    PeriodicBorders(const BoundingBox<R4Py_DiscretePoint>& bounds);

    ~PeriodicBorders() {}

    void transform(const PointType* pt, Point<PointType>& transformed_pt);
};

template<typename PointType>
PeriodicBorders<PointType>::PeriodicBorders(const BoundingBox<R4Py_DiscretePoint>& bounds) : 
    bounds_{bounds}
{
    if (bounds_.x_extent_ > 0 && bounds_.y_extent_ > 0 && bounds_.z_extent_ > 0) {
        transform_ = &transformXYZ<PointType>;
    } else if (bounds_.x_extent_ > 0 && bounds_.y_extent_ > 0) {
        transform_ = &transformXY<PointType>;
    } else {
        transform_ = &transformX<PointType>;
    }
}

template<typename PointType>
void PeriodicBorders<PointType>::transform(const PointType* pt, Point<PointType>& transformed_pt) {
    transform_(pt, transformed_pt, bounds_);
}

using GridPeriodicBorders = PeriodicBorders<R4Py_DiscretePoint>;
using CSPeriodicBorders = PeriodicBorders<R4Py_ContinuousPoint>;

}



#endif