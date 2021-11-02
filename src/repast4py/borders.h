// Copyright 2021, UChicago Argonne, LLC 
// All Rights Reserved
// Software Name: repast4py
// By: Argonne National Laboratory
// License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

#ifndef SRC_BORDERS_H
#define SRC_BORDERS_H

#include "geometry.h"

namespace repast4py {


template<typename PointType>
class StickyBorders {

private:
    BoundingBox bounds_;

public:
    using coord_type  = typename TypeSelector<PointType>::type;
    StickyBorders(const BoundingBox& bounds);

    ~StickyBorders() {}

    void transform(const PointType* pt, Point<PointType>& transformed_pt);
    void transform(const PointType* pt, PointType* transformed_pt);
};

template<typename PointType>
StickyBorders<PointType>::StickyBorders(const BoundingBox& bounds) : 
    bounds_{bounds}
{}

template<typename PointType>
struct Offset {
    
};

template<>
struct Offset<R4Py_DiscretePoint> {
    constexpr static long value = 1;
};

template<>
struct Offset<R4Py_ContinuousPoint> {
    constexpr static double value = 0.00000001;
};

template <typename PointType>
void StickyBorders<PointType>::transform(const PointType* pt, PointType* transformed_pt) {
    coord_type* data = (coord_type *)PyArray_DATA(pt->coords);
    coord_type* t_data = (coord_type *)PyArray_DATA(transformed_pt->coords);


    // coord_type v = (bounds_.xmax_ - 1) < data[0] ? (bounds_.xmax_ - 1) : data[0];
    // transformed_pt.x = bounds_.xmin_ > v ? bounds_.xmin_ : v;

    // v = (bounds_.ymax_ - 1) < data[1] ? (bounds_.ymax_ - 1) : data[1];
    // transformed_pt.y = bounds_.ymin_ > v ? bounds_.ymin_ : v;

    // v = (bounds_.zmax_ - 1) < data[2] ? (bounds_.zmax_ - 1) : data[2];
    // transformed_pt.z = bounds_.zmin_ > v ? bounds_.zmin_ : v;

    t_data[0] = std::max((coord_type)bounds_.xmin_, std::min((coord_type)bounds_.xmax_ - Offset<PointType>::value, data[0]));
    t_data[1] = std::max((coord_type)bounds_.ymin_, std::min((coord_type)bounds_.ymax_ - Offset<PointType>::value, data[1]));
    t_data[2] = std::max((coord_type)bounds_.zmin_, std::min((coord_type)bounds_.zmax_ - Offset<PointType>::value, data[2]));
}

template<typename PointType>
void StickyBorders<PointType>::transform(const PointType* pt, Point<PointType>& transformed_pt) {
    coord_type* data = (coord_type*) PyArray_DATA(pt->coords);

    // coord_type v = (bounds_.xmax_ - 1) < data[0] ? (bounds_.xmax_ - 1) : data[0];
    // transformed_pt.x = bounds_.xmin_ > v ? bounds_.xmin_ : v;

    // v = (bounds_.ymax_ - 1) < data[1] ? (bounds_.ymax_ - 1) : data[1];
    // transformed_pt.y = bounds_.ymin_ > v ? bounds_.ymin_ : v;

    // v = (bounds_.zmax_ - 1) < data[2] ? (bounds_.zmax_ - 1) : data[2];
    // transformed_pt.z = bounds_.zmin_ > v ? bounds_.zmin_ : v;

    transformed_pt.x = std::max((coord_type)bounds_.xmin_, std::min((coord_type)bounds_.xmax_ - Offset<PointType>::value, data[0]));
    transformed_pt.y = std::max((coord_type)bounds_.ymin_, std::min((coord_type)bounds_.ymax_ - Offset<PointType>::value, data[1]));
    transformed_pt.z = std::max((coord_type)bounds_.zmin_, std::min((coord_type)bounds_.zmax_ - Offset<PointType>::value, data[2]));
}

using GridStickyBorders = StickyBorders<R4Py_DiscretePoint>;
using CSStickyBorders = StickyBorders<R4Py_ContinuousPoint>;

template <typename PointType>
void transformX(const PointType* pt, Point<PointType>& transformed_pt, const BoundingBox& bounds) {
    using coord_type  = typename TypeSelector<PointType>::type;
    coord_type* data = (coord_type*) PyArray_DATA(pt->coords);

    coord_type nc = fmod((data[0] - bounds.xmin_), bounds.x_extent_);
    transformed_pt.x = nc < 0 ? bounds.xmax_ + nc : bounds.xmin_ + nc;
}

template <typename PointType>
void transformXY(const PointType* pt, Point<PointType>& transformed_pt, const BoundingBox& bounds) {
    using coord_type  = typename TypeSelector<PointType>::type;
    coord_type* data = (coord_type*) PyArray_DATA(pt->coords);

    coord_type nc = fmod(data[0] - bounds.xmin_, bounds.x_extent_);
    transformed_pt.x = nc < 0 ? bounds.xmax_ + nc : bounds.xmin_ + nc;

    nc = fmod(data[1] - bounds.ymin_, bounds.y_extent_);
    transformed_pt.y = nc < 0 ? bounds.ymax_ + nc : bounds.ymin_ + nc; 
}

template <typename PointType>
void transformXYZ(const PointType* pt, Point<PointType>& transformed_pt, const BoundingBox& bounds) {
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
    using trans_func = void(*)(const PointType*, Point<PointType>&, const BoundingBox&);

    BoundingBox bounds_;
    trans_func transform_;
    Point<PointType> temp_pt;

public:
    using coord_type  = typename TypeSelector<PointType>::type;
    PeriodicBorders(const BoundingBox& bounds);

    ~PeriodicBorders() {}

    void transform(const PointType* pt, Point<PointType>& transformed_pt);
    void transform(const PointType* pt, PointType* transformed_pt);
};

template<typename PointType>
PeriodicBorders<PointType>::PeriodicBorders(const BoundingBox& bounds) : 
    bounds_{bounds}, temp_pt{0, 0, 0}
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

template<typename PointType>
void PeriodicBorders<PointType>::transform(const PointType* pt, PointType* transformed_pt) {
    transform_(pt, temp_pt, bounds_);
    coord_type* data = (coord_type *)PyArray_DATA(transformed_pt->coords);
    data[0] = temp_pt.x;
    data[1] = temp_pt.y;
    data[2] = temp_pt.z;
}

using GridPeriodicBorders = PeriodicBorders<R4Py_DiscretePoint>;
using CSPeriodicBorders = PeriodicBorders<R4Py_ContinuousPoint>;

struct R4Py_GridStickyBorders
{
    PyObject_HEAD
    GridStickyBorders* borders;
    
};

struct R4Py_GridPeriodicBorders
{
    PyObject_HEAD
    GridPeriodicBorders* borders;
    
};

}
#endif