// Copyright 2021, UChicago Argonne, LLC 
// All Rights Reserved
// Software Name: repast4py
// By: Argonne National Laboratory
// License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

#include "geometry.h"

namespace repast4py {

R4Py_DiscretePoint* create_point(PyTypeObject* pt_type, const Point<R4Py_DiscretePoint>& wpt) {
    R4Py_DiscretePoint* pt = (R4Py_DiscretePoint*)pt_type->tp_new(pt_type, NULL, NULL);
    update_point(pt, wpt);
    return pt;
}

bool point_equals(R4Py_DiscretePoint* pt, const Point<R4Py_DiscretePoint>& coords) {
    if (pt) {
        long* data = (long*)PyArray_DATA(pt->coords);
        //printf("%lu,%lu,%lu  -- %lu,%lu,%lu\n", data[0], data[1], data[2],
        //    coords.x, coords.y, coords.z);
        return data[0] == coords.x && data[1] == coords.y && data[2] == coords.z;
    }
    return false;
}

void extract_coords(R4Py_DiscretePoint* pt, Point<R4Py_DiscretePoint>& coords) {
    long* data = (long*)PyArray_DATA(pt->coords);
    coords.x = data[0];
    coords.y = data[1];
    coords.z = data[2];
}

void update_point(R4Py_DiscretePoint* pt, const Point<R4Py_DiscretePoint>& coords) {
    long* data = (long*)PyArray_DATA(pt->coords);
    data[0] = coords.x;
    data[1] = coords.y;
    data[2] = coords.z;

    //printf("Updated Point: %lu,%lu,%lu\n", data[0], data[1], data[2]);
}

R4Py_ContinuousPoint* create_point(PyTypeObject* pt_type, const Point<R4Py_ContinuousPoint>& wpt) {
    R4Py_ContinuousPoint* pt = (R4Py_ContinuousPoint*)pt_type->tp_new(pt_type, NULL, NULL);
    update_point(pt, wpt);
    return pt;
}

bool point_equals(R4Py_ContinuousPoint* pt, const Point<R4Py_ContinuousPoint>& coords) {
    if (pt) {
        double* data = (double*)PyArray_DATA(pt->coords);
        //printf("%lu,%lu,%lu  -- %lu,%lu,%lu\n", data[0], data[1], data[2],
        //    coords.x, coords.y, coords.z);
        return data[0] == coords.x && data[1] == coords.y && data[2] == coords.z;
    }
    return false;

}
// sets coords.xyz from pt.xyz
void extract_coords(R4Py_ContinuousPoint* pt, Point<R4Py_ContinuousPoint>& coords) {
    double* data = (double*)PyArray_DATA(pt->coords);
    coords.x = data[0];
    coords.y = data[1];
    coords.z = data[2];
}

// sets pt.xyz from coords.xyz
void update_point(R4Py_ContinuousPoint* pt, const Point<R4Py_ContinuousPoint>& coords) {
    double* data = (double*)PyArray_DATA(pt->coords);
    data[0] = coords.x;
    data[1] = coords.y;
    data[2] = coords.z;
}

std::ostream& operator<<(std::ostream& os, const BoundingBox& box) {
    os << "BoundingBox(" << box.xmin_ << ", " << box.x_extent_ << ", "
        << box.ymin_ << ", " << box.y_extent_ << ", "
        << box.zmin_ << ", " << box.z_extent_ << ")";
    return os;
}


BoundingBox::BoundingBox(coord_type xmin, coord_type x_extent, coord_type ymin, coord_type y_extent,
            coord_type zmin, coord_type z_extent) : xmin_{xmin}, xmax_{xmin + x_extent}, ymin_{ymin},
            ymax_{ymin + y_extent}, zmin_{zmin}, zmax_{zmin + z_extent}, x_extent_{x_extent}, y_extent_{y_extent},
            z_extent_{z_extent}, num_dims{1} {

    if (y_extent_ > 0) num_dims = 2;
    if (z_extent_ > 0) num_dims = 3;

}

BoundingBox::BoundingBox(const BoundingBox& other) : xmin_{other.xmin_}, 
    xmax_{other.xmin_ + other.x_extent_}, ymin_{other.ymin_}, ymax_{other.ymin_ + other.y_extent_}, 
    zmin_{other.zmin_}, zmax_{other.zmin_ + other.z_extent_}, x_extent_{other.x_extent_}, 
    y_extent_{other.y_extent_}, z_extent_{other.z_extent_}, num_dims(1)
{
    if (y_extent_ > 0) num_dims = 2;
    if (z_extent_ > 0) num_dims = 3;

}


void BoundingBox::reset(coord_type xmin, coord_type x_extent, coord_type ymin, coord_type y_extent,
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


bool BoundingBox::contains(const R4Py_DiscretePoint* pt) const {
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

bool BoundingBox::contains(const Point<R4Py_DiscretePoint>& pt) const {
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

 bool BoundingBox::contains(const R4Py_ContinuousPoint* pt) const {
    using pt_type = typename TypeSelector<R4Py_ContinuousPoint>::type;
    pt_type* data = (pt_type*)PyArray_DATA(pt->coords);

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

bool BoundingBox::contains(const Point<R4Py_ContinuousPoint>& pt) const {
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


}
