// Copyright 2021, UChicago Argonne, LLC 
// All Rights Reserved
// Software Name: repast4py
// By: Argonne National Laboratory
// License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

#include "SpatialTree.h"

namespace repast4py {

NodePoint::NodePoint(double x, double y, double z) : x_{x}, y_{y}, z_{z} {}

// ignore the z coordinate
Box2D::Box2D(NodePoint& min, double max_x, double max_y, double max_z) : min_{min}, max_{max_x, max_y, 0}, x_extent{max_.x_ - min.x_}, 
    y_extent{max_.y_ - min.y_}, z_extent{0} {}

bool Box2D::contains(R4Py_ContinuousPoint* pt) { 
    double* data = (double*)PyArray_DATA(pt->coords);
    // std::cout << min_.x_ << ", " << max_.x_ << ", " << min_.y_ << ", " << max_.y_ << std::endl;
    return data[0] >= min_.x_ && data[0] <= max_.x_ 
        && data[1] >= min_.y_ && data[1] <= max_.y_;
}

bool Box2D::intersects(const BoundingBox& bbox) {
    if (min_.x_ > bbox.xmax_ || bbox.xmin_ > max_.x_) return false;
    if (min_.y_ > bbox.ymax_ || bbox.ymin_ > max_.y_) return false;
    return true;
}

Box3D::Box3D(NodePoint& min,  double max_x, double max_y, double max_z) : min_{min}, max_{max_x, max_y, max_z}, x_extent{max_x - min.x_}, 
    y_extent{max_y - min.y_}, z_extent{max_z - min.z_} {}

bool Box3D::contains(R4Py_ContinuousPoint* pt) { 
    double* data = (double*)PyArray_DATA(pt->coords);
    return data[0] >= min_.x_ && data[0] <= max_.x_ 
        && data[1] >= min_.y_ && data[1] <= max_.y_
        && data[2] >= min_.z_ && data[2] <= max_.z_;
}

bool Box3D::intersects(const BoundingBox& bbox) {
    if (min_.x_ > bbox.xmax_ || bbox.xmin_ > max_.x_) return false;
    if (min_.y_ > bbox.ymax_ || bbox.ymin_ > max_.y_) return false;
    if (min_.z_ > bbox.zmax_ || bbox.zmin_ > max_.z_) return false;

    return true;
}


}