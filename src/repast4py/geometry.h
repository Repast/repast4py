// Copyright 2021, UChicago Argonne, LLC 
// All Rights Reserved
// Software Name: repast4py
// By: Argonne National Laboratory
// License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

#ifndef R4PY_SRC_GEOMETRY_H
#define R4PY_SRC_GEOMETRY_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <memory>


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// See https://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#define NO_IMPORT_ARRAY_API
//#define PY_ARRAY_UNIQUE_SYMBOL REPAST4PY_ARRAY_API
#include "numpy/arrayobject.h"

#include "types.h"

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
    using type = long_t;
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

    bool operator()(const Point<PointType>& p1, const Point<PointType>& p2) const {
        if (p1.x != p2.x) return p1.x < p2.x;
        if (p1.y != p2.y) return p1.y < p2.y;
        return p1.z < p2.z;
    }
};


template<typename PointType>
struct PtrPointComp {
    using coord_type  = typename TypeSelector<PointType>::type;

    bool operator()(const PointType* p1, const PointType* p2) {
        coord_type* p1_data = (coord_type*)PyArray_DATA(p1->coords);
        coord_type* p2_data = (coord_type*)PyArray_DATA(p2->coords);
        if (p1_data[0] != p2_data[0]) return p1_data[0] < p2_data[0];
        if (p1_data[1] != p2_data[1]) return p1_data[1] < p2_data[1];
        return p1_data[2] < p2_data[2];
    }

    bool operator()(const PointType* p1, const PointType* p2) const {
        coord_type* p1_data = (coord_type*)PyArray_DATA(p1->coords);
        coord_type* p2_data = (coord_type*)PyArray_DATA(p2->coords);
        if (p1_data[0] != p2_data[0]) return p1_data[0] < p2_data[0];
        if (p1_data[1] != p2_data[1]) return p1_data[1] < p2_data[1];
        return p1_data[2] < p2_data[2];
    }
};

struct BoundingBox {
    using coord_type  = typename TypeSelector<R4Py_DiscretePoint>::type;
    coord_type xmin_, xmax_;
    coord_type ymin_, ymax_;
    coord_type zmin_, zmax_;
    coord_type x_extent_, y_extent_, z_extent_;
    unsigned int num_dims;

    BoundingBox(coord_type xmin, coord_type x_extent, coord_type ymin, coord_type y_extent,
            coord_type zmin = 0, coord_type z_extent = 0);
    BoundingBox(const BoundingBox& other);

    ~BoundingBox() {}


    void reset(coord_type xmin, coord_type x_extent, coord_type ymin, coord_type y_extent,
            coord_type zmin = 0, coord_type z_extent = 0);
    bool contains(const R4Py_DiscretePoint* pt) const;
    bool contains(const Point<R4Py_DiscretePoint>& pt) const;
    bool contains(const R4Py_ContinuousPoint* pt) const;
    bool contains(const Point<R4Py_ContinuousPoint>& pt) const;
};


std::ostream& operator<<(std::ostream& os, const BoundingBox& box); 

}

#endif