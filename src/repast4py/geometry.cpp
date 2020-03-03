
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

}
