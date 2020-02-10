#ifndef R4PY_SRC_DISTRIBUTEDSPACE_H
#define R4PY_SRC_DISTRIBUTEDSPACE_H

#define PY_SSIZE_T_CLEAN

#include <vector>

#include "mpi.h"

#include "space.h"


namespace repast4py {

struct CTNeighbor {
    int rank;
    int cart_coord_x, cart_coord_y, cart_coord_z;
};


class CartesianTopology {

private:
    int num_dims_;
    int* procs_per_dim;
    MPI_Comm comm_;
    bool periodic_;

public:
    CartesianTopology(MPI_Comm, int num_dims, bool periodic);
    CartesianTopology(MPI_Comm, const std::vector<int>& procs_per_dimension, bool periodic);
    ~CartesianTopology();

    void getBounds(int rank, const BoundingBox<R4Py_DiscretePoint>& global_bounds, 
        BoundingBox<R4Py_DiscretePoint>& local_bounds);    
    void getCoords(int rank, std::vector<int>& coords);
    void getNeighbors(int rank, std::vector<CTNeighbor>& neighbors);


};


}


#endif