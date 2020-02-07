#ifndef R4PY_SRC_DISTRIBUTEDSPACE_H
#define R4PY_SRC_DISTRIBUTEDSPACE_H

#define PY_SSIZE_T_CLEAN

#include "mpi.h"

#include "space.h"


namespace repast4py {


class CartesianTopology {

private:
    int num_dims_;
    int* procs_per_dim;
    MPI_Comm comm_;

public:
    CartesianTopology(MPI_Comm, int num_dims, bool periodic);
    ~CartesianTopology();

    void getBounds(int rank, const BoundingBox<R4Py_DiscretePoint>& global_bounds, 
        BoundingBox<R4Py_DiscretePoint>& local_bounds);    


};


}


#endif