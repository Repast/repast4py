#include <algorithm>

#include "distributed_space.h"

namespace repast4py {

using LongBox = BoundingBox<R4Py_DiscretePoint>;

CartesianTopology::CartesianTopology(MPI_Comm comm, int num_dims, bool periodic) : num_dims_{num_dims}, 
    procs_per_dim{nullptr} 
{
    int size;
    MPI_Comm_size(comm, &size);

    procs_per_dim = new int[num_dims];
    
    std::fill_n(procs_per_dim, num_dims, 0);
    MPI_Dims_create(size, num_dims, procs_per_dim);
    
    int periods[num_dims];
    std::fill_n(periods, num_dims, periodic ? 1 : 0);

    MPI_Cart_create(comm, num_dims, procs_per_dim, periods, 0, &comm_);
}

CartesianTopology::~CartesianTopology() {
    MPI_Comm_free(&comm_);
    delete[] procs_per_dim;
}

void CartesianTopology::getBounds(int rank, const BoundingBox<R4Py_DiscretePoint>& global_bounds, 
        BoundingBox<R4Py_DiscretePoint>& local_bounds) 
{
    int coords[num_dims_];
    MPI_Cart_coords(comm_, rank, num_dims_, coords);
    long x_extent = global_bounds.x_extent_ / procs_per_dim[0];
    long xmin = global_bounds.xmin_ +  x_extent * coords[0];

    long ymin = 0;
    long y_extent = 0;

    long zmin = 0;
    long z_extent = 0;

    if (num_dims_ > 1) {
        y_extent = global_bounds.y_extent_ / procs_per_dim[1];
        ymin = global_bounds.ymin_ +  y_extent * coords[1];
    }

    if (num_dims_ == 3) {
        z_extent = global_bounds.z_extent_ / procs_per_dim[2];
        zmin = global_bounds.zmin_ +  z_extent * coords[2];
    }

    local_bounds.reset(xmin, x_extent, ymin, y_extent, zmin, z_extent);
} 


}