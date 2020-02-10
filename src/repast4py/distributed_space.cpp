#include <algorithm>

#include "distributed_space.h"

namespace repast4py {


CartesianTopology::CartesianTopology(MPI_Comm comm, int num_dims, bool periodic) : num_dims_{num_dims}, 
    procs_per_dim{nullptr}, periodic_{periodic}
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

CartesianTopology::CartesianTopology(MPI_Comm comm, const std::vector<int>& procs_per_dimension, bool periodic) :
    num_dims_{procs_per_dimension.size()}, procs_per_dim{nullptr}, periodic_{periodic}
{
    int size;
    MPI_Comm_size(comm, &size);
    int total_requested_size = 0;
    for (auto val : procs_per_dimension) {
        total_requested_size += val;
    }

    if (total_requested_size > size) {
        throw std::invalid_argument("Unable to create topology - request number of processes per dimension is greater than available processors");
    }

    procs_per_dim = new int[num_dims_];
    for (int i = 0; i < num_dims_; ++i) {
        procs_per_dim[i] = procs_per_dimension[i];
    }

    int periods[num_dims_];
    std::fill_n(periods, num_dims_, periodic ? 1 : 0);

    MPI_Cart_create(comm, num_dims_, procs_per_dim, periods, 0, &comm_);
}

CartesianTopology::~CartesianTopology() {
    MPI_Comm_free(&comm_);
    delete[] procs_per_dim;
}

void CartesianTopology::getCoords(int rank, std::vector<int>& coords) {
    coords.reserve(num_dims_);
    MPI_Cart_coords(comm_, rank, num_dims_, &coords[0]);
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

void CartesianTopology::getNeighbors(int rank, std::vector<CTNeighbor>& neighbors) {
    int coords[num_dims_];
    MPI_Cart_coords(comm_, rank, num_dims_, coords);
    
    if (num_dims_ == 1) {
        std::list<int> offsets{-1, 1};
        if (!periodic_) {
            if (coords[0] + offsets.front()) offsets.pop_front();
            else if (coords[0] + offsets.back() >= procs_per_dim[0]) offsets.pop_back();
        }
        for (int xd : offsets) {
            int working[] = {coords[0] + xd};
            int n_rank;
            MPI_Cart_rank(comm_, working, &n_rank);
            MPI_Cart_coords(comm_, n_rank, num_dims_, working);
            neighbors.push_back({n_rank, working[0], -1, -1});
        }

    } else if (num_dims_ == 2) {
        std::list<int> xoffsets{-1, 0, 1};
        std::list<int> yoffsets{-1, 0, 1};
        if (!periodic_) {
            if (coords[0] + xoffsets.front() < 0) xoffsets.pop_front();
            else if (coords[0] + xoffsets.back() >= procs_per_dim[0]) xoffsets.pop_back();

            if (coords[1] + yoffsets.front() < 0) yoffsets.pop_front();
            else if (coords[1] + yoffsets.back() >= procs_per_dim[1]) yoffsets.pop_back();
        }
        // for (auto v : xoffsets) {
        //     printf("%d\n", v);
        // }
        for (int xd : xoffsets) {
            for (int yd : yoffsets) {
                if (!(xd == 0 && yd == 0)) {
                    int working[] = {coords[0] + xd, coords[1] + yd};
                    int n_rank;
                    MPI_Cart_rank(comm_, working, &n_rank);
                    MPI_Cart_coords(comm_, n_rank, num_dims_, working);
                    neighbors.push_back({n_rank, working[0], working[1], -1});
                }
            }
        }

    } else if (num_dims_ == 3) {
        std::list<int> xoffsets{-1, 0, 1};
        std::list<int> yoffsets{-1, 0, 1};
        std::list<int> zoffsets{-1, 0, 1};
        if (!periodic_) {
            if (coords[0] + xoffsets.front() < 0) xoffsets.pop_front();
            else if (coords[0] + xoffsets.back() >= procs_per_dim[0]) xoffsets.pop_back();

            if (coords[1] + yoffsets.front() < 0) yoffsets.pop_front();
            else if (coords[1] + yoffsets.back() >= procs_per_dim[1]) yoffsets.pop_back();

            if (coords[2] + zoffsets.front() < 0) zoffsets.pop_front();
            else if (coords[2] + zoffsets.back() >= procs_per_dim[2]) zoffsets.pop_back();
        }
        for (int xd : xoffsets) {
            for (int yd : yoffsets) {
                for (int zd : zoffsets) {
                    if (!(xd == 0 && yd == 0 && zd == 0)) {
                        int working[] = {coords[0] + xd, coords[1] + yd, coords[2] + zd};
                        int n_rank;
                        MPI_Cart_rank(comm_, working, &n_rank);
                        MPI_Cart_coords(comm_, n_rank, num_dims_, working);
                        neighbors.push_back({n_rank, working[0], working[1], working[2]});
                    }
                }
            }
        }
    }

}


}