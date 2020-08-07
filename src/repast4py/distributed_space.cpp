#include <algorithm>

#include "distributed_space.h"

namespace repast4py {

void compute_buffer_bounds(CTNeighbor& ngh, int offsets[], int num_dims, BoundingBox& local_bounds, 
    unsigned int buffer_size) 
{
    long xmin = local_bounds.xmin_, xmax = local_bounds.xmax_;
    long ymin = 0, ymax = 0;
    if (num_dims == 2) {
        ymin = local_bounds.ymin_; 
        ymax = local_bounds.ymax_;
    }
    long zmin = 0, zmax = 0;
    if (num_dims == 3) {
        ymin = local_bounds.ymin_; 
        ymax = local_bounds.ymax_;

        zmin = local_bounds.zmin_;
        zmax = local_bounds.zmax_;
    }

    if (offsets[0] == -1 || offsets[0] == 2) {
        xmin = local_bounds.xmin_;
        xmax = xmin + buffer_size;
    } else if (offsets[0] == 1 || offsets[0] == -2) {
        xmin = local_bounds.xmax_ - buffer_size;
        xmax = xmin + buffer_size;
    }

    if (offsets[1] == -1 || offsets[1] == 2) {
        ymin = local_bounds.ymin_;
        ymax = ymin + buffer_size;
    } else if (offsets[1] == 1 || offsets[1] == -2) {
        ymin = local_bounds.ymax_ - buffer_size;
        ymax = ymin + buffer_size;
    }

    if (offsets[2] == -1 || offsets[2] == 2) {
        zmin = local_bounds.zmin_;
        zmax = zmin + buffer_size;
    } else if (offsets[2] == 1 || offsets[2] == -2) {
        zmin = local_bounds.zmax_ - buffer_size;
        zmax = zmin + buffer_size;
    }

    ngh.buffer_info = Py_BuildValue("(i(llllll))", ngh.rank, xmin, xmax, ymin, ymax,
        zmin, zmax);
}

void compute_neighbor_buffers(std::vector<CTNeighbor>& nghs, std::vector<int>& cart_coords, 
    BoundingBox& local_bounds, int num_dims, unsigned int buffer_size)
{
    int offsets[3] = {0, 0, 0};

    if (num_dims == 1) {
        for (auto& ngh : nghs) {
            offsets[0] = ngh.cart_coord_x - cart_coords[0];
            compute_buffer_bounds(ngh, offsets, num_dims, local_bounds, buffer_size);
        }
    } else if (num_dims == 2) {
        for (auto& ngh : nghs) {
            offsets[0] = ngh.cart_coord_x - cart_coords[0];
            offsets[1] = ngh.cart_coord_y - cart_coords[1];
            compute_buffer_bounds(ngh, offsets, num_dims, local_bounds, buffer_size);
        }
    } else if (num_dims == 3) {
        for (auto& ngh : nghs) {
            offsets[0] = ngh.cart_coord_x - cart_coords[0];
            offsets[1] = ngh.cart_coord_y - cart_coords[1];
            offsets[2] = ngh.cart_coord_z - cart_coords[2];
            // if (rank == 0) {
            //     printf("Offsets: %d - %d, %d, %d\n", ngh.rank, offsets[0], offsets[1], offsets[2]);
            // }
            compute_buffer_bounds(ngh, offsets, num_dims, local_bounds, buffer_size);
        }
    }
}


CartesianTopology::CartesianTopology(MPI_Comm comm, MPI_Comm* cart_comm, int num_dims, const BoundingBox& global_bounds, bool periodic) : 
    num_dims_{num_dims},  procs_per_dim{nullptr}, periodic_{periodic}, bounds_(global_bounds), x_remainder{0},
    y_remainder{0}, z_remainder{0}, comm_{}
{
    int size;
    MPI_Comm_size(comm, &size);

    procs_per_dim = new int[num_dims];
    std::fill_n(procs_per_dim, num_dims, 0);
    MPI_Dims_create(size, num_dims, procs_per_dim);

    int periods[num_dims];
    std::fill_n(periods, num_dims, periodic ? 1 : 0);

    MPI_Cart_create(comm, num_dims, procs_per_dim, periods, 0, cart_comm);
    comm_ = *cart_comm;

    x_remainder = bounds_.x_extent_ % procs_per_dim[0];
    if (num_dims_ > 1) {
        y_remainder = bounds_.y_extent_ % procs_per_dim[1];
        if (num_dims_ == 3) {
            z_remainder = bounds_.z_extent_ % procs_per_dim[2];
        }
    }
}

CartesianTopology::CartesianTopology(MPI_Comm comm, MPI_Comm* cart_comm, const std::vector<int>& procs_per_dimension, 
    const BoundingBox& global_bounds,bool periodic) :
    num_dims_{(int)procs_per_dimension.size()}, procs_per_dim{nullptr}, 
    periodic_{periodic}, bounds_(global_bounds), x_remainder{0},
    y_remainder{0}, z_remainder{0}, comm_{}
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

    MPI_Cart_create(comm, num_dims_, procs_per_dim, periods, 0, cart_comm);
    comm_ = *cart_comm;
    x_remainder = bounds_.x_extent_ % procs_per_dim[0];
    if (num_dims_ > 1) {
        y_remainder = bounds_.y_extent_ % procs_per_dim[1];
        if (num_dims_ == 3) {
            z_remainder = bounds_.z_extent_ % procs_per_dim[2];
        }
    }
}

CartesianTopology::~CartesianTopology() {
    delete[] procs_per_dim;
}

void CartesianTopology::getCoords(std::vector<int>& coords) {
    coords.reserve(num_dims_);
    MPI_Cart_coords(comm_, getRank(), num_dims_, &coords[0]);
}

static void adjust_min_extent(int coord, int remainder, long* min, long* extent) {
    if (remainder > 0) {
        (*extent) += coord < remainder ?  1 : 0;
        if (coord > 0 && coord < remainder) {
            (*min) += remainder - (remainder - coord);
        } else if (coord >= remainder) {
            (*min) += remainder;
        }
    }
}

int CartesianTopology::getRank() {
    int rank;
    MPI_Comm_rank(comm_, &rank);
    return rank;
}

void CartesianTopology::getBounds(int rank, BoundingBox& local_bounds) {
    int coords[num_dims_];
    MPI_Cart_coords(comm_, rank, num_dims_, coords);
    long x_extent = floor(bounds_.x_extent_ / procs_per_dim[0]);
    long xmin = bounds_.xmin_ + x_extent * coords[0];
    adjust_min_extent(coords[0], x_remainder, &xmin, &x_extent);

    long ymin = 0;
    long y_extent = 0;
    long zmin = 0;
    long z_extent = 0;

    if (num_dims_ > 1) {
        y_extent = floor(bounds_.y_extent_ / procs_per_dim[1]);
        ymin = bounds_.ymin_ +  y_extent * coords[1];
        adjust_min_extent(coords[1], y_remainder, &ymin, &y_extent);
    }

    if (num_dims_ == 3) {
        z_extent = floor(bounds_.z_extent_ / procs_per_dim[2]);
        zmin = bounds_.zmin_ +  z_extent * coords[2];
        adjust_min_extent(coords[2], z_remainder, &zmin, &z_extent);
    }

    local_bounds.reset(xmin, x_extent, ymin, y_extent, zmin, z_extent);

}

void CartesianTopology::getBounds(BoundingBox& local_bounds) {
    int rank = getRank();
    getBounds(rank, local_bounds);
}

void CartesianTopology::getNeighbors(std::vector<CTNeighbor>& neighbors) {
    int coords[num_dims_];
    MPI_Cart_coords(comm_, getRank(), num_dims_, coords);

    // we can get duplicate neighbors when we have periodic space 
    // with only two procs in that dimension -- i.e. 1 and -1 offset 
    // point to the same neighbor rank
    std::map<int, CTNeighbor> ngh_map;

    if (num_dims_ == 1) {
        std::list<int> offsets{-1, 1};
        if (!periodic_) {
            if (coords[0] + offsets.front() < 0) offsets.pop_front();
            if (coords[0] + offsets.back() >= procs_per_dim[0]) offsets.pop_back();
        }
        for (int xd : offsets) {
            int working[] = {coords[0] + xd};
            int n_rank;
            MPI_Cart_rank(comm_, working, &n_rank);
            MPI_Cart_coords(comm_, n_rank, num_dims_, working);
            ngh_map.emplace(n_rank, CTNeighbor{n_rank, working[0], -1, -1, nullptr,
                {0, 0, 0, 0}});
        }

    } else if (num_dims_ == 2) {
        std::list<int> xoffsets{-1, 0, 1};
        std::list<int> yoffsets{-1, 0, 1};
        if (!periodic_) {
            if (coords[0] + xoffsets.front() < 0) xoffsets.pop_front();
            if (coords[0] + xoffsets.back() >= procs_per_dim[0]) xoffsets.pop_back();

            if (coords[1] + yoffsets.front() < 0) yoffsets.pop_front();
            if (coords[1] + yoffsets.back() >= procs_per_dim[1]) yoffsets.pop_back();
        }

        for (int xd : xoffsets) {
            for (int yd : yoffsets) {
                if (!(xd == 0 && yd == 0)) {
                    int working[] = {coords[0] + xd, coords[1] + yd};
                    //printf("%d: (%d, %d)\n", getRank(), working[0], working[1]);
                    int n_rank;
                    MPI_Cart_rank(comm_, working, &n_rank);
                    MPI_Cart_coords(comm_, n_rank, num_dims_, working);
                    ngh_map.emplace(n_rank, CTNeighbor{n_rank, working[0], working[1], -1, nullptr,
                    {0, 0, 0, 0}});
                }
            }
        }

    } else if (num_dims_ == 3) {
        std::list<int> xoffsets{-1, 0, 1};
        std::list<int> yoffsets{-1, 0, 1};
        std::list<int> zoffsets{-1, 0, 1};
        if (!periodic_) {
            if (coords[0] + xoffsets.front() < 0) xoffsets.pop_front();
            if (coords[0] + xoffsets.back() >= procs_per_dim[0]) xoffsets.pop_back();

            if (coords[1] + yoffsets.front() < 0) yoffsets.pop_front();
            if (coords[1] + yoffsets.back() >= procs_per_dim[1]) yoffsets.pop_back();

            if (coords[2] + zoffsets.front() < 0) zoffsets.pop_front();
            if (coords[2] + zoffsets.back() >= procs_per_dim[2]) zoffsets.pop_back();
        }
        for (int xd : xoffsets) {
            for (int yd : yoffsets) {
                for (int zd : zoffsets) {
                    if (!(xd == 0 && yd == 0 && zd == 0)) {
                        int working[] = {coords[0] + xd, coords[1] + yd, coords[2] + zd};
                        int n_rank;
                        MPI_Cart_rank(comm_, working, &n_rank);
                        MPI_Cart_coords(comm_, n_rank, num_dims_, working);
                        // if (getRank() == 0) {
                        //     printf("rank: %d, nds: %d, %d, %d, coords: %d, %d, %d, offsets: %d, %d, %d\n", n_rank, xd, yd, zd,
                        //     coords[0], coords[1], coords[2], working[0], working[1], working[2]);
                        // }
                        ngh_map.emplace(n_rank, CTNeighbor{n_rank, working[0], working[1], working[2], nullptr,
                        {0, 0, 0, 0}});
                    }
                }
            }
        }
    }

    for (auto& kv : ngh_map) {
        neighbors.push_back(kv.second);
    }
 
    for (auto& ngh : neighbors) {
        getBounds(ngh.rank, ngh.local_bounds);
    }
}


}