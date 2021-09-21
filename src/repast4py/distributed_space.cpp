// Copyright 2021, UChicago Argonne, LLC 
// All Rights Reserved
// Software Name: repast4py
// By: Argonne National Laboratory
// License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

#include <algorithm>
#include <set>

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

    // 1: ngh to right of me.
    // -1: ngh to left of me.
    // > 1: ngh is wrapped to left
    // < -1: ngh is wrapped to right
    if (offsets[0] == -1 || offsets[0] > 1) {
        xmin = local_bounds.xmin_;
        xmax = xmin + buffer_size;
    } else if (offsets[0] == 1 || offsets[0] < -1) {
        xmin = local_bounds.xmax_ - buffer_size;
        xmax = xmin + buffer_size;
    }

    if (offsets[1] == -1 || offsets[1] > 1) {
        ymin = local_bounds.ymin_;
        ymax = ymin + buffer_size;
    } else if (offsets[1] == 1 || offsets[1] < -1) {
        ymin = local_bounds.ymax_ - buffer_size;
        ymax = ymin + buffer_size;
    }

    if (offsets[2] == -1 || offsets[2] > 1) {
        zmin = local_bounds.zmin_;
        zmax = zmin + buffer_size;
    } else if (offsets[2] == 1 || offsets[2] < -1) {
        zmin = local_bounds.zmax_ - buffer_size;
        zmax = zmin + buffer_size;
    }

    ngh.buffer_info = Py_BuildValue("(i(llllll))", ngh.rank, xmin, xmax, ymin, ymax,
        zmin, zmax);
}

void compute_neighbor_buffers(std::vector<CTNeighbor>& nghs, std::vector<int>& cart_coords, 
    BoundingBox& local_bounds, int num_dims, const int* procs_per_dim, unsigned int buffer_size)
{
    int offsets[3] = {0, 0, 0};

    std::set<int> p_ranks;
    if (num_dims == 1) {
        for (auto& ngh : nghs) {
            offsets[0] = ngh.cart_coord_x - cart_coords[0];
            // if in p_ranks:
            //    coord_x > card_coords[0] -- to the right of me, so offsets to -1 as if to the left of me.
            //    coord_x < card_coords[0] -- to the left of me , so offsets to 1 as if to right of me
            if (p_ranks.find(ngh.rank) != p_ranks.end()) {
                if (offsets[0] < 0) offsets[0] = 1;
                else offsets[0] = -1;
            }
            compute_buffer_bounds(ngh, offsets, num_dims, local_bounds, buffer_size);
            p_ranks.emplace(ngh.rank);
        }
    } else if (num_dims == 2) {
        // bool is_0 = cart_coords[0] == 0 && cart_coords[1] == 0;
        
        // first element is number of times ngh appears, 2 and 3 and cart coord
        // differences
        std::map<int, std::vector<int>> coord_offset_counts;
        // if (is_0) std::cout << "num nghs: " << nghs.size() << std::endl;
        for (auto& ngh : nghs) {
            if (coord_offset_counts.find(ngh.rank) == coord_offset_counts.end()) {
                int ox = ngh.cart_coord_x - cart_coords[0];
                int oy = ngh.cart_coord_y - cart_coords[1];
                // if (is_0) std::cout << "ngh offsets for " << ngh.rank << ": " << ox << ", " << oy << std::endl;
                coord_offset_counts.emplace(ngh.rank, std::vector<int>{1, ox, oy});
            } else {
                coord_offset_counts[ngh.rank][0] += 1;
                // int ox = ngh.cart_coord_x - cart_coords[0];
                // int oy = ngh.cart_coord_y - cart_coords[1];
                // if (is_0) std::cout << "ngh offsets for " << ngh.rank << ": " << ox << ", " << oy << std::endl;
            }
        }

        bool x2 = procs_per_dim[0] == 2;
        bool y2 = procs_per_dim[1] == 2;

        std::map<int, std::vector<int>> fixed_offsets;
        for (auto kv : coord_offset_counts) {
            auto& val = kv.second;
            if (val[0] == 1) {
                fixed_offsets.emplace(kv.first, std::vector<int>{val[1], val[2]});
            } else if (val[0] == 2) {
                // which dimension is duplicated
                if (val[1] != 0 && x2) {
                    fixed_offsets.emplace(kv.first, std::vector<int>{1, val[2], -1 , val[2]});
                } 
                
                if (val[2] != 0 && y2) {
                    fixed_offsets.emplace(kv.first, std::vector<int>{val[1], 1, val[1], -1});
                }
            } else if (val[0] == 4) {
                // both dimensions duplicated
                fixed_offsets.emplace(kv.first, std::vector<int>{1, 1, 1, -1, -1, 1, -1, -1});
            } else if (val[0] == 6) {
                // this will occur with 2 procs and periodic -- meet at left / right or top / bottom
                // and 4 corners
                if (x2) {
                    fixed_offsets.emplace(kv.first, std::vector<int>{-1, -1, -1, 1, 1, -1, 1, 1, -1, 0, 1, 0});
                } else if (y2) {
                    fixed_offsets.emplace(kv.first, std::vector<int>{-1, -1, -1, 1, 1, -1, 1, 1, 0, -1, 0, 1});
                }
            }
        }

        std::map<int, int> ngh_offsets_idx;
        for (auto& ngh : nghs) {
            int idx = ngh_offsets_idx[ngh.rank];
            ngh_offsets_idx[ngh.rank] += 2;
            auto& ngh_offsets = fixed_offsets[ngh.rank];
            offsets[0] = ngh_offsets[idx];
            offsets[1] = ngh_offsets[idx + 1];
            compute_buffer_bounds(ngh, offsets, num_dims, local_bounds, buffer_size);
            // if (is_0) std::cout << "fixed ngh offsets for " << ngh.rank << ": " << offsets[0] << ", " << offsets[1] << std::endl;
        }

        // for (auto& ngh : nghs) {
        //     offsets[0] = ngh.cart_coord_x - cart_coords[0];
        //     offsets[1] = ngh.cart_coord_y - cart_coords[1];
        //     if (is_0) std::cout << "ngh offsets for " << ngh.rank << ": " << offsets[0] << ", " << offsets[1] << std::endl;
        //     if (p_ranks.find(ngh.rank) != p_ranks.end()) {
        //         if (x2) {
        //             if (offsets[0] < 0) offsets[0] = 1;
        //             else if (offsets[0] > 0) offsets[0] = -1;
        //         }
        //         if (y2) {
        //             if (offsets[1] < 0) offsets[1] = 1;
        //             else if (offsets[1] > 0) offsets[1] = -1;
        //         }
        //         if (is_0) std::cout << "ngh offsets for " << ngh.rank << ": " << offsets[0] << ", " << offsets[1] << std::endl;
        //     }
        //     compute_buffer_bounds(ngh, offsets, num_dims, local_bounds, buffer_size);
        //     p_ranks.emplace(ngh.rank);
        // }
    } else if (num_dims == 3) {
        bool x2 = procs_per_dim[0] == 2;
        bool y2 = procs_per_dim[1] == 2;
        bool z2 = procs_per_dim[2] == 2;
        for (auto& ngh : nghs) {
            offsets[0] = ngh.cart_coord_x - cart_coords[0];
            offsets[1] = ngh.cart_coord_y - cart_coords[1];
            offsets[2] = ngh.cart_coord_z - cart_coords[2];
            if (p_ranks.find(ngh.rank) != p_ranks.end()) {
                if (x2) {
                    if (offsets[0] < 0) offsets[0] = 1;
                    else if (offsets[0] > 0) offsets[0] = -1;

                }
                if (y2) {
                    if (offsets[1] < 0) offsets[1] = 1;
                    else if (offsets[1] > 0) offsets[1] = -1;
                }
                if (z2) {
                    if (offsets[2] < 0) offsets[2] = 1;
                    else if (offsets[2] > 0) offsets[2] = -1;
                }

            }
            compute_buffer_bounds(ngh, offsets, num_dims, local_bounds, buffer_size);
            p_ranks.emplace(ngh.rank);
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
    // std::map<int, CTNeighbor> ngh_map;

    int rank = getRank();

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
            // ngh_map.emplace(n_rank, CTNeighbor{n_rank, working[0], -1, -1, nullptr,
            //    {0, 0, 0, 0}});
            neighbors.push_back(CTNeighbor{n_rank, working[0], -1, -1, nullptr,
                {0, 0, 0, 0}});
        }

    } else if (num_dims_ == 2) {
        std::list<int> xoffsets{-1, 0, 1};
        std::list<int> yoffsets{-1, 0, 1};
        if (periodic_) {
            // avoid making self a neighbor
            // if (procs_per_dim[0] == 1) {
            //     xoffsets.pop_front();
            //     xoffsets.pop_back();
            // }
            // if (procs_per_dim[1] == 1) {
            //     yoffsets.pop_front();
            //     yoffsets.pop_back();
            // }
        } else {
            if (coords[0] + xoffsets.front() < 0) xoffsets.pop_front();
            if (coords[0] + xoffsets.back() >= procs_per_dim[0]) xoffsets.pop_back();

            if (coords[1] + yoffsets.front() < 0) yoffsets.pop_front();
            if (coords[1] + yoffsets.back() >= procs_per_dim[1]) yoffsets.pop_back();
        }
        for (int xd : xoffsets) {
            for (int yd : yoffsets) {
                if (!(xd == 0 && yd == 0)) {
                    int working[] = {coords[0] + xd, coords[1] + yd};
                    // printf("%d: (%d, %d)\n", getRank(), working[0], working[1]);
                    int n_rank;
                    MPI_Cart_rank(comm_, working, &n_rank);
                    MPI_Cart_coords(comm_, n_rank, num_dims_, working);
                    if (n_rank != rank) {
                        //ngh_map.emplace(n_rank, CTNeighbor{n_rank, working[0], working[1], -1, nullptr,
                        //{0, 0, 0, 0}});
                        neighbors.push_back(CTNeighbor{n_rank, working[0], working[1], -1, nullptr,
                            {0, 0, 0, 0}});
                    }
                }
            }
        }
    } else if (num_dims_ == 3) {
        std::list<int> xoffsets{-1, 0, 1};
        std::list<int> yoffsets{-1, 0, 1};
        std::list<int> zoffsets{-1, 0, 1};
        if (periodic_) {
            // avoid making self a neighbor
            if (procs_per_dim[0] == 1) {
                xoffsets.pop_front();
                xoffsets.pop_back();
            }
            if (procs_per_dim[1] == 1) {
                yoffsets.pop_front();
                yoffsets.pop_back();
            }
            if (procs_per_dim[2] == 1) {
                zoffsets.pop_front();
                zoffsets.pop_back();
            }
        } else {
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
                        //ngh_map.emplace(n_rank, CTNeighbor{n_rank, working[0], working[1], working[2], nullptr,
                        //{0, 0, 0, 0}});
                        neighbors.push_back(CTNeighbor{n_rank, working[0], working[1], working[2], nullptr,
                            {0, 0, 0, 0}});
                    }
                }
            }
        }
    }

    //for (auto& kv : ngh_map) {
    //    neighbors.push_back(kv.second);
    //}
 
    for (auto& ngh : neighbors) {
        getBounds(ngh.rank, ngh.local_bounds);
    }
}


}