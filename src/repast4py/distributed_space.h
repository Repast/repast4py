#ifndef R4PY_SRC_DISTRIBUTEDSPACE_H
#define R4PY_SRC_DISTRIBUTEDSPACE_H

#define PY_SSIZE_T_CLEAN

#include <vector>
#include <memory>

#include "mpi.h"
#include "core.h"
#include "space.h"


namespace repast4py {

struct CTNeighbor {
    int rank;
    int cart_coord_x, cart_coord_y, cart_coord_z;
    BoundingBox<R4Py_DiscretePoint> buffer_bounds;
    BoundingBox<R4Py_DiscretePoint> local_bounds;
};

struct RankPtPair {
    int rank;
    Point<R4Py_DiscretePoint> pt;
};


class CartesianTopology {

private:
    int num_dims_;
    int* procs_per_dim;
    MPI_Comm comm_;
    bool periodic_;
    BoundingBox<R4Py_DiscretePoint>& bounds_;
    unsigned int x_remainder, y_remainder, z_remainder;

    void getBounds(int rank, BoundingBox<R4Py_DiscretePoint>& local_bounds);

public:
    CartesianTopology(MPI_Comm, int num_dims, BoundingBox<R4Py_DiscretePoint>& global_bounds, bool periodic);
    CartesianTopology(MPI_Comm, const std::vector<int>& procs_per_dimension, BoundingBox<R4Py_DiscretePoint>& global_bounds, bool periodic);
    ~CartesianTopology();

    int getRank();
    void getBounds(BoundingBox<R4Py_DiscretePoint>& local_bounds);    
    void getCoords(std::vector<int>& coords);
    void getNeighbors(std::vector<CTNeighbor>& neighbors);
    MPI_Comm getCartesianComm();
};


template<typename BaseGrid>
class DistributedGrid {

private:
    std::unique_ptr<BaseGrid> grid;
    BoundingBox<R4Py_DiscretePoint> local_bounds;
    std::map<R4Py_AgentID*, R4Py_Agent*> ghosts;
    std::map<R4Py_AgentID*, R4Py_Agent*> out_of_bounds_agents;
    unsigned int buffer_size_;
    MPI_Comm cart_comm;
    std::vector<CTNeighbor> nghs;
    int rank;

    void calcBufferBounds(int ngh_cart_coords[], int offsets[]);

public:
    DistributedGrid(const std::string& name, const BoundingBox<R4Py_DiscretePoint>& bounds, 
        unsigned int buffer_size, MPI_Comm comm);
    virtual ~DistributedGrid() {}

    bool add(R4Py_Agent* agent);
    bool remove(R4Py_Agent* agent);
    R4Py_Agent* getAgentAt(R4Py_DiscretePoint* pt);
    AgentList getAgentsAt(R4Py_DiscretePoint* pt);
    R4Py_DiscretePoint* getLocation(R4Py_Agent* agent);
    R4Py_DiscretePoint* move(R4Py_Agent* agent, R4Py_DiscretePoint* to);
};

template<typename BaseGrid>
DistributedGrid<BaseGrid>::DistributedGrid(const std::string& name, const BoundingBox<R4Py_DiscretePoint>& bounds, 
        unsigned int buffer_size, MPI_Comm comm) : grid {std::unique_ptr<BaseGrid>(new BaseGrid(name, bounds))}, local_bounds(0, 0, 0, 0),
        ghosts(), buffer_size_(buffer_size), cart_comm(), nghs(), rank(-1)
{
    int dims = 1;
    if (bounds.y_extent_ > 0) ++dims;
    if (bounds.z_extent_ > 0) ++dims;
    CartesianTopology ct(comm, dims, bounds, is_periodic<BaseGrid>());
    rank = ct.getRank();
    ct.getBounds(local_bounds);
    cart_comm = ct.getCartesianComm();
    ct.getNeighbors(nghs);
    std::vector<int> coords;
    ct.getCoords(coords);
    

    if (dims == 1) {
        int x_offsets[2] = {-1, 1};
        for (int i = 0; i < 2; ++i) {
            int ngh_card_coords[] = {coords[0] + x_offsets[i]};
            int offsets[] = {x_offsets[i], 0, 0};
            calcBufferBounds(ngh_card_coords, offsets);
        }
    } else if (dims == 2) {
        int coord_offsets[3] = {-1, 0, 1};
        for (int x = 0; x < 3; ++x) {
            int x_offset = coord_offsets[x];
            for (int y = 0; y < 3; ++y) {
                int y_offset = coord_offsets[y];
                int ngh_card_coords[] = {coords[0] + x_offset, coords[1] + y_offset};
                int offsets[] = {x_offset, y_offset, 0};
                if (!(x == 0 && y == 0)) {
                    calcBufferBounds(ngh_card_coords, offsets);
                }
            }
        }
    } else if (dims == 2) {
        int coord_offsets[3] = {-1, 0, 1};
        for (int x = 0; x < 3; ++x) {
            int x_offset = coord_offsets[x];
            for (int y = 0; y < 3; ++y) {
                int y_offset = coord_offsets[y];
                for (int z = 0; z < 3; ++z) {
                    int z_offset = coord_offsets[z];
                    int ngh_card_coords[] = {coords[0] + x_offset, coords[1] + y_offset, 
                        coords[2] + z_offset};
                    int offsets[] = {x_offset, y_offset, z_offset};
                    if (!(x == 0 && y == 0 && z == 0)) {
                        calcBufferBounds(ngh_card_coords, offsets);
                    }
                }
            }
        }
    }
}


template<typename BaseGrid>
void DistributedGrid<BaseGrid>::calcBufferBounds(int ngh_cart_coords[], int offsets[]) {
    int n_rank;
    MPI_Cart_rank(cart_comm, ngh_cart_coords, &n_rank);
    if (n_rank != MPI_PROC_NULL) {
        for (auto& ngh : nghs) {
            if (ngh.rank == n_rank) {
                BoundingBox<R4Py_DiscretePoint>& box = ngh.buffer_bounds;
                if (offsets[0] == -1) {
                    box.xmin_ = local_bounds.xmin_;
                    box.x_extent_ = buffer_size_;
                } else if (offsets[0] == 1) {
                    box.xmin_ = local_bounds.xmax_ - buffer_size_;
                    box.x_extent_ = buffer_size_;
                }

                if (offsets[1] == -1) {
                    box.ymin_ = local_bounds.ymin_;
                    box.y_extent_ = buffer_size_;
                } else if (offsets[1] == 1) {
                    box.ymin_ = local_bounds.ymax_ - buffer_size_;
                    box.y_extent_ = buffer_size_;
                }

                if (offsets[2] == -1) {
                    box.zmin_ = local_bounds.zmin_;
                    box.z_extent_ = buffer_size_;
                } else if (offsets[2] == 1) {
                    box.zmin_ = local_bounds.zmax_ - buffer_size_;
                    box.z_extent_ = buffer_size_;
                }

                break;
            }
        }
    }
}


template<typename BaseGrid>
bool DistributedGrid<BaseGrid>::add(R4Py_Agent* agent) {
    return grid->add(agent);
}

template<typename BaseGrid>
bool DistributedGrid<BaseGrid>::remove(R4Py_Agent* agent) {
    return grid->remove(agent);
}

template<typename BaseGrid>
R4Py_Agent* DistributedGrid<BaseGrid>::getAgentAt(R4Py_DiscretePoint* pt) {
    return grid->getAgentAt(pt);
}

template<typename BaseGrid>
AgentList DistributedGrid<BaseGrid>::getAgentsAt(R4Py_DiscretePoint* pt) {
    return grid->getAgentsAt(pt);
}

template<typename BaseGrid>
R4Py_DiscretePoint* DistributedGrid<BaseGrid>::getLocation(R4Py_Agent* agent) {
    return grid->getLocation(agent);
}

template<typename BaseGrid>
R4Py_DiscretePoint* DistributedGrid<BaseGrid>::move(R4Py_Agent* agent, R4Py_DiscretePoint* to) {
   
    R4Py_DiscretePoint* pt = grid->move(agent, to);
    // pt will be null if the move fails for a valid reason -- e.g.,
    // space is already occupied
    if (pt) {
        if (local_bounds.contains(pt)) {
            out_of_bounds_agents.erase(agent->aid);
        } else {
            // what neighbor now contains the agent
            for (auto& ngh : nghs) {
                if (ngh.local_bounds.contains(pt)) {
                    RankPtPair rpp;
                    extract_coords(pt, rpp.pt);
                    out_of_bounds_agents[agent->aid] = rpp;
                }
            }
        }
    }
    return pt;
}

}


#endif