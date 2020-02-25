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

struct R4Py_MoveAddress {
    PyObject_HEAD
    R4Py_AgentID* aid;
    int rank;
    R4Py_DiscretePoint* pt;
};

class CartesianTopology {

private:
    int num_dims_;
    int* procs_per_dim;
    MPI_Comm comm_;
    bool periodic_;
    BoundingBox<R4Py_DiscretePoint> bounds_;
    unsigned int x_remainder, y_remainder, z_remainder;

    void getBounds(int rank, BoundingBox<R4Py_DiscretePoint>& local_bounds);

public:
    CartesianTopology(MPI_Comm, int num_dims, const BoundingBox<R4Py_DiscretePoint>& global_bounds, bool periodic);
    CartesianTopology(MPI_Comm, const std::vector<int>& procs_per_dimension, const BoundingBox<R4Py_DiscretePoint>& global_bounds, bool periodic);
    ~CartesianTopology();

    int getRank();
    void getBounds(BoundingBox<R4Py_DiscretePoint>& local_bounds);    
    void getCoords(std::vector<int>& coords);
    void getNeighbors(std::vector<CTNeighbor>& neighbors);
    MPI_Comm getCartesianComm();
};


using AIDPyObjMapT = std::map<R4Py_AgentID*, PyObject*>;

template<typename BaseGrid>
class DistributedGrid {

private:
    std::unique_ptr<BaseGrid> grid;
    BoundingBox<R4Py_DiscretePoint> local_bounds;
    std::map<R4Py_AgentID*, R4Py_Agent*> ghosts;
    // value: tuple ((aid.id, aid.type, aid.rank), ngh.rank, pt)
    std::shared_ptr<AIDPyObjMapT> out_of_bounds_agents;
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
    std::shared_ptr<std::map<R4Py_AgentID*, PyObject*>> getOOBData();
    BoundingBox<R4Py_DiscretePoint> getLocalBounds() const;
};

template<typename BaseGrid>
DistributedGrid<BaseGrid>::DistributedGrid(const std::string& name, const BoundingBox<R4Py_DiscretePoint>& bounds, 
        unsigned int buffer_size, MPI_Comm comm) : grid {std::unique_ptr<BaseGrid>(new BaseGrid(name, bounds))}, 
        local_bounds(0, 0, 0, 0), ghosts(), out_of_bounds_agents(std::make_shared<AIDPyObjMapT>()), 
        buffer_size_(buffer_size), cart_comm(), nghs(), rank(-1)
{
    int dims = 1;
    if (bounds.y_extent_ > 0) ++dims;
    if (bounds.z_extent_ > 0) ++dims;
    CartesianTopology ct(comm, dims, bounds, is_periodic<BaseGrid>::value);
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
            out_of_bounds_agents->erase(agent->aid);
        } else {
            // what neighbor now contains the agent
            for (auto& ngh : nghs) {
                if (ngh.local_bounds.contains(pt)) {
                    Py_INCREF(pt);
                    PyObject* obj = Py_BuildValue("((liI), I, O)", 
                        agent->aid->id, agent->aid->type, agent->aid->rank, 
                        ngh.rank, pt);
                    (*out_of_bounds_agents)[agent->aid] = obj;
                }
            }
        }
    }
    return pt;
}

template<typename BaseGrid>
std::shared_ptr<std::map<R4Py_AgentID*, PyObject*>> DistributedGrid<BaseGrid>::getOOBData() {
    return out_of_bounds_agents;
}


template<typename BaseGrid>
BoundingBox<R4Py_DiscretePoint> DistributedGrid<BaseGrid>::getLocalBounds() const {
    return local_bounds;
}

class ISharedGrid {

public:
    virtual ~ISharedGrid() = 0;

    virtual bool add(R4Py_Agent* agent) = 0;
    virtual bool remove(R4Py_Agent* agent) = 0;
    virtual R4Py_Agent* getAgentAt(R4Py_DiscretePoint* pt) = 0;
    virtual AgentList getAgentsAt(R4Py_DiscretePoint* pt) = 0;
    virtual R4Py_DiscretePoint* getLocation(R4Py_Agent* agent) = 0;
    virtual R4Py_DiscretePoint* move(R4Py_Agent* agent, R4Py_DiscretePoint* to) = 0;
    virtual std::shared_ptr<std::map<R4Py_AgentID*, PyObject*>> getOOBData() = 0;
    virtual BoundingBox<R4Py_DiscretePoint> getLocalBounds() const = 0;
    
};

inline ISharedGrid::~ISharedGrid() {}

template<typename DelegateType>
class SharedGrid : public ISharedGrid {

private:
    std::unique_ptr<DelegateType> delegate;

public:
    SharedGrid(const std::string& name, const BoundingBox<R4Py_DiscretePoint>& bounds, 
        unsigned int buffer_size, MPI_Comm comm);
    virtual ~SharedGrid() {}
    bool add(R4Py_Agent* agent) override;
    bool remove(R4Py_Agent* agent) override;
    R4Py_Agent* getAgentAt(R4Py_DiscretePoint* pt) override;
    AgentList getAgentsAt(R4Py_DiscretePoint* pt) override;
    R4Py_DiscretePoint* getLocation(R4Py_Agent* agent) override;
    R4Py_DiscretePoint* move(R4Py_Agent* agent, R4Py_DiscretePoint* to) override;
    std::shared_ptr<std::map<R4Py_AgentID*, PyObject*>> getOOBData() override;
    BoundingBox<R4Py_DiscretePoint> getLocalBounds() const override;
};

template<typename DelegateType>
SharedGrid<DelegateType>::SharedGrid(const std::string& name, const BoundingBox<R4Py_DiscretePoint>& bounds, 
        unsigned int buffer_size, MPI_Comm comm) : 
    delegate{std::unique_ptr<DelegateType>(new DelegateType(name, bounds, buffer_size, comm))} {}

template<typename DelegateType>
bool SharedGrid<DelegateType>::add(R4Py_Agent* agent) {
    return delegate->add(agent);
}

template<typename DelegateType>
bool SharedGrid<DelegateType>::remove(R4Py_Agent* agent) {
    return delegate->remove(agent);
}

template<typename DelegateType>
R4Py_Agent* SharedGrid<DelegateType>::getAgentAt(R4Py_DiscretePoint* pt) {
    return delegate->getAgentAt(pt);
}

template<typename DelegateType>
AgentList SharedGrid<DelegateType>::getAgentsAt(R4Py_DiscretePoint* pt) {
    return delegate->getAgentsAt(pt);
}

template<typename DelegateType>
R4Py_DiscretePoint* SharedGrid<DelegateType>::getLocation(R4Py_Agent* agent) {
    return delegate->getLocation(agent);
}

template<typename DelegateType>
R4Py_DiscretePoint* SharedGrid<DelegateType>::move(R4Py_Agent* agent, R4Py_DiscretePoint* to) {
    return delegate->move(agent, to);
}

template<typename DelegateType>
std::shared_ptr<std::map<R4Py_AgentID*, PyObject*>> SharedGrid<DelegateType>::getOOBData() {
    return delegate->getOOBData();
}

template<typename DelegateType>
BoundingBox<R4Py_DiscretePoint> SharedGrid<DelegateType>::getLocalBounds() const {
    return delegate->getLocalBounds();
}


struct R4Py_SharedGrid {
    PyObject_HEAD
    ISharedGrid* grid;
};

}


#endif