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
    // Py Tuple with info for who and what range to cover when sending 
    // buffered agents - tuple: (rank, (xmin, xmax, ymin, ymax, zmin, zmax))
    PyObject* buffer_info;
    BoundingBox<R4Py_DiscretePoint> local_bounds;
};

struct GetBufferInfo {

    PyObject* operator()(CTNeighbor& ct) {
        return ct.buffer_info;
    }
};

class CartesianTopology {

private:
    int num_dims_;
    int* procs_per_dim;
    bool periodic_;
    BoundingBox<R4Py_DiscretePoint> bounds_;
    int x_remainder, y_remainder, z_remainder;
    MPI_Comm comm_;

    void getBounds(int rank, BoundingBox<R4Py_DiscretePoint>& local_bounds);

public:
    CartesianTopology(MPI_Comm, MPI_Comm* cart_comm, int num_dims, const BoundingBox<R4Py_DiscretePoint>& global_bounds, bool periodic);
    CartesianTopology(MPI_Comm, MPI_Comm* cart_comm, const std::vector<int>& procs_per_dimension, const BoundingBox<R4Py_DiscretePoint>& global_bounds, bool periodic);
    ~CartesianTopology();

    int getRank();
    void getBounds(BoundingBox<R4Py_DiscretePoint>& local_bounds);    
    void getCoords(std::vector<int>& coords);
    void getNeighbors(std::vector<CTNeighbor>& neighbors);
};


using AIDPyObjMapT = std::map<R4Py_AgentID*, PyObject*>;

template<typename BaseGrid>
class DistributedGrid {

private:
    std::unique_ptr<BaseGrid> grid;
    BoundingBox<R4Py_DiscretePoint> local_bounds;
    // value: tuple ((aid.id, aid.type, aid.rank), ngh.rank, pt)
    std::shared_ptr<AIDPyObjMapT> out_of_bounds_agents;
    unsigned int buffer_size_;
    MPI_Comm cart_comm;
    std::shared_ptr<std::vector<CTNeighbor>> nghs;
    int rank;

    void calcBufferBounds(CTNeighbor& ngh, int offsets[], int num_dims);

public:
    DistributedGrid(const std::string& name, const BoundingBox<R4Py_DiscretePoint>& bounds, 
        unsigned int buffer_size, MPI_Comm comm);
    virtual ~DistributedGrid();

    bool add(R4Py_Agent* agent);
    bool remove(R4Py_Agent* agent);
    bool remove(R4Py_AgentID* aid);
    R4Py_Agent* getAgentAt(R4Py_DiscretePoint* pt);
    AgentList getAgentsAt(R4Py_DiscretePoint* pt);
    R4Py_DiscretePoint* getLocation(R4Py_Agent* agent);
    R4Py_DiscretePoint* move(R4Py_Agent* agent, R4Py_DiscretePoint* to);
    R4Py_DiscretePoint* moveBufferAgent(R4Py_Agent* agent, R4Py_DiscretePoint* to);
    std::shared_ptr<std::map<R4Py_AgentID*, PyObject*>> getOOBData();
    std::shared_ptr<std::vector<CTNeighbor>> getNeighborData();
    void clearOOBData();
    BoundingBox<R4Py_DiscretePoint> getLocalBounds() const;
    MPI_Comm getCartesianCommunicator();

};

template<typename BaseGrid>
DistributedGrid<BaseGrid>::DistributedGrid(const std::string& name, const BoundingBox<R4Py_DiscretePoint>& bounds, 
        unsigned int buffer_size, MPI_Comm comm) : grid {std::unique_ptr<BaseGrid>(new BaseGrid(name, bounds))}, 
        local_bounds{0, 0, 0, 0}, out_of_bounds_agents{std::make_shared<AIDPyObjMapT>()}, 
        buffer_size_{buffer_size}, cart_comm{}, nghs{std::make_shared<std::vector<CTNeighbor>>()}, rank{-1}
{
    int dims = 1;
    if (bounds.y_extent_ > 0) ++dims;
    if (bounds.z_extent_ > 0) ++dims;
    CartesianTopology ct(comm, &cart_comm, dims, bounds, is_periodic<BaseGrid>::value);
    rank = ct.getRank();
    ct.getBounds(local_bounds);
    ct.getNeighbors(*nghs);
    std::vector<int> coords;
    ct.getCoords(coords);


    int offsets[3] = {0, 0, 0};

    if (dims == 1) {
        for (auto& ngh : (*nghs)) {
            offsets[0] = ngh.cart_coord_x - coords[0];
            calcBufferBounds(ngh, offsets, dims);
        }
    } else if (dims == 2) {
        for (auto& ngh : (*nghs)) {
            offsets[0] = ngh.cart_coord_x - coords[0];
            offsets[1] = ngh.cart_coord_y - coords[1];
            calcBufferBounds(ngh, offsets, dims);
        }
    } else if (dims == 3) {
        for (auto& ngh : (*nghs)) {
            offsets[0] = ngh.cart_coord_x - coords[0];
            offsets[1] = ngh.cart_coord_y - coords[1];
            offsets[2] = ngh.cart_coord_z - coords[2];
            // if (rank == 0) {
            //     printf("Offsets: %d - %d, %d, %d\n", ngh.rank, offsets[0], offsets[1], offsets[2]);
            // }
            calcBufferBounds(ngh, offsets, dims);
        }
    }
}

template<typename BaseGrid>
DistributedGrid<BaseGrid>::~DistributedGrid() {
    MPI_Comm_free(&cart_comm);
}


template<typename BaseGrid>
void DistributedGrid<BaseGrid>::calcBufferBounds(CTNeighbor& ngh, int offsets[], int num_dims) {
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
        xmax = xmin + buffer_size_;
    } else if (offsets[0] == 1 || offsets[0] == -2) {
        xmin = local_bounds.xmax_ - buffer_size_;
        xmax = xmin + buffer_size_;
    }

    if (offsets[1] == -1 || offsets[1] == 2) {
        ymin = local_bounds.ymin_;
        ymax = ymin + buffer_size_;
    } else if (offsets[1] == 1 || offsets[1] == -2) {
        ymin = local_bounds.ymax_ - buffer_size_;
        ymax = ymin + buffer_size_;
    }

    if (offsets[2] == -1 || offsets[2] == 2) {
        zmin = local_bounds.zmin_;
        zmax = zmin + buffer_size_;
    } else if (offsets[2] == 1 || offsets[2] == -2) {
        zmin = local_bounds.zmax_ - buffer_size_;
        zmax = zmin + buffer_size_;
    }

    ngh.buffer_info = Py_BuildValue("(i(llllll))", ngh.rank, xmin, xmax, ymin, ymax,
        zmin, zmax);
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
bool DistributedGrid<BaseGrid>::remove(R4Py_AgentID* aid) {
    return grid->remove(aid);
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
R4Py_DiscretePoint* DistributedGrid<BaseGrid>::moveBufferAgent(R4Py_Agent* agent, R4Py_DiscretePoint* to) {
    R4Py_DiscretePoint* pt = grid->move(agent, to);
    return pt;
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
            for (auto& ngh : (*nghs)) {
                if (ngh.local_bounds.contains(pt)) {
                    PyObject* aid_tuple = agent->aid->as_tuple;
                    Py_INCREF(aid_tuple);
                    PyArrayObject* pt_array = pt->coords;
                    Py_INCREF(pt_array);
                    PyObject* obj = Py_BuildValue("(O, I, O)", aid_tuple, ngh.rank, pt_array);
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
std::shared_ptr<std::vector<CTNeighbor>> DistributedGrid<BaseGrid>::getNeighborData() {
    return nghs;
}

template<typename BaseGrid>
void DistributedGrid<BaseGrid>::clearOOBData() {
    out_of_bounds_agents->clear();
}

template<typename BaseGrid>
BoundingBox<R4Py_DiscretePoint> DistributedGrid<BaseGrid>::getLocalBounds() const {
    return local_bounds;
}

template<typename BaseGrid>
MPI_Comm DistributedGrid<BaseGrid>::getCartesianCommunicator() {
    return cart_comm;
}

class ISharedGrid {

public:
    virtual ~ISharedGrid() = 0;

    virtual bool add(R4Py_Agent* agent) = 0;
    virtual bool remove(R4Py_Agent* agent) = 0;
    virtual bool remove(R4Py_AgentID* aid) = 0;
    virtual R4Py_Agent* getAgentAt(R4Py_DiscretePoint* pt) = 0;
    virtual AgentList getAgentsAt(R4Py_DiscretePoint* pt) = 0;
    virtual R4Py_DiscretePoint* getLocation(R4Py_Agent* agent) = 0;
    virtual R4Py_DiscretePoint* move(R4Py_Agent* agent, R4Py_DiscretePoint* to) = 0;
    virtual R4Py_DiscretePoint* moveBufferAgent(R4Py_Agent* agent, R4Py_DiscretePoint* to) = 0;
    virtual std::shared_ptr<std::map<R4Py_AgentID*, PyObject*>> getOOBData() = 0;
    virtual std::shared_ptr<std::vector<CTNeighbor>> getNeighborData() = 0;
    virtual void clearOOBData() = 0;
    virtual BoundingBox<R4Py_DiscretePoint> getLocalBounds() const = 0;
    virtual MPI_Comm getCartesianCommunicator() = 0;
    
    
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
    bool remove(R4Py_AgentID* aid) override;
    R4Py_Agent* getAgentAt(R4Py_DiscretePoint* pt) override;
    AgentList getAgentsAt(R4Py_DiscretePoint* pt) override;
    R4Py_DiscretePoint* getLocation(R4Py_Agent* agent) override;
    R4Py_DiscretePoint* move(R4Py_Agent* agent, R4Py_DiscretePoint* to) override;
    R4Py_DiscretePoint* moveBufferAgent(R4Py_Agent* agent, R4Py_DiscretePoint* to) override;
    std::shared_ptr<std::map<R4Py_AgentID*, PyObject*>> getOOBData() override;
    std::shared_ptr<std::vector<CTNeighbor>> getNeighborData() override;
    void clearOOBData() override;
    BoundingBox<R4Py_DiscretePoint> getLocalBounds() const override;
    MPI_Comm getCartesianCommunicator() override;
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
bool SharedGrid<DelegateType>::remove(R4Py_AgentID* aid) {
    return delegate->remove(aid);
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
R4Py_DiscretePoint* SharedGrid<DelegateType>::moveBufferAgent(R4Py_Agent* agent, R4Py_DiscretePoint* to) {
    return delegate->moveBufferAgent(agent, to);
}

template<typename DelegateType>
std::shared_ptr<std::map<R4Py_AgentID*, PyObject*>> SharedGrid<DelegateType>::getOOBData() {
    return delegate->getOOBData();
}

template<typename DelegateType>
std::shared_ptr<std::vector<CTNeighbor>> SharedGrid<DelegateType>::getNeighborData() {
    return delegate->getNeighborData();
}

template<typename DelegateType>
void SharedGrid<DelegateType>::clearOOBData() {
    return delegate->clearOOBData();
}

template<typename DelegateType>
BoundingBox<R4Py_DiscretePoint> SharedGrid<DelegateType>::getLocalBounds() const {
    return delegate->getLocalBounds();
}

template<typename DelegateType>
MPI_Comm SharedGrid<DelegateType>::getCartesianCommunicator() {
    return delegate->getCartesianCommunicator();
}

struct R4Py_SharedGrid {
    PyObject_HEAD
    ISharedGrid* grid;
    PyObject* cart_comm;
};

}


#endif