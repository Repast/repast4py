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
    BoundingBox local_bounds;
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
    BoundingBox bounds_;
    int x_remainder, y_remainder, z_remainder;
    MPI_Comm comm_;

    void getBounds(int rank, BoundingBox& local_bounds);

public:
    CartesianTopology(MPI_Comm, MPI_Comm* cart_comm, int num_dims, const BoundingBox& global_bounds, bool periodic);
    CartesianTopology(MPI_Comm, MPI_Comm* cart_comm, const std::vector<int>& procs_per_dimension, const BoundingBox& global_bounds, bool periodic);
    ~CartesianTopology();

    int getRank();
    void getBounds(BoundingBox& local_bounds);    
    void getCoords(std::vector<int>& coords);
    void getNeighbors(std::vector<CTNeighbor>& neighbors);
};


using AIDPyObjMapT = std::map<R4Py_AgentID*, PyObject*>;

template<typename BaseSpaceType, typename PointType>
class DistributedCartesianSpace {

private:
    std::unique_ptr<BaseSpaceType> base_space;
    BoundingBox local_bounds;
    // value: tuple ((aid.id, aid.type, aid.rank), ngh.rank, pt)
    std::shared_ptr<AIDPyObjMapT> out_of_bounds_agents;
    unsigned int buffer_size_;
    MPI_Comm cart_comm;
    std::shared_ptr<std::vector<CTNeighbor>> nghs;
    int rank;

    void calcBufferBounds(CTNeighbor& ngh, int offsets[], int num_dims);

public:
    DistributedCartesianSpace(const std::string& name, const BoundingBox& bounds, 
        unsigned int buffer_size, MPI_Comm comm);
    DistributedCartesianSpace(const std::string& name, const BoundingBox& bounds, 
        unsigned int buffer_size, MPI_Comm comm, int tree_threshold);
    virtual ~DistributedCartesianSpace();

    bool add(R4Py_Agent* agent);
    bool remove(R4Py_Agent* agent);
    bool remove(R4Py_AgentID* aid);
    R4Py_Agent* getAgentAt(PointType* pt);
    AgentList getAgentsAt(PointType* pt);
    PointType* getLocation(R4Py_Agent* agent);
    PointType* move(R4Py_Agent* agent, PointType* to);
    PointType* moveBufferAgent(R4Py_Agent* agent, PointType* to);
    std::shared_ptr<std::map<R4Py_AgentID*, PyObject*>> getOOBData();
    std::shared_ptr<std::vector<CTNeighbor>> getNeighborData();
    void clearOOBData();
    BoundingBox getLocalBounds() const;
    MPI_Comm getCartesianCommunicator();
    void getAgentsWithin(const BoundingBox& bounds, std::shared_ptr<std::vector<R4Py_Agent*>>& agents);

};

template<typename BaseSpaceType, typename PointType>
DistributedCartesianSpace<BaseSpaceType, PointType>::DistributedCartesianSpace(const std::string& name, const BoundingBox& bounds, 
        unsigned int buffer_size, MPI_Comm comm) : base_space {std::unique_ptr<BaseSpaceType>(new BaseSpaceType(name, bounds))}, 
        local_bounds{0, 0, 0, 0}, out_of_bounds_agents{std::make_shared<AIDPyObjMapT>()}, 
        buffer_size_{buffer_size}, cart_comm{}, nghs{std::make_shared<std::vector<CTNeighbor>>()}, rank{-1}
{
    int dims = 1;
    if (bounds.y_extent_ > 0) ++dims;
    if (bounds.z_extent_ > 0) ++dims;
    CartesianTopology ct(comm, &cart_comm, dims, bounds, is_periodic<BaseSpaceType>::value);
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

template<typename BaseSpaceType, typename PointType>
DistributedCartesianSpace<BaseSpaceType, PointType>::DistributedCartesianSpace(const std::string& name, const BoundingBox& bounds, 
        unsigned int buffer_size, MPI_Comm comm, int tree_threshold) : base_space {std::unique_ptr<BaseSpaceType>(new BaseSpaceType(name, bounds, tree_threshold))}, 
        local_bounds{0, 0, 0, 0}, out_of_bounds_agents{std::make_shared<AIDPyObjMapT>()}, 
        buffer_size_{buffer_size}, cart_comm{}, nghs{std::make_shared<std::vector<CTNeighbor>>()}, rank{-1}
{
    int dims = 1;
    if (bounds.y_extent_ > 0) ++dims;
    if (bounds.z_extent_ > 0) ++dims;
    CartesianTopology ct(comm, &cart_comm, dims, bounds, is_periodic<BaseSpaceType>::value);
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

template<typename BaseSpaceType, typename PointType>
DistributedCartesianSpace<BaseSpaceType, PointType>::~DistributedCartesianSpace() {
    MPI_Comm_free(&cart_comm);
}


template<typename BaseSpaceType, typename PointType>
void DistributedCartesianSpace<BaseSpaceType, PointType>::calcBufferBounds(CTNeighbor& ngh, int offsets[], int num_dims) {
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

template<typename BaseSpaceType, typename PointType>
bool DistributedCartesianSpace<BaseSpaceType, PointType>::add(R4Py_Agent* agent) {
    return base_space->add(agent);
}

template<typename BaseSpaceType, typename PointType>
bool DistributedCartesianSpace<BaseSpaceType, PointType>::remove(R4Py_Agent* agent) {
    return base_space->remove(agent);
}

template<typename BaseSpaceType, typename PointType>
bool DistributedCartesianSpace<BaseSpaceType, PointType>::remove(R4Py_AgentID* aid) {
    return base_space->remove(aid);
}

template<typename BaseSpaceType, typename PointType>
R4Py_Agent* DistributedCartesianSpace<BaseSpaceType, PointType>::getAgentAt(PointType* pt) {
    return base_space->getAgentAt(pt);
}

template<typename BaseSpaceType, typename PointType>
AgentList DistributedCartesianSpace<BaseSpaceType, PointType>::getAgentsAt(PointType* pt) {
    return base_space->getAgentsAt(pt);
}

template<typename BaseSpaceType, typename PointType>
PointType* DistributedCartesianSpace<BaseSpaceType, PointType>::getLocation(R4Py_Agent* agent) {
    return base_space->getLocation(agent);
}

template<typename BaseSpaceType, typename PointType>
void DistributedCartesianSpace<BaseSpaceType, PointType>::getAgentsWithin(const BoundingBox& bounds, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) {
    base_space->getAgentsWithin(bounds, agents);
}

template<typename BaseSpaceType, typename PointType>
PointType* DistributedCartesianSpace<BaseSpaceType, PointType>::moveBufferAgent(R4Py_Agent* agent, PointType* to) {
    PointType* pt = base_space->move(agent, to);
    return pt;
}

template<typename BaseSpaceType, typename PointType>
PointType* DistributedCartesianSpace<BaseSpaceType, PointType>::move(R4Py_Agent* agent, PointType* to) {
   
    PointType* pt = base_space->move(agent, to);
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
                    break;
                }
            }
        }
    }
    return pt;
}

template<typename BaseSpaceType, typename PointType>
std::shared_ptr<std::map<R4Py_AgentID*, PyObject*>> DistributedCartesianSpace<BaseSpaceType, PointType>::getOOBData() {
    return out_of_bounds_agents;
}

template<typename BaseSpaceType, typename PointType>
std::shared_ptr<std::vector<CTNeighbor>> DistributedCartesianSpace<BaseSpaceType, PointType>::getNeighborData() {
    return nghs;
}

template<typename BaseSpaceType, typename PointType>
void DistributedCartesianSpace<BaseSpaceType, PointType>::clearOOBData() {
    out_of_bounds_agents->clear();
}

template<typename BaseSpaceType, typename PointType>
BoundingBox DistributedCartesianSpace<BaseSpaceType, PointType>::getLocalBounds() const {
    return local_bounds;
}

template<typename BaseSpaceType, typename PointType>
MPI_Comm DistributedCartesianSpace<BaseSpaceType, PointType>::getCartesianCommunicator() {
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
    virtual BoundingBox getLocalBounds() const = 0;
    virtual MPI_Comm getCartesianCommunicator() = 0;
    
    
};

inline ISharedGrid::~ISharedGrid() {}

template<typename DelegateType>
class SharedGrid : public ISharedGrid {

private:
    std::unique_ptr<DelegateType> delegate;

public:
    SharedGrid(const std::string& name, const BoundingBox& bounds, 
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
    BoundingBox getLocalBounds() const override;
    MPI_Comm getCartesianCommunicator() override;
};

template<typename DelegateType>
SharedGrid<DelegateType>::SharedGrid(const std::string& name, const BoundingBox& bounds, 
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
BoundingBox SharedGrid<DelegateType>::getLocalBounds() const {
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

/////////////// ContinuousSpace ///////////////////

class ISharedContinuousSpace {

public:
    virtual ~ISharedContinuousSpace() = 0;

    virtual bool add(R4Py_Agent* agent) = 0;
    virtual bool remove(R4Py_Agent* agent) = 0;
    virtual bool remove(R4Py_AgentID* aid) = 0;
    virtual R4Py_Agent* getAgentAt(R4Py_ContinuousPoint* pt) = 0;
    virtual AgentList getAgentsAt(R4Py_ContinuousPoint* pt) = 0;
    virtual R4Py_ContinuousPoint* getLocation(R4Py_Agent* agent) = 0;
    virtual R4Py_ContinuousPoint* move(R4Py_Agent* agent, R4Py_ContinuousPoint* to) = 0;
    virtual R4Py_ContinuousPoint* moveBufferAgent(R4Py_Agent* agent, R4Py_ContinuousPoint* to) = 0;
    virtual std::shared_ptr<std::map<R4Py_AgentID*, PyObject*>> getOOBData() = 0;
    virtual std::shared_ptr<std::vector<CTNeighbor>> getNeighborData() = 0;
    virtual void clearOOBData() = 0;
    virtual BoundingBox getLocalBounds() const = 0;
    virtual MPI_Comm getCartesianCommunicator() = 0;
    virtual void getAgentsWithin(const BoundingBox& bounds, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) = 0;
};

inline ISharedContinuousSpace::~ISharedContinuousSpace() {}

template<typename DelegateType>
class SharedContinuousSpace : public ISharedContinuousSpace {

private:
    std::unique_ptr<DelegateType> delegate;

public:
    SharedContinuousSpace(const std::string& name, const BoundingBox& bounds, 
        unsigned int buffer_size, MPI_Comm comm, int tree_threshold);
    virtual ~SharedContinuousSpace() {}
    bool add(R4Py_Agent* agent) override;
    bool remove(R4Py_Agent* agent) override;
    bool remove(R4Py_AgentID* aid) override;
    R4Py_Agent* getAgentAt(R4Py_ContinuousPoint* pt) override;
    AgentList getAgentsAt(R4Py_ContinuousPoint* pt) override;
    R4Py_ContinuousPoint* getLocation(R4Py_Agent* agent) override;
    R4Py_ContinuousPoint* move(R4Py_Agent* agent, R4Py_ContinuousPoint* to) override;
    R4Py_ContinuousPoint* moveBufferAgent(R4Py_Agent* agent, R4Py_ContinuousPoint* to) override;
    std::shared_ptr<std::map<R4Py_AgentID*, PyObject*>> getOOBData() override;
    std::shared_ptr<std::vector<CTNeighbor>> getNeighborData() override;
    void clearOOBData() override;
    BoundingBox getLocalBounds() const override;
    MPI_Comm getCartesianCommunicator() override;
    void getAgentsWithin(const BoundingBox& bounds, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) override;
};

template<typename DelegateType>
SharedContinuousSpace<DelegateType>::SharedContinuousSpace(const std::string& name, const BoundingBox& bounds, 
        unsigned int buffer_size, MPI_Comm comm, int tree_threshold) : 
    delegate{std::unique_ptr<DelegateType>(new DelegateType(name, bounds, buffer_size, comm, tree_threshold))} {}

template<typename DelegateType>
bool SharedContinuousSpace<DelegateType>::add(R4Py_Agent* agent) {
    return delegate->add(agent);
}

template<typename DelegateType>
bool SharedContinuousSpace<DelegateType>::remove(R4Py_Agent* agent) {
    return delegate->remove(agent);
}

template<typename DelegateType>
bool SharedContinuousSpace<DelegateType>::remove(R4Py_AgentID* aid) {
    return delegate->remove(aid);
}

template<typename DelegateType>
R4Py_Agent* SharedContinuousSpace<DelegateType>::getAgentAt(R4Py_ContinuousPoint* pt) {
    return delegate->getAgentAt(pt);
}

template<typename DelegateType>
AgentList SharedContinuousSpace<DelegateType>::getAgentsAt(R4Py_ContinuousPoint* pt) {
    return delegate->getAgentsAt(pt);
}

template<typename DelegateType>
R4Py_ContinuousPoint* SharedContinuousSpace<DelegateType>::getLocation(R4Py_Agent* agent) {
    return delegate->getLocation(agent);
}

template<typename DelegateType>
R4Py_ContinuousPoint* SharedContinuousSpace<DelegateType>::move(R4Py_Agent* agent, R4Py_ContinuousPoint* to) {
    return delegate->move(agent, to);
}

template<typename DelegateType>
R4Py_ContinuousPoint* SharedContinuousSpace<DelegateType>::moveBufferAgent(R4Py_Agent* agent, R4Py_ContinuousPoint* to) {
    return delegate->moveBufferAgent(agent, to);
}

template<typename DelegateType>
std::shared_ptr<std::map<R4Py_AgentID*, PyObject*>> SharedContinuousSpace<DelegateType>::getOOBData() {
    return delegate->getOOBData();
}

template<typename DelegateType>
std::shared_ptr<std::vector<CTNeighbor>> SharedContinuousSpace<DelegateType>::getNeighborData() {
    return delegate->getNeighborData();
}

template<typename DelegateType>
void SharedContinuousSpace<DelegateType>::clearOOBData() {
    return delegate->clearOOBData();
}

template<typename DelegateType>
BoundingBox SharedContinuousSpace<DelegateType>::getLocalBounds() const {
    return delegate->getLocalBounds();
}

template<typename DelegateType>
MPI_Comm SharedContinuousSpace<DelegateType>::getCartesianCommunicator() {
    return delegate->getCartesianCommunicator();
}

template<typename DelegateType>
void SharedContinuousSpace<DelegateType>::getAgentsWithin(const BoundingBox& bounds, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) {
    delegate->getAgentsWithin(bounds, agents);
}

struct R4Py_SharedCSpace {
    PyObject_HEAD
    ISharedContinuousSpace* space;
    PyObject* cart_comm;
};

}


#endif