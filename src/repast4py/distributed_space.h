// Copyright 2021, UChicago Argonne, LLC 
// All Rights Reserved
// Software Name: repast4py
// By: Argonne National Laboratory
// License: BSD-3 - https://github.com/Repast/repast4py/blob/master/LICENSE.txt

#ifndef R4PY_SRC_DISTRIBUTEDSPACE_H
#define R4PY_SRC_DISTRIBUTEDSPACE_H

#define PY_SSIZE_T_CLEAN

#include <vector>
#include <set>
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

void compute_neighbor_buffers(std::vector<CTNeighbor>& nghs, std::vector<int>& cart_coords, 
    BoundingBox& local_bounds, int num_dims, const int* procs_per_dim, unsigned int buffer_size);


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
    int numDims() const {
        return num_dims_;
    }
    
    const int* procsPerDim() const {
        return procs_per_dim;
    }
};

struct R4Py_CartesianTopology {
    PyObject_HEAD
    CartesianTopology* topo;
    PyObject* cart_comm;
};


using AIDPyObjMapT = std::map<R4Py_AgentID*, PyObject*, agent_id_comp>;

template<typename BaseSpaceType>
class DistributedCartesianSpace {

private:
    std::unique_ptr<BaseSpaceType> base_space;
    BoundingBox local_bounds;
    // value: tuple ((aid.id, aid.type, aid.rank), ngh.rank, pt)
    std::shared_ptr<AIDPyObjMapT> out_of_bounds_agents;
    unsigned int buffer_size_;
    MPI_Comm cart_comm;
    std::shared_ptr<std::vector<CTNeighbor>> nghs;
    std::vector<std::pair<BoundingBox, int>> all_bounds;
    int rank;

    void getAllBounds(MPI_Comm comm);
    void eraseOOB(R4Py_AgentID *aid);

public:
    using PointType = typename BaseSpaceType::PointType;
    DistributedCartesianSpace(const std::string& name, const BoundingBox& bounds, 
        unsigned int buffer_size, MPI_Comm comm);
    DistributedCartesianSpace(const std::string& name, const BoundingBox& bounds, 
        unsigned int buffer_size, MPI_Comm comm, int tree_threshold);
    virtual ~DistributedCartesianSpace();

    bool add(R4Py_Agent* agent);
    bool remove(R4Py_Agent* agent);
    bool remove(R4Py_AgentID* aid);
    R4Py_Agent* getAgentAt(PointType* pt);
    AgentListPtr getAgentsAt(PointType* pt);
    PointType* getLocation(R4Py_Agent* agent);
    PointType* move(R4Py_Agent* agent, PointType* to);
    std::shared_ptr<AIDPyObjMapT> getOOBData();
    std::shared_ptr<std::vector<CTNeighbor>> getNeighborData();
    void clearOOBData();
    BoundingBox getLocalBounds() const;
    MPI_Comm getCartesianCommunicator();
    void getAgentsWithin(const BoundingBox& bounds, std::shared_ptr<std::vector<R4Py_Agent*>>& agents);
    const std::string name() const;
    bool contains(R4Py_Agent* agent) const;
};

template<typename BaseSpaceType>
DistributedCartesianSpace<BaseSpaceType>::DistributedCartesianSpace(const std::string& name, const BoundingBox& bounds, 
        unsigned int buffer_size, MPI_Comm comm) : base_space {std::unique_ptr<BaseSpaceType>(new BaseSpaceType(name, bounds))}, 
        local_bounds{0, 0, 0, 0}, out_of_bounds_agents{std::make_shared<AIDPyObjMapT>()}, 
        buffer_size_{buffer_size}, cart_comm{}, nghs{std::make_shared<std::vector<CTNeighbor>>()}, all_bounds{},
        rank{-1}
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

    compute_neighbor_buffers(*nghs, coords, local_bounds, dims, ct.procsPerDim(), buffer_size_);
    getAllBounds(comm);
}

template<typename BaseSpaceType>
DistributedCartesianSpace<BaseSpaceType>::DistributedCartesianSpace(const std::string& name, const BoundingBox& bounds, 
        unsigned int buffer_size, MPI_Comm comm, int tree_threshold) : base_space {std::unique_ptr<BaseSpaceType>(new BaseSpaceType(name, bounds, tree_threshold))}, 
        local_bounds{0, 0, 0, 0}, out_of_bounds_agents{std::make_shared<AIDPyObjMapT>()}, 
        buffer_size_{buffer_size}, cart_comm{}, nghs{std::make_shared<std::vector<CTNeighbor>>()}, all_bounds{},
        rank{-1}
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

    compute_neighbor_buffers(*nghs, coords, local_bounds, dims, ct.procsPerDim(), buffer_size_);
    getAllBounds(comm);

}

template<typename BaseSpaceType>
DistributedCartesianSpace<BaseSpaceType>::~DistributedCartesianSpace() {
    MPI_Comm_free(&cart_comm);
}

template<typename T>
struct MPIType_Selector {
    static MPI_Datatype type;
};

template<typename BaseSpaceType>
void DistributedCartesianSpace<BaseSpaceType>::getAllBounds(MPI_Comm comm) {
    // coord_type is always long, I think, but use it here anyway
    std::vector<BoundingBox::coord_type> box{local_bounds.xmin_, local_bounds.x_extent_,
                                                       local_bounds.ymin_, local_bounds.y_extent_,
                                                       local_bounds.zmin_, local_bounds.z_extent_};
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    BoundingBox::coord_type* all_bb = new BoundingBox::coord_type[comm_size * 6];
    MPI_Datatype dtype = MPIType_Selector<BoundingBox::coord_type>::type;
    MPI_Allgather(&box[0], 6, dtype, all_bb, 6, dtype, comm);

    std::set<int> ngh_ranks;
    for (auto& ngh : (*nghs)) {
        all_bounds.emplace(all_bounds.end(), ngh.local_bounds, ngh.rank);
        ngh_ranks.emplace(ngh.rank);
    }

    int rank;
    MPI_Comm_rank(comm, &rank);

    for (int i = 0; i < comm_size; ++i) {
        if (i != rank && ngh_ranks.find(i) == ngh_ranks.end()) {
            int idx = 6 * i;
            BoundingBox::coord_type x_min = all_bb[idx];
            BoundingBox::coord_type x_extent = all_bb[idx + 1];
            BoundingBox::coord_type y_min = all_bb[idx + 2];
            BoundingBox::coord_type y_extent = all_bb[idx + 3];
            BoundingBox::coord_type z_min = all_bb[idx + 4];
            BoundingBox::coord_type z_extent = all_bb[idx + 5];

            BoundingBox box(x_min, x_extent, y_min, y_extent, z_min, z_extent);
            all_bounds.emplace(all_bounds.end(), box, i);
        }
    }
    delete[] all_bb;
}

template<typename BaseSpaceType>
bool DistributedCartesianSpace<BaseSpaceType>::add(R4Py_Agent* agent) {
    return base_space->add(agent);
}

template<typename BaseSpaceType>
bool DistributedCartesianSpace<BaseSpaceType>::remove(R4Py_Agent* agent) {
    return this->remove(agent->aid);
}

template<typename BaseSpaceType>
bool DistributedCartesianSpace<BaseSpaceType>::contains(R4Py_Agent* agent) const {
    return base_space->contains(agent);
}

template<typename BaseSpaceType>
bool DistributedCartesianSpace<BaseSpaceType>::remove(R4Py_AgentID* aid) {
    eraseOOB(aid);
    return base_space->remove(aid);
}

template<typename BaseSpaceType>
R4Py_Agent* DistributedCartesianSpace<BaseSpaceType>::getAgentAt(PointType* pt) {
    return base_space->getAgentAt(pt);
}

template<typename BaseSpaceType>
AgentListPtr DistributedCartesianSpace<BaseSpaceType>::getAgentsAt(PointType* pt) {
    return base_space->getAgentsAt(pt);
}

template<typename BaseSpaceType>
typename DistributedCartesianSpace<BaseSpaceType>::PointType* DistributedCartesianSpace<BaseSpaceType>::getLocation(R4Py_Agent* agent) {
    return base_space->getLocation(agent);
}

template<typename BaseSpaceType>
void DistributedCartesianSpace<BaseSpaceType>::getAgentsWithin(const BoundingBox& bounds, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) {
    base_space->getAgentsWithin(bounds, agents);
}

template <typename BaseSpaceType>
void DistributedCartesianSpace<BaseSpaceType>::eraseOOB(R4Py_AgentID* aid) {
    auto iter = out_of_bounds_agents->find(aid);
    if (iter != out_of_bounds_agents->end()) {
        Py_DECREF(PyTuple_GET_ITEM(iter->second, 0));
        Py_DECREF(PyTuple_GET_ITEM(iter->second, 2));
        // if decref destroys the tuple, then it's members are decreffed
        Py_DECREF(iter->second);
        out_of_bounds_agents->erase(iter);
    }
}

    template <typename BaseSpaceType>
    typename DistributedCartesianSpace<BaseSpaceType>::PointType *DistributedCartesianSpace<BaseSpaceType>::move(R4Py_Agent *agent, PointType *to)
{

    PointType* pt = base_space->move(agent, to);
    // pt will be null if the move fails for a valid reason -- e.g.,
    // space is already occupied
    if (pt) {
        eraseOOB(agent->aid);
        if (!local_bounds.contains(pt)) {
            // what rank now contains the agent
            for (auto &box_rank : all_bounds) {
                if (box_rank.first.contains(pt)) {
                    PyObject *aid_tuple = agent->aid->as_tuple;
                    Py_INCREF(aid_tuple);
                    PyArrayObject *pt_array = pt->coords;
                    Py_INCREF(pt_array);
                    // Py_BuildValue increments aid_tuple and pt_array ref counts
                    PyObject *obj = Py_BuildValue("(O, I, O)", aid_tuple, box_rank.second, pt_array);
                    (*out_of_bounds_agents)[agent->aid] = obj;
                    break;
                }
            }
        }
    }
    return pt;
}

template<typename BaseSpaceType>
std::shared_ptr<AIDPyObjMapT> DistributedCartesianSpace<BaseSpaceType>::getOOBData() {
    return out_of_bounds_agents;
}

template<typename BaseSpaceType>
std::shared_ptr<std::vector<CTNeighbor>> DistributedCartesianSpace<BaseSpaceType>::getNeighborData() {
    return nghs;
}

template<typename BaseSpaceType>
void DistributedCartesianSpace<BaseSpaceType>::clearOOBData() {
    for (auto& kv: (*out_of_bounds_agents)) {
        // Note: key wasn't inc'reffed
        Py_DECREF(PyTuple_GET_ITEM(kv.second, 0));
        Py_DECREF(PyTuple_GET_ITEM(kv.second, 2));
        Py_DECREF(kv.second);
    }
    out_of_bounds_agents->clear();
}

template<typename BaseSpaceType>
BoundingBox DistributedCartesianSpace<BaseSpaceType>::getLocalBounds() const {
    return local_bounds;
}

template<typename BaseSpaceType>
MPI_Comm DistributedCartesianSpace<BaseSpaceType>::getCartesianCommunicator() {
    return cart_comm;
}

template<typename BaseSpaceType>
const std::string DistributedCartesianSpace<BaseSpaceType>::name() const {
    return base_space->name();
}

class ISharedGrid {

public:
    virtual ~ISharedGrid() = 0;

    virtual bool add(R4Py_Agent* agent) = 0;
    virtual bool remove(R4Py_Agent* agent) = 0;
    virtual bool remove(R4Py_AgentID* aid) = 0;
    virtual R4Py_Agent* getAgentAt(R4Py_DiscretePoint* pt) = 0;
    virtual AgentListPtr getAgentsAt(R4Py_DiscretePoint* pt) = 0;
    virtual R4Py_DiscretePoint* getLocation(R4Py_Agent* agent) = 0;
    virtual R4Py_DiscretePoint* move(R4Py_Agent* agent, R4Py_DiscretePoint* to) = 0;
    virtual std::shared_ptr<AIDPyObjMapT> getOOBData() = 0;
    virtual std::shared_ptr<std::vector<CTNeighbor>> getNeighborData() = 0;
    virtual void clearOOBData() = 0;
    virtual BoundingBox getLocalBounds() const = 0;
    virtual MPI_Comm getCartesianCommunicator() = 0;
    virtual const std::string name() const = 0;
    virtual bool contains(R4Py_Agent* agent) const = 0;
    
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
    AgentListPtr getAgentsAt(R4Py_DiscretePoint* pt) override;
    R4Py_DiscretePoint* getLocation(R4Py_Agent* agent) override;
    R4Py_DiscretePoint* move(R4Py_Agent* agent, R4Py_DiscretePoint* to) override;
    std::shared_ptr<AIDPyObjMapT> getOOBData() override;
    std::shared_ptr<std::vector<CTNeighbor>> getNeighborData() override;
    void clearOOBData() override;
    BoundingBox getLocalBounds() const override;
    MPI_Comm getCartesianCommunicator() override;
    const std::string name() const override;
    virtual bool contains(R4Py_Agent* agent) const override;
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
bool SharedGrid<DelegateType>::contains(R4Py_Agent* agent) const {
    return delegate->contains(agent);
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
AgentListPtr SharedGrid<DelegateType>::getAgentsAt(R4Py_DiscretePoint* pt) {
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
std::shared_ptr<AIDPyObjMapT> SharedGrid<DelegateType>::getOOBData() {
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

template<typename DelegateType>
const std::string SharedGrid<DelegateType>::name() const {
    return delegate->name();
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
    virtual AgentListPtr getAgentsAt(R4Py_ContinuousPoint* pt) = 0;
    virtual R4Py_ContinuousPoint* getLocation(R4Py_Agent* agent) = 0;
    virtual R4Py_ContinuousPoint* move(R4Py_Agent* agent, R4Py_ContinuousPoint* to) = 0;
    virtual std::shared_ptr<AIDPyObjMapT> getOOBData() = 0;
    virtual std::shared_ptr<std::vector<CTNeighbor>> getNeighborData() = 0;
    virtual void clearOOBData() = 0;
    virtual BoundingBox getLocalBounds() const = 0;
    virtual MPI_Comm getCartesianCommunicator() = 0;
    virtual void getAgentsWithin(const BoundingBox& bounds, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) = 0;
    virtual const std::string name() const = 0;
    virtual bool contains(R4Py_Agent* agent) const = 0;
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
    AgentListPtr getAgentsAt(R4Py_ContinuousPoint* pt) override;
    R4Py_ContinuousPoint* getLocation(R4Py_Agent* agent) override;
    R4Py_ContinuousPoint* move(R4Py_Agent* agent, R4Py_ContinuousPoint* to) override;
    std::shared_ptr<AIDPyObjMapT> getOOBData() override;
    std::shared_ptr<std::vector<CTNeighbor>> getNeighborData() override;
    void clearOOBData() override;
    BoundingBox getLocalBounds() const override;
    MPI_Comm getCartesianCommunicator() override;
    void getAgentsWithin(const BoundingBox& bounds, std::shared_ptr<std::vector<R4Py_Agent*>>& agents) override;
    const std::string name() const;
    bool contains(R4Py_Agent* agent) const override;
    
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
bool SharedContinuousSpace<DelegateType>::contains(R4Py_Agent* agent) const {
    return delegate->contains(agent);
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
AgentListPtr SharedContinuousSpace<DelegateType>::getAgentsAt(R4Py_ContinuousPoint* pt) {
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
std::shared_ptr<AIDPyObjMapT> SharedContinuousSpace<DelegateType>::getOOBData() {
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

template<typename DelegateType>
const std::string SharedContinuousSpace<DelegateType>::name() const {
    return delegate->name();
}


struct R4Py_SharedCSpace {
    PyObject_HEAD
    ISharedContinuousSpace* space;
    PyObject* cart_comm;
};
}


#endif