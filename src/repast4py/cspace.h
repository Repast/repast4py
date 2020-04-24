#ifndef R4PY_SRC_CSPACE_H
#define R4PY_SRC_CSPACE_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "space.h"

namespace repast4py {

    class ICSpace {

public:
    virtual ~ICSpace() = 0;

    virtual bool add(R4Py_Agent* agent) = 0;
    virtual bool remove(R4Py_Agent* agent) = 0;
    virtual bool remove(R4Py_AgentID* aid) = 0;
    virtual R4Py_Agent* getAgentAt(R4Py_ContinuousPoint* pt) = 0;
    virtual AgentList getAgentsAt(R4Py_ContinuousPoint* pt) = 0;
    virtual R4Py_ContinuousPoint* getLocation(R4Py_Agent* agent) = 0;
    virtual R4Py_ContinuousPoint* move(R4Py_Agent* agent, R4Py_ContinuousPoint* to) = 0;
};

inline ICSpace::~ICSpace() {}

template<typename DelegateType>
class CSpace : public ICSpace {

private:
    std::unique_ptr<DelegateType> delegate;

public:
    CSpace(const std::string& name, const BoundingBox& bounds);
    virtual ~CSpace() {}
    bool add(R4Py_Agent* agent) override;
    bool remove(R4Py_Agent* agent) override;
    bool remove(R4Py_AgentID* aid) override;
    R4Py_Agent* getAgentAt(R4Py_ContinuousPoint* pt) override;
    AgentList getAgentsAt(R4Py_ContinuousPoint* pt) override;
    R4Py_ContinuousPoint* getLocation(R4Py_Agent* agent) override;
    R4Py_ContinuousPoint* move(R4Py_Agent* agent, R4Py_ContinuousPoint* to) override;
};

template<typename DelegateType>
CSpace<DelegateType>::CSpace(const std::string& name, const BoundingBox& bounds) : 
    delegate{std::unique_ptr<DelegateType>(new DelegateType(name, bounds))} {}

template<typename DelegateType>
bool CSpace<DelegateType>::add(R4Py_Agent* agent) {
    return delegate->add(agent);
}

template<typename DelegateType>
bool CSpace<DelegateType>::remove(R4Py_Agent* agent) {
    return delegate->remove(agent);
}

template<typename DelegateType>
bool CSpace<DelegateType>::remove(R4Py_AgentID* aid) {
    return delegate->remove(aid);
}

template<typename DelegateType>
R4Py_Agent* CSpace<DelegateType>::getAgentAt(R4Py_ContinuousPoint* pt) {
    return delegate->getAgentAt(pt);
}

template<typename DelegateType>
AgentList CSpace<DelegateType>::getAgentsAt(R4Py_ContinuousPoint* pt) {
    return delegate->getAgentsAt(pt);
}

template<typename DelegateType>
R4Py_ContinuousPoint* CSpace<DelegateType>::getLocation(R4Py_Agent* agent) {
    return delegate->getLocation(agent);
}

template<typename DelegateType>
R4Py_ContinuousPoint* CSpace<DelegateType>::move(R4Py_Agent* agent, R4Py_ContinuousPoint* to) {
    return delegate->move(agent, to);
}

// aliases for  CSpace with multi occupancy and sticky borders
using ContinuousMOType = MultiOccupancyAccessor<LocationMapType<R4Py_ContinuousPoint>, R4Py_ContinuousPoint>;
using MOSCSpace = BaseSpace<R4Py_ContinuousPoint, ContinuousMOType, CSStickyBorders>;
using MOPCSpace = BaseSpace<R4Py_ContinuousPoint, ContinuousMOType, CSPeriodicBorders>;

template<>
struct is_periodic<MOSCSpace> {
    static constexpr bool value {false};
};

template<>
struct is_periodic<MOPCSpace> {
    static constexpr bool value {true};
};

struct R4Py_CSpace {
    PyObject_HEAD
    ICSpace* space;
};


}

#endif