#ifndef R4PY_SRC_CORE_H
#define R4PY_SRC_CORE_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <list>
#include <memory>

namespace repast4py {

struct R4Py_AgentID {
    long id;
    int type;
    unsigned int rank;
    PyObject* as_tuple;
};

struct agent_id_comp {
    bool operator()(const R4Py_AgentID* a1, const R4Py_AgentID* a2) const {
        if (a1->id != a2->id) return a1->id < a2->id;
        if (a1->type != a2->type) return a1->type < a2->type;
        return a1->rank < a2->rank;
    }
};

struct R4Py_Agent {
    PyObject_HEAD
    R4Py_AgentID* aid;
};

class AgentIter {

protected:
    bool incr;

public:
    AgentIter() : incr{false} {}
    virtual ~AgentIter() {}
    virtual R4Py_Agent* next() = 0;
    virtual bool hasNext() = 0;
    virtual void reset() = 0;
};

template<typename T>
AgentIter* create_iter(T* iterable);

struct R4Py_AgentIter {
    PyObject_HEAD
    AgentIter* iter;
};

template<typename IterableT>
class TAgentIter : public AgentIter {

private:
    std::shared_ptr<IterableT> iterable_;
    typename IterableT::iterator iter_;

public:
    TAgentIter(std::shared_ptr<IterableT>);
    virtual ~TAgentIter() {}

    R4Py_Agent* next() override;
    bool hasNext() override;
    void reset() override;
};

template<typename IterableT>
TAgentIter<IterableT>::TAgentIter(std::shared_ptr<IterableT> iterable) : AgentIter(),
    iterable_{iterable}, iter_(iterable_->begin()) {}

template<typename IterableT>
bool TAgentIter<IterableT>::hasNext() {
    return iter_ != iterable_->end();
}

template<typename IterableT>
void TAgentIter<IterableT>::reset() {
    iter_ = iterable_->begin();
}

template<typename IterableT>
R4Py_Agent* TAgentIter<IterableT>::next() {
    R4Py_Agent* agent = *iter_;
    ++iter_;
    return agent;
}

class PyObjectIter {

protected:
    bool incr;

public:
    PyObjectIter() : incr{false} {}
    virtual ~PyObjectIter() {}
    virtual PyObject* next() = 0;
    virtual bool hasNext() = 0;
    virtual void reset() = 0;
};


struct R4Py_PyObjectIter {
    PyObject_HEAD
    PyObjectIter* iter;
};

template<typename MapT>
class ValueIter : public PyObjectIter {

private:
    std::shared_ptr<MapT> iterable_;
    typename MapT::const_iterator iter_;

public:
    ValueIter(std::shared_ptr<MapT>);
    virtual ~ValueIter() {}

    PyObject* next() override;
    bool hasNext() override;
    void reset() override;
};

template<typename MapT>
ValueIter<MapT>::ValueIter(std::shared_ptr<MapT> iterable) : PyObjectIter(),
    iterable_{iterable}, iter_(iterable_->begin()) {}

template<typename MapT>
bool ValueIter<MapT>::hasNext() {
    return iter_ != iterable_->end();
}

template<typename MapT>
void ValueIter<MapT>::reset() {
    iter_ = iterable_->begin();
}

template<typename MapT>
PyObject* ValueIter<MapT>::next() {
    PyObject* obj = iter_->second;
    ++iter_;
    return obj;
}




}

#endif