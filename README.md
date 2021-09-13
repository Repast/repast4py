# Repast for Python (Repast4Py)

[![codecov](https://codecov.io/gh/Repast/repast4py/branch/develop/graph/badge.svg?token=JCDU2LT8G2)](https://codecov.io/gh/Repast/repast4py/branch/develop)

## Build Status

<table>
  <tr>
    <td><b>Master</b></td>
    <td><b>Develop</b></td>
  </tr>
  <tr>
    <td><a href="https://circleci.com/gh/Repast/repast4py/tree/master"><img src="https://circleci.com/gh/Repast/repast4py/tree/master.svg?style=shield&circle-token=8eabe328704119bf3f175172e1613c52f9310c65" alt="Build Status" /></a></td>
    <td><a href="https://circleci.com/gh/Repast/repast4py/tree/develop"><img src="https://circleci.com/gh/Repast/repast4py/tree/develop.svg?style=shield&circle-token=8eabe328704119bf3f175172e1613c52f9310c65" alt="Build Status" /></a></td>
  </tr>
</table>

Repast for Python (Repast4Py) is a distributed agent-based simulation toolkit written in Python.
Modern CPUs typically contain multiple cores, each of which is capable of running concurrently.
Repast4Py attempts to leverage this hardware by distributing a simulation over multiple processes
running in parallel on these cores. A typical agent-based simulation consists of a population of agents 
each of which performs some behavior each timestep or at some frequency. In practice, this
is often implemented as a loop over the agent population in which each agent executes its behavior. 
The time it takes to complete the loop depends on the number of agents and the complexity of the behavior.
By distributing the agent population across multiple processes running in parallel, each process 
executes its own loop over only a subset of the population, allowing for larger agent populations and more 
complex behavior without increasing the runtime. Repast4Py is also a natural fit for implementing
agent-based simulations on high performance computers and clusters, such as those hosted by
universities, national laboratories, and cloud computing providers. Such machines can have
thousands or 10s of thousands of processor cores available, allowing for very large and
complex simulations.

Repast4Py is part of the [Repast](https://repast.github.io) family of agent-based modeling toolkits
which have been successfully used in many application domains including social science, consumer products, 
supply chains, hydrogen infrastructures, manufacturing, epidemiology, biomedical systems modeling, and ancient 
pedestrian traffic to name a few.

### Installation ###

pip install repast4py

or 

If need to install from source,

install mpi -- apt get install mpich-dev etc.

CC=mpicxx CXX=mpicxx pip install repast4py

### Documentation ###

Links to docs