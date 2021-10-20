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

## Repast4Py

Repast for Python (Repast4Py) is the newest member of the [Repast Suite](https://repast.github.io) of free and open source agent-based modeling and simulation software.
It builds on [Repast HPC](https://repast.github.io/repast_hpc.html), and provides the ability to build large, distributed agent-based models (ABMs) that span multiple processing cores. 
Distributed ABMs enable the development of complex systems models that capture the scale and relevant details of many problems of societal importance. Where Repast HPC is implemented in C++ and is more HPC expert focused, Repast4Py is a Python package and is designed to provide an easier on-ramp for researchers from diverse scientific communities to apply large-scale distributed ABM methods. See our paper on Repast4Py for additional information about the design and implementation.

### Requirements

Repast4Py requires Python 3.7+

Repast4Py can run on Linux, macOS and Windows provided there is a working MPI implementation
installed and mpi4py is supported. Repast4Py is developed and tested on Linux. We recommend
that Windows users use the Windows Subsystem for Linux (WSL). Installation instructions for
WSL can be found [here](https://docs.microsoft.com/en-us/windows/wsl/install).

Under Linux, MPI can be installed using your OS's package manager. For example, 
under Ubuntu 20.04 (and thus WSL), the mpich MPI implementation can be installed with:

```bash
$ sudo apt install mpich
```

A typical campus cluster, or HPC resource will have MPI and mpi4py installed. 
Check the resource's documentation on available software for more details.

### Installation

Repast4Py can be downloaded and installed from PyPI using pip. 
However, since Repast4Py includes native MPI C++ code that needs to be compiled,
the C compiler `CC` environment variable must be set
to the `mpicxx` (or `mpic++`) compiler wrapper provided by your MPI installation.

```
env CC=mpicxx pip install repast4py
```

### Documentation

* [User's Guide](https://jozik.github.io/goes_bing/guide/user_guide.html)
* [API Docs](https://jozik.github.io/goes_bing/apidoc/index.html)
* [Example Models](https://jozik.github.io/goes_bing/examples/examples.html)

### Contact and Support

* [GitHub Issues](https://github.com/Repast/repast4py/issues)
* [GitHub Repository](https://github.com/Repast/repast4pyV)

In addition to filing issues on GitHub, support is also available via Stack Overflow. 
Please use the `repast4py` tag to ensure that we are notified of your question. 

Jonathan Ozik is the Repast project lead. Please contact him through 
the [Argonne Staff Directory](https://www.anl.gov/staff-directory) if you
have project-related questions.

