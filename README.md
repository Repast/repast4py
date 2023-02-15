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

Repast for Python (Repast4Py) is the newest member of the [Repast Suite](https://repast.github.io) of 
free and open source agent-based modeling and simulation software.
It builds on  [Repast HPC](https://repast.github.io/repast_hpc.html), and provides the ability to build large, distributed agent-based models (ABMs) that span multiple processing cores. 
Distributed ABMs enable the development of complex systems models that capture the 
scale and relevant details of many problems of societal importance. Where Repast HPC is 
implemented in C++ and is more HPC expert focused, Repast4Py is a Python package and is 
designed to provide an easier on-ramp for researchers from diverse scientific communities to apply large-scale distributed ABM methods. 
Repast4Py is released under the BSD-3 open source license, and leverages [Numba](https://numba.pydata.org),
[NumPy](https://numpy.org), and [PyTorch](https://pytorch.org) packages, and the Python C API 
to create a scalable modeling system that can exploit the largest HPC resources and emerging computing architectures. See our paper on Repast4Py for additional information about the design and implementation.

Collier, N. T., Ozik, J., & Tatara, E. R. (2020). Experiences in Developing a Distributed Agent-based Modeling Toolkit with Python. 2020 IEEE/ACM 9th Workshop on Python for High-Performance and Scientific Computing (PyHPC), 1–12. https://doi.org/10.1109/PyHPC51966.2020.00006

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

Installation instructions for MPI on macOS can be found [here](https://repast.github.io/repast4py.site/macos_mpi_install.html).

A typical campus cluster, or HPC resource will have MPI and mpi4py installed. 
Check the resource's documentation on available software for more details.

### Installation

Repast4Py can be downloaded and installed from PyPI using pip. 
Since Repast4Py includes native MPI C++ code that needs to be compiled,
the C compiler `CC` environment variable must be set
to the `mpicxx` (or `mpic++`) compiler wrapper provided by your MPI installation.

```
env CC=mpicxx pip install repast4py
```

__NOTE__: If you see an error message about a missing `python.h` header file when
installing Repast4Py under Ubuntu (or other Linuxes), you will need to install
a python dev package using your OS's package manager. For example, assuming
Python 3.8, `sudo apt install python3.8-dev` will work for Ubuntu.

### Documentation

* [User's Guide](https://repast.github.io/repast4py.site/guide/user_guide.html)
* [API Docs](https://repast.github.io/repast4py.site/apidoc/index.html)
* [Example Models](https://repast.github.io/repast4py.site/examples/examples.html)

### Contact and Support

* [GitHub Issues](https://github.com/Repast/repast4py/issues)
* [GitHub Repository](https://github.com/Repast/repast4pyV)

In addition to filing issues on GitHub, support is also available via
[Stack Overflow](https://stackoverflow.com/questions/tagged/repast4py). 
Please use the `repast4py` tag to ensure that we are notified of your question. 
Software announcements will be made on the 
[repast-interest](http://lists.sourceforge.net/lists/listinfo/repast-interest) mailing list.

Jonathan Ozik is the Repast project lead. Please contact him through 
the [Argonne Staff Directory](https://www.anl.gov/staff-directory) if you
have project-related questions.
