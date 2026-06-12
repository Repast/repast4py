# Development Notes #

### Compiling and Testing ###
Compile with: 

`CC=mpicxx CXX=mpicxx python setup.py build_ext --inplace`

or for debugging:

`CC=mpicxx CXX=mpicxx CFLAGS="-O0 -g" CXXFLAGS="-O0 -g" python setup.py build_ext --inplace`


There are 3 types of python unit tests:

1. Ordinary single process tests. Run with:

`python -m unittest discover tests` 

2. Multiprocess (9 procs) mpi tests for 2D spaces. Run with:

```
mpirun -n 9 python -m unittest tests.shared_obj_tests
mpirun -n 9 python -m unittest tests.shared_vl_tests
mpirun -n 9 python -m unittest tests.ctopo_tests
```

3. Multiprocess (18 procs) mpi tests for 3D spaces. Run with:

```
mpirun -n 18 python -m unittest tests.shared_obj_tests.SharedGridTests.test_buffer_data_3d
mpirun -n 18 python -m unittest tests.shared_obj_tests.SharedGridTests.test_buffer_data_3d_periodic
```

4. Multiprocess (4 procs) mpi tests for logging and network support. Run with:

```
mpirun -n 4 python -m unittest tests.logging_tests
mpirun -n 4 python -m unittest tests.shared_network_tests
```


There are also some C++ unitests. C++ tests can be compiled with the Makefile in the tests directory. 
Copy the Makefile to a `Release` or `Debug` directory at the top level, and edit it as necessary.
The  makefile target 'tests' will compile a `unit_tests` executable. Run the tests with:

```
mpirun -n 9 ./unit_tests --gtest_filter=CartesianTopology.*
./unit_tests --gtest_filter=SpatialTreeTests.*
```

### Testing both build modes with tox ###

`tox` builds and runs the test suite for both build modes across Python versions.
The environments are generative: `py{39,310,311,312,313}-{mpi,mock}`.

* `*-mpi` envs build the default native-MPI variant (`CC=CXX=mpicxx`) and run the
  single-process tests plus the `mpirun` parallel suites.
* `*-mock` envs build the single-rank variant (`R4PY_SINGLE_RANK=1`, no native MPI
  and no mpi4py) and run the single-process tests.

Common invocations:

```
tox                 # all environments (both modes, all Python versions)
tox -f mpi          # default native-MPI mode, all Python versions
tox -f mock         # single-rank mock mode, all Python versions
tox -e py312-mpi    # a single environment
tox -e py312-mock
```

tox builds the package itself per environment and sets `CC`/`CXX` to `mpicxx` for
the `mpi` envs (taken from `PATH`), so those do not need to be set on the command
line. The `mpi` envs require an MPI installation (`mpicxx`, `mpirun`); the `mock`
envs require neither.

## Requirements

* Python 3.8+
* mpi4py
* PyTorch
* NumPy >= 1.18
* nptyping (`pip install nptyping`)
* numba
* typing-extensions if < 3.8
* pyyaml

## Linting ##

flake8 - configuration, exclusions etc. are in setup.cfg

## Documentation Guidelines ##

* Use Sphinx and readthedocs.io

https://docs.python-guide.org/writing/documentation/#sphinx

* Use restructured text:

https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html

* Use google style doc strings:

https://www.sphinx-doc.org/en/1.8/usage/extensions/example_google.html?highlight=docstring

* Pep 484 type hints - when these are present then they do not need to be included in the doc string

## Generating API Docs ##

If a new module is added then, from within the docs directory `sphinx-apidoc -e -o source ../src/repast4py` 
to generate the rst for that module.

And `make html` to create the html docs.

`make clean` followed by `make html` will build from scratch.


## Generating ASCIIDoc Manual ##

### Prerequisites ###

1. Install asciidoctor.
  * With apt (Ubuntu): `sudo apt-get install -y asciidoctor`
  * Other OSes, see `https://docs.asciidoctor.org/asciidoctor/latest/install/`
2. Install pygments for code syntax highlighting.
  * Ubuntu: `gem install --user-install pygments.rb`
  * Other OSes, see `https://docs.asciidoctor.org/asciidoctor/latest/syntax-highlighting/pygments/`

Currently (08/09/2021) the docs are generated as single html page. If we want multiple
pages, see `https://github.com/owenh000/asciidoctor-multipage`

### Generating the Docs ###

```
cd docs/guide
asciidoctor user_guide.adoc
```

This generates a user_guide.html that can be viewed in a browser.

### Creating a Distribution ###

`CC=mpicxx CXX=mpicxx python -m build`

creates a source tar.gz and a wheel in `dist/`

https://packaging.python.org/guides/distributing-packages-using-setuptools/#packaging-your-project
https://setuptools.readthedocs.io/en/latest/userguide/index.html
https://packaging.python.org/tutorials/packaging-projects/#packaging-python-projects

Note that a whl created on linux cannot be uploaded. See the many linux project:
https://github.com/pypa/manylinux

Testing the source dist in a virtual env with tox — see "Testing both build modes
with tox" above for the environment names. The `-r` flag recreates the virtual
envs if needed:

`tox -r -f mpi`

If using conda for Python, switch to the appropriate
environment, and then use tox's -e argument to select
the env that matches the activated conda environment:

```
tox -e py312-mpi
tox -e py312-mock
```

### Uploading to PyPI (or testpypi)

This uses ~/.pypirc for API token authentication

python3 -m twine upload --repository repast4py dist/*.tar.gz

### Testing from testPyPI

CC=mpicxx pip3 install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple repast4py

## Multi Process Seg Fault Debugging

See https://www.open-mpi.org/faq/?category=debugging#serial-debuggers for using gdb with mpi.

General idea is to add this code

```
  {
         volatile int i = 0;
         char hostname[256];
         gethostname(hostname, sizeof(hostname));
         printf("PID %d on %s ready for attach\n", getpid(), hostname);
         fflush(stdout);
         while (0 == i)
             sleep(5);
    }
```

to the module code function call that's triggering the segfault, and follow the directions in the link above. Note that might have to run gdb via sudo.