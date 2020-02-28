# repast4py

Compile with: 

`CC=mpicc CXX=mpicxx python setup.py build_ext --inplace`

or for debugging:

`CC=mpicc CXX=mpicxx CFLAGS="-O0 -g" CXXFLAGS="-O0 -g" python setup.py build_ext --inplace`

## Tests ##

`python -m unittest discover tests` will run all the unit tests in `tests` that
begin with 'test'. 

Multiprocess tests can be run with:

`mpirun -n 9 python -m unittest tests.shared_obj_tests.SharedContextTests.test_buffer_3x3`

C++ tests can be compiled with makefile target 'tests' and run with:

`mpirun -n 9 ./unit_tests`