# repast4py

## Build Status

<table>
  <tr>
    <td><b>Master</b></td>
    <td><b>Develop</b></td>
  </tr>
  <tr>
    <td><a href="https://circleci.com/gh/Repast/repast4py/tree/master"><img src="https://circleci.com/gh/Repast/repast4py/tree/master.svg?style=shield&circle-token=7c12be81746f1285510fd8f96ce5700f6a44ae13" alt="Build Status" /></a></td>
    <td><a href="https://circleci.com/gh/Repast/repast4py/tree/develop"><img src="https://circleci.com/gh/Repast/repast4py/tree/develop.svg?style=shield&circle-token=7c12be81746f1285510fd8f96ce5700f6a44ae13" alt="Build Status" /></a></td>
  </tr>
</table>

Compile with: 

`CC=mpicc CXX=mpicxx python setup.py build_ext --inplace`

or for debugging:

`CC=mpicc CXX=mpicxx CFLAGS="-O0 -g" CXXFLAGS="-O0 -g" python setup.py build_ext --inplace`

## Tests ##

There are 3 types of python unit tests:

1. Ordinary single process tests. Run with:

`python -m unittest discover tests` 

2. Multiprocess (9 procs) mpi tests for 2D spaces. Run with:

```
mpirun -n 9 python -m unittest tests.shared_obj_tests
mpirun -n 9 python -m unittest tests.shared_vl_tests
```

3. Multiprocess (18 procs) mpi tests for 3D spaces. Run with:

```
mpirun -n 18 python -m unittest tests.shared_obj_tests.SharedGridTests.test_buffer_data_3d
mpirun -n 18 python -m unittest tests.shared_obj_tests.SharedGridTests.test_buffer_data_3d_periodic
mpirun -n 18 python -m unittest tests.shared_vl_tests.SharedValueLayerTests.test_buffers_3x3x3_periodic
mpirun -n 18 python -m unittest tests.shared_vl_tests.SharedValueLayerTests.test_buffers_3x3x3_sticky
```

4. Multiprocess (4 procs) mpi tests for logging and network support. Run with:

```
mpirun -n 4 python -m unittest tests.logging_tests
mpirun -n 4 python -m unittest tests.shared_network_tests
```


There are also some C++ unitests. C++ tests can be compiled with makefile target 'tests' and run with:

`mpirun -n 9 ./unit_tests`

## Zombies ##

Requires compiled repast4py.

Run with: 

`PYTHONPATH=./src python src/zombies/zombies.py src/zombies/zombie_model.props`

Supplying a json string as a second argument will override the props in model.props

`PYTHONPATH=./src python src/zombies/zombies.py src/zombies/zombie_model.props "{\"random.seed\" : 12, \"stop.at\" : 20, \"human.count\" : 10, \"zombie.count\" : 5, \"world.width\" : 10, \"world.height\" : 22, \"run.number\" : 2}"`


