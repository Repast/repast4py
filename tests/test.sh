#!/bin/bash
commands=(
  "coverage run -m unittest discover tests"
  "mpirun -n 9 coverage run --rcfile=coverage.rc -m unittest tests.shared_obj_tests"
  "mpirun -n 9 coverage run --rcfile=coverage.rc -m unittest tests.shared_vl_tests"
  "mpirun -n 18 coverage run --rcfile=coverage.rc -m unittest tests.shared_obj_tests.SharedGridTests.test_buffer_data_3d"
  "mpirun -n 18 coverage run --rcfile=coverage.rc -m unittest tests.shared_obj_tests.SharedGridTests.test_buffer_data_3d_periodic"
  "mpirun -n 4 coverage run --rcfile=coverage.rc -m unittest tests.logging_tests"
  "mpirun -n 9 python -m unittest tests.ctopo_tests"
  "mpirun -n 4 python -m unittest tests.shared_network_tests"
)

declare -i result

for cmd in "${commands[@]}"
do
  echo $cmd
  echo ""
  $cmd
  (( result = $result + $? ))
  echo ""
done

exit $result
