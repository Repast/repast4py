#include "mpi.h"
#include "gtest/gtest.h"


int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	MPI_Init(&argc, &argv);
	int ret_val = RUN_ALL_TESTS();
	MPI_Finalize();
	return ret_val;
}

