#include "mpi.h"
#include "gtest/gtest.h"


int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	MPI_Init(&argc, &argv);
	return RUN_ALL_TESTS();
}

