/*
 * NetworkTests.cpp
 *
 *  Created on: Nov 2, 2015
 *      Author: nick
 */

#include "gtest/gtest.h"
#include "mpi.h"

#include "distributed_space.h"

using namespace repast4py;

// Run with: -n 9

void create_test_comm(MPI_Comm* test_comm) {
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    const int exclude[1] = {8};
    MPI_Group test_group;
    MPI_Group_excl(world_group, 1, exclude, &test_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, test_group, 0, test_comm);
}

TEST(CartesianTopology, testAutoProcsPerDim) {
    MPI_Comm test_comm;
    create_test_comm(&test_comm);

    if (test_comm != MPI_COMM_NULL) {

        BoundingBox<R4Py_DiscretePoint> gb(0, 100, 0, 200);
        CartesianTopology ct(test_comm, 2, gb, true);
        BoundingBox<R4Py_DiscretePoint> lb(0, 0, 0, 0);

        int rank;
        MPI_Comm_rank(test_comm, &rank);

        ct.getBounds(rank, lb);
        if (rank == 0) {
            ASSERT_EQ(0, lb.xmin_);
            ASSERT_EQ(25, lb.x_extent_);
            ASSERT_EQ(0, lb.ymin_);
            ASSERT_EQ(100, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 1) {
            ASSERT_EQ(0, lb.xmin_);
            ASSERT_EQ(25, lb.x_extent_);
            ASSERT_EQ(100, lb.ymin_);
            ASSERT_EQ(100, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 2) {
            ASSERT_EQ(25, lb.xmin_);
            ASSERT_EQ(25, lb.x_extent_);
            ASSERT_EQ(0, lb.ymin_);
            ASSERT_EQ(100, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 3) {
            ASSERT_EQ(25, lb.xmin_);
            ASSERT_EQ(25, lb.x_extent_);
            ASSERT_EQ(100, lb.ymin_);
            ASSERT_EQ(100, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 4) {
            ASSERT_EQ(50, lb.xmin_);
            ASSERT_EQ(25, lb.x_extent_);
            ASSERT_EQ(0, lb.ymin_);
            ASSERT_EQ(100, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 5) {
            ASSERT_EQ(50, lb.xmin_);
            ASSERT_EQ(25, lb.x_extent_);
            ASSERT_EQ(100, lb.ymin_);
            ASSERT_EQ(100, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 6) {
            ASSERT_EQ(75, lb.xmin_);
            ASSERT_EQ(25, lb.x_extent_);
            ASSERT_EQ(0, lb.ymin_);
            ASSERT_EQ(100, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 7) {
            ASSERT_EQ(75, lb.xmin_);
            ASSERT_EQ(25, lb.x_extent_);
            ASSERT_EQ(100, lb.ymin_);
            ASSERT_EQ(100, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        }
    }
}

TEST(CartesianTopology, testRemainders) {
    MPI_Comm test_comm;
    create_test_comm(&test_comm);

    if (test_comm != MPI_COMM_NULL) {

        BoundingBox<R4Py_DiscretePoint> gb(0, 107, 0, 201);
        CartesianTopology ct(test_comm, 2, gb, true);
        BoundingBox<R4Py_DiscretePoint> lb(0, 0, 0, 0);

        int rank;
        MPI_Comm_rank(test_comm, &rank);

        ct.getBounds(rank, lb);
        if (rank == 0) {
            ASSERT_EQ(0, lb.xmin_);
            ASSERT_EQ(27, lb.x_extent_);
            ASSERT_EQ(0, lb.ymin_);
            ASSERT_EQ(101, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 1) {
            ASSERT_EQ(0, lb.xmin_);
            ASSERT_EQ(27, lb.x_extent_);
            ASSERT_EQ(101, lb.ymin_);
            ASSERT_EQ(100, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 2) {
            ASSERT_EQ(27, lb.xmin_);
            ASSERT_EQ(27, lb.x_extent_);
            ASSERT_EQ(0, lb.ymin_);
            ASSERT_EQ(101, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 3) {
            ASSERT_EQ(27, lb.xmin_);
            ASSERT_EQ(27, lb.x_extent_);
            ASSERT_EQ(101, lb.ymin_);
            ASSERT_EQ(100, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 4) {
            ASSERT_EQ(54, lb.xmin_);
            ASSERT_EQ(27, lb.x_extent_);
            ASSERT_EQ(0, lb.ymin_);
            ASSERT_EQ(101, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 5) {
            ASSERT_EQ(54, lb.xmin_);
            ASSERT_EQ(27, lb.x_extent_);
            ASSERT_EQ(101, lb.ymin_);
            ASSERT_EQ(100, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 6) {
            ASSERT_EQ(81, lb.xmin_);
            ASSERT_EQ(26, lb.x_extent_);
            ASSERT_EQ(0, lb.ymin_);
            ASSERT_EQ(101, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 7) {
            ASSERT_EQ(81, lb.xmin_);
            ASSERT_EQ(26, lb.x_extent_);
            ASSERT_EQ(101, lb.ymin_);
            ASSERT_EQ(100, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        }
    }
}

CTNeighbor find_neighbor(int rank, std::vector<CTNeighbor>& nghs) {
    for (auto n : nghs) {
        if (n.rank == rank) return n;
    }
    return {-1, -1, -1, -1};
}

void test_ngh(const CTNeighbor& n, int x, int y, int z) {
    ASSERT_TRUE(n.rank != -1);
    ASSERT_EQ(x, n.cart_coord_x);
    ASSERT_EQ(y, n.cart_coord_y);
    ASSERT_EQ(z, n.cart_coord_z);
}

TEST(CartesianTopology, testGetNeighbors) {
    BoundingBox<R4Py_DiscretePoint> gb(0, 100, 0, 200);
    CartesianTopology ct(MPI_COMM_WORLD, 2, gb, false);

    std::vector<CTNeighbor> nghs;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ct.getNeighbors(rank, nghs);

    if (rank == 0) {
        // for (int i = 0; i < 9; ++i) {
        //     std::vector<int> coords;
        //     ct.getCoords(i, coords);
        //     std::cout << i << ": " << coords[0] << ", " << coords[1] << std::endl;
        // }
        SCOPED_TRACE("0");
        ASSERT_EQ(3, nghs.size());
        test_ngh(find_neighbor(1, nghs), 0, 1, -1);
        test_ngh(find_neighbor(3, nghs), 1, 0, -1);
        test_ngh(find_neighbor(4, nghs), 1, 1, -1);
    } else if (rank == 1) {
        SCOPED_TRACE("1");
        ASSERT_EQ(5, nghs.size());
        test_ngh(find_neighbor(0, nghs), 0, 0, -1);
        test_ngh(find_neighbor(2, nghs), 0, 2, -1);
        test_ngh(find_neighbor(3, nghs), 1, 0, -1);
        test_ngh(find_neighbor(4, nghs), 1, 1, -1);
        test_ngh(find_neighbor(5, nghs), 1, 2, -1);
    } else if (rank == 2) {
        SCOPED_TRACE("2");
        ASSERT_EQ(3, nghs.size());
        test_ngh(find_neighbor(1, nghs), 0, 1, -1);
        test_ngh(find_neighbor(4, nghs), 1, 1, -1);
        test_ngh(find_neighbor(5, nghs), 1, 2, -1);
    } else if (rank == 3) {
        SCOPED_TRACE("3");
        ASSERT_EQ(5, nghs.size());
        test_ngh(find_neighbor(0, nghs), 0, 0, -1);
        test_ngh(find_neighbor(1, nghs), 0, 1, -1);
        test_ngh(find_neighbor(4, nghs), 1, 1, -1);
        test_ngh(find_neighbor(6, nghs), 2, 0, -1);
        test_ngh(find_neighbor(7, nghs), 2, 1, -1);

    } else if (rank == 4) {
        SCOPED_TRACE("4");
        ASSERT_EQ(8, nghs.size());
        test_ngh(find_neighbor(0, nghs), 0, 0, -1);
        test_ngh(find_neighbor(1, nghs), 0, 1, -1);
        test_ngh(find_neighbor(2, nghs), 0, 2, -1);
        test_ngh(find_neighbor(3, nghs), 1, 0, -1);
        test_ngh(find_neighbor(5, nghs), 1, 2, -1);
        test_ngh(find_neighbor(6, nghs), 2, 0, -1);
        test_ngh(find_neighbor(7, nghs), 2, 1, -1);
        test_ngh(find_neighbor(8, nghs), 2, 2, -1);
        
    } else if (rank == 5) {
        SCOPED_TRACE("5");
        ASSERT_EQ(5, nghs.size());
        test_ngh(find_neighbor(1, nghs), 0, 1, -1);
        test_ngh(find_neighbor(2, nghs), 0, 2, -1);
        test_ngh(find_neighbor(4, nghs), 1, 1, -1);
        test_ngh(find_neighbor(7, nghs), 2, 1, -1);
        test_ngh(find_neighbor(8, nghs), 2, 2, -1);
        
    } else if (rank == 6) {
        SCOPED_TRACE("6");
        ASSERT_EQ(3, nghs.size());
        test_ngh(find_neighbor(3, nghs), 1, 0, -1);
        test_ngh(find_neighbor(4, nghs), 1, 1, -1);
        test_ngh(find_neighbor(7, nghs), 2, 1, -1);
    } else if (rank == 7) {
        SCOPED_TRACE("7");
        ASSERT_EQ(5, nghs.size());
        test_ngh(find_neighbor(3, nghs), 1, 0, -1);
        test_ngh(find_neighbor(4, nghs), 1, 1, -1);
        test_ngh(find_neighbor(5, nghs), 1, 2, -1);
        test_ngh(find_neighbor(6, nghs), 2, 0, -1);
        test_ngh(find_neighbor(8, nghs), 2, 2, -1);
    } else if (rank == 8) {
        SCOPED_TRACE("8");
        ASSERT_EQ(3, nghs.size());
        test_ngh(find_neighbor(4, nghs), 1, 1, -1);
        test_ngh(find_neighbor(5, nghs), 1, 2, -1);
        test_ngh(find_neighbor(7, nghs), 2, 1, -1);
    }
}

TEST(CartesianTopology, testGetNeighborsPeriodic) {
    BoundingBox<R4Py_DiscretePoint> gb(0, 100, 0, 200);
    CartesianTopology ct(MPI_COMM_WORLD, 2, gb, true);

    std::vector<CTNeighbor> nghs;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ct.getNeighbors(rank, nghs);

    // test_ngh(find_neighbor(0, nghs), 0, 0, -1);
    // test_ngh(find_neighbor(1, nghs), 0, 1, -1);
    // test_ngh(find_neighbor(2, nghs), 0, 2, -1);
    // test_ngh(find_neighbor(3, nghs), 1, 0, -1);
    // test_ngh(find_neighbor(4, nghs), 1, 1, -1);
    // test_ngh(find_neighbor(5, nghs), 1, 2, -1);
    // test_ngh(find_neighbor(6, nghs), 2, 0, -1);
    // test_ngh(find_neighbor(7, nghs), 2, 1, -1);
    // test_ngh(find_neighbor(8, nghs), 2, 2, -1);

    if (rank == 0) {
        SCOPED_TRACE("0");
        ASSERT_EQ(8, nghs.size());
        test_ngh(find_neighbor(6, nghs), 2, 0, -1);
        test_ngh(find_neighbor(8, nghs), 2, 2, -1);
        test_ngh(find_neighbor(2, nghs), 0, 2, -1);
        test_ngh(find_neighbor(1, nghs), 0, 1, -1);
        test_ngh(find_neighbor(3, nghs), 1, 0, -1);
        test_ngh(find_neighbor(4, nghs), 1, 1, -1);   
        test_ngh(find_neighbor(5, nghs), 1, 2, -1);
        test_ngh(find_neighbor(7, nghs), 2, 1, -1);
    } else if (rank == 1) {
        SCOPED_TRACE("1");
        ASSERT_EQ(8, nghs.size());
        test_ngh(find_neighbor(0, nghs), 0, 0, -1);
        test_ngh(find_neighbor(2, nghs), 0, 2, -1);
        test_ngh(find_neighbor(3, nghs), 1, 0, -1);
        test_ngh(find_neighbor(4, nghs), 1, 1, -1);
        test_ngh(find_neighbor(5, nghs), 1, 2, -1);
        test_ngh(find_neighbor(6, nghs), 2, 0, -1);
        test_ngh(find_neighbor(7, nghs), 2, 1, -1);
        test_ngh(find_neighbor(8, nghs), 2, 2, -1);
    } else if (rank == 2) {
        SCOPED_TRACE("2");
        ASSERT_EQ(8, nghs.size());
        test_ngh(find_neighbor(0, nghs), 0, 0, -1);
        test_ngh(find_neighbor(1, nghs), 0, 1, -1);
        test_ngh(find_neighbor(3, nghs), 1, 0, -1);
        test_ngh(find_neighbor(4, nghs), 1, 1, -1);
        test_ngh(find_neighbor(5, nghs), 1, 2, -1);
        test_ngh(find_neighbor(6, nghs), 2, 0, -1);
        test_ngh(find_neighbor(7, nghs), 2, 1, -1);
        test_ngh(find_neighbor(8, nghs), 2, 2, -1);
    } else if (rank == 3) {
        SCOPED_TRACE("3");
        ASSERT_EQ(8, nghs.size());
        test_ngh(find_neighbor(0, nghs), 0, 0, -1);
        test_ngh(find_neighbor(1, nghs), 0, 1, -1);
        test_ngh(find_neighbor(2, nghs), 0, 2, -1);
        test_ngh(find_neighbor(4, nghs), 1, 1, -1);
        test_ngh(find_neighbor(5, nghs), 1, 2, -1);
        test_ngh(find_neighbor(6, nghs), 2, 0, -1);
        test_ngh(find_neighbor(7, nghs), 2, 1, -1);
        test_ngh(find_neighbor(8, nghs), 2, 2, -1);
    } else if (rank == 4) {
        SCOPED_TRACE("4");
        ASSERT_EQ(8, nghs.size());
        test_ngh(find_neighbor(0, nghs), 0, 0, -1);
        test_ngh(find_neighbor(1, nghs), 0, 1, -1);
        test_ngh(find_neighbor(2, nghs), 0, 2, -1);
        test_ngh(find_neighbor(3, nghs), 1, 0, -1);
        test_ngh(find_neighbor(5, nghs), 1, 2, -1);
        test_ngh(find_neighbor(6, nghs), 2, 0, -1);
        test_ngh(find_neighbor(7, nghs), 2, 1, -1);
        test_ngh(find_neighbor(8, nghs), 2, 2, -1);
        
    } else if (rank == 5) {
        SCOPED_TRACE("5");
        ASSERT_EQ(8, nghs.size());
        test_ngh(find_neighbor(0, nghs), 0, 0, -1);
        test_ngh(find_neighbor(1, nghs), 0, 1, -1);
        test_ngh(find_neighbor(2, nghs), 0, 2, -1);
        test_ngh(find_neighbor(3, nghs), 1, 0, -1);
        test_ngh(find_neighbor(4, nghs), 1, 1, -1);
        test_ngh(find_neighbor(6, nghs), 2, 0, -1);
        test_ngh(find_neighbor(7, nghs), 2, 1, -1);
        test_ngh(find_neighbor(8, nghs), 2, 2, -1);
    } else if (rank == 6) {
        SCOPED_TRACE("6");
        ASSERT_EQ(8, nghs.size());
        test_ngh(find_neighbor(0, nghs), 0, 0, -1);
        test_ngh(find_neighbor(1, nghs), 0, 1, -1);
        test_ngh(find_neighbor(2, nghs), 0, 2, -1);
        test_ngh(find_neighbor(3, nghs), 1, 0, -1);
        test_ngh(find_neighbor(4, nghs), 1, 1, -1);
        test_ngh(find_neighbor(5, nghs), 1, 2, -1);
        test_ngh(find_neighbor(7, nghs), 2, 1, -1);
        test_ngh(find_neighbor(8, nghs), 2, 2, -1);

    } else if (rank == 7) {
        SCOPED_TRACE("7");
        ASSERT_EQ(8, nghs.size());
        test_ngh(find_neighbor(0, nghs), 0, 0, -1);
        test_ngh(find_neighbor(1, nghs), 0, 1, -1);
        test_ngh(find_neighbor(2, nghs), 0, 2, -1);
        test_ngh(find_neighbor(3, nghs), 1, 0, -1);
        test_ngh(find_neighbor(4, nghs), 1, 1, -1);
        test_ngh(find_neighbor(5, nghs), 1, 2, -1);
        test_ngh(find_neighbor(6, nghs), 2, 0, -1);
        test_ngh(find_neighbor(8, nghs), 2, 2, -1);
    } else if (rank == 8) {
        SCOPED_TRACE("8");
        ASSERT_EQ(8, nghs.size());
        test_ngh(find_neighbor(0, nghs), 0, 0, -1);
        test_ngh(find_neighbor(1, nghs), 0, 1, -1);
        test_ngh(find_neighbor(2, nghs), 0, 2, -1);
        test_ngh(find_neighbor(3, nghs), 1, 0, -1);
        test_ngh(find_neighbor(4, nghs), 1, 1, -1);
        test_ngh(find_neighbor(5, nghs), 1, 2, -1);
        test_ngh(find_neighbor(6, nghs), 2, 0, -1);
        test_ngh(find_neighbor(7, nghs), 2, 1, -1);
    }
}

TEST(CartesianTopology, testSpecifyProcsPerDim) {

    MPI_Comm test_comm;
    create_test_comm(&test_comm);

    if (test_comm != MPI_COMM_NULL) {
        std::vector<int> dims{2, 4};
        BoundingBox<R4Py_DiscretePoint> gb(0, 100, 0, 200);
        CartesianTopology ct(MPI_COMM_WORLD, dims, gb, false);
        BoundingBox<R4Py_DiscretePoint> lb(0, 0, 0, 0);

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // std::vector<int> coords;
        // if (rank == 0) {
        //     for (int i = 0; i < 8; ++i) {
        //         ct.getCoords(i, coords);
        //         std::cout << i << ": " << coords[0] << ", " << coords[1] << std::endl;
        //     }
        // }

        ct.getBounds(rank, gb);
        if (rank == 0) {
            ASSERT_EQ(0, lb.xmin_);
            ASSERT_EQ(50, lb.x_extent_);
            ASSERT_EQ(0, lb.ymin_);
            ASSERT_EQ(50, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 1) {
            ASSERT_EQ(0, lb.xmin_);
            ASSERT_EQ(50, lb.x_extent_);
            ASSERT_EQ(25, lb.ymin_);
            ASSERT_EQ(50, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 2) {
            ASSERT_EQ(0, lb.xmin_);
            ASSERT_EQ(50, lb.x_extent_);
            ASSERT_EQ(50, lb.ymin_);
            ASSERT_EQ(50, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 3) {
            ASSERT_EQ(0, lb.xmin_);
            ASSERT_EQ(25, lb.x_extent_);
            ASSERT_EQ(75, lb.ymin_);
            ASSERT_EQ(50, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 4) {
            ASSERT_EQ(25, lb.xmin_);
            ASSERT_EQ(50, lb.x_extent_);
            ASSERT_EQ(0, lb.ymin_);
            ASSERT_EQ(50, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 5) {
            ASSERT_EQ(25, lb.xmin_);
            ASSERT_EQ(50, lb.x_extent_);
            ASSERT_EQ(25, lb.ymin_);
            ASSERT_EQ(50, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 6) {
            ASSERT_EQ(25, lb.xmin_);
            ASSERT_EQ(25, lb.x_extent_);
            ASSERT_EQ(50, lb.ymin_);
            ASSERT_EQ(25, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 7) {
            ASSERT_EQ(25, lb.xmin_);
            ASSERT_EQ(50, lb.x_extent_);
            ASSERT_EQ(75, lb.ymin_);
            ASSERT_EQ(50, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        }
    }
}

