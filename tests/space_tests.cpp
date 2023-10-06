#include "gtest/gtest.h"
#include "mpi.h"
#include "Python.h"

#include "distributed_space.h"
#include "SpatialTree.h"


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
        MPI_Comm cart_comm;
        BoundingBox gb(0, 100, 0, 200);
        CartesianTopology ct(test_comm, &cart_comm, 2, gb, true);
        BoundingBox lb(0, 0, 0, 0);

        int rank = ct.getRank();

        ct.getBounds(lb);
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
        MPI_Comm cart_comm;
        BoundingBox gb(0, 107, 0, 201);
        CartesianTopology ct(test_comm, &cart_comm, 2, gb, true);
        BoundingBox lb(0, 0, 0, 0);

        int rank = ct.getRank();
        ct.getBounds(lb);
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
    return {-1, -1, -1, -1, nullptr, {0, 0, 0, 0}};
}

void test_ngh(const CTNeighbor& n, int x, int y, int z) {
    ASSERT_TRUE(n.rank != -1);
    ASSERT_EQ(x, n.cart_coord_x);
    ASSERT_EQ(y, n.cart_coord_y);
    ASSERT_EQ(z, n.cart_coord_z);
}

TEST(CartesianTopology, testGetNeighbors) {
    BoundingBox gb(0, 100, 0, 200);
    MPI_Comm cart_comm;
    CartesianTopology ct(MPI_COMM_WORLD, &cart_comm, 2, gb, false);

    std::vector<CTNeighbor> nghs;
    int rank = ct.getRank();
    ct.getNeighbors(nghs);

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
    BoundingBox gb(0, 100, 0, 200);
    MPI_Comm cart_comm;
    CartesianTopology ct(MPI_COMM_WORLD, &cart_comm, 2, gb, true);

    std::vector<CTNeighbor> nghs;
    int rank = ct.getRank();
    ct.getNeighbors(nghs);

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
        BoundingBox gb(0, 100, 0, 200);
        MPI_Comm cart_comm;
        CartesianTopology ct(test_comm, &cart_comm, dims, gb, false);
        BoundingBox lb(0, 0, 0, 0);

        int rank = ct.getRank();
        ct.getBounds(lb);
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
            ASSERT_EQ(50, lb.ymin_);
            ASSERT_EQ(50, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 2) {
            ASSERT_EQ(0, lb.xmin_);
            ASSERT_EQ(50, lb.x_extent_);
            ASSERT_EQ(100, lb.ymin_);
            ASSERT_EQ(50, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 3) {
            ASSERT_EQ(0, lb.xmin_);
            ASSERT_EQ(50, lb.x_extent_);
            ASSERT_EQ(150, lb.ymin_);
            ASSERT_EQ(50, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 4) {
            ASSERT_EQ(50, lb.xmin_);
            ASSERT_EQ(50, lb.x_extent_);
            ASSERT_EQ(0, lb.ymin_);
            ASSERT_EQ(50, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 5) {
            ASSERT_EQ(50, lb.xmin_);
            ASSERT_EQ(50, lb.x_extent_);
            ASSERT_EQ(50, lb.ymin_);
            ASSERT_EQ(50, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 6) {
            ASSERT_EQ(50, lb.xmin_);
            ASSERT_EQ(50, lb.x_extent_);
            ASSERT_EQ(100, lb.ymin_);
            ASSERT_EQ(50, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        } else if (rank == 7) {
            ASSERT_EQ(50, lb.xmin_);
            ASSERT_EQ(50, lb.x_extent_);
            ASSERT_EQ(150, lb.ymin_);
            ASSERT_EQ(50, lb.y_extent_);
            ASSERT_EQ(0, lb.zmin_);
            ASSERT_EQ(0, lb.z_extent_);
        }
    }
}

class SpatialTreeTests : public testing::Test {

protected:

static PyObject* space;
static PyObject* core;
static PyObject* cpt_class;
static PyObject* agent_class;

static void SetUpTestSuite() {
    Py_Initialize();
    space = PyImport_ImportModule("repast4py.space");
    if (space == nullptr) {
        FAIL();
    }

    cpt_class = PyObject_GetAttrString(space, "ContinuousPoint");
}

static void TearDownTestSuite() {
    Py_XDECREF(space);
    Py_XDECREF(cpt_class);
    Py_Finalize();
}

static R4Py_ContinuousPoint* create_pt(float x, float y, int id) {
    PyObject* arg_list = Py_BuildValue("dd", x, y);
    R4Py_ContinuousPoint* pt = (R4Py_ContinuousPoint*)PyObject_CallObject(cpt_class, arg_list);
    Py_DECREF(arg_list);

    return pt;
}

};

PyObject* SpatialTreeTests::space = nullptr;
PyObject* SpatialTreeTests::cpt_class = nullptr;

TEST_F(SpatialTreeTests, testSOI) {
    

    SOItems<R4Py_ContinuousPoint> soi;
    ASSERT_EQ(0, soi.size());

    auto s1 = create_pt(1, 1, 0);
    soi.add(s1);
    ASSERT_EQ(1, soi.size());

    auto s2 = create_pt(1, 10, 1);
    soi.add(s2);
    ASSERT_EQ(2, soi.size());

    ASSERT_TRUE(soi.remove(s1));
    ASSERT_EQ(1, soi.size());

    ASSERT_TRUE(soi.remove(s2));
    ASSERT_EQ(0, soi.size());

    ASSERT_FALSE(soi.remove(s2));

    soi.add(s1);
    soi.add(s2);
    ASSERT_EQ(2, soi.size());
    soi.clear();
    ASSERT_EQ(0, soi.size());
}


TEST_F(SpatialTreeTests, testMOI) {

    MOItems<R4Py_ContinuousPoint> moi;
    ASSERT_EQ(0, moi.size());

    auto s1 = create_pt(1, 1, 0);
    moi.add(s1);
    ASSERT_EQ(1, moi.size());

    auto s2 = create_pt(1, 10, 1);
    moi.add(s2);
    ASSERT_EQ(2, moi.size());

    auto s3 = create_pt(1, 10, 2);
    moi.add(s3);
    // still 2 -- we are counting the points
    ASSERT_EQ(2, moi.size());
    
    ASSERT_TRUE(moi.remove(s2));
    ASSERT_EQ(2, moi.size());

    ASSERT_TRUE(moi.remove(s3));
    ASSERT_EQ(1, moi.size());

    ASSERT_FALSE(moi.remove(s3));
    ASSERT_EQ(1, moi.size());

    moi.add(s2);
    ASSERT_EQ(2, moi.size());

    moi.clear();
    ASSERT_EQ(0, moi.size());
}

