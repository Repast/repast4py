src = geometry.cpp SpatialTree.cpp distributed_space.cpp
test_src = space_tests.cpp test_main.cpp

CXX = mpicxx
CXXLD = mpicxx

INCLUDES = -I/home/nick/anaconda3/lib/python3.8/site-packages/numpy/core/include
INCLUDES += -I/home/nick/anaconda3/include/python3.8

GTEST_HOME = $(HOME)/sfw/googletest-1.10.0
GTEST_LIB = $(GTEST_HOME)/lib/libgtest.a
TEST_INCLUDES = -I $(GTEST_HOME)/include -I../src/repast4py

SRC_DIR=../src
BUILD_DIR = ./build


# objects used by both executable and tests
OBJECTS :=
OBJECTS += $(subst .cpp,.o, $(addprefix $(BUILD_DIR)/, $(src)))

TEST_OBJECTS := $(OBJECTS)
TEST_OBJECTS += $(subst .cpp,.o, $(addprefix $(BUILD_DIR)/, $(test_src)))

VPATH = ../src/repast4py ../tests

CXX_RELEASE_FLAGS = -Wall -O2 -g0 -std=c++11 -MMD -MP
CXX_DEBUG_FLAGS = -Wall -O0 -g3 -std=c++11 -MMD -MP
CXX_FLAGS = $(CXX_DEBUG_FLAGS)

TEST_NAME = unit_tests

-include $(TEST_OBJECTS:.o=.d)

.PHONY: all

all : tests

tests : $(TEST_DEPS) $(TEST_OBJECTS)
	$(CXXLD) $(filter-out %.d, $^) -L/home/nick/anaconda3/lib -lpython3.8 -o $(TEST_NAME) -pthread $(GTEST_LIB)


$(BUILD_DIR)/%.o : %.cpp
	@-mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $(TEST_INCLUDES) -c $< -o $@
	
clean:
	rm -fv $(NAME) $(TEST_NAME)
	rm -rf $(BUILD_DIR)/*

