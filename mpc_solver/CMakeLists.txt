cmake_minimum_required(VERSION 3.0.0)
project(mpc_motion_planner VERSION 0.1.0)

set (CMAKE_CXX_STANDARD 11) 
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(pinocchio REQUIRED)
find_package(ruckig REQUIRED)
#find_package(pybind11 REQUIRED)  # Required for creating Python bindings
#find_package(PythonLibs REQUIRED)


add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=1000000) # 1 MB 
include_directories(../submodules/polympc/src ../robot_utils ../submodules/pybind11/include)

# Original library for linking with executables
#add_library(motionPlannerLib SHARED motionPlanner.cpp ../robot_utils/pandaWrapper.cpp)
#add_library(motionPlannerLib SHARED motionPlanner.cpp ../robot_utils/kukaWrapper.cpp)
add_library(motionPlannerLib SHARED motionPlanner.cpp ../robot_utils/armRobotWrapper.cpp)
target_link_libraries(motionPlannerLib Eigen3::Eigen pinocchio::pinocchio ruckig::ruckig ${Python_LIBRARIES})
target_include_directories(motionPlannerLib PUBLIC ${PYTHON_INCLUDE_DIRS} "." "../submodules/polympc/src" "../robot_utils")



# Python module
#pybind11_add_module(motionPlanner MODULE motionPlanner.cpp ../robot_utils/pandaWrapper.cpp)
#pybind11_add_module(motionPlanner MODULE motionPlanner.cpp ../robot_utils/kukaWrapper.cpp)
#pybind11_add_module(motionPlanner MODULE motionPlanner.cpp ../robot_utils/armRobotWrapper.cpp)
#target_link_libraries(motionPlanner PRIVATE Eigen3::Eigen pinocchio::pinocchio ruckig::ruckig)

pybind11_add_module(motion_planning_lib MODULE motionPlanner.cpp ../robot_utils/armRobotWrapper.cpp)
target_link_libraries(motion_planning_lib PRIVATE Eigen3::Eigen pinocchio::pinocchio ruckig::ruckig)

