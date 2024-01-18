cmake_minimum_required(VERSION 3.0.0)
project(mpc_motion_planner_examples VERSION 0.1.0)

set(CMAKE_BUILD_TYPE Release)

add_subdirectory(submodules/pybind11)
add_subdirectory(mpc_solver)
add_subdirectory(submodules/ruckig)

# Using motionPlannerLib for linking with executables
# add_executable(mpc_benchmark examples/benchmark.cpp)
# target_link_libraries(mpc_benchmark PUBLIC motionPlannerLib)

# add_executable(offline_trajectory examples/offline_trajectory.cpp)
#target_link_libraries(offline_trajectory PUBLIC motionPlannerLib)

