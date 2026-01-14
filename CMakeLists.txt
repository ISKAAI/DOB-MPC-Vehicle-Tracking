cmake_minimum_required(VERSION 3.10)
project(MPC_Simulation_Project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(Armadillo REQUIRED)
find_package(osqp REQUIRED)

add_executable(mpc_simulation
    main.cpp
    Vehicle.cpp
    PathGenerator.cpp
    NoiseGenerator.cpp
    filter.cpp 
    MpcController.cpp
)

target_include_directories(mpc_simulation
    PRIVATE
    ${ARMADILLO_INCLUDE_DIRS}
    ${OSQP_INCLUDE_DIRS}  
)

target_link_libraries(mpc_simulation
    PRIVATE
    ${ARMADILLO_LIBRARIES}
    osqp::osqp
)

add_custom_target(run
    COMMAND ./mpc_simulation
    DEPENDS mpc_simulation
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)