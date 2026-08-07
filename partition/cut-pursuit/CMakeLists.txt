cmake_minimum_required(VERSION 2.8)
project(cut_pursuit)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -O3")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fopenmp")

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS python)
include_directories(${Boost_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SOURCE_FILES "src/*.cpp")
add_library(cp SHARED ${SOURCE_FILES} src/libcp_wrapper.cpp)

target_link_libraries(cp ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
