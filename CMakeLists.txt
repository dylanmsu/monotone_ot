cmake_minimum_required(VERSION 3.16)

project(monotone_ot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(USE_CIMAGE true)
if(USE_CIMAGE)
    add_definitions(-D_cimg_)
    add_definitions(-Dcimg_display=0)

    find_package(PNG)
    if(PNG_FOUND)
        add_definitions(-Dcimg_use_png)
        set(ALLLIBS ${ALLLIBS} PNG::PNG)
    endif()

    find_package(JPEG)
    if(JPEG_FOUND)
        add_definitions(-Dcimg_use_jpeg)
        set(ALLLIBS ${ALLLIBS} JPEG::JPEG)
    endif()

   if(WIN32)
      add_definitions(-Dcimg_OS=2)
   else()
      add_definitions(-Dcimg_OS=1)
   endif()
endif()

include_directories(src)
include_directories(extern)
include_directories(extern/eigen/)
include_directories(extern/cppad/include)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_subdirectory(extern/cppad)

set(PROJECT_SOURCES
    main.cpp
    src/bvh.cpp
    src/poisson_solver.cpp
    src/mesh.cpp
    src/normal_integration/normal_integration.cpp
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${ALLLIBS} ${CERES_LIBRARIES})