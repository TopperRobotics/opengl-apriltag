cmake_minimum_required(VERSION 3.16)
project(apriltag_opengl LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenGL REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(PkgConfig REQUIRED QUIET)
find_package(OpenCV REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Embedded SQLite
add_library(sqlite3 STATIC external/sqlite/sqlite3.c)
target_include_directories(sqlite3 PUBLIC external/sqlite)

# libapriltags source files
set(APRILTAG_SRCS
    external/libapriltags/src/Edge.cc
    external/libapriltags/src/FloatImage.cc
    external/libapriltags/src/GLine2D.cc
    external/libapriltags/src/GLineSegment2D.cc
    external/libapriltags/src/Gaussian.cc
    external/libapriltags/src/GrayModel.cc
    external/libapriltags/src/Homography33.cc
    external/libapriltags/src/MathUtil.cc
    external/libapriltags/src/Quad.cc
    external/libapriltags/src/Segment.cc
    external/libapriltags/src/TagDetection.cc
    external/libapriltags/src/TagDetector.cc
    external/libapriltags/src/TagFamily.cc
    external/libapriltags/src/UnionFindSimple.cc
)

add_library(apriltags_lib STATIC ${APRILTAG_SRCS})
target_include_directories(apriltags_lib PUBLIC
    external/libapriltags/include
    ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(apriltags_lib PUBLIC ${OpenCV_LIBS})

set(MAIN_SRC
    src/main.cpp
    src/GpuDetector.cpp
    src/CpuPostProcessor.cpp
    src/ConfigManager.cpp
    src/HttpServer.cpp
    src/DetectionResult.cpp
)

add_executable(apriltag ${MAIN_SRC})

target_include_directories(apriltag PRIVATE
    include
    external/libapriltags/include
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    external/sqlite
    external/include
)

target_link_libraries(apriltag PRIVATE
    apriltags_lib
    OpenGL::GL
    GLEW::GLEW
    glfw
    sqlite3
    pthread
    ${OpenCV_LIBS}
)

target_compile_options(apriltag PRIVATE -Wall -Wextra -O2)

install(TARGETS apriltag DESTINATION bin)
