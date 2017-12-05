project(FaceAlignment)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
find_package(Caffe REQUIRED)

include_directories(${Caffe_INCLUDE_DIRS})
include(lib3000fps.cmake)

set(SRC src/main.cpp
        src/mtcnn.cpp
        src/prepare.cpp
        src/test.cpp
        src/train.cpp)

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} ${Caffe_LIBRARIES})
target_link_libraries(${PROJECT_NAME} lib3000fps ${OpenCV_LIBS})
