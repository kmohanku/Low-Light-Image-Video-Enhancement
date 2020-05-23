cmake_minimum_required(VERSION 3.16)
project(Video-Brightning VERSION 1.0.0)

SET("OpenCV_DIR" "/usr/local/include/opencv4")
find_package(OpenCV REQUIRED)
find_package(TensorflowCC REQUIRED)
find_library(PROTOBUF_LIBRARY libprotobuf.so.17)

MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

include_directories(include)
set(HEADER_FILES include/enhancementModel.h include/audioHandler.h)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../bin")
add_executable(video-enhancement src/main.cc src/enhancementModel.cc src/audioHandler.cc ${HEADER_FILES})
target_link_libraries(video-enhancement ${OpenCV_LIBS} TensorflowCC::TensorflowCC ${PROTOBUF_LIBRARY})