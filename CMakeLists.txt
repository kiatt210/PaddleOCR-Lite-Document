cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(ocr-server)

macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD 17)
  endif ()
endmacro(use_cxx11)

find_package(OpenCV REQUIRED)
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/proto/ocr.proto)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})

set(GRPC_SRCS ${CMAKE_CURRENT_BINARY_DIR}/ocr.grpc.pb.cc)
set(GRPC_HDRS ${CMAKE_CURRENT_BINARY_DIR}/ocr.grpc.pb.h)

set(SOURCES src/clipper.cpp src/cls_process.cc src/crnn_process.cc src/db_post_process.cc src/main.cpp src/ocr.cpp)

set(EXTRA_LIBS ${OpenCV_LIBS} /usr/local/lib/ncnn/libncnn.a)

set(EXECUTABLE_OUTPUT_PATH "../")

include_directories(include ${OpenCV_INCLUDE_DIRS} /usr/local/include/ncnn)

add_custom_command(
    OUTPUT ${GRPC_SRCS} ${GRPC_HDRS}
    COMMAND protobuf::protoc
    ARGS
        --grpc_out=${CMAKE_CURRENT_BINARY_DIR}
        --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
        -I ${CMAKE_CURRENT_SOURCE_DIR}/proto
        ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
)


add_link_options(-fopenmp -s -ldl -pthread -lgomp -rdynamic)

add_executable(ocr-server
    ${SOURCES}
    src/main.cpp
    src/ocr_service.cpp
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

target_include_directories(ocr-server
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ocr-server
    PRIVATE
        gRPC::grpc++
        protobuf::libprotobuf
        ${EXTRA_LIBS}
)
