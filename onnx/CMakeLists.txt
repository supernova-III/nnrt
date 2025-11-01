find_package(Protobuf REQUIRED)

add_library(onnx STATIC onnx.proto)

set(ONNX_OUT_PATH ${PROJECT_SOURCE_DIR}/onnx/proto)

protobuf_generate(
  TARGET onnx
  LANGUAGES cpp
  OUT_VAR onnx_SOURCES
  PROTOC_OUT_DIR ${ONNX_OUT_PATH}
)

target_sources(onnx PRIVATE ${onnx_SOURCES})
target_link_libraries(onnx PUBLIC protobuf::libprotobuf)
target_include_directories(onnx PUBLIC ${PROJECT_SOURCE_DIR}/onnx)
