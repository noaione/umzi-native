cmake_minimum_required(VERSION 3.23)
project(umzi_vulkan LANGUAGES CXX)

find_package(Vulkan 1.3.268 REQUIRED COMPONENTS glslc)
find_program(SPIRV_VAL spirv-val HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin" REQUIRED)
if(NOT WIN32)
    find_package(OpenSSL REQUIRED COMPONENTS Crypto)
endif()

set(shader_headers)
foreach(shader conv element reduce attention)
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}.comp")
    set(spv "${CMAKE_CURRENT_BINARY_DIR}/${shader}.spv")
    set(header "${CMAKE_CURRENT_BINARY_DIR}/${shader}_spv.h")
    add_custom_command(OUTPUT "${header}"
        COMMAND Vulkan::glslc --target-env=vulkan1.2 -O "${source}" -o "${spv}"
        COMMAND "${SPIRV_VAL}" --target-env vulkan1.2 "${spv}"
        COMMAND Vulkan::glslc --target-env=vulkan1.2 -O -mfmt=num "${source}" -o "${header}"
        DEPENDS "${source}" shaders/schedule.glsl
        VERBATIM)
    list(APPEND shader_headers "${header}")
endforeach()

add_library(umzi_native_vulkan SHARED api.cpp session.cpp context.cpp ../runtime/model_package.cpp ${shader_headers})
target_compile_features(umzi_native_vulkan PRIVATE cxx_std_20)
target_compile_definitions(umzi_native_vulkan PRIVATE UMZI_BUILD)
target_include_directories(umzi_native_vulkan PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(umzi_native_vulkan PRIVATE Vulkan::Vulkan)
if(WIN32)
    target_link_libraries(umzi_native_vulkan PRIVATE bcrypt)
else()
    target_link_libraries(umzi_native_vulkan PRIVATE OpenSSL::Crypto)
endif()
set_target_properties(umzi_native_vulkan PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
