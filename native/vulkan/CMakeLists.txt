cmake_minimum_required(VERSION 3.23)
project(umzi_vulkan LANGUAGES CXX)

find_package(Vulkan 1.3.268 REQUIRED COMPONENTS glslc)
find_program(SPIRV_VAL spirv-val HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin" REQUIRED)
if(NOT WIN32)
    find_package(OpenSSL REQUIRED COMPONENTS Crypto)
endif()

set(UMZI_VULKAN_SHADER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common/shaders")
function(umzi_vulkan_shader name source_name)
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${source_name}.comp")
    set(spv "${CMAKE_CURRENT_BINARY_DIR}/${name}.spv")
    set(header "${CMAKE_CURRENT_BINARY_DIR}/${name}_spv.h")
    set(depfile "${CMAKE_CURRENT_BINARY_DIR}/${name}.d")
    set(flags --target-env=vulkan1.2 -O -I "${UMZI_VULKAN_SHADER_INCLUDE_DIR}" ${ARGN})
    add_custom_command(OUTPUT "${header}"
        BYPRODUCTS "${spv}"
        COMMAND Vulkan::glslc ${flags} -MD -MF "${depfile}" -MT "${header}" "${source}" -o "${spv}"
        COMMAND "${SPIRV_VAL}" --target-env vulkan1.2 "${spv}"
        COMMAND Vulkan::glslc ${flags} -mfmt=num "${source}" -o "${header}"
        DEPENDS "${source}"
        DEPFILE "${depfile}"
        VERBATIM)
    set(shader_headers ${shader_headers} "${header}" PARENT_SCOPE)
endfunction()

add_library(umzi_native_vulkan SHARED
    common/compute.cpp
    common/context.cpp
    runtime/api.cpp
    runtime/registry.cpp
    ../runtime/model_package.cpp)
target_compile_features(umzi_native_vulkan PRIVATE cxx_std_20)
target_compile_definitions(umzi_native_vulkan PRIVATE UMZI_BUILD)
target_link_libraries(umzi_native_vulkan PRIVATE Vulkan::Vulkan)
if(WIN32)
    target_link_libraries(umzi_native_vulkan PRIVATE bcrypt)
else()
    target_link_libraries(umzi_native_vulkan PRIVATE OpenSSL::Crypto)
endif()
set_target_properties(umzi_native_vulkan PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)

add_subdirectory(kernels/gaterv3)
