cmake_minimum_required(VERSION 3.23)
project(umzi_vulkan LANGUAGES CXX)

find_package(Vulkan 1.3.268 REQUIRED COMPONENTS glslc)
find_program(SPIRV_VAL spirv-val HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin" REQUIRED)
if(NOT WIN32)
    find_package(OpenSSL REQUIRED COMPONENTS Crypto)
endif()

set(shader_headers)
function(add_shader name source_name)
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${source_name}.comp")
    set(spv "${CMAKE_CURRENT_BINARY_DIR}/${name}.spv")
    set(header "${CMAKE_CURRENT_BINARY_DIR}/${name}_spv.h")
    add_custom_command(OUTPUT "${header}"
        COMMAND Vulkan::glslc --target-env=vulkan1.2 -O ${ARGN} "${source}" -o "${spv}"
        COMMAND "${SPIRV_VAL}" --target-env vulkan1.2 "${spv}"
        COMMAND Vulkan::glslc --target-env=vulkan1.2 -O ${ARGN} -mfmt=num "${source}" -o "${header}"
        DEPENDS "${source}" shaders/schedule.glsl
        VERBATIM)
    set(shader_headers ${shader_headers} "${header}" PARENT_SCOPE)
endfunction()

foreach(shader conv element reduce attention conv_fast mean_fast norm_fast gates)
    add_shader(${shader} ${shader})
endforeach()
foreach(shader element reduce attention conv_fast mean_fast norm_fast gates)
    add_shader(${shader}_half ${shader} -DHALF_STORAGE)
endforeach()
add_shader(conv_native_half conv_fast -DHALF_STORAGE -DHALF_MATH)

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
