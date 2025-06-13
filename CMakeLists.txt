cmake_minimum_required(VERSION 3.12.0)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
project(vortex CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(spdlog 1.3.1 REQUIRED)

find_package(derecho CONFIG REQUIRED)
find_package(cascade CONFIG REQUIRED)

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(Boost REQUIRED COMPONENTS system filesystem numpy)
include_directories(${Boost_INCLUDE_DIRS})

include(GNUInstallDirs)
if (NOT DEFINED CMAKE_INSTALL_BINDIR)
    set(CMAKE_INSTALL_BINDIR bin)
endif()

if (NOT DEFINED CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR lib)
endif()

find_package(Faiss QUIET)
if (Faiss_FOUND)
    message(STATUS "Faiss found. Enabling Faiss-related features.")
    set(USE_FAISS ON)
else()
    set(USE_FAISS OFF)
endif()
find_package(CURL QUIET)
if (CURL_FOUND)
    message(STATUS "CURL found. Enabling CURL-related features.")
    set(USE_CURL ON)
else()
    set(USE_CURL OFF)
endif()
find_package(CUDAToolkit QUIET)
if (CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit found. Enabling CUDA-related features.")
    set(USE_CUDA ON)
else()
    set(USE_CUDA OFF)
endif()


add_subdirectory(applications)

add_custom_target(prepare_vortex_config ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/cfg
        ${CMAKE_CURRENT_BINARY_DIR}/cfg
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/cfg/layout.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json.tmp
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/cfg/dfgs.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/dfgs.json.tmp
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/cfg/udl_dlls.cfg.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/udl_dlls.cfg.tmp
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/applications/python
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/applications/python/python_udls
        ${CMAKE_CURRENT_BINARY_DIR}/applications/python/python_udls
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/dfgs.json.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/dfgs.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/dfgs.json.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/dfgs.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/dfgs.json.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/dfgs.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/dfgs.json.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/dfgs.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/dfgs.json.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/dfgs.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/udl_dlls.cfg.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/udl_dlls.cfg
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/udl_dlls.cfg.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/udl_dlls.cfg
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/udl_dlls.cfg.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/udl_dlls.cfg
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/udl_dlls.cfg.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/udl_dlls.cfg
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/udl_dlls.cfg.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/udl_dlls.cfg
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/applications/python/python_udls ${CMAKE_CURRENT_BINARY_DIR}/cfg/python_udls
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/applications/python/python_udls ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/python_udls
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/applications/python/python_udls ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/python_udls
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/applications/python/python_udls ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/python_udls
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/applications/python/python_udls ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/python_udls
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/applications/python/python_udls ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/python_udls
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/clear_log.sh
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/clear_log.sh
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/clear_log.sh
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/clear_log.sh
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/clear_log.sh
    COMMAND chmod 755 ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/cfg/layout.json.tmp
        ${CMAKE_CURRENT_SOURCE_DIR}/cfg/dfgs.json.tmp
        ${CMAKE_CURRENT_SOURCE_DIR}/cfg/udl_dlls.cfg.tmp
        ${CMAKE_CURRENT_SOURCE_DIR}/cfg/clear_log.sh.tmp
    COMMENT "Prepare vortex UDLs and group configuration"
)

# Run this after building applications (e.g., python_udl_manager)
add_dependencies(prepare_vortex_config python_udl_manager)