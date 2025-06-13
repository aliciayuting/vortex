# Build Python UDLs
find_package(Python3 3.10.0 QUIET COMPONENTS Interpreter Development Development.Module Development.Embed NumPy)
# find_package(QUIET COMPONENTS Interpreter Development Development.Module Development.Embed NumPy)


if (Python3_FOUND)

    # Configurations
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

    # main target
    add_library(python_udl_manager SHARED ${CMAKE_CURRENT_SOURCE_DIR}/python/python_udl_manager.cpp)
    target_include_directories(python_udl_manager PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    )

    target_link_libraries(python_udl_manager cascade Python3::Python Python3::NumPy)

endif()