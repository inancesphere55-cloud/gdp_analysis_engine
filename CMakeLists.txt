cmake_minimum_required(VERSION 3.15)
project(gdp_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

pybind11_add_module(gdp_engine
    src/bindings.cpp
    src/add.cpp
    src/gdp_engine.cpp
    src/backtest.cpp
    src/statistics.cpp
)

target_include_directories(gdp_engine
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gdp_engine
    PRIVATE
        pybind11::headers
)

set_target_properties(gdp_engine
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

install(TARGETS gdp_engine DESTINATION ${Python_SITELIB})
