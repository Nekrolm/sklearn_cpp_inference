cmake_minimum_required(VERSION 3.4)

project(sklearn_inference_lib)

file(GLOB LIB_SRC
    "src/classifier/*.cpp"
    "src/classifier/ensemble/*.cpp"
    "src/classifier/nn/*.cpp"
    "src/transformer/*.cpp"
    "src/transformer/nn/*.cpp"
)

include_directories(include)
include_directories(src)

add_library(sklearn SHARED ${LIB_SRC})
add_compile_definitions(SKLEARN_EXPORTS)

set_target_properties(sklearn PROPERTIES
            CXX_STANDARD 17
)

set(Boost_USE_STATIC_LIBS OFF)

find_package(Boost COMPONENTS filesystem system)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(sklearn ${Boost_LIBRARIES})
else()
    message("Can't find boost")
    return()
endif()
