cmake_minimum_required(VERSION 3.12)

project(homework13)

set(CMAKE_CXX_STANDARD 17)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(Boost_NO_WARN_NEW_VERSIONS 1)
set(Boost_USE_STATIC_LIBS ON)

find_package(Boost REQUIRED COMPONENTS program_options)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})

	add_executable(homework13
        src/main.cpp
        src/AccuracyFinder.cpp
        src/classifier.cpp
        src/helpers.cpp
        src/mlp_classifier.cpp
    )

    target_link_libraries(homework13 ${Boost_LIBRARIES})
endif()

target_include_directories(homework13
    PRIVATE "${CMAKE_BINARY_DIR}"
)

install(TARGETS homework13 RUNTIME DESTINATION bin)
set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_FILE_NAME "homework13")
set(CPACK_PACKAGE_CONTACT art.shestakov@icloud.com)
include(CPack)
