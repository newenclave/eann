cmake_minimum_required( VERSION 2.8 )

set( PROJECT_NAME eann_test )

project( ${PROJECT_NAME} )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED 11)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/eann/include )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/catch )

execute_process( COMMAND "git" "submodule" "init"
                 WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

execute_process( COMMAND "git" "submodule" "update" "catch"
                 WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

list( APPEND src ./eann/include/eann/
                 ./eann/include/eann/traits/
                 ./tests
                 )

foreach( src_dir ${src} )
    aux_source_directory(${src_dir} lib_src)
    file(GLOB headers ${src_dir}/*.h)
    list(APPEND lib_src ${headers})
endforeach( )

add_executable( ${PROJECT_NAME} ${lib_src} )
