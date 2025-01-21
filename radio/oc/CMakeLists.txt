
add_library(oai_ocdevif MODULE oc_lib.cpp)
target_link_libraries(oai_ocdevif PRIVATE UTIL)
set_target_properties(oai_ocdevif PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_custom_command(TARGET oai_ocdevif POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E create_symlink liboai_ocdevif.so liboai_device.so
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
