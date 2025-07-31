cmake_minimum_required(VERSION 3.10)
project(Multhreading_camera)

# Ative C++11 ou superior
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Busca o OpenCV
find_package(OpenCV REQUIRED)

set(SOURCES
    main.cpp)

add_executable(camera_app ${SOURCES})

# Inclui o diretório local e o do OpenCV
target_include_directories(camera_app PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Linka o OpenCV
target_link_libraries(camera_app PRIVATE ${OpenCV_LIBS})
