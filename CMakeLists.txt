cmake_minimum_required(VERSION 3.10)

project(mujoco_camera)

# Compiler settings
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# finding modules
find_package(OpenCV REQUIRED) 
find_package(PCL REQUIRED)

# Specify directories
set(MUJOCO_INCLUDE_DIR ~/.mujoco/mujoco210/include)
set(MUJOCO_LINK_DIR ~/.mujoco/mujoco210/bin)
set(MUJOCO_LIBS libmujoco210.so
                libglfw.so.3
                libGL.so
                libglew.so
                m
                dl)

include_directories(${MUJOCO_INCLUDE_DIR}
                    include
                    ${OpenCV_INCLUDE_DIRS}
                    ${PCL_INCLUDE_DIRS})

                    
link_directories(${MUJOCO_LINK_DIR}
                ${PCL_LIBRARY_DIRS})



add_executable(camera_test src/camera.cpp)                

target_link_libraries(camera_test 
            ${MUJOCO_LIBS}
            ${OpenCV_LIBS}
            ${PCL_LIBRARIES})     

add_executable(multi_win_test src/two_window.cpp
                                src/mujoco_rgbd_camera.cpp)     

target_link_libraries(multi_win_test
            ${MUJOCO_LIBS}
            ${OpenCV_LIBS}
            ${PCL_LIBRARIES})              