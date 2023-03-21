cmake_minimum_required(VERSION 3.5)
project(boost_global_planner)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Boost REQUIRED COMPONENTS graph)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Declare a C++ executable
add_executable(boost_global_planner src/boost_global_planner.cpp)

# Specify include directories
target_include_directories(boost_global_planner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${Boost_INCLUDE_DIRS}
)

# Specify libraries for the executable
ament_target_dependencies(boost_global_planner
  rclcpp
  nav_msgs
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
)


target_link_libraries(boost_global_planner
  ${Boost_LIBRARIES}
)

install(TARGETS boost_global_planner
  DESTINATION lib/${PROJECT_NAME}
)

# Export information to downstream packages
ament_package()
