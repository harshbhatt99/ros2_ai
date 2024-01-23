cmake_minimum_required(VERSION 3.8)
project(ros2_autoencoder)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# (HB) Start ---------------------------------------------------------

# (HB) Find additional packages
# find_package(ament_cmake_python REQUIRED)
# find_package(rclpy REQUIRED)
# find_package(std_msgs REQUIRED)

# (HB) Add extra executables

#ament_python_install_package(${PROJECT_NAME})

#add_executable(autoencoder_node scripts/autoencoder_node.py)
#ament_target_dependencies(autoencoder_node rclpy std_msgs)

# install(DIRECTORY scripts
#  DESTINATION share/${PROJECT_NAME})
  
# install(PROGRAMS scripts/autoencoder_node.py
#  DESTINATION lib/${PROJECT_NAME}
#)
  
#install(TARGETS
#  autoencoder_node
#  DESTINATION lib/${PROJECT_NAME}
#)

# (HB) End -----------------------------------------------------------

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
