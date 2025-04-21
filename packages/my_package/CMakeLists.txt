cmake_minimum_required(VERSION 2.8.3)
project(my_package)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs  # include any message packages you use
)

catkin_package()

catkin_install_python(PROGRAMS
  src/motion.py
  src/motion_bot.py
  src/camera_reader_node.py
  src/shutdown.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

