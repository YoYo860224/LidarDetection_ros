cmake_minimum_required(VERSION 2.8.3)
project(lidar_detection_msg)

find_package(catkin REQUIRED
  std_msgs
  sensor_msgs
  message_generation
  jsk_recognition_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Clusters.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  jsk_recognition_msgs
)

catkin_package(
  CATKIN_DEPENDS std_msgs sensor_msgs jsk_recognition_msgs
)
