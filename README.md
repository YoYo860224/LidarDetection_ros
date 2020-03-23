# ARS408_ros

## Dependence
* Ubuntu 18.04
* ROS melodic
* [ouster_example](https://github.com/ouster-lidar/ouster_example)

## Usage
```bash
# bulid
cd <Your ROS Workspace>\src
git clone https://github.com/ouster-lidar/ouster_example.git
git clone https://github.com/YoYo860224/LidarDetection_ros.git
cd ..
rosdep install -i --from-paths src --os=ubuntu:bionic
catkin_make

# run
roslaunch lidar_detection_ros os1.launch replay:=true
```