# LidarDetection_ros

## Dependence
* Ubuntu 18.04
* ROS melodic
* ros-Velodyne
* [ouster_example](https://github.com/ouster-lidar/ouster_example)
* [YoPCNet](https://github.com/YoYo860224/My3DWork)

## Usage
``` bash
# bulid
cd <Where you want to save your model>
git clone https://github.com/YoYo860224/My3DWork.git

cd <Your ROS Workspace>\src
git clone https://github.com/ouster-lidar/ouster_example.git
git clone https://github.com/YoYo860224/LidarDetection_ros.git
cd ..
rosdep install -i --from-paths src --os=ubuntu:bionic
catkin_make

# run
roslaunch lidar_detection_ros os1.launch replay:=true
rosbag play -l <os1 Bag>

roslaunch lidar_detection_ros velo.launch
rosbag play -l <velo Bag>
```
