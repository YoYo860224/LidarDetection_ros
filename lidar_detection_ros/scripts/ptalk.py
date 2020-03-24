#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from sensor_msgs.point_cloud2 import PointCloud2

from lidar_detection_msg.msg import clus


import pypcd.numpy_pc2
import pypcd

import os
import math
import natsort
import numpy as np


def talker():
    rospy.init_node('talker', anonymous=True)
    pub = rospy.Publisher('/clus', clus, queue_size=10)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        a = clus()
        a.cs
        pub.publish(a)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
