#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from sensor_msgs.point_cloud2 import PointCloud2

from lidar_detection_msg.msg import clus
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray 


import pypcd.numpy_pc2
import pypcd

import os
import math
import natsort
import numpy as np

import torch

from pointNetX.model import PointNetCls, feature_transform_regularizer

nPoints = 300

classifier = PointNetCls(k=2, feature_transform=True)
classifier.load_state_dict(torch.load("./pth/3D.pth"))
classifier.cuda()
classifier = classifier.eval()

BB = BoundingBoxArray
pub = rospy.Publisher('/PeopleBB', BoundingBoxArray, queue_size=10)

def callback(data: clus):
    cluPoints = np.random.random((0, nPoints, 3))
    for cluMsg in data.csx:
        pcdata = pypcd.PointCloud.from_msg(cluMsg)
        # pc = np.asarray(pcdata.pc_data[['x', 'y', 'z', 'intensity']].tolist(), dtype=float)
        pc = np.asarray(pcdata.pc_data[['x', 'y', 'z']].tolist(), dtype=float)
        choice = np.random.choice(len(pc), nPoints, replace=True)
        point_set = pc[choice, :]
        point_set = np.expand_dims(point_set, axis=0)
        cluPoints = np.concatenate([cluPoints, point_set])

    BB = data.bbarr

    print(cluPoints.shape)
    print(len(BB.boxes))

    if (cluPoints.shape[0] > 0):
        cluPs_cuda = GetTorchInputForPointNet(cluPoints)
        cluPs_cuda = cluPs_cuda.cuda()

        pred, _, _ = classifier(cluPs_cuda)
        pred_choice = pred.data.max(1)[1].cpu().numpy()==1
        trueID_pos = np.where(pred_choice==True)[0].tolist()

        BB.boxes = [BB.boxes[i] for i in trueID_pos]
        pub.publish(BB)

def listener():
    rospy.init_node('listener', anonymous=False)
    rospy.Subscriber("/clus", clus, callback, queue_size=1)

    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():

        rate.sleep()

def GetTorchInputForPointNet(point_set):
    '''
    Input point_set with: (Batch, NumOfPoints, 3)
                                            └───＞ xyz
    '''
    # To Center
    point_set = point_set - np.expand_dims(np.mean(point_set, axis = 1), 1)
    # Do Scale
    distAll = np.sum(point_set ** 2, axis = -1)
    distMax = np.max(distAll, axis = -1)
    point_set = point_set / distMax[:, None, None]     
    # Torch Format
    points = torch.tensor(point_set, dtype=torch.float)
    points = points.transpose(2, 1)

    return points   

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
