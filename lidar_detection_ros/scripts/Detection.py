#!/usr/bin/env python

import rospy
import rospkg
from std_msgs.msg import String
from sensor_msgs.point_cloud2 import PointCloud2
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray 

from lidar_detection_msg.msg import Clusters

import pypcd.numpy_pc2
import pypcd

import os
import sys
import time
import math
import natsort
import numpy as np

import cv2
import torch

from NetUtil import GetTorchInputForPointNet

sys.path.append("/home/yoyo/文件/My3DWork/YoPCNet/")
from pointnetX.model import PointNetCls, feature_transform_regularizer
from Util.Pcd2Img import ImgFlow
from Util.ArtFeature import f15_Haar


np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)

objLimit = 20
nPoints = 300

# rospkg = rospkg.RosPack()
# pth = os.path.join(rospkg.get_path('lidar_detection_ros'), "scripts")
# os.chdir(pth)

# Load Model
print("- Loading Model...")
classifier = PointNetCls(mtype="PI", k=4, feature_transform=True)
classifier.load_state_dict(torch.load("/home/yoyo/文件/My3DWork/YoPCNet/pth/v32/CE.PI.wF_v32_model_0025_0.880.pth"))
classifier.cuda()
classifier = classifier.eval()

# Make ros publisher
global pub_bb, pub_pc


def callback(data: Clusters):
    global pub_bb, pub_pc

    # Remove oudate data
    nowT = rospy.Time().now()
    dataT = data.bboxArray.header.stamp
    if  nowT.to_time() - dataT.to_time() > 0.3:
        return

    # Make input
    cluPoints = np.random.random((0, nPoints, 4))
    imgs = np.random.random((0, 224, 224, 3))
    artfs = np.random.random((0, 256))

    numOfObjs = 0
    for cluMsg in data.pointcloudArray:
        # Point Cloud
        pcdata = pypcd.PointCloud.from_msg(cluMsg)
        pc = np.asarray(pcdata.pc_data[['x', 'y', 'z', 'intensity']].tolist(), dtype=float)
        choice = np.random.choice(len(pc), nPoints, replace=True)
        point_set = pc[choice, :]
        point_set = np.expand_dims(point_set, axis=0)
        cluPoints = np.concatenate([cluPoints, point_set])
        # Image
        img = ImgFlow(pc)
        imgs = np.concatenate([imgs, [img]])
        # Arti feature
        # artf = f15_Haar(img)
        # artfs = np.concatenate([artfs, [artf]])

        numOfObjs += 1
        if numOfObjs > objLimit:
            break

    bboxArray = data.bboxArray
    bboxArray.boxes = bboxArray.boxes[0:numOfObjs]

    if (numOfObjs > 0):
        # Get cuda input
        cluPs_cuda = GetTorchInputForPointNet(cluPoints)
        cluPs_cuda = cluPs_cuda.cuda()
        imgs_cuda = torch.from_numpy(imgs).float().cuda()
        # artfs_cuda = torch.from_numpy(artfs).float().cuda()

        # Predict
        pred, _, _ = classifier(cluPs_cuda, imgs_cuda, 0)
        pred_Res = pred.data.max(1)[1].cpu().numpy()
        print(pred_Res)

        for idx, box in enumerate(bboxArray.boxes):
            # '*11' for choice good color
            box.label = pred_Res[idx]*11

        # 0 is Other object
        trueID_pos = np.where(pred_Res!=0)[0].tolist()

        # Only publish true bbox
        bboxArray.boxes = [bboxArray.boxes[i] for i in trueID_pos]
        bboxArray.header.stamp = rospy.Time().now()
        pub_bb.publish(bboxArray)

    ppc = data.procPointCloud
    tSpent = rospy.Time().now().to_time() - ppc.header.stamp.to_time()
    ppc.header.stamp = rospy.Time().now()
    pub_pc.publish(data.procPointCloud)

    # t2 = time.time()
    # print("N of clus:  ", cluPoints.shape[0])
    print("Time spent: ", tSpent)
    print("FPS:        ", 1.0 / tSpent)
    # print("PC Process: ", ts - (t2 - t0))
    # print("PC to IMG:  ", t1-t0)
    # print("Model Dect: ", t2-t1)
    # print("=======================================")

def Detection():
    global pub_bb, pub_pc
    rospy.init_node('Detection', anonymous=False)
    pub_bb = rospy.Publisher('/DetectBBox', BoundingBoxArray, queue_size=1)
    pub_pc = rospy.Publisher('/DetectProcPC', PointCloud2, queue_size=1)
    sub = rospy.Subscriber("/clusters", Clusters, callback, queue_size=1)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == '__main__':
    try:
        Detection()
    except rospy.ROSInterruptException:
        pass
