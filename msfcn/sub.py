#! /usr/bin/python


model_path = "/home/stormbreaker/catkin_ws/src/dl_ros/src/classifier_kitti_50_epochs_50-50_kitti_split.pt"

import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import torch
import PIL
from PIL import Image as PILImage

import torch.nn as nn
import torchvision.models
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import time
from PIL import ImageOps
from matplotlib import cm
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils import data
import torch.optim as optim

import helpers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

resnet_enc = helpers.resnet_encoder().cuda()

model= helpers.MSFCN_3()
model.to(device)
model.load_state_dict(torch.load(model_path))
ms = "model loaded"
print(ms)

transform_train = transforms.Compose([
    
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])

counter = 0 
A1 = None
A2 = None
# Instantiate CvBridge
bridge = CvBridge()


# def call(data):
#     try:
#         self.image_pub.publish(bridge.cv2_to_imgmsg(data, "bgr8"))
#     except CvBridgeError as e:
#         print(e)

image_pub=rospy.Publisher("image_topic_2",Image)

def image_callback(msg):
    global counter
    global A1
    global A2

    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        counter += 1 
        k = 10 #step
        if counter%k == 1:
            cv2.imwrite('camera_image.jpeg', cv2_img)
            im = PILImage.open("camera_image.jpeg")
            #cv2_img = cv2_img[0:720,:]


            x3 = transform_train(im).unsqueeze(0).cuda()
            tic = time.time()
            A3 = resnet_enc(x3)
            if counter == 1:
                A1 = A3
            elif counter == 1+k:
                A2 = A3
            else:
                
                X = helpers.join(A1,A2,A3)
                y = model(X)
                toc = time.time()
                y1 = 255*(np.asarray(nn.Softmax(dim=1)(y).cpu().detach())>0.5)
                print(cv2_img.shape)

                print(str(1000*(toc-tic))+" ms")
                cv2_img = cv2.resize(cv2_img,(224,224))
                for i in range(224):
                    for j in range(224):
                        if(y1[0][1][i][j]==255):
                            cv2_img[i][j][1] = 255
                image_pub.publish(bridge.cv2_to_imgmsg(cv2_img, "bgr8"))
                A1 = A2
                A2 = A3

        


def main():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/camera/image_color"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()
     
