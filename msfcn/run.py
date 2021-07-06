
print("Enter the paths of the sequential images one by one, the first image refers to the image at (t-2)th second and the third image is the image taken a (t)th second")
x = input("Enter the path for the first image\n")
y = input("Enter the path for the second image\n")
z = input("Enter the path for the third image\n")

path = input("Enter the path of the model")

import torch
import random
import torch.nn as nn
import torchvision.models
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import time
from PIL import Image
from PIL import ImageOps
from IPython.display import display
from IPython.display import clear_output
from matplotlib import cm
import numpy as np
import cv2 as cv
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils import data
import torch.optim as optim

#function for flipping PIL images given a probability
def hflip(img,img1,img2,img3,img4,p=0.5):
        if random.random() < p:
            return img.transpose(Image.FLIP_LEFT_RIGHT),img1.transpose(Image.FLIP_LEFT_RIGHT),img2.transpose(Image.FLIP_LEFT_RIGHT),img3.transpose(Image.FLIP_LEFT_RIGHT),img4.transpose(Image.FLIP_LEFT_RIGHT)
        else:
          return img,img1,img2,img3,img4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

#FUNCTIONS TO MANIPULATE RESNET ENCODINGS TO PUT THEM IN A SINGLE TENSOR FOR FORWARD PASS

#Concatenation all activations of the resnet to a single tensor for forward pass 
def concatenate(a,b,c,d,e):
  a = a.reshape(-1,16,224,224)
  b = b.reshape(-1,16,224,224)
  c = c.reshape(-1,8,224,224)
  d = d.reshape(-1,4,224,224)
  e = e.reshape(-1,2,224,224)
  return torch.cat((a,b,c,d,e),1)

#Function for unwrapping all tensors IN forward pass
def unwrap(cat):
  a = cat[:,0:16,:,:]
  a = a.reshape(-1,16,224,224)
  a = a.reshape(-1,64,112,112)

  b = cat[:,16:32,:,:]
  b = b.reshape(-1,16,224,224)
  b = b.reshape(-1,256,56,56)

  c = cat[:,32:40,:,:]
  c = c.reshape(-1,8,224,224)
  c = c.reshape(-1,512,28,28)

  d = cat[:,40:44,:,:]
  d = d.reshape(-1,4,224,224)
  d = d.reshape(-1,1024,14,14)

  e = cat[:,44:46,:,:]
  e = e.reshape(-1,2,224,224)
  e = e.reshape(-1,2048,7,7)
  return a,b,c,d,e

#Get the activations from the forward pass
def get_activations(cat):
  a = cat[:,0:46,:,:]
  b = cat[:,46:92,:,:]
  c = cat[:,92:138,:,:]
  return a,b,c

def join(a,b,c):
  return torch.cat((a,b,c),1)


class resnet_encoder(nn.Module):
    def __init__(self):
        super(resnet_encoder,self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        child = nn.Sequential(*list(resnet50.children())[:])
        self.Conv1 = child[0]
        self.Batchnorm1 = child[1]
        self.Maxpool1 = child[3]
        self.Sequential1 = child[4]  #conv2
        self.Sequential2 = child[5]  #conv3
        self.Sequential3 = child[6]  #conv4
        self.Sequential4 = child[7]  #conv5Speed Reduce
        #self.AvgPool1 = child[8]  
        #self.Linear1 = child[9]
        
        
    def forward(self,x):
        out = self.Conv1(x)
        out = self.Batchnorm1(out)
        out = F.relu(out)
        out1 = self.Maxpool1(out)
        out2 = self.Sequential1(out1)
        out3 = self.Sequential2(out2)
        out4 = self.Sequential3(out3)
        out5 = self.Sequential4(out4)
       
      
        return concatenate(out.detach(),out2.detach(),out3.detach(),out4.detach(),out5.detach())
      
#resnet encoder
resnet_enc = resnet_encoder().cuda()


class MSFCN_3(nn.Module):
    

    def __init__(self):
        super(MSFCN_3,self).__init__()
        
        
        
        self.compact5 = nn.Conv2d(6144, 2048, 1)
        self.bnc5 = nn.BatchNorm2d(2048)
        self.conv5 = nn.Conv2d(2048, 2048, 3,padding = 1)
        torch.nn.init.xavier_uniform(self.compact5.weight)
        self.bn5 = nn.BatchNorm2d(2048)
        self.compact4 = nn.Conv2d(5120, 1024, 1)
        self.bnc4 = nn.BatchNorm2d(1024)
        self.conv4 = nn.Conv2d(1024,1024, 3 ,padding = 1)
        self.bn4 = nn.BatchNorm2d(1024)
        torch.nn.init.xavier_uniform(self.compact4.weight)
        self.compact3 = nn.Conv2d(2560, 512, 1)
        self.bnc3 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512 , 512, 3 ,padding = 1)
        self.bn3 = nn.BatchNorm2d(512)
        torch.nn.init.xavier_uniform(self.compact3.weight)
        self.compact2 = nn.Conv2d(1280, 256, 1)
        self.bnc2 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256,256, 3 ,padding = 1)
        self.bn2 = nn.BatchNorm2d(256)
        torch.nn.init.xavier_uniform(self.compact2.weight)
        self.compact1 = nn.Conv2d(448, 64, 1)
        self.bnc1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64,64, 3 ,padding = 1)
        self.bn1 = nn.BatchNorm2d(64)
        torch.nn.init.xavier_uniform(self.compact1.weight)
        
        
        
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.convf1 = nn.Conv2d(64 , 16 , 3 , padding = 1)
        self.bnf1 = nn.BatchNorm2d(16)
        torch.nn.init.xavier_uniform(self.convf1.weight)
        self.convf2 = nn.Conv2d(16 , 8 , 1 )
        self.bnf2 = nn.BatchNorm2d(8)
        torch.nn.init.xavier_uniform(self.convf2.weight)
        self.convf3 = nn.Conv2d(8 , 4 , 1 )
        self.convf4 = nn.Conv2d(4 , 2 , 1)
        torch.nn.init.xavier_uniform(self.convf3.weight)
        torch.nn.init.xavier_uniform(self.convf4.weight)
        #self.convf5 = nn.Conv2d(2 , 1 , 1 )
        #torch.nn.init.xavier_uniform(self.convf5.weight)
        #self.softmax = torch.nn.Sigmoid()
        

        
    def forward(self,x):
   
     
    
    
    
        
      
      
      
      
      
      
        t1,t2,t3 = get_activations(x)
        #first input,second input,third input
      
        a1,b1,c1,d1,e1 = unwrap(t1)
        a2,b2,c2,d2,e2 = unwrap(t2)
        a3,b3,c3,d3,e3 = unwrap(t3)
      
      

        
        features5 = torch.cat([e1,e2,e3],dim =1)#7 7 6144
        features5_compacted = F.relu(self.compact5(features5))#1X1 convolution
        features5_compacted = self.bnc5(features5_compacted)
        features5_compacted = F.relu(self.conv5(features5_compacted))
        features5_compacted = self.bn5(features5_compacted)
        up5 = F.relu(self.upsample(features5_compacted))#14 14 2048
        
        
        features4 = torch.cat([d1,d2,d3,up5],dim =1)
        features4_compacted = F.relu(self.compact4(features4))#1X1 convolution
        features4_compacted = self.bnc4(features4_compacted)
        features4_compacted = F.relu(self.conv4(features4_compacted))
        features4_compacted = self.bn4(features4_compacted)
        up4 = F.relu(self.upsample(features4_compacted))# 28 28 1024
        
        features3 = torch.cat([c1,c2,c3,up4],dim =1)
        features3_compacted = F.relu(self.compact3(features3))
        features3_compacted = self.bnc3(features3_compacted)
        features3_compacted = F.relu(self.conv3(features3_compacted))
        features3_compacted = self.bn3(features3_compacted)
        up3 = F.relu(self.upsample(features3_compacted))#56 56 512
        
        features2 = torch.cat([b1,b2,b3,up3],dim =1)
        features2_compacted = F.relu(self.compact2(features2))
        features2_compacted = self.bnc2(features2_compacted)
        features2_compacted = F.relu(self.conv2(features2_compacted))
        features2_compacted = self.bn2(features2_compacted)
        up2 = F.relu(self.upsample(features2_compacted))#112 112 256
        
        features1 = torch.cat([a1,a2,a3,up2],dim =1)
        features1_compacted = F.relu(self.compact1(features1))
        features1_compacted = self.bnc1(features1_compacted)
        features1_compacted = F.relu(self.conv1(features1_compacted))
        features1_compacted = self.bn1(features1_compacted)
        up1 = F.relu(self.upsample(features1_compacted))#224 224 64
        
        f1= F.relu(self.convf1(up1))#224 224 16
        f2= F.relu(self.convf2(f1))
        f3= F.relu(self.convf3(f2))
        f4= F.relu(self.convf4(f3))
        #f5= F.relu(self.convf5(f4))
               
        #predict = self.softmax(f3)

        return f4
    
        

      
model= MSFCN_3()
model.to(device)





transform_train = transforms.Compose([
    
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])

model.load_state_dict(torch.load(path))

im1 = Image.open(x)
im2 = Image.open(y)
im3 = Image.open(z)

im1_t = transform_train(im1).unsqueeze(0).cuda()
im2_t = transform_train(im2).unsqueeze(0).cuda()
im3_t = transform_train(im3).unsqueeze(0).cuda()


tic = time.time()
#resnet encoding
A1 = resnet_enc(im1_t)
A2 = resnet_enc(im2_t)
A3 = resnet_enc(im3_t)

X = join(A1,A2,A3)
y = model(X)

toc = time.time()


print("Time taken for forward pass is:"+str(1000*(toc-tic))+"ms")


y1 = 255*(np.asarray(nn.Softmax(dim=1)(y).cpu().detach())>0.5)

cv.imwrite('output.png',y1[0][1])

print("The output is saved as output.png")