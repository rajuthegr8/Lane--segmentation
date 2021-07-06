# Parameters
params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 0}#Keep numworkers 0 as cuda cannot be used with multiprocessing
max_epochs = 100

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Currently using "+str(device))

#function for flipping PIL images given a probability
def hflip(img,img1,img2,img3,img4,p=0.5):
        if random.random() < p:
            return img.transpose(Image.FLIP_LEFT_RIGHT),img1.transpose(Image.FLIP_LEFT_RIGHT),img2.transpose(Image.FLIP_LEFT_RIGHT),img3.transpose(Image.FLIP_LEFT_RIGHT),img4.transpose(Image.FLIP_LEFT_RIGHT)
        else:
          return img,img1,img2,img3,img4
    
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
    

image_path = os.getcwd()+"/data/train/images/"
label_path = os.getcwd()+"/data/train/labels/"

partition = {}

list_train = []
list_test = []


label = {}


for i in sorted(os.listdir(image_path)):
  for j in range(1,10000):
    
    if j == 1:
      image1 = image_path+i+"/"+str(j)+".png"
      continue
    elif j == 2:
      image2 = image_path+i+"/"+str(j)+".png"
      continue
    
      
    if os.path.isfile(image_path+i+"/"+str(j)+".png")==False:
      print((i)+" "+str(j))
      break
    if os.path.isfile(label_path+i+"/"+str(j)+".png")==False:
      print("no label")
      break
    
    image3 = image_path+i+"/"+str(j)+".png"
    
    image_ID = image1+"^"+image2+"^"+image3
    
    label_ID = label_path+i+"/"+str(j)+".png"
    
    
    
    
    
    list_train.append(image_ID)
    
    label[image_ID]= label_ID
    image1 = image2
    image2 = image3

image_path = os.getcwd()+"/data/test/images/"
label_path = os.getcwd()+"/data/test/labels/"
#creating test set
for i in sorted(os.listdir(image_path)):
  for j in range(1,10000):
    
    if j == 1:
      image1 = image_path+i+"/"+str(j)+".png"
      continue
    elif j == 2:
      image2 = image_path+i+"/"+str(j)+".png"
      continue
    
      
    if os.path.isfile(image_path+i+"/"+str(j)+".png")==False:
      print((i)+" "+str(j))
      break
    if os.path.isfile(label_path+i+"/"+str(j)+".png")==False:
      print("no label")
      break
    
    image3 = image_path+i+"/"+str(j)+".png"
    
    image_ID = image1+"^"+image2+"^"+image3
    
    label_ID = label_path+i+"/"+str(j)+".png"
    
    
    
    
    
    list_test.append(image_ID)
    
    label[image_ID]= label_ID
    image1 = image2
    image2 = image3

#will create label given address of image  
#def label(im):
#  return label_path+im.split("/")[-2]+"/"+im.split("/")[-1] 

#lists of all addresses
partition['train'] = list_train
partition['test'] = list_test


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
    
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.1,scale=(0.02, 0.08)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])


from torch.utils import data

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs,labels):
        'Initialization'
        super(Dataset,self).__init__()
        self.list_IDs = list_IDs
        self.labels = labels

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        
        #The three sequential images
        x,y,z = ID.split("^")
        im1 = Image.open(x)
        im2 = Image.open(y)
        im3 = Image.open(z)
        
        #The label
        lb = cv.imread(self.labels[ID],0)
        
        #Converting road pixels to white
        lb1 = 255*(lb==49)
        lb1 = lb1.astype(np.uint8)
        lb = Image.fromarray(lb1,mode='L')
        
        #To create the negative of the image for the second class 
        lb_n = ImageOps.invert(lb)
        
        x,y,z,w,s = hflip(im1,im2,im3,lb,lb_n)#horizontal flipping with a probability of 0.5
        
        x = transform_train(x)
        y = transform_train(y)
        z = transform_train(z)
        
        #resnet encoding
        A1 = resnet_enc(x.unsqueeze(0).cuda())
        A2 = resnet_enc(y.unsqueeze(0).cuda())
        A3 = resnet_enc(z.unsqueeze(0).cuda())
        
        
        
        
        
        #Concatanating for a forward pass
        X = join(A1,A2,A3)
        
        y1 = transforms.ToTensor()(w)
        y2 = transforms.ToTensor()(s)
        
        y = torch.cat((y1,y2),0).long()#creating labels
        
        return X.squeeze(), y1
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=0)


# Generators
training_set = Dataset(partition['train'],label)
training_generator = data.DataLoader(training_set, **params)

test = Dataset(partition['test'],label)
test_generator = data.DataLoader(test, **params)

# Loop over epochs
for epoch in range(max_epochs):
    # Training
    running_loss = 0.0
    i = 0
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        i = i + 1
        
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        
        outputs = model(local_batch)
        lb = local_labels.squeeze().long()
        if len(lb)==2:
         lb = lb.unsqueeze(0)
        
          
        loss = criterion(outputs, lb)
        loss.backward(retain_graph=False)
        running_loss += loss.item()
        optimizer.step()
        print("batch no."+str(i))
      
    model_save_name = "classifier_epoch_"+ str(epoch+1) +".pt"
    path =  os.getcwd()+"/"+model_save_name 
    torch.save(model.state_dict(), path)
        
    print('[%d] loss_training: %.6f' %(epoch + 1, running_loss ))        
    ls = 0.0
    i = 0
    #Validation
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in test_generator:
            i = i + 1
            
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            lb = local_labels.squeeze().long()
            if len(lb)==2:
              lb = lb.unsqueeze(0)
            if(outputs.shape[0]!=lb.shape[0]):
              continue
            loss = criterion(outputs, lb)
            ls = ls + loss.item()
            print("batch no.(test)"+str(i))
        print('[%d] loss_test: %.6f' %(epoch + 1, ls ))