# Road Segmentation Approaches
All branches are the different models currently being experimented on
## Refine Net

The refineNet folder contains Colab notebooks for training and testing refineNet model as well as pretrained model on CityScapes of DeepLabV3 for estimating Depth Images

Models Train on Kitti
Link of the dataset used : https://drive.google.com/drive/folders/1XpJ9RAR7RI8BcFL41Hd2ybt4bejO5uXa?usp=sharing

Link of the trained model : https://drive.google.com/file/d/1--CYpopkHa53fXxLr8RRJZz2fqhRBzuY/view?usp=sharing

IOU on Kitti test = 87% 

Models Train on CityScapes

Link of dataset used

For Image : https://drive.google.com/drive/folders/1lSc520BfXSMbzRTlyN0PKW1o27Q0AEXT?usp=sharing

For Depth : https://drive.google.com/drive/folders/1UgFGMzavRVL9w_J6HvGMipGX7AYFkHhB?usp=sharing

Link of the trained model

For refineNet : https://drive.google.com/file/d/1-b5h0qoGMsX7ltPShkRFbOR_W0YjwU5K/view?usp=sharing

For DeepLab : https://drive.google.com/file/d/1I4IZ8pRD8eumF_jSqxXy91EO-BNXf_vF/view?usp=sharing


IOU on CityScapes test = 86%

Link of output video : https://drive.google.com/file/d/1MLdJ5j4Q93ASXg-25Fu5K_pV4tUVHm9l/view?usp=sharing

Result on 2.2 : https://drive.google.com/file/d/1MLdJ5j4Q93ASXg-25Fu5K_pV4tUVHm9l/view

### Sample Results
<img src="https://github.com/raanyild/Lane-Segmentation/blob/master/refineNet/examples/outputs255.png" width="800"/>
<img src="https://github.com/raanyild/Lane-Segmentation/blob/master/refineNet/examples/outputs256.png" width="800"/>
<img src="https://github.com/raanyild/Lane-Segmentation/blob/master/refineNet/examples/outputs512.png" width="800"/>

### Results on Roads of KGP
<img src="https://github.com/raanyild/Lane-Segmentation/blob/master/refineNet/sample_result/test1.png" width="400"/> <img src="https://github.com/raanyild/Lane-Segmentation/blob/master/refineNet/sample_result/test2.png" width="400"/>
<img src="https://github.com/raanyild/Lane-Segmentation/blob/master/refineNet/sample_result/test3.png" width="400"/> <img src="https://github.com/raanyild/Lane-Segmentation/blob/master/refineNet/sample_result/test4.png" width="400"/>
<img src="https://github.com/raanyild/Lane-Segmentation/blob/master/refineNet/sample_result/test5.png" width="400"/> <img src="https://github.com/raanyild/Lane-Segmentation/blob/master/refineNet/sample_result/test6.png" width="400"/>

Forward pass takes on an average for refineNet 0.04 s.

Here is the colab notebook link of SAD experimentation of RefineNet with VGG
https://colab.research.google.com/drive/1m3eEQa9EgjFifa3SdDkocVtSg2bbz_bl

Sample output
<img src="https://github.com/raanyild/Lane-Segmentation/blob/master/refineNet/examples/refineNetwihVGG.png" width="400"/>

## Graph Cut
Using graph cut method the region of road is segmented. This method is quite slow compared to the current state of the art deep learning appoaches. 
To get segmented image use main.py file.
### Sample Results
<img src="https://github.com/raanyild/Lane-Segmentation/blob/master/graph_cut/sample_results/input1.png" width="400"/> <img src="https://github.com/raanyild/Lane-Segmentation/blob/master/graph_cut/sample_results/output1.png" width="400"/>
<img src="https://github.com/raanyild/Lane-Segmentation/blob/master/graph_cut/sample_results/input2.png" width="400"/> <img src="https://github.com/raanyild/Lane-Segmentation/blob/master/graph_cut/sample_results/output2.png" width="400"/>
<img src="https://github.com/raanyild/Lane-Segmentation/blob/master/graph_cut/sample_results/input3.png" width="400"/> <img src="https://github.com/raanyild/Lane-Segmentation/blob/master/graph_cut/sample_results/output3.png" width="400"/>

## Multistream CNN

The code to run the model is is the folder msfcn with the name run.py

The model takes three consecutive images which have been previously resized to (224,224) and are in png format . To run you need the enter the paths of the images and the path to the trained model

sub.py is the code required to run the model in ROS kinetic. The code subscribes the images from the node "/camera/image_color" which is stored in the variable image_topic defined in the main function. The output is published on the topic "image_topic_2"

Link of the dataset used : https://drive.google.com/open?id=1ZE_KMgrmazHKCtMcPNQwqcSSTRYLFTQs

Link of the trained model : https://drive.google.com/open?id=1QnfjKH_1GIUMh1Ytl9649BubaO_c1C4m

After the code has been run the output is saved as output.png and the time for forward pass is displayed

RUNNING train.py FOR TRAINING MSFCN

### Keep the data in the following format:
1.) make a folder called "data" which has folders "test" and "train"


2.) "test" and "train" will have each have two folders "images" and "labels"


3.) "images" and "labels" will have subfolders with numeric names (1 2 etc) which will contain SEQUENCES OF IMAGES in the following format :

      1> They will be of the size 224X224
      
      2> They will be in png format
      
      3> They sequence will have an acsending order of names (1.png 2.png)


4.) Folder "n" in "images" will have the sequence corresponding to Folder "n" in labels.


5.) Therefore for example /data/train/images/3/56.png will have the label image at /data/test/images/3/56.png

### RESULTS

ACCURACY ON CITYSCAPES: 98%


IOU ON CITYSCAPES : 93%

![33](https://user-images.githubusercontent.com/43606115/68527836-83df5380-0311-11ea-91be-94c32b64db35.png)
![31](https://user-images.githubusercontent.com/43606115/68527837-8641ad80-0311-11ea-8a7f-34bcc318a012.png)
![21](https://user-images.githubusercontent.com/43606115/68527838-8772da80-0311-11ea-8aa1-984b645a8c19.png)
![17](https://user-images.githubusercontent.com/43606115/68527839-893c9e00-0311-11ea-9df3-6abd9fd6e559.png)
![10](https://user-images.githubusercontent.com/43606115/68527841-8a6dcb00-0311-11ea-81c6-200be3f0f9cf.png)
![9](https://user-images.githubusercontent.com/43606115/68527843-8c378e80-0311-11ea-9f93-70cf5a13a911.png)



RESULTs ON 2.2:
https://drive.google.com/file/d/1AVtZlIzjx0Zp6JeTB2bGtnJs7l_ArTHG/view?usp=sharing https://drive.google.com/file/d/1Klec59x0r34yhjiqwo6BVXwKKyz-kY62/view?usp=sharing


## Depth Model

Link to required files: https://drive.google.com/open?id=1aQYMIWeyXZzcoRACmQUQROwzG-325JKq

### Instruction to run

Run the notebook in depth_model in the same folder which has the folder of required files from the above link

### Results on 2.2

<img src="https://user-images.githubusercontent.com/43606115/72667467-27e82800-3a42-11ea-8185-bd6a250869ab.jpg" height="192" width="640"> ![](https://user-images.githubusercontent.com/43606115/72667474-38989e00-3a42-11ea-823b-db1ed8e1c1e2.png)

<img src="https://user-images.githubusercontent.com/43606115/72667484-464e2380-3a42-11ea-93d0-1388054766e5.jpg" height="192" width="640"> ![](https://user-images.githubusercontent.com/43606115/72667495-549c3f80-3a42-11ea-939b-bc36f0010e91.png)

<img src=https://user-images.githubusercontent.com/43606115/72667636-fc663d00-3a43-11ea-8a11-78dc944581d1.jpg height="192" width="640">  ![](https://user-images.githubusercontent.com/43606115/72667653-261f6400-3a44-11ea-87f7-f143d8bea834.png)
 

## Domain adaptation

Paper: [Learning to Adapt Structured Output Space for Semantic Segmentation](https://arxiv.org/abs/1802.10349)
Repo: https://github.com/wasidennis/AdaptSegNet

Results on 2.2: https://drive.google.com/open?id=1dAJDNy0Yyiv7HrOJIKu6URO288NkZTzx

## Enet + Self Attention Distillation
Paper: https://arxiv.org/abs/1908.00821

Model Trained on Cityscapes
Dataset used : https://drive.google.com/drive/folders/1lSc520BfXSMbzRTlyN0PKW1o27Q0AEXT?usp=sharing

ACCURACY ON CITYSCAPES: 93.39%

### Results

<img src=https://user-images.githubusercontent.com/45457504/73071278-649b9f80-3ed8-11ea-8783-36e4553a7f21.png>

<img src=https://user-images.githubusercontent.com/45457504/73071284-69605380-3ed8-11ea-8775-4491e635a70c.png>

<img src=https://user-images.githubusercontent.com/45457504/73071296-6e250780-3ed8-11ea-9af7-e9c02c782628.png>

Attention Maps:
Image:

<img src=https://user-images.githubusercontent.com/45457504/73071406-a7f60e00-3ed8-11ea-9cf9-31fd4d3d5ec8.jpg>

Attention Map 1:

<img src=https://user-images.githubusercontent.com/45457504/73071322-7bda8d00-3ed8-11ea-8d90-a604630d2a64.png>

Attention Map 2:

<img src=https://user-images.githubusercontent.com/45457504/73071332-8137d780-3ed8-11ea-81b0-05472d3d299c.png>

Attention Map 3:

<img src=https://user-images.githubusercontent.com/45457504/73071354-872db880-3ed8-11ea-8686-f721f6a70b87.png>

Attention Map 4:

<img src=https://user-images.githubusercontent.com/45457504/73071371-8eed5d00-3ed8-11ea-98a5-00633e3da659.png>

# TODO
HR ENet and other architectures
Domain Adaptation
CutMix
