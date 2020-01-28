#!/usr/bin/env python
# coding: utf-8

# In[12]:


import json
import matplotlib.pyplot as plt
from random import seed
from random import randint
# seed random number generator
seed(1)


# In[13]:


path_for_json_file = 'C:\\Users\\spider2\\Desktop\\New_training_data_Std_Format_files.json'
with open(path_for_json_file,'r') as f:
    distros_dic=json.load(f)


# In[14]:


targetlist=list(distros_dic["frames"].keys())
import cv2


# In[15]:


#Funtions for doing data augmentation

def make_upper_patch(x1, y1, x2, y2, shape):
    if (y1-4)>=0:
        return [x1, y1-4, x2, y2-4]
    else:
        return [-1, -1, -1, -1]

def make_left_patch(x1, y1, x2, y2, shape):
    if (x1-4)>=0:
        return [x1-4, y1, x2-4, y2]
    else:
        return [-1, -1, -1, -1]
    
def make_bottom_patch(x1, y1, x2, y2, shape):
    if (y2+4)<shape[0]:
        return [x1, y1+4, x2, y2+4]
    else:
        return [-1, -1, -1, -1]
    
def make_right_patch(x1, y1, x2, y2, shape):
    if (x2+4)<shape[1]:
        return [x1+4, y1, x2+4, y2]
    else:
        return [-1, -1, -1, -1]

def check_seq(seq, shape):
    if ((seq[0]>0) & (seq[1]>0) & (seq[2]<shape[1]) & (seq[3]<shape[0])):
        return seq
    else:
        return [-1, -1 , -1, -1]
    


# In[16]:


#This part makes the patches from the given data images and compiles them in a list

listoftargetpatches=[]
o=0
for imgname in targetlist:
     o=o+1
     if imgname!='normalsoilmetalincomprun.png':
         path_for_training_images = "C:\\Users\\spider2\\Desktop\\Data\\New_training_data_Std_Format_files\\"
         im1=cv2.imread(path_for_training_images+imgname)
         for j in range(len(distros_dic["frames"][imgname])) :
            x1=distros_dic["frames"][imgname][j]['x1']
            x2=distros_dic["frames"][imgname][j]['x2']
            y1=distros_dic["frames"][imgname][j]['y1']
            y2=distros_dic["frames"][imgname][j]['y2']
            seq = [x1, y1, x2, y2]
            seq = check_seq(seq, im1.shape)
            badseq = [-1, -1, -1, -1]
            
            #Main patch
            if seq!=badseq:
                im2=im1[int(y1):int(y2),int(x1):int(x2)]
                listoftargetpatches.append(im2)

                
                
#If you Require Augmentation, please uncomment the code given below


#             #Upper Augmentation
#             seq = make_upper_patch(x1, y1, x2, y2, im1.shape)
#             if seq!=badseq:
#                print(seq)
#                im2=im1[int(seq[1]):int(seq[3]),int(seq[0]):int(seq[2])]
#                listoftargetpatches.append(im2)
#                print("Upper_aug1:")
# #               plt.figure()
# #               plt.imshow(im2,cmap='gray') 
# #               plt.show()
#                seq = make_upper_patch(seq[0], seq[1], seq[2], seq[3], im1.shape)
#                if seq!=badseq:
#                    im2=im1[int(seq[1]):int(seq[3]),int(seq[0]):int(seq[2])]
#                    listoftargetpatches.append(im2)
# #                    print("Upper_aug2:")
# #                    plt.figure()
# #                    plt.imshow(im2,cmap='gray') 
# #                    plt.show()
            
#             #Lower Augmentation
#             seq = make_bottom_patch(x1, y1, x2, y2, im1.shape)
#             if seq!=badseq:
#                im2=im1[int(seq[1]):int(seq[3]),int(seq[0]):int(seq[2])]
#                listoftargetpatches.append(im2)
# #                plt.figure()
# #                plt.imshow(im2,cmap='gray') 
# #                plt.show()
#                seq = make_bottom_patch(seq[0], seq[1], seq[2], seq[3], im1.shape)
#                if seq!=badseq:
#                    im2=im1[int(seq[1]):int(seq[3]),int(seq[0]):int(seq[2])]
#                    listoftargetpatches.append(im2)
# #                    plt.figure()
# #                    plt.imshow(im2,cmap='gray') 
# #                    plt.show()
            
            
#             #Right Augmentation
#             seq = make_right_patch(x1, y1, x2, y2, im1.shape)
#             if seq!=badseq:
#                im2=im1[int(seq[1]):int(seq[3]),int(seq[0]):int(seq[2])]
#                listoftargetpatches.append(im2)
# #                plt.figure()
# #                plt.imshow(im2,cmap='gray') 
# #                plt.show()
                
#             #Left Augmentation 
#             seq = make_left_patch(x1, y1, x2, y2, im1.shape)
#             if seq!=badseq:
#                im2=im1[int(seq[1]):int(seq[3]),int(seq[0]):int(seq[2])]
#                listoftargetpatches.append(im2)
# #                plt.figure()
# #                plt.imshow(im2,cmap='gray') 
# #                plt.show()
# print(len(listoftargetpatches))


# In[17]:


#This part makes patches for non-object from the given images

def make_non_threat_patch(listofnontargetpatches, shape):
    xm = int(im1.shape[1]/2)
    ym = int(im1.shape[0]/2)
    x1=xm-16
    x2=xm+16
    y1=ym-16
    y2=ym+16
    listofnontargetpatches.append([x1,y1,x2,y2])
    listofnontargetpatches.append([x1-2, y1, x2-2, y2])
    listofnontargetpatches.append([x1+2, y1, x2+2, y2])
    for i in range(6):
        yr = randint(0, (int(shape[0]/2) -16))
        listofnontargetpatches.append([x1-2, y1+yr, x2-2, y2+yr])
    for i in range(6):
        yr = randint(0, (int(shape[0]/2) -16))
        listofnontargetpatches.append([x1-2, y1-yr, x2-2, y2-yr])
    for i in range(6):
        yr = randint(0, (int(shape[0]/2) -16))
        listofnontargetpatches.append([x1+2, y1+yr, x2+2, y2+yr])
    for i in range(6):
        yr = randint(0, (int(shape[0]/2) -16))
        listofnontargetpatches.append([x1+2, y1-yr, x2+2, y2-yr])
    


# In[18]:


#This part compiles all the non target patches in a list

listofnontargetpatchesseq=[]
listofnontargetpatches=[]
for imgname in targetlist:
     if imgname!='normalsoilmetalincomprun.png':
#          print(imgname)
         im1=cv2.imread("C:\\Users\\spider2\\Desktop\\Data\\New_training_data_Std_Format_files\\"+imgname)
         make_non_threat_patch(listofnontargetpatchesseq, im1.shape)
        
for i in range(len(listofnontargetpatchesseq)):
    im2=im1[int(listofnontargetpatchesseq[i][1]):int(listofnontargetpatchesseq[i][3]),int(listofnontargetpatchesseq[i][0]):int(listofnontargetpatchesseq[i][2])]
    listofnontargetpatches.append(im2)
print(len(listofnontargetpatches))


# In[19]:


#This part resizes all the target patches to a single size (here 32*32)

listoftargetpatches2=[]
maxh=0
maxw=0
for j in range(len(listoftargetpatches)):
    im=cv2.resize(listoftargetpatches[j],(32,32))
    listoftargetpatches2.append(im)


# In[20]:


#sample some random patch images for view

im1=listoftargetpatches2[14]
plt.figure()
plt.imshow(im1,cmap='gray') 
plt.show()

im1=listofnontargetpatches[1234]
plt.figure()
plt.imshow(im1,cmap='gray') 
plt.show()


# In[34]:


#This is the place where we really save the patch images in a designated folder
#For threat (known object) patches

from PIL import Image
row = []
dataset = []
for i in range(len(listoftargetpatches2)):
    img = listoftargetpatches2[i]
    img_name = str(i)+"_threat.jpeg"
    img = Image.fromarray(img, 'RGB')
    root_dir = "C://Users//spider2//Downloads//Final//threatdata_img//" #Location for threat patches save location
    row = [img_name, 1]
    dataset.append(row)
    img = img.save(root_dir+img_name)


# In[35]:


#For threat (known non-object) patches

from numpy import savetxt
import pandas as pd
row = []
for i in range(len(listofnontargetpatches)):
    img = listofnontargetpatches[i]
    img_name = str(i)+"_nonthreat.jpeg"
    img = Image.fromarray(img, 'RGB')
    root_dir = "C://Users//spider2//Downloads//Final//nonthreatdata_img//"  #Location for non-threat patches save location
    row = [img_name, 0]
    dataset.append(row)
    img = img.save(root_dir+img_name)

#Make a csv file for all the patches along with the patch names and types
import numpy as np    
dataset = np.array(dataset)
df = pd.DataFrame.from_records(dataset)
gprs = df.to_csv("C://Users//spider2//Downloads//Final//gprs.csv")



#Thats it, you have all the images saved to your folder in the sub-directories along with a csv file, ready for training


# In[ ]:




