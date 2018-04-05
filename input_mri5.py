# _*_coding:utf-8

import os
import tensorflow as tf
import numpy as np
import random
import nibabel as nib

width = 61
height = 73
depth = 61
channel = 1
batch_index = 0
test_batch_index = 0
num_class = 2

PATH='/workspace/3dcnn/mycode_2017.7_2/file.list'
def get_all_data(path=PATH):
    image_file=[]
    label=[]
    data=[]
    f=open(path,'r')
    lines=list(f)
    for index in range(len(lines)):
        line = lines[index].strip('\n').split()
        image_file.append(line[0])
        label.append(line[1])
    label = list(map(int,label))
    for i in range(len(label)):
        data.append([image_file[i],label[i]])
    random.shuffle(data)
    return data


def get_data_mri(train_filenames,batch_size):
    global batch_index

    max = len(train_filenames)
    begin = batch_index
    end = batch_index + batch_size
    batch_index += batch_size
    if end >= max:
        end = max
        batch_index=0
    real_batch_size=end-begin

    x_data = np.array([],np.float32)
    y_data = np.zeros((real_batch_size,num_class))
    index = 0

    for i in range(begin,end):
        image_path = train_filenames[i][0]
        image_org = nib.load(image_path)
        image_data = image_org.get_data()
        x_data = np.append(x_data, np.asarray(image_data, dtype='float32'))
        y_data[index][train_filenames[i][1]]=1
        index += 1
 
    x_data = x_data.reshape(real_batch_size, width*height*depth)

    return x_data, y_data,end

def get_test_mri(test_filenames,batch_size):
    global test_batch_index

    max = len(test_filenames)
    if test_batch_index>=max:
       test_batch_index=0
    begin = test_batch_index
    end = test_batch_index + batch_size
    test_batch_index += batch_size
    real_batch_size=end-begin

    x_data = np.array([],np.float32)
    y_data = np.zeros((real_batch_size,num_class))
    index = 0

    for i in range(begin,end):
        image_path = test_filenames[i][0]
        image_org = nib.load(image_path)
        image_data = image_org.get_data()
        x_data = np.append(x_data, np.asarray(image_data, dtype='float32'))
        y_data[index][test_filenames[i][1]]=1
        index += 1
 
    x_data = x_data.reshape(real_batch_size, width*height*depth)

    return x_data, y_data, image_path

#for i in range(17):
   # _,y=get_test_mri(batch_size=1)
    #print(y)
#for i in range(50):
    #print(i)
    #x_data, y_data = get_data_mri(6)
    #print(x_data.shape)
    #print(y_data)
    #x_data, y_data = get_data_mri(1)
    #print(x_data.shape)
    #print(y_data)
# imagePath = "E:/3Ddata/nii_61x73x61/myHC_ADNI1_/_smwc1ADNI_005_S_0610_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20070904175641004_S26265_I71325.nii"
# FA_org = nib.load(imagePath)
# FA_data = FA_org.get_data()
# print(FA_data.shape)
# # print(np.max(FA_data))
# resized_image = tf.image.resize_images(images=FA_data, size=(width,height), method=1)
# print(resized_image.shape)
# # batch = get_data_mri(sess=sess)
# x_data = np.array([],np.float32)
# x_data = np.append(x_data, np.asarray(FA_data, dtype='float32'))
# print(x_data.shape)
