#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:33:27 2019

@author: weidongming
"""

import os
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

def threscut(subject_data, threshold_min=0, threshold_max=1700):# 1000,2000
    subject_data[subject_data > threshold_max] = threshold_max
    subject_data[subject_data < threshold_min] = threshold_min

    return subject_data


def normalize_3D(img_3D, nor_min=0, nor_max=1700):
    """ The shape of img_3D should be (depth, width, height)"""
    data_3D = img_3D - nor_min
    data_3D = data_3D / np.float32(nor_max-nor_min)
    #data_3D = img_3D - img_3D.min()
    #data_3D = data_3D / np.float32(img_3D.max())
    return np.asarray(data_3D, np.float32)

def hist_of_3d_array(array_3d):
    array_1d = np.reshape(array_3d, [np.size(array_3d)])
    num_bins=10
    hist = plt.hist(array_1d, num_bins, facecolor='blue', alpha=0.5)
    return(hist)

def data_augumentation_flip(array_3d):
    flip_index = np.random.randint(0,3)
    if flip_index == 0:
        array_3d = array_3d[::-1,:,:]
    if flip_index == 1:
        array_3d = array_3d[:,::-1,:]
    if flip_index == 2:
        array_3d = array_3d[:,:,::-1]
    return array_3d

def data_augumentation_noise(img_3d):
    # Create a list of intensity modifying filters, which we apply to the given images
    filter_list = []
    
    # Smoothing filters
    
    filter_list.append(sitk.SmoothingRecursiveGaussianImageFilter())
    filter_list[-1].SetSigma(2.0)
    
    filter_list.append(sitk.DiscreteGaussianImageFilter())
    filter_list[-1].SetVariance(4.0)
    
    filter_list.append(sitk.BilateralImageFilter())
    filter_list[-1].SetDomainSigma(4.0)
    filter_list[-1].SetRangeSigma(8.0)
    
    filter_list.append(sitk.MedianImageFilter())
    filter_list[-1].SetRadius(8)
    
    # Noise filters using default settings
    
    # Filter control via SetMean, SetStandardDeviation.
    filter_list.append(sitk.AdditiveGaussianNoiseImageFilter())
    filter_list[-1].SetMean(0.2)
    filter_list[-1].SetStandardDeviation(0.7)

    # Filter control via SetProbability
    filter_list.append(sitk.SaltAndPepperNoiseImageFilter())
    
    # Filter control via SetScale
    filter_list.append(sitk.ShotNoiseImageFilter())
    
    # Filter control via SetStandardDeviation
    filter_list.append(sitk.SpeckleNoiseImageFilter())

    filter_list.append(sitk.AdaptiveHistogramEqualizationImageFilter())
    filter_list[-1].SetAlpha(1.0)
    filter_list[-1].SetBeta(0.0)

    filter_list.append(sitk.AdaptiveHistogramEqualizationImageFilter())
    filter_list[-1].SetAlpha(0.0)
    filter_list[-1].SetBeta(1.0)
    
    aug_image_lists = [] # Used only for display purposes in this notebook.
    f = filter_list[4]
    aug_image = f.Execute(img_3d)
    return aug_image

def dilation(img_3d):
    d = sitk.GrayscaleDilateImageFilter()
    for i in range(8):
        img_3d = d.Execute(img_3d)
    return img_3d

def pre_process(img_3d):
    img_data = sitk.GetArrayFromImage(img_3d)
    img_data = threscut(img_data)
    img_data = normalize_3D(img_data)
    img = sitk.GetImageFromArray(img_data)
    img = dilation(img)
    return img

def generate_single_batch_train_data(data_path, batch_size):
    while 1:
        #img_data_batch = np.zeros([30,512,512,batch_size])
        #label_batch = np.zeros([1,2,batch_size])
        #for i in os.listdir(data_path):
        for j in range(batch_size):
            img_data_sum = np.zeros([1,256,128,128,1])
            size = int(img_data_sum.shape[1]/2)
            #label_sum = np.zeros([1,2])
            class_ind = np.random.randint(0,2)
                        
            #if class_ind == 0:
            data_path_1 = data_path + '/CT_full_size'
            i = np.random.randint(len(os.listdir(data_path_1)))
            
            img = sitk.ReadImage(os.path.join(data_path_1, sorted(os.listdir(data_path_1))[i]))
            img = pre_process(img)
            img_data = sitk.GetArrayFromImage(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = np.expand_dims(img_data, axis=4)
                
            img_data_sum[0,:size,:,:,:] = img_data
            #label = np.expand_dims(label, axis=0)
            
                
            #else:
            data_path_1 = data_path + '/Probe_full_size'
            

            img = sitk.ReadImage(os.path.join(data_path_1, sorted(os.listdir(data_path_1))[i]))
            img = pre_process(img)
            img_data = sitk.GetArrayFromImage(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = np.expand_dims(img_data, axis=4)
            img_data_sum[0,size:2*size,:,:,:] = img_data
            if class_ind ==0:
                label = np.array([1,0])
            if class_ind == 1:
                label = np.array([0,1])
            label = np.expand_dims(label, axis=0)
            
            if class_ind == 0:
                yield [img_data_sum[:,:size,:,:,:], img_data_sum[:,size:2*size,:,:,:]], label
            else:
                yield [img_data_sum[:,size:,:,:,:], img_data_sum[:,:size,:,:,:]], label
def generate_single_batch_validation_data(data_path, batch_size):
    while 1:
        for j in range(batch_size):
            img_data_sum = np.zeros([1,256,128,128,1])
            size = int(img_data_sum.shape[1]/2)
            #label_sum = np.zeros([1,2])
            class_ind = np.random.randint(0,2)
                        
            #if class_ind == 0:
            data_path_1 = data_path + '/CT_full_size'
            i = np.random.randint(len(os.listdir(data_path_1)))
            
            img = sitk.ReadImage(os.path.join(data_path_1, sorted(os.listdir(data_path_1))[i]))
            img = pre_process(img)
            img_data = sitk.GetArrayFromImage(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = np.expand_dims(img_data, axis=4)
                
            img_data_sum[0,:size,:,:,:] = img_data
            #label = np.expand_dims(label, axis=0)
            
                
            #else:
            data_path_1 = data_path + '/Probe_full_size'
            

            img = sitk.ReadImage(os.path.join(data_path_1, sorted(os.listdir(data_path_1))[i]))
            img = pre_process(img)
            img_data = sitk.GetArrayFromImage(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = np.expand_dims(img_data, axis=4)
            img_data_sum[0,size:2*size,:,:,:] = img_data
            if class_ind ==0:
                label = np.array([1,0])
            if class_ind == 1:
                label = np.array([0,1])
            label = np.expand_dims(label, axis=0)
            
            if class_ind == 0:
                yield [img_data_sum[:,:size,:,:,:], img_data_sum[:,size:2*size,:,:,:]], label
            else:
                yield [img_data_sum[:,size:,:,:,:], img_data_sum[:,:size,:,:,:]], label
        
