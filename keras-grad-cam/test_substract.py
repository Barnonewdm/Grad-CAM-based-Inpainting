#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 14:15:39 2019

@author: weidongming
"""

from tensorflow.keras import layers
from tensorflow.keras import Model
import SimpleITK as sitk
import numpy as np
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

target_size = (128, 128, 128)
def dilation(img_3d):
    d = sitk.GrayscaleDilateImageFilter()
    for i in range(16):
        img_3d = d.Execute(img_3d)
    return img_3d    
    

def model():
    input_1 = layers.Input(shape = target_size + (1,))
    input_2 = layers.Input(shape = target_size + (1,))
    
    subtract = layers.Subtract()([input_2, input_1])
    #subtract = layers.Subtract()([subtract, input_1])
    x = layers.MaxPooling3D(8)(subtract)
    
    model = Model(inputs = [input_1, input_2], outputs = x)
    
    return model

if __name__=='__main__':
    model = model()
    IMG1 = sitk.ReadImage('/data/wdm_Projects/X-ray-images-classification-with-Keras-TensorFlow/data/training/CT_full_size_dilation/2_CT.nii')
    IMG2 = sitk.ReadImage('/data/wdm_Projects/X-ray-images-classification-with-Keras-TensorFlow/data/training/Probe_full_size_dilation/2_Probe.nii')
    IMG1 = dilation(IMG1)
    IMG2 = dilation(IMG2)
    img1 = sitk.GetArrayFromImage(IMG1)
    img2 = sitk.GetArrayFromImage(IMG2)
    img1 = np.expand_dims(img1, axis=0)
    img1 = np.expand_dims(img1, axis=4)
    img2 = np.expand_dims(img2, axis=0)
    img2 = np.expand_dims(img2, axis=4)
    data = model.predict([img1, img2])
    result = sitk.GetImageFromArray(data[0,:,:,:,0])
    sitk.WriteImage(result, './test2.nii.gz')
    
