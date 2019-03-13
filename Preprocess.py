#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 20:55:11 2019

@author: weidongming
"""
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
class Preprocess(object):
    def __init__(self):
        #threshold cut
        self.threshold_min = 0
        self.threshold_max = 1700
        # histogram normalization
        self.nor_min = 0
        self.nor_max = 1700
        # histogram plot
        self.num_bins = 10
        # number of grayscale dilation
        self.num_dilation = 6
    
    def threscut(self, subject_data):# 1000,2000
        # CT image threshold cut
        subject_data[subject_data > self.threshold_max] = self.threshold_max
        subject_data[subject_data < self.threshold_min] = self.threshold_min
    
        return subject_data
    
    
    def normalize_3D(self, img_3D):
        """ histogram normalized to 0~1"""
        data_3D = img_3D - self.nor_min
        data_3D = data_3D / np.float32(self.nor_max-self.nor_min)
        #data_3D = img_3D - img_3D.min()
        #data_3D = data_3D / np.float32(img_3D.max())
        return np.asarray(data_3D, np.float32)
    
    def hist_of_3d_array(self, array_3d):
        array_1d = np.reshape(array_3d, [np.size(array_3d)])
        num_bins=self.num_bins
        hist = plt.hist(array_1d, num_bins, facecolor='blue', alpha=0.5)
        return(hist)
    
    def data_augumentation_flip(self, array_3d):
        # flip
        flip_index = np.random.randint(0,3)
        if flip_index == 0:
            array_3d = array_3d[::-1,:,:]
        if flip_index == 1:
            array_3d = array_3d[:,::-1,:]
        if flip_index == 2:
            array_3d = array_3d[:,:,::-1]
        return array_3d
    
    def data_augumentation_noise(self, img_3d):
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
        
        f = filter_list[4]
        aug_image = f.Execute(img_3d)
        return aug_image
    
    def dilation(self, img_3d):
        # GrayeScale Dilation
        d = sitk.GrayscaleDilateImageFilter()
        for i in range(self.num_dilation):
            img_3d = d.Execute(img_3d)
        return img_3d
    
    def pre_process(self, img_3d):
        # user defined pipeline
        img_data = sitk.GetArrayFromImage(img_3d)
        img_data = self.threscut(img_data)
        img_data = self.normalize_3D(img_data)
        img = sitk.GetImageFromArray(img_data)
        img = self.dilation(img)
        return img