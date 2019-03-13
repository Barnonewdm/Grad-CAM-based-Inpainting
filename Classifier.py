#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:01:51 2019

@author: weidongming
"""

import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from datagenerator import generate_single_batch_train_data
from datagenerator import generate_single_batch_validation_data

os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
base_dir = '/data/wdm_Projects/X-ray-images-classification-with-Keras-TensorFlow/data'
class_names = ['CT','Probe']
img_width = img_height = 512
color_channels = 3

#layers & optimizer
target_width = target_height = 512
target_size = (128, 128, 128)#(30, 256, 256)#(100,306,386)#(100, 306, 386)
pooling_window = 2
conv_window = 3
kernel_size = 4
dropout=0.01
activation = "relu"
optimizer = 'Adam'
loss='categorical_crossentropy'
#metrics= ['acc']
metrics=['categorical_accuracy']
class_mode = 'categorical'
batch_size = 200
epochs = 100
verbose = 2
lr = 0.0001

# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for the three color channels: R, G, and B
img_input1 = layers.Input(shape=target_size + (1,))
img_input2 = layers.Input(shape=target_size + (1,))
img_input = layers.Subtract()([img_input2, img_input1])
net = layers.Conv3D(kernel_size, conv_window, activation=activation, padding='valid', strides=4)(img_input)
x = layers.MaxPooling3D(pooling_window)(net)
x = layers.Conv3D(2*kernel_size, conv_window, activation=activation, padding='same', strides=1)(x)
x = layers.MaxPooling3D(pooling_window)(x)
x = layers.Conv3D(6*kernel_size, conv_window, activation=activation, padding='same', strides=1)(x)
x = layers.Conv3D(6*kernel_size, conv_window, activation=activation, padding='same')(x)
x = layers.Conv3D(6*kernel_size, conv_window, activation=activation, padding='same', name='conv_3d')(x)
x = layers.Flatten(name='Flatten')(x)
x = layers.Dense(128, activation=activation, name='Dense_512')(x)
output = layers.Dense(len(class_names), activation = 'softmax', name='Dense_2') (x)
model = Model(inputs=[img_input1,img_input2], outputs=output)

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=metrics)

train_generator = generate_single_batch_train_data(base_dir+'/training', batch_size)
validation_generator = generate_single_batch_validation_data(base_dir + '/validation', batch_size)

from keras.callbacks import ModelCheckpoint
filepath="./models/weights-improvement-{epoch:02d}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max', period=2)
callbacks_list = [checkpoint]
#model.load_weights('./models/work_model.h5')
history = model.fit_generator(
    train_generator,
    steps_per_epoch = batch_size,
    validation_data = validation_generator, 
    validation_steps =2,
    callbacks=callbacks_list,
    epochs = epochs)