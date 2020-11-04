# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:14:01 2019

@author: ADML
"""
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten,Reshape
from keras.layers.merge import concatenate
from keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose,UpSampling3D,BatchNormalization
from metrics import dice_coefficient_loss, dice_coefficient, IoU_mean,IoU_2,IoU_0,IoU_1,tversky_loss
from keras.losses import categorical_crossentropy

def voxmodel(process_size,input_shape,n_labels, INIT_LR,include_label_wise_dice_coefficients =False ):
    model1_in =Input(shape=(process_size,))
    model1 = Dense(32,activation='relu')(model1_in)
    model1=Dense(64,activation='relu')(model1)
    
    model2_in = Input(shape=input_shape)
    
    conv_1 = Conv3D(16,(3,3,3),padding='same',activation='relu')(model2_in)
    #conv_1 = BatchNormalization(axis=1)(conv_1)
    conv_1 = Conv3D(32,(3,3,3),padding='same',activation='relu')(conv_1)
    #conv_1 = BatchNormalization(axis=1)(conv_1)
    pool_1 = MaxPooling3D()(conv_1)
    conv_2 = Conv3D(32,(3,3,3),padding='same',activation='relu')(pool_1)
    #conv_2 = BatchNormalization(axis=1)(conv_2)
    conv_2 = Conv3D(64,(3,3,3),padding='same',activation='relu')(conv_2)
    #conv_2 = BatchNormalization(axis=1)(conv_2)
    pool_2 = MaxPooling3D()(conv_2)
    conv_3 = Conv3D(64,(3,3,3),padding='same',activation='relu')(pool_2)
    #conv_3 = BatchNormalization(axis=1)(conv_3)
    conv_3 = Conv3D(128,(3,3,3),padding='same',activation='relu')(conv_3)
    #conv_3 = BatchNormalization(axis=1)(conv_3)
    pool_3 = MaxPooling3D()(conv_3)

    
    flatlayer = Flatten()(pool_3)
    flatlayer = Dense(256, activation='relu')(flatlayer)
    flatlayer = Dense(64, activation='relu')(flatlayer)

    merge4 = concatenate([flatlayer,model1])
    merge4 = Dense(128,activation='relu')(merge4)
    merge4 = Dropout(0.5)(merge4)
    merge4 = Dense(256,activation='relu')(merge4)
    merge4 = Dropout(0.5)(merge4)
    merge4 = Dense(16384,activation='relu')(merge4)
    merge4 = Reshape((16,16,16,4))(merge4)
    #up_conv4 = Conv3DTranspose(128,(2,2,2),strides=(2, 2, 2))(merge4)
    
    conv_4 = Conv3D(128,(3,3,3),padding='same',activation='relu')(merge4)
    #conv_4 = BatchNormalization(axis=1)(conv_4)
    conv_4 = Conv3D(256,(3,3,3),padding='same',activation='relu')(conv_4)
    #conv_4 = BatchNormalization(axis=1)(conv_4)
    up_conv3 = Conv3DTranspose(256,(2,2,2),strides=(2, 2, 2))(conv_4)
    #up_conv3 = UpSampling3D()(conv_4)
    merge3 = concatenate([conv_3,up_conv3])
    merge3 = Conv3D(128,(3,3,3),padding='same',activation='relu')(merge3)
    #merge3 = BatchNormalization(axis=1)(merge3)
    merge3 = Conv3D(128,(3,3,3),padding='same',activation='relu')(merge3)
    #merge3 = BatchNormalization(axis=1)(merge3)
    up_conv2 = Conv3DTranspose(128,(2,2,2),strides=(2, 2, 2))(merge3)
    #up_conv2 = UpSampling3D()(merge3)
    merge2 = concatenate([conv_2,up_conv2])
    merge2 = Conv3D(64,(3,3,3),padding='same',activation='relu')(merge2)
    #merge2 = BatchNormalization(axis=1)(merge2)
    merge2 = Conv3D(64,(3,3,3),padding='same',activation='relu')(merge2)
    #merge2 = BatchNormalization(axis=1)(merge2)
    up_conv1 = Conv3DTranspose(64,(2,2,2),strides=(2, 2, 2))(merge2)
    #up_conv1 = UpSampling3D()(merge2)
    merge1 = concatenate([conv_1,up_conv1])
    merge1 = Conv3D(32,(3,3,3),padding='same',activation='relu')(merge1)
    #merge1 = BatchNormalization(axis=1)(merge1)
    merge1 = Conv3D(32,(3,3,3),padding='same',activation='relu')(merge1)
    #merge1 = BatchNormalization(axis=1)(merge1)
    out = Conv3D(n_labels,(1,1,1),activation='softmax')(merge1)
    #with tf.device("/cpu:0"):
    model = Model([model1_in,model2_in],out)
    #model = multi_gpu_model(model,gpus=2)
    print("[INFO] compiling model...") 
    opt = keras.optimizers.Adam(lr=INIT_LR)  
    metrics=dice_coefficient
    metrics = [metrics]+[IoU_mean]+[IoU_0]+[IoU_1]+[IoU_2]
    #model.compile(loss = categorical_crossentropy,optimizer = opt,metrics=['acc'])
    model.compile(loss = tversky_loss, optimizer = opt, metrics=metrics)
    return model