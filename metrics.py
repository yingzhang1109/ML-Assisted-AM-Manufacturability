# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:51:12 2019

@author: ADML
"""

from functools import partial

from keras import backend as K

def IoU(y_true, y_pred):
# =============================================================================
#     y_pred = K.one_hot(K.argmax(y_pred,axis=-1),3)
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (intersection)/(K.sum(y_true_f) + K.sum(y_pred_f)-intersection)
# =============================================================================
    #y_true = y_true[:,:,:,1:3]
    #y_pred = y_pred[:,:,:,1:3]
    pred = K.argmax(y_pred,axis=-1)
    pred = K.one_hot(pred,num_classes=3)
    intersection = K.sum(K.abs(y_true * pred), axis=[1,2,3])
    union = K.sum(y_true,axis=[1,2,3])+K.sum(pred,axis=[1,2,3])-intersection
    iou = (intersection) / (union)
    return iou

def IoU_mean(y_true, y_pred):
    return K.mean(IoU(y_true, y_pred),axis=0)

def IoU_2(y_true, y_pred):
    return IoU(y_true, y_pred)[:,2]

def IoU_0(y_true, y_pred):
    return IoU(y_true, y_pred)[:,0]

def IoU_1(y_true, y_pred):
    return IoU(y_true, y_pred)[:,1]

def dice_coefficient(y_true, y_pred, smooth=1):
    #y_true_f = K.flatten(y_true)
    #y_pred_f = K.flatten(y_pred)
    #y_true = y_true[:,:,:,1:3]
    #y_pred = y_pred[:,:,:,1:3]
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth)) 
    #intersection = K.sum(y_true_f * y_pred_f)
    #return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def tversky_loss(y_true, y_pred, smooth=1):
    beta = 0.5
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    denominator = intersection + K.sum((beta*(1-y_true)*(y_pred) + (1-beta)*y_true*(1-y_pred)),axis=[1,2,3])
    pre_loss = (intersection+smooth)/(denominator +smooth)
    loss = 0.25*pre_loss[:,0]+0.25*pre_loss[:,1]+0.5*pre_loss[:,2]
    return 1-loss

def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss