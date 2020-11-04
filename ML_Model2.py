# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 09:30:11 2019

@author: ADML
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from preprocessing import preprocessing
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
from VoxModel import voxmodel

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NUM_EPOCHS = 100 
INIT_LR = 1e-5
x,y,z = 128,128,128
process_size = 11 
labels = (0,1,2)
n_labels = len(labels)
batch_size = 1

dataset = pd.read_pickle('data\dataset_only0_all.pickle')#dataset_only0_all.pickle
train_dataset,valid_dataset = train_test_split(dataset, test_size=0.2)
testdata = pd.read_pickle('data\dataset_test.pickle')
X1_train,X2_train,y_train,vec = preprocessing(train_dataset,0)
X1_valid,X2_valid,y_valid,vec = preprocessing(valid_dataset,vec)
X1_test,X2_test = preprocessing(testdata,vec,0)

model = voxmodel(process_size,(x,y,z,1),n_labels,INIT_LR)
#from keras.utils.vis_utils import plot_model
#plot_model(model, to_file='3dmodel_plot.png', show_shapes=True, show_layer_names=True)

print("[INFO] training network...")    
ML=model.fit([X1_train,X2_train], y_train,
          batch_size=batch_size,
          epochs=NUM_EPOCHS,
          validation_data=([X1_valid,X2_valid], y_valid), 
          verbose =1)    
y_valid_pred = model.predict([X1_valid,X2_valid],batch_size = 1)

#score = model.evaluate([X1_test,X2_test], y_test, verbose=1)
#y_test_pred = model.predict([X1_test,X2_test])
#  y_train_pred = model.predict([X1_train,X2_train])


#from sklearn.externals import joblib  
#
## Save the model
#joblib.dump(model, "trained-model.pkl")