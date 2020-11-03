# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 13:07:22 2019

@author: Ying
"""
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
# definine the total number of epochs to train for along with the
# initial learning rate

config = tf.ConfigProto( device_count = {'CPU': 12} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

NUM_EPOCHS = 40
INIT_LR = 1e-5
x,y,z = 32,32,32
process_size = 12
num_classes = 1
batch_size = 8

def poly_decay(epoch):
	# initialize the maximum number of epochs, base learning rate,
	# and power of the polynomial
	maxEpochs = NUM_EPOCHS
	baseLR = INIT_LR
	power = 1.0
	# compute the new learning rate based on polynomial decay
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
	# return the new learning rate
	return alpha


dataset = pd.read_pickle('data\dataset_norepeat32.pickle') #dataset_norepeat
train_dataset,valid_dataset = train_test_split(dataset, test_size=0.2)
#train_dataset,valid_dataset = train_test_split(train_dataset, test_size=0.2)
def preprocessing(dataset,vec, testornot = 1):
    obj = dataset['stl'].values
    obj=np.concatenate(obj).astype(None).reshape(obj.shape[0],32,32,32)
    # =============================================================================
    # parameters_text  = dataset[['lattice_type','material','machine']]
    # parameters_value = dataset[['strut_radius','cell_size','internal_density','density']].values
    # =============================================================================
    parameters  = dataset[['material','material_brand','machine_brand',
                           'machine_type','material density']]
    parameters = parameters.to_dict('records')

    if vec ==0:
        vec = DictVectorizer()
        
        X1 = vec.fit_transform(parameters).toarray()
        X1 = np.float16(X1)
        X2 = obj.reshape(obj.shape[0],x,y,z,1)
        X2 = np.float16(X2)
    else:
        X1 = vec.transform(parameters).toarray()
        X1 = np.float16(X1)
        X2 = obj.reshape(obj.shape[0],x,y,z,1)
        X2 = np.float16(X2)
    if testornot == 0:
        return X1,X2
    else:
        label = dataset['fabricated?']
        #encoded_label = to_categorical(label)
        encoded_label =label   
    
        encoded_label = encoded_label.values
        encoded_label = np.float16(encoded_label)
        return X1,X2,encoded_label,vec

# =============================================================================
# ex = obj[59]
# image = ex[:,:,5]
# plt.imshow(image)
# plt.gray()
# plt.show()
# =============================================================================

X1_train,X2_train,y_train,vec = preprocessing(train_dataset,0)
X1_valid,X2_valid,y_valid,vec = preprocessing(valid_dataset,vec)
callbacks = [LearningRateScheduler(poly_decay)]

#testdata = pd.read_pickle('data\dataset_test.pickle')
#X1_test,X2_test = preprocessing(testdata,vec,0)
# sample 0 is the overhang
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.merge import concatenate
from keras.layers import Conv3D, MaxPooling3D


model1_in =Input(shape=(process_size,))
model1 = Dense(64,activation='relu')(model1_in)
model1=Dense(128,activation='relu')(model1)

model2_in = Input(shape=(x,y,z,1))

model2 = Conv3D(16,(5,5,5),activation='relu')(model2_in)
model2 = MaxPooling3D()(model2)
model2 = Conv3D(32,(3,3,3),activation='relu')(model2)
model2 = MaxPooling3D()(model2)
model2 = Conv3D(64,(3,3,3),activation='relu')(model2)
model2 = MaxPooling3D()(model2)
#model2 = Dropout(0.25)(model2)
model2 = Flatten()(model2)
model2 = Dense(1024, activation='relu')(model2)
model2 = Dropout(0.5)(model2)
model2 = Dense(128, activation='relu')(model2)
#model2 = Dropout(0.5)(model2)

merged = concatenate([model1, model2])
merged = Dense(256,activation='relu')(merged)
merged = Dropout(0.5)(merged)
merged = Dense(512,activation='relu')(merged)
merged = Dropout(0.5)(merged)
out = Dense(num_classes, activation = 'sigmoid')(merged)

print("[INFO] training with CPUs...")
model = Model([model1_in,model2_in],out)

print("[INFO] compiling model...")    
opt = keras.optimizers.Adam(lr=INIT_LR)  

model.compile(loss = keras.losses.binary_crossentropy, 
              optimizer = opt, metrics=['accuracy'])

# =============================================================================
# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# =============================================================================
print("[INFO] training network...")    
ML=model.fit([X1_train,X2_train], y_train,
          batch_size=batch_size,
          epochs=NUM_EPOCHS,
          validation_data=([X1_valid,X2_valid], y_valid),
          callbacks = [LearningRateScheduler(poly_decay)],
          verbose =1)    
#weights = model.save_weights("layers_weights_inh5format")
# =============================================================================
# for layer in model.layers:
#     g=layer.get_config()
#     h=layer.get_weights()
#     print (g)
#     print (h)    
# =============================================================================
score = model.evaluate([X1_valid,X2_valid], y_valid, verbose=1)
# =============================================================================
# y_test_pred = model.predict([X1_test,X2_test])
# real_y_pred = []
# for item in y_test_pred:
#     if item >0.5:
#         real_y_pred.append(1)
#     else:
#         real_y_pred.append(0)  
# real_y_pred = np.asarray(real_y_pred)
# =============================================================================
print('Test loss:', score[0])
print('Test accuracy:', score[1])
pyplot.plot(ML.history['loss'])
pyplot.plot(ML.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()
pyplot.plot(ML.history['acc'])
pyplot.plot(ML.history['val_acc'])
pyplot.title('model accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper left')
pyplot.show()



