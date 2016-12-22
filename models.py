# For every new model you want to make, 
# make a new function in this models.py script that runs the model
# 
#

import xgboost as xgb
import numpy as np
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score

from keras.layers.pooling import MaxPooling1D, AveragePooling1D, MaxPooling2D, AveragePooling2D
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Model
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam


def cnn_1(weights_path = None):

    inputs = Input(shape=(240000, 16))

    pool1 = AveragePooling1D(10)(inputs)
    #norm1 = BatchNormalization(axis = 2)(pool1)

    conv1 = Convolution1D(32, 10, activation='relu', border_mode='same')(pool1)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Convolution1D(32, 10, activation='relu', border_mode='same')(conv1)
    conv1 = Dropout(0.1)(conv1)

    pool2 = MaxPooling1D(10)(conv1)

    conv2 = Convolution1D(32, 10, activation='relu', border_mode='same')(pool2)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Convolution1D(32, 10, activation='relu', border_mode='same')(conv2)
    conv2 = Dropout(0.1)(conv2)

    pool3 = MaxPooling1D(10)(conv2)

    conv3 = Convolution1D(64, 10, activation='relu', border_mode='same')(pool3)
    conv3 = Dropout(0.1)(conv3)
    conv3 = Convolution1D(64, 10, activation='relu', border_mode='same')(conv3)
    conv3 = Dropout(0.5)(conv3)

    pool4 = MaxPooling1D(10)(conv3)

    outputs = Convolution1D(1, 24, activation='sigmoid', border_mode='valid')(pool4)
    outputs = Flatten()(outputs)

    model = Model(input=inputs, output=outputs)

    if weights_path:
        model.load_weights(weights_path)
        
    adam = Adam(lr=.001, clipnorm = 1)
    model.compile(optimizer=adam, loss = 'binary_crossentropy', metrics=['binary_crossentropy'])
    
    return model

def cnn_2(weights_path = None):

    inputs = Input(shape=(240000, 16))

    pool1 = AveragePooling1D(10)(inputs)
    norm1 = BatchNormalization(axis = 1)(pool1)

    conv1 = Convolution1D(32, 10, activation='relu', border_mode='same')(norm1)
    #conv1 = Dropout(0.1)(conv1)
    conv1 = Convolution1D(32, 10, activation='relu', border_mode='same')(conv1)
    #conv1 = Dropout(0.1)(conv1)

    pool2 = MaxPooling1D(10)(conv1)

    conv2 = Convolution1D(32, 10, activation='relu', border_mode='same')(pool2)
    #conv2 = Dropout(0.1)(conv2)
    conv2 = Convolution1D(32, 10, activation='relu', border_mode='same')(conv2)
    #conv2 = Dropout(0.1)(conv2)

    pool3 = MaxPooling1D(10)(conv2)

    conv3 = Convolution1D(64, 10, activation='relu', border_mode='same')(pool3)
    #conv3 = Dropout(0.1)(conv3)
    conv3 = Convolution1D(64, 10, activation='relu', border_mode='same')(conv3)
    #conv3 = Dropout(0.5)(conv3)

    pool4 = MaxPooling1D(10)(conv3)

    outputs = Convolution1D(1, 24, activation='sigmoid', border_mode='valid')(pool4)
    outputs = Flatten()(outputs)

    model = Model(input=inputs, output=outputs)

    if weights_path:
        model.load_weights(weights_path)
        
    adam = Adam(lr=.001, clipnorm = 1)
    model.compile(optimizer=adam, loss = 'binary_crossentropy', metrics=['binary_crossentropy'])
    
    return model

def cnn_spectro(weights_path=None):

    inputs = Input(shape=(16,150, 685))

    norm1 = BatchNormalization(axis = 1)(inputs)

    conv1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same',)(norm1)
    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same',)(conv1)

    pool1 = MaxPooling2D((2,3), strides=(2,3))(conv2)

    conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same',)(pool1)
    conv4 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', )(conv3)

    pool2 = MaxPooling2D((2,3), strides=(2,3))(conv4)

    conv5 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', )(pool2)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', )(conv5)

    pool3 = MaxPooling2D((2,3), strides=(2,3))(conv6)

    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', )(pool3)
    conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', )(conv7)

    flat = Flatten()(conv8)
    outputs = Dense(128, activation = 'relu')(flat)
    outputs = Dense(128, activation = 'relu')(outputs)
    outputs = Dense(128, activation = 'relu')(outputs)
    outputs = Dense(1, activation = 'sigmoid')(outputs)

    model = Model(input=inputs, output=outputs)

    if weights_path:
        model.load_weights(weights_path)
        
    ad = Adam(lr=0.001, clipnorm = 1)
    model.compile(optimizer=ad, loss = 'binary_crossentropy', metrics=['binary_crossentropy'])

    return model 


def cnn_train(train_gen, test_gen, auc_gen, patient, fold, test_size):

    model = cnn_spectro()

    checkpoint = ModelCheckpoint('cnn_weights_'+patient+'_'+fold+'_{epoch:02d}.h5')

    model.fit_generator(train_gen, 5000, 5, validation_data = test_gen, nb_val_samples = 256, callbacks=[checkpoint], 
                        nb_worker = 6)
    #class_weight = {0: .1, 1: .9}

    #preds = model.predict_generator(pred_gen, 250, nb_worker = 2)
    preds = model.predict_generator(auc_gen, test_size, nb_worker = 4)
    return preds

    #print roc_auc_score(y_cv, preds)



def xgb_train(X, y, X_cv, y_cv, j, fold, train_mode):

    #xgb params
    if j == '1':
        param = {'max_depth':3, 'eta':0.1, 'gamma':0, 'silent':0, 'objective':'binary:logistic', 'colsample_bytree':.9,
                 'subsample':.9, 'nthread':6, 'min_child_weight':1.4, 'eval_metric': 'auc'} # 'scale_pos_weight':.885/.115, 
        numround = 30
    if j == '2':
        param = {'max_depth':3, 'eta':0.1, 'gamma':0, 'silent':0, 'objective':'binary:logistic', 'colsample_bytree':.5,
                 'subsample':.9, 'nthread':6, 'min_child_weight':1.4, 'eval_metric': 'auc',} # 'scale_pos_weight':.93/.07, 
        numround = 40
    if j == '3':
        param = {'max_depth':3, 'eta':0.1, 'gamma':0, 'silent':0, 'objective':'binary:logistic', 'colsample_bytree':.5,
                 'subsample':.9, 'nthread':6, 'min_child_weight':1.4, 'eval_metric': 'auc'} # ,'scale_pos_weight':.94/.06,
        numround = 40

    #62: 3, .2, .5, 50 and 40

    #feature selection
    if j == '1':
        #standard deviations and FFT correlation matrix+ range(752, 888)
        features = range(32) 
    if j == '2':
        #just standard deviations range(1664, 2304) + range(752, 888)
        features = range(32) 
    if j == '3':
        #standard deviations, raw FFT, and FFT correlation matrix + range(752, 888) + range(752)
        features = range(32) 


    X = X[:, features]


    #train model
    if train_mode: 

        X_cv = X_cv[:, features]

        dtrain = xgb.DMatrix(X, label=y, missing=0)

        dtest = xgb.DMatrix(X_cv, label=y_cv)
        
        bst = xgb.train(param, dtrain, numround, [(dtest,'eval'), (dtrain,'train')])

        bst.save_model('xgb_model_'+j+'_'+str(fold)+'.model')
        
        preds = bst.predict(dtest)

        return preds

    else: 
        dtrain = xgb.DMatrix(X, label=y, missing = 0)
        
        bst = xgb.train(param, dtrain, numround, [(dtrain,'train')])

        bst.save_model('xgb_model_'+j+'.model')



def train(X, y, X_cv, y_cv, j, train_mode):

	return xgb_train(X, y, X_cv, y_cv, j, train_mode)
