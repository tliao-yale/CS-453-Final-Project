#must create utils.py file with get_data() function

#must create a model.py file with model() and train(model, X,y, X_cv, Y_cv) functions
    #train function returns out of sample predictions on the test set


import numpy as np
import pandas as pd
import models
import utils
from sklearn.metrics import roc_auc_score
from glob import glob
import re

scores = list()

#get data
files = np.array(glob('../train_*/*'))

#define CV
n_folds = 5

label_finder = re.compile('_(\d)\.mat')
y = map(lambda x: int(label_finder.findall(x)[0]), files)
y = np.array(y)

skf = utils.sequence_stratified_Kfold(y, n_folds=n_folds)

#oos_class_preds = np.zeros(y.shape)
#auc_out = np.zeros(n_folds)

#train with Cross validation
for i, (train, test) in enumerate(skf):
    print "Running Fold", i+1, "/", n_folds 
    
    train_batch = 16
    test_batch = 6
    test_size = 1188

    traingen = utils.data_label_spectro_gen(files[np.array(train)], train_batch)
    testgen =  utils.data_label_spectro_gen(files[np.array(test)], train_batch)

    aucgen = utils.data_label_spectro_gen(files[test], test_batch)

    preds = models.cnn_train(traingen, testgen, aucgen, '0', str(i+1), test_size)
    y_test = y[test][:test_size]

    print 'AUC score', roc_auc_score(y_test, preds)



