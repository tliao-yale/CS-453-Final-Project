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

for j in ['1', '2', '3']:
	#get data
	files = np.array(glob('../train_'+j+'/*'))

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
	    test_batch = 18
	    if j == '1':
	    	test_size = 252
	    if j == '2':
	    	test_size = 468
	    if j == '3':
	    	test_size = 468

	    traingen = utils.data_label_gen(files[train], train_batch)
	    testgen =  utils.data_label_gen(files[test], train_batch)

	    aucgen = utils.data_label_gen(files[test], test_batch)

	    preds = models.cnn_train(traingen, testgen, aucgen, j, str(i+1), test_size)
	    y_test = y[test][:test_size]

	    print 'AUC score', roc_auc_score(y_test, preds)

	#np.save('oos_preds_'+j+'.npy', oos_class_preds)
	#np.save('labels_'+j+'.npy', y)

	#scores.append(roc_auc_score(y, oos_class_preds))


#get total score
preds1 = np.load('oos_preds_1.npy')
preds2 = np.load('oos_preds_2.npy')
preds3 = np.load('oos_preds_3.npy')

preds = np.hstack((preds1, preds2, preds3))

labels1 = np.load('labels_1.npy')
labels2 = np.load('labels_2.npy')
labels3 = np.load('labels_3.npy')

labels = np.hstack((labels1, labels2, labels3))

print scores
print 'ROC AUC Score:', roc_auc_score(labels, preds)