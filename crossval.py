#must create utils.py file with get_data() function

#must create a model.py file with model() and train(model, X,y, X_cv, Y_cv) functions
    #train function returns out of sample predictions on the test set

import numpy as np
import pandas as pd
import models
import utils
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold

scores = list()

for j in ['1', '2', '3']:
	#get data
	X, y = utils.get_data('fft_simple_features_train_'+j+'.csv')

	#define CV
	n_folds = 5

	skf = utils.sequence_stratified_Kfold(y, n_folds=n_folds)

	oos_class_preds = np.zeros(y.shape)
	auc_out = np.zeros(n_folds)

	#train with Cross validation
	for i, (train, test) in enumerate(skf):
	    print "Running Fold", i+1, "/", n_folds 
	    
	    oos_class_preds[test] = models.xgb_train(X[train], y[train], X[test], y[test], j, i+1, True)

	np.save('oos_preds_'+j+'.npy', oos_class_preds)
	np.save('labels_'+j+'.npy', y)

	scores.append(roc_auc_score(y, oos_class_preds))


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




