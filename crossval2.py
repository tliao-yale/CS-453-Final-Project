#must create utils.py file with get_data() function

#must create a model.py file with model() and train(model, X,y, X_cv, Y_cv) functions
    #train function returns out of sample predictions on the test set

import numpy as np
import pandas as pd
import models
import utils
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
import re
from scipy.io import loadmat

scores = list()

segment = re.compile('_(\d+)_(0|1)')

for j in ['1', '2', '3']:
	data = pd.read_csv('fft_simple_features_train_'+j+'.csv') 

	files = data['filename']

	data['segment'] = data['filename'].map(lambda x: int(segment.findall(x)[0][0]))

	data = data.sort_values(['label', 'segment'])
	data['seq'] = data['segment'].map(lambda x: (int(x) - 1)//6)
	data['seq'] = data['seq'].where(data['label'] == 0, -1 * (data['seq'] + 1))
	   
	seq_outcome = data[['seq', 'label']].drop_duplicates()

	skf = StratifiedKFold(seq_outcome['label'], n_folds=5, random_state=1)
	oos_class_preds = np.zeros(len(files))
	auc_out = np.zeros(5)

	labels = data['label'].as_matrix()

	for train_index, test_index in skf:
	    seq_train = np.array(seq_outcome.iloc[train_index]['seq'])
	    seq_test = np.array(seq_outcome.iloc[test_index]['seq'])

	    train_data = data[data.seq.isin(seq_train)]
	    test_data = data[data.seq.isin(seq_test)]

	    X_train = train_data.drop(['Unnamed: 0', 'filename','segment', 'label', 'seq'], axis=1).as_matrix()
	    y_train = train_data['label'].as_matrix()
	    X_test = test_data.drop(['Unnamed: 0', 'filename', 'segment', 'label', 'seq'], axis=1).as_matrix()
	    y_test = test_data['label'].as_matrix()

	    oos_class_preds[data.seq.isin(seq_test).as_matrix()] = models.xgb_train(X_train, y_train, X_test, y_test, j, 1, True)
	    print np.sum(oos_class_preds)
	    print np.sum([i > 0 for i in oos_class_preds])

	np.save('oos_preds_'+j+'.npy', oos_class_preds)
	np.save('labels_'+j+'.npy', labels)

	scores.append(roc_auc_score(labels, oos_class_preds))


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