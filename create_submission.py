import numpy as np
import pandas as pd
import models
import utils
import xgboost as xgb
import re

for j in ['1', '2', '3']:
	#train full model
	X, y = utils.get_xgb_data('fft_simple_features_train_'+j+'.csv')
	dtrain = xgb.DMatrix(X, label=y, missing=0)

	models.xgb_train(X, y, [], [], j, 0, False)

	bst = xgb.Booster({'nthread':4}) #init model
	bst.load_model('xgb_model_'+j+'.model')


	#get test data
	X, files = utils.get_test_data('fft_simple_features_test_'+j+'.csv', j)
	dtest = xgb.DMatrix(X)

	preds = bst.predict(dtest)

	# preds = np.zeros((5, len(files)))

	# for i, fold in enumerate(['1', '2', '3', '4', '5']):
	# 	bst = xgb.Booster({'nthread':4}) #init model
	# 	bst.load_model('xgb_model_'+j+'_'+fold+'.model')

	# 	#predict
	# 	preds[i,:] = bst.predict(dtest)

	# preds = np.mean(preds, axis = 0)

	if j == '1':
		all_preds = list(preds)
		all_files = list(files)
	else:
		all_preds += list(preds)
		all_files += list(files)


slash = re.compile('/')
all_files = map(lambda x: slash.split(x)[-1], all_files)

with open('submission.csv', 'w+') as f:
    f.write('File,Class\n')
    for i in range(len(all_preds)):
        s = all_files[i] + ',' + str(all_preds[i])
        f.write(s + '\n')


