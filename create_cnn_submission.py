import numpy as np
import pandas as pd
import models
import utils
import re
from glob import glob

for j in ['1', '2', '3']:
	#load model
	if j == '1':
		model = models.cnn_1('cnn_weights_1_3_09.h5')
	if j == '2':
		model = models.cnn_2('cnn_weights_2_3_09.h5')
	if j == '3':
		model = models.cnn_2('cnn_weights_3_3_09.h5')

	#get data
	files = np.array(glob('../test_'+j+'/*'))

	if j == '1':
		#1584, 18*88
		batch = 18
	if j == '2':
		#2256, 24*94 or 12*188
		batch = 24
	if j == '3':
		#2286, 127*18
		batch = 18

	testgen =  utils.data_label_gen(files, batch)

	print 'running patient', j
	preds = model.predict_generator(testgen, len(files), nb_worker = 2)

	if j == '1':
		all_preds = list(preds)
		all_files = list(files)
	else:
		all_preds += list(preds)
		all_files += list(files)


slash = re.compile('/')
all_files = map(lambda x: slash.split(x)[-1], all_files)

with open('cnn_submission.csv', 'w+') as f:
    f.write('File,Class\n')
    for i in range(len(all_preds)):
        s = all_files[i] + ',' + str(all_preds[i])
        f.write(s + '\n')

