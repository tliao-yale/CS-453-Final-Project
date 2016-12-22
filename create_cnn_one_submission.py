import numpy as np
import pandas as pd
import models
import utils
import re
from glob import glob

#get data
files = np.array(glob('../test_*/*'))

#load model
model = models.cnn_spectro('cnn_weights_0_1_05.h5')

batch = 6

testgen =  utils.data_label_spectro_gen(files, batch)

print 'running patient'
preds = model.predict_generator(testgen, len(files), nb_worker = 4)[:,0]


slash = re.compile('/')
files = map(lambda x: slash.split(x)[-1], files)

with open('cnn_submission.csv', 'w+') as f:
    f.write('File,Class\n')
    for i in range(len(preds)):
        s = files[i] + ',' + str(preds[i])
        f.write(s + '\n')