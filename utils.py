import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import re
from scipy.io import loadmat
from scipy import signal
import threading

def get_xgb_data(filename):

	data = pd.read_csv(filename)

	#sort
	# nums = re.compile('_(\d+)_(0|1).mat')
	# files = data['filename']
	# file_number = map(lambda x: int(nums.findall(x)[0][0]), files)
	# data.insert(1, 'number', file_number)
	# data = data.sort('number')

	#clean and return
	data = data.drop(['Unnamed: 0', 'filename'], 1)

	data = data.as_matrix()

	y = data[:,0]
	X = data[:,1:]

	return X,y

def get_test_data(filename, j):

	data = pd.read_csv(filename)

	#clean and return
	data = data.drop(['Unnamed: 0'], 1)
	data = data.as_matrix()

	files = data[:,0]
	X = data[:,1:]

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

	return X,files

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def data_label_gen(files, batch_size):
	label_finder = re.compile('_(\d)\.mat')

	batch = batch_size

	while True:
		for j in range(len(files) / batch):
			X = np.zeros((batch, 240000, 16))
			y = np.zeros(batch)

			for i in range(batch):
				try:
					datadict = loadmat(files[batch * j + i])
					X[i,:,:] = datadict['dataStruct']['data'][0][0]
					y[i] = int(label_finder.findall(files[batch * j + i])[0])
				except:
					pass
 

			yield np.array(X), np.array(y)

@threadsafe_generator
def data_label_spectro_gen(files, batch_size):
	label_finder = re.compile('_(\d)\.mat')

	batch = batch_size

	while True:
		for j in range(len(files) / batch):
			X = np.zeros((batch, 16, 150, 685))
			y = np.zeros(batch)

			for i in range(batch):
				try:
					datadict = loadmat(files[batch * j + i])
					data = datadict['dataStruct']['data'][0][0]
					for k in range(16):
						f,t,Sxx = signal.spectrogram(data[:,k], 400, nperseg = 400)
						X[i,k,:,:] = Sxx[:150,:]
					y[i] = int(label_finder.findall(files[batch * j + i])[0])
				except:
					pass
 

			yield np.array(X), np.array(y)


def sequence_stratified_Kfold(labels, n_folds):
	#random.seed(1)
	np.random.seed(1)

	result = list()

	test = list()

	positives = [i for i in range(len(labels)) if labels[i] == 1]
	negatives = [i for i in range(len(labels)) if labels[i] == 0]

	pos_chunks = range(len(positives) / 6)
	neg_chunks = range(len(negatives) / 6)
	np.random.shuffle(pos_chunks)
	np.random.shuffle(neg_chunks)

	for i in range(n_folds):
	    this_test = list()

	    test_pos_size = len(pos_chunks) / n_folds
	    test_neg_size = len(neg_chunks) / n_folds
	    
	    test_pos_chunks = pos_chunks[i * test_pos_size : (i+1) * test_pos_size]
	    test_neg_chunks = neg_chunks[i * test_neg_size : (i+1) * test_neg_size]
	    

	    for j in test_pos_chunks:
	        this_test += positives[j * 6: (j+1) * 6]
	    for j in test_neg_chunks:
	        this_test += negatives[j * 6: (j+1) * 6]

	    test.append(np.random.permutation(this_test))

	for t in test:
		train = np.random.permutation([i for i in range(len(labels)) if i not in t])

		result.append((train, t))

	return result



def get_data(filename):

	return get_xgb_data(filename)
