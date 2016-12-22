#change the filepath!!!!

import pandas as pd
from sklearn.preprocessing import scale
from scipy.signal import resample
import numpy as np
from multiprocessing import Pool
from glob import glob
from scipy.io import loadmat
import re
from scipy import stats


#define functions to get features
#gets FFT with 1-47 Hz
def fft(time_data):
    return np.log10(np.absolute(np.fft.rfft(np.transpose(time_data), axis=1)[:,1:48]))

#takes only upper triangular elements (used to eliminate redundant information from symmetric matrices)
def upper_right_triangle(mat):
    dim = mat.shape[0]
    result = np.zeros(dim * (dim + 1) / 2 - dim)
    k = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            result[k] = mat[i,j]
            k += 1
    return result

#gets the correlation coefficients in frequency domain and eigenvalues between channels of EEG in FFT
def freq_corr(fft_data):
    scaled = scale(fft_data, axis=0)
    corr_matrix = np.corrcoef(scaled)
    eigenvalues = np.absolute(np.linalg.eig(corr_matrix)[0])
    eigenvalues.sort()
    corr_coefficients = upper_right_triangle(corr_matrix) # custom func
    return np.concatenate((corr_coefficients, eigenvalues))

#gets the correlation coefficients in time domain and eigenvalues between channels of EEG in FFT
def time_corr(data):
    data = np.transpose(data)
    resampled = resample(data, 400, axis=1) if data.shape[-1] > 400 else data
    scaled = scale(resampled, axis=0)
    corr_matrix = np.corrcoef(scaled)
    eigenvalues = np.absolute(np.linalg.eig(corr_matrix)[0])
    corr_coefficients = upper_right_triangle(corr_matrix) # custom func
    return np.concatenate((corr_coefficients, eigenvalues))

def get_averages(time_data):
    time_data = np.transpose(time_data)

    means = np.zeros((time_data.shape[0], time_data.shape[1] / 6000))
    stds = np.zeros((time_data.shape[0], time_data.shape[1] / 6000))
    for i in range(means.shape[1]):
        means[:, i] = np.mean(time_data[:, i * 6000: (i+1) * 6000])
        stds[:, i] = np.std(time_data[:, i * 6000: (i+1) * 6000])

    return np.concatenate((means.ravel(), stds.ravel()))

def get_simple_averages(time_data):
    time_data = np.transpose(time_data)

    means = np.mean(time_data, axis=1)
    stds = np.std(time_data, axis=1)
    skews = stats.skew(time_data, axis = 1)
    kurt = stats.kurtosis(time_data, axis = 1)

    return np.concatenate((means.ravel(), stds.ravel(), skews.ravel(), kurt.ravel()))

#performs all feature extraction from a given csv file
def transform(filename):
    print filename

    try: 
        datadict = loadmat(filename)
        data = datadict['dataStruct']['data'][0][0]
        if np.amax(data) == 0 and np.amin(data) == 0:
            return np.zeros(2304)

        fft_out = fft(data)
        freq_corr_out = freq_corr(fft_out)
        time_corr_out = time_corr(data)

        averages = get_averages(data)

        return np.concatenate((fft_out.ravel(), freq_corr_out, time_corr_out, averages))

    except:
        return np.zeros(2304)

#performs all feature extraction from a given csv file
def simple_transform(filename):
    print filename

    try: 
        datadict = loadmat(filename)
        data = datadict['dataStruct']['data'][0][0]
        if np.amax(data) == 0 and np.amin(data) == 0:
            return np.zeros(64)

        simple = get_simple_averages(data)

        return simple

    except:
        return np.zeros(64)


def train_features():
    for i in ['1','2','3']:
        #get all files
        files = glob('../train_'+i+'/*')

        #use multiprocessing to parallelize mapping
        pool = Pool(2)
        result = pool.map(transform, files)
        result = np.array(result)


        #get labels from the file names
        label_finder = re.compile('_(\d)\.mat')
        labels = map(lambda x: int(label_finder.findall(x)[0]), files)

        #append filenames, labels, and features into a csv
        labels = np.expand_dims(labels, 1)
        files = np.expand_dims(files, 1)

        output = np.concatenate((files, labels, result), 1) 
        columns = ['filename', 'label'] + range(result.shape[1]) 
        columns = np.array(columns)


        df = pd.DataFrame(output, columns=columns)
        df.to_csv('fft_avg_features_train_'+i+'.csv')

def train_simple_features():
    for i in ['1','2','3']:
        #get all files
        files = glob('../train_'+i+'/*')

        #use multiprocessing to parallelize mapping
        pool = Pool(2)
        result = pool.map(simple_transform, files)
        result = np.array(result)

        #get labels from the file names
        label_finder = re.compile('_(\d)\.mat')
        labels = map(lambda x: int(label_finder.findall(x)[0]), files)

        #append filenames, labels, and features into a csv
        labels = np.expand_dims(labels, 1)
        files = np.expand_dims(files, 1)

        print result.shape
        print files.shape
        print labels.shape

        output = np.concatenate((files, labels, result), 1) 
        columns = ['filename', 'label'] + range(result.shape[1]) 
        columns = np.array(columns)


        df = pd.DataFrame(output, columns=columns)
        df.to_csv('fft_simple_features_train_'+i+'.csv')

def test_features():
    for i in ['1','2','3']:
        #get all files
        files = glob('../test_'+i+'/*')

        #use multiprocessing to parallelize mapping
        pool = Pool(2)
        result = pool.map(transform, files)
        result = np.array(result)

        #append filenames, labels, and features into a csv
        files = np.expand_dims(files, 1)

        output = np.concatenate((files, result), 1) 
        columns = ['filename'] + range(result.shape[1]) 
        columns = np.array(columns)


        df = pd.DataFrame(output, columns=columns)
        df.to_csv('fft_avg_features_test_'+i+'.csv')

def test_simple_features():
    for i in ['1','2','3']:
        #get all files
        files = glob('../test_'+i+'/*')

        #use multiprocessing to parallelize mapping
        pool = Pool(2)
        result = pool.map(simple_transform, files)
        result = np.array(result)

        #append filenames, labels, and features into a csv
        files = np.expand_dims(files, 1)

        output = np.concatenate((files, result), 1) 
        columns = ['filename'] + range(result.shape[1]) 
        columns = np.array(columns)


        df = pd.DataFrame(output, columns=columns)
        df.to_csv('fft_simple_features_test_'+i+'.csv')

train_simple_features()
test_simple_features()
