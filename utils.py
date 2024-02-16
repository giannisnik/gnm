import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_moons


def load_dataset(dataset_name):
    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    if dataset_name == 'noisy_moons':
        noisy_moons = make_moons(n_samples=1000, noise=0.1)
        X = np.array([item for item in noisy_moons[0]])
        y = noisy_moons[1]
    elif dataset_name == 'yeast':
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data', sep='\s+', header=None)
        X = df.iloc[:,1:9].values
        y = df.iloc[:,9].values
        
        le = LabelEncoder()
        y = le.fit_transform(y)
    elif dataset_name == 'wireless':
        if not os.path.exists('datasets/wireless+indoor+localization') :
            urllib.request.urlretrieve("https://archive.ics.uci.edu/static/public/422/wireless+indoor+localization.zip", "datasets/wireless+indoor+localization.zip")
            with zipfile.ZipFile('datasets/wireless+indoor+localization.zip', 'r') as zip_ref:
                zip_ref.extractall('datasets/wireless+indoor+localization')
        
        df = pd.read_csv('datasets/wireless+indoor+localization/wifi_localization.txt', delimiter='\t', header=None)
        X = df.iloc[:,:7].values
        y = df.iloc[:,7].values
        y -= 1
    elif dataset_name == 'wine':
        if not os.path.exists('datasets/wine+quality') :
            urllib.request.urlretrieve("https://archive.ics.uci.edu/static/public/186/wine+quality.zip", "datasets/wine+quality.zip")
            with zipfile.ZipFile('datasets/wine+quality.zip', 'r') as zip_ref:
                zip_ref.extractall('datasets/wine+quality')
        
        X = np.loadtxt('datasets/wine+quality/winequality-white.csv', skiprows=1, delimiter=';', usecols=(0,1,2,3,4,5,6,7,8,9,10))
        y = np.loadtxt('datasets/wine+quality/winequality-white.csv', skiprows=1, delimiter=';', usecols=11)

    return X, y