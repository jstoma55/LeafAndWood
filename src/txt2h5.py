import pandas as pd
import numpy as np
import h5py
import sys
sys.path.append("..")
from os import listdir
from os.path import isfile, join
data_dir = "../data/labeled/"
h5_data_dir = "../data/h5/"
files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
for file in files:
    df = pd.read_csv(data_dir+file, sep=" ", header=None)
    data = df.to_numpy()
    coords = data[:,0:3]
    feats = data[:,3:5]
    labels = data[:,-1]
    with h5py.File(h5_data_dir+file.replace(".txt","")+".h5py", 'w') as h5f:
        h5f.create_dataset('coords', data = coords)
        h5f.create_dataset("feats", data = feats)
        h5f.create_dataset('labels', data = labels)
