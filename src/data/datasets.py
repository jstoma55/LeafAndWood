import sys
sys.path.append("..")
import h5py
import numpy as np
from torch.utils.data import Dataset
import MinkowskiEngine as ME

class TrainingDataset(Dataset):
    def __init__(self, data_path, quantization_size=0.005, debug=False):
        self.data_path = data_path
        self.quantization_size=quantization_size
        self.data = h5py.File(data_path, "r")
        self.keys = list(self.data.keys())
        self.debug = debug
        print("Train set size: ",len(self.keys))

    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, idx):
        key = self.keys[idx]
        coords = self.data[key]['coords'][:]
        features = self.data[key]['feats'][:]
        labels = np.int_(self.data[key]['labels'][:])
        #if self.debug:
        #    print("coords:")
        #    print(coords)
        #    print("feats:")
        #    print(features)
        #    print("labels:")
        #    print(labels)
        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coordinates=coords,
            features=features,
            labels=labels,
            quantization_size=self.quantization_size,
            ignore_label=-100)
        return discrete_coords, unique_feats, unique_labels
    

class PredictionDataset(Dataset):
    def __init__(self, data_path, quantization_size=0.005, debug=False):
        self.data_path = data_path
        self.quantization_size=quantization_size
        self.data = h5py.File(data_path, "r")
        self.keys = list(self.data.keys())
        self.debug = debug
        print("Test set size: ",len(self.keys))

    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, idx):
        key = self.keys[idx]
        coords = self.data[key]['coords'][:]/self.quantization_size
        feats = self.data[key]['feats'][:]
        labels = np.int_(self.data[key]['labels'][:])
        #if self.debug:
        #    print("coords:")
        #    print(coords)
        #    print("feats:")
        #    print(feats)
        #    print("labels:")
        #    print(labels)
        return coords, feats, labels
    
