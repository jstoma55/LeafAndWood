import h5py

def read_voxel_data(data_path):
    with h5py.File(data_path, "r") as hf:   
        voxels = hf["voxels"][:]
        labels = hf["labels"][:]
        print("Voxels data shape:", voxels.shape)
        print("Labels data shape:", labels.shape)
    return voxels, labels

