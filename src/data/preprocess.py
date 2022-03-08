import pandas as pd
import numpy as np
import h5py
import sys
sys.path.append("..")
from os import listdir
from os.path import isfile, join
import random
# random.seed(42)
from scipy.spatial import KDTree

# Default paths
LABELED_DATA_DIR = "../data/labeled/"
H5_PATH = "../data/h5_meter/data.h5py"
H5_TRAIN_PATH = "../data/h5_meter/train.h5py"
H5_TEST_PATH = "../data/h5_meter/test.h5py"
H5_DEV_PATH = "../data/h5_meter/dev.h5py"

# Default parameters
GRID_TILE_SIZE = 1
MIN_POINT_NUM = 5
TRAINING_SIZE = 0.8
TEST_SIZE = 0.1
DEV_SIZE = 0.1

# Slice all big tiles into GRID_TILE_SIZE
def slice_tiles(
    labeled_data_dir=LABELED_DATA_DIR,
    h5_output_path=H5_PATH,
    grid_tile_size=GRID_TILE_SIZE,
    min_point_num=MIN_POINT_NUM):

    files = [f for f in listdir(labeled_data_dir) if isfile(join(labeled_data_dir, f))]
    count = 0

    with h5py.File(h5_output_path, 'w') as h5f:
        tiles = h5f.create_group("tiles")
        for file in files:
            df = pd.read_csv(join(labeled_data_dir, file), sep=" ", header=None)
            data = df.to_numpy()
            xyz_max = np.array([np.max(data[:,0]), np.max(data[:,1]), np.max(data[:,2])])
            xyz_min = np.array([np.min(data[:,0]), np.min(data[:,1]), np.min(data[:,2])])
            x_min, y_min, z_min = xyz_min
            xyz_len = xyz_max - xyz_min
            xyz_grid_num = np.ceil(xyz_len / grid_tile_size)

            print("Processing Tile:",  file)
            print('Grid Shape: ', xyz_grid_num)

            processed_tiles = []
            deleted_tiles = []

            # Perform Splitting
            for x in range(int(xyz_grid_num[0])):
                bool_x_array = np.logical_and(data[:,0] <= (x_min + grid_tile_size * (x + 1)), data[:,0] >= (x_min + grid_tile_size * x))
                for y in range(int(xyz_grid_num[1])):
                    bool_y_array = np.logical_and(data[:,1] <= (y_min + grid_tile_size * (y + 1)), data[:,1] >= (y_min + grid_tile_size * y))
                    for z in range(int(xyz_grid_num[2])):
                        bool_z_array = np.logical_and(data[:,2] <= (z_min + grid_tile_size * (z + 1)), data[:,2] >= (z_min + grid_tile_size * z))
                        tile_data = data[np.logical_and(bool_x_array, np.logical_and(bool_y_array, bool_z_array))]
                        
                        # if tile contains zero points skip it
                        if tile_data.shape[0] == 0:
                            print("Skipping: tile contains 0 points.")
                            continue

                        coords = tile_data[:,0:3]
                        feats = tile_data[:,3:5]
                        labels = tile_data[:,-1]

                        tile_name = "tile_" + str(count)
                        processed_tiles.append(tile_name)
                        tile = tiles.create_group(tile_name)
                        tile.create_dataset('coords', data=coords)
                        tile.create_dataset("feats", data=feats)
                        tile.create_dataset('labels', data=labels)
                        
                        print("Number of Mini Tiles",  count)
                        count += 1


            # Perfrom tile with less than min points merging
            for tile in processed_tiles:
                if tiles[tile]["coords"].shape[0] < min_point_num:
                    print("Tile with less than minimum points:", tile)
                    coords = tiles[tile]["coords"][:]
                    feats = tiles[tile]["feats"][:]
                    labels = tiles[tile]["labels"][:]
                    xyz_max = np.array([np.max(coords[:,0]), np.max(coords[:,1]), np.max(coords[:,2])])
                    xyz_min = np.array([np.min(coords[:,0]), np.min(coords[:,1]), np.min(coords[:,2])])
                    closest_tile = None
                    closest_max = None 
                    closest_min = None 

                    # Find closest tile
                    for tile_2 in processed_tiles:
                        if tile_2 == tile or tile_2 in deleted_tiles:
                            continue
                        current_tile_data = tiles[tile_2]["coords"][:]
                        close_max = current_tile_data[KDTree(current_tile_data).query(xyz_max)[1]]
                        close_min = current_tile_data[KDTree(current_tile_data).query(xyz_min)[1]]
                        if closest_tile is None:
                            closest_tile = tile_2
                            closest_max = close_max
                            closest_min = close_min
                        else:
                            compare_max_arr = [closest_max, close_max]
                            temp_closest_max = compare_max_arr[KDTree(compare_max_arr).query(xyz_max)[1]]
                            compare_min_arr = [closest_min, close_min]
                            temp_closest_min = compare_min_arr[KDTree(compare_min_arr).query(xyz_max)[1]]
                            if not np.array_equal(temp_closest_max, closest_max) and not np.array_equal(temp_closest_min, closest_min):
                                closest_tile = tile_2
                                closest_max = temp_closest_max
                                closest_min = temp_closest_min
                    print("Closest:", closest_tile)
                    print("Merging:", tile, "to", closest_tile, "\n")
                    np.concatenate((tiles[closest_tile]["coords"][:], coords))
                    np.concatenate((tiles[closest_tile]["feats"][:], feats))
                    np.concatenate((tiles[closest_tile]["labels"][:], labels))
                    del tiles[tile]
                    deleted_tiles.append(tile)
            print("Total Merged Tiles:", str(len(deleted_tiles)))

# Split data into train/test/dev
def split_for_training(
    data_path=H5_PATH,
    h5_train=H5_TRAIN_PATH,
    h5_test=H5_TEST_PATH,
    h5_dev=H5_DEV_PATH,
    training_size=TRAINING_SIZE,
    test_size=TEST_SIZE,
    dev_size=DEV_SIZE):

    with h5py.File(data_path, 'r') as h5f:

        # Get number of tiles per split
        tiles = h5f['tiles']
        tile_keys = list(tiles.keys())
        num_tiles = len(tile_keys)
        num_train_tiles = int(num_tiles * training_size)
        num_test_tiles = int(num_tiles * test_size)
        num_validation_tiles = int(num_tiles * dev_size)
        print("Train Number of Tiles:", num_train_tiles)
        print("Test Number of Tiles:", num_test_tiles)
        print("Validation Number of Tiles:", num_validation_tiles)

        # Shuffle the data
        random.shuffle(tile_keys)

        # Write train h5py
        with h5py.File(h5_train, 'w') as train:
            train_tiles = train.create_group("tiles")
            for tile in tile_keys[:num_train_tiles]:
                tiles.copy(tiles[tile], train_tiles)
        get_label_freq(h5_train)
        print("Produced train dataset:", h5_train)
        # Write test h5py
        with h5py.File(h5_test, 'w') as test:
            test_tiles = test.create_group("tiles")
            for tile in tile_keys[num_train_tiles:num_train_tiles + num_test_tiles]:
                tiles.copy(tiles[tile], test_tiles)
        get_label_freq(h5_test)
        print("Produced test dataset:", h5_test)
        # Write dev h5py
        with h5py.File(h5_dev, 'w') as dev:
            dev_tiles = dev.create_group("tiles")
            for tile in tile_keys[num_train_tiles + num_test_tiles:]:
                tiles.copy(tiles[tile], dev_tiles)
        get_label_freq(h5_dev)
        print("Produced dev dataset:", h5_dev)

# Get the frequency of labels per dataset
def get_label_freq(data_path):
    label_freq = {}
    with h5py.File(data_path, 'r+') as h5f:
        tiles = h5f['tiles']
        tile_keys = list(tiles.keys())
        for tile in tile_keys:
            labels, freq = np.unique(tiles[tile]['labels'][:], return_counts=True)
            for i, label in enumerate(labels):
                label = str(int(label))
                if label in label_freq:
                    label_freq[label] += freq[i]
                else:
                    label_freq[label] = freq[i]
        tiles.attrs['freq'] = list(label_freq.values())
    return label_freq