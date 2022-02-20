
import os
import h5py
import numpy as np

class Voxelizer:
    def __init__(self, input_path):
        self.data = np.genfromtxt(input_path, delimiter=' ')
        self.x, self.y, self.z = (0, 1, 2)


    def __call__(self, size=100):
        xyz_range = self.get_xyz_range()
        list_matrix = self.get_matrix(xyz_range)
        return self.get_3D_matrix(list_matrix, xyz_range, size)

    def get_matrix(self,  xyz_range):
        X = self.data
        low_x, low_y, low_z, high_x, high_y, high_z = xyz_range
        return X[
            (X[:, self.x] >= low_x) &
            (X[:, self.x] < high_x) &
            (X[:, self.y] >= low_y) &
            (X[:, self.y] < high_y) &
            (X[:, self.z] >= low_z) &
            (X[:, self.z] < high_z)
        ]

    def get_3D_matrix(self, full_list, xyz_range, mag_coeff):
        low_x, low_y, low_z, high_x, high_y, high_z = xyz_range
        magnifier = int(mag_coeff) / (high_x - low_x)
        dx, dy, dz = map(int, [
            (high_x - low_x) * magnifier,
            (high_y - low_y) * magnifier,
            400] # this is just an arbitrary number, heights are different in each tile thus we need to figure out the max height
        )
        full_3D_matrix = np.zeros(shape=[dx, dy, dz, self.data.shape[1]], dtype=np.float32)
        no_label_3D_matrix = np.zeros(shape=[dx, dy, dz, full_list.shape[1] - 1], dtype=np.float32)
        label = np.zeros(shape=[dx, dy, dz], dtype=np.float32)
        for point in full_list:
            x, y, z = point[[self.x, self.y, self.z]]
            full_3D_matrix[int((x - low_x) * magnifier) - 1][int((y - low_y) * magnifier) - 1][int((z - low_z) * magnifier) - 1] = point
            no_label_3D_matrix[int((x - low_x) * magnifier) - 1][int((y - low_y) * magnifier) - 1][int((z - low_z) * magnifier) - 1] = point[:7]
            label[int((x - low_x) * magnifier) - 1][int((y - low_y) * magnifier) - 1][int((z - low_z) * magnifier) - 1] = point[-1]
        return full_3D_matrix, no_label_3D_matrix, label
    
    def get_xyz_range(self):
        return [
            int(self.data[:, self.x].min()),
            int(self.data[:, self.y].min()),
            int(self.data[:, self.z].min()),
            int(self.data[:, self.x].max()),
            int(self.data[:, self.y].max()),
            int(self.data[:, self.z].max()),
        ]

def voxelize(input_dir_path, output_file_path, size=100):
    voxels = []
    labels = []
    print('Voxel Size: %s' % size)
    for file in os.listdir(input_dir_path):
        input_path = os.path.join(input_dir_path, file)  
        voxelizer = Voxelizer(input_path)
        full_3D_matrix,  no_label_3D_matrix, label = voxelizer(size)

        # this condition is needed due to some data being 10x9 meters
        if (full_3D_matrix.shape[:2] == (100, 100)):
            voxels.append(no_label_3D_matrix)
            labels.append(label)
            print('Voxelized: %s | Shape: %s' % (file, full_3D_matrix.shape))
    h5f = h5py.File(output_file_path, 'w')
    h5f.create_dataset('voxels', data = voxels)
    h5f.create_dataset('labels', data = labels)
    print('Output: %s' % (output_file_path))