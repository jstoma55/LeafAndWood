import laspy
import os 
from ..visualization import visualizasion

DATA = os.path.abspath("../data/")

class LasFile:
    def __init__(self, path):
        self.name = os.path.basename(path)
        self.data = laspy.read(os.path.join(DATA, path))
        self.points = len(self.data.points)

    def visualize(self):
        visualizasion.las_plt(self.data)

    def get_name(self):
        return self.name

    def get_data(self):
        return self.data

    def get_points(self):
        return self.points



