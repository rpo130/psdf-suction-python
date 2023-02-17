import os
import numpy as np
from scipy.spatial.transform import Rotation as R

DEVICE = "cuda:0"
EPSILON = 1e-6

class Config:
    def __init__(self):
        self.package_name = "psdf_suction"
        self.path = os.path.dirname(os.path.dirname(__file__))

        # vaccum cup
        self.gripper_radius = 0.01
        self.gripper_height = 0.02
        self.gripper_vertices = 8
        self.gripper_angle_threshold = 45
        self.vacuum_length = 0.125

        # PSDF
        self.volume_range = np.array([0.5, 0.5, 1])
        self.volume_resolution = 0.002
        self.volume_shape = np.ceil(self.volume_range / self.volume_resolution).astype(np.int32).tolist()
        self.T_volume_to_world = np.eye(4)
        self.T_volume_to_world[:3, 3] = [0.0, -0.25, 0.02]

        self.T_world_to_volume = np.eye(4)
        self.T_world_to_volume[:3, 3] = -self.T_volume_to_world[:3, 3]

config = Config()