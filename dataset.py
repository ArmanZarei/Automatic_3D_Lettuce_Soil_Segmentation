from numpy.linalg.linalg import norm
from utils import normalize_points
from torch.utils.data import Dataset
import os
import open3d as o3d
import numpy as np
from utils import normalize_points
import torch


class LettucePointCloudDataset(Dataset):
    def __init__(self, files_dir):
        self.files = []
        for f in os.listdir(files_dir):
            self.files.append((f.replace('.ply', ''), os.path.join(files_dir, f)))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        f, f_path = self.files[idx]
        pcd = o3d.io.read_point_cloud(f_path)
        points = torch.from_numpy(np.array(pcd.points))

        return f, normalize_points(points)