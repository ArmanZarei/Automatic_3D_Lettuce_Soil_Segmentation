from torch.utils.data import Dataset
import os
import open3d as o3d
import numpy as np


class LettucePointCloudDataset(Dataset):
    def __init__(self, files_dir):
        self.files = []
        for f in os.listdir(files_dir):
            self.files.append(os.path.join(files_dir, f))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.files[idx]['pcd_path'])
        points = np.array(pcd.points)

        return points