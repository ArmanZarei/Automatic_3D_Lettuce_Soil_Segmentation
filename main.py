from models.pointnet import PointNet
from utils import get_model_output, knn
from torch.utils import data
from dataset import LettucePointCloudDataset
import torch
from models.pointnet2 import PointNet2
import numpy as np
import multiprocessing as mp
import time


n_samples = 1500
K = 5

dataset = LettucePointCloudDataset(files_dir='./data')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}\n{"-"*30}')

model = PointNet2(2).to(device)
model.load_state_dict(torch.load('./pretrained_models/PointNet2.pth'))

for (f_path, points) in dataset:
    sampled_indices = np.random.choice(points.shape[0], n_samples)
    sampled_indices_lookup = set(sampled_indices)
    sampled_points = points[sampled_indices]
    sampled_labels = get_model_output(model, sampled_points.float().unsqueeze(0).to(device)).argmax(1).squeeze().cpu()

    labels = torch.zeros(points.shape[0], dtype=int)
    labels[sampled_indices] = sampled_labels


    p = mp.Pool(2) # mp.cpu_count() - 2
    start_time = time.time()
    print("START")
    other_labels = p.starmap(knn, [(points[i], sampled_points, labels, K) for i in range(points.shape[0]) if i not in sampled_indices_lookup])
    print('Time:', time.time()-start_time)
    others_indices_mask = np.ones(points.shape[0], dtype=bool)
    others_indices_mask[sampled_indices] = False

    labels[others_indices_mask] = other_labels

    print(labels.shape)
    np.save(f_path.replace('.ply', '.npy'), labels)
    exit()