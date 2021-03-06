from models.pointnet import PointNet
from utils import get_model_output, knn, get_model
from torch.utils import data
from dataset import LettucePointCloudDataset
import torch
import numpy as np
import multiprocessing as mp
import time
import argparse
from tqdm import tqdm
import os


if not os.path.isdir('./data'):
    raise Exception("./data dir does not exist.")


parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default=51)
parser.add_argument('--n_samples', type=str, default=1500)
parser.add_argument('--model', type=str, default='dgcnn', choices=['pointnet', 'pointnet2', 'randlanet', 'dgcnn'])
args = parser.parse_args()

dataset = LettucePointCloudDataset(files_dir='./data')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

model = get_model(args.model, device)
print(f'Model: {type(model).__name__}\n{"-"*15}')
model.eval()

print("Segmenting PointClouds...")
for (f_path, points) in tqdm(dataset):
    sampled_indices = np.random.choice(points.shape[0], args.n_samples, replace=False)
    sampled_indices_lookup = set(sampled_indices)
    sampled_points = points[sampled_indices]
    sampled_labels = get_model_output(model, torch.from_numpy(sampled_points).float().unsqueeze(0).to(device)).argmax(1).squeeze().cpu().numpy()
    
    labels = np.zeros(points.shape[0], dtype=int)
    labels[sampled_indices] = sampled_labels

    p = mp.Pool(mp.cpu_count() - 2)

    start_time = time.time()
    other_labels = p.starmap(knn, [(points[i], sampled_points, sampled_labels, args.K) for i in range(points.shape[0]) if i not in sampled_indices_lookup])
    # print('KNN Log ===> Shape:', points.shape, '----', 'Time:', time.time()-start_time)

    others_indices_mask = np.ones(points.shape[0], dtype=bool)
    others_indices_mask[sampled_indices] = False

    labels[others_indices_mask] = other_labels

    np.save(f_path.replace('.ply', '.npy'), labels)
