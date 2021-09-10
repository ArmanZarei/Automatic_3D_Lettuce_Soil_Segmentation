import numpy as np
from models.pointnet2 import PointNet2
from models.pointnet import PointNet
from models.randlanet import RandLANet
import torch


def normalize_points(points):
    points = points - points.mean(axis=0)
    points /= np.linalg.norm(points, axis=1).max()

    return points

def get_model_output(model, input):
    """
    Returns the output of the model according to model type
    Parameters:
        model (Type[nn.Module]): Model
        input (Tensor): input
    """
    if isinstance(model, PointNet):
        outputs, _, _ = model(input)
        return outputs
    elif isinstance(model, (RandLANet, PointNet2)):
        return model(input)
    
    raise Exception("Model should be of type PointNet or RandLANet or PointNet++ (PointNet2)")

def knn(point, sampled_points, sampled_labels, K):
    distances = torch.sqrt(torch.sum(torch.pow(sampled_points - point, 2), axis=1))
    scores = torch.zeros(2)
    for idx in torch.argsort(distances)[:K]:
        scores[sampled_labels[idx]] += 1/distances[idx]

    #return torch.argmax(np.bincount(sampled_labels[np.argsort(distances)[:K]]))
    return scores.argmax()