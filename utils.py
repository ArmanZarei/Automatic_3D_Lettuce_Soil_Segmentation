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

def knn(point, sampled_points, sampled_labels, K, scored_knn=True):
    distances = np.sqrt(np.sum(np.power(sampled_points - point, 2), axis=1))

    if not scored_knn:
        return np.argmax(np.bincount(sampled_labels[np.argsort(distances)[:K]]))

    scores = np.zeros(2)
    for idx in np.argsort(distances)[:K]:
        scores[sampled_labels[idx]] += 1/(distances[idx]+1e-10)

    return scores.argmax()

def get_model(model_name, device):
    if model_name == 'pointnet':
        model = PointNet().to(device)
        model.load_state_dict(torch.load('./pretrained_models/PointNet.pth', map_location=device))
    elif model_name == 'pointnet2':
        model = PointNet2(2).to(device)
        model.load_state_dict(torch.load('./pretrained_models/PointNet2.pth', map_location=device))
    elif model_name == 'randlanet':
        model = RandLANet(d_in=3, num_classes=2, num_neighbors=16, decimation=4, device=device).to(device)
        model.load_state_dict(torch.load('./pretrained_models/RandLANet.pth', map_location=device))
    else:
        raise Exception("invalid model.")
    
    return model