import numpy as np
from models.pointnet2 import PointNet2
from models.pointnet import PointNet
from models.randlanet import RandLANet


def normalize_points(points):
    points = points - points.mean(axis=0)
    points /= np.linalg.norm(points, axis=1).max()

    return points

def get_model_output_and_loss(model, input, labels, calculate_loss=True):
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