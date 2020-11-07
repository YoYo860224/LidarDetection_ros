import numpy as np
import torch

def GetTorchInputForPointNet(point_set):
    '''
    Input point_set with: (Batch, NumOfPoints, 3)
                                            └───＞ xyz
    '''
    # To Center
    point_set = point_set - np.expand_dims(np.mean(point_set, axis = 1), 1)
    # Do Scale
    distAll = np.sum(point_set ** 2, axis = -1)
    distMax = np.max(distAll, axis = -1)
    point_set = point_set / distMax[:, None, None]     
    # Torch Format
    points = torch.tensor(point_set, dtype=torch.float)
    points = points.transpose(2, 1)

    return points