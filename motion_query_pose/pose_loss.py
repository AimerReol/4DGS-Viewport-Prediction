import math
import torch
from torch import nn

METHODS = ['AnchorGNN','AnchorNoGraph','AnchorUniformGNN','AnchorXYZRGBGNN','FixedCellGNN']

def rotvec_quat(vector):
    theta = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    coefficient = .5*torch.sinc(theta/(2*math.pi))
    return torch.cat([vector*coefficient, torch.cos(theta/2)], dim=-1)

def geodesic_radians(pred, target):
    pq,tq = rotvec_quat(pred),rotvec_quat(target)
    a,b = -tq[...,:3],pq[...,:3]
    aw,bw = tq[...,3:],pq[...,3:]
    vector = aw*b+bw*a+torch.linalg.cross(a,b)
    scalar = (aw*bw).squeeze(-1)-(a*b).sum(-1)
    return 2*torch.atan2(torch.linalg.vector_norm(vector,dim=-1),scalar.abs())

def pose_loss(pred_normalized, target_raw, output_scale, loss_scales):
    pred = pred_normalized*output_scale
    position = ((pred[...,:3]-target_raw[...,:3])**2).sum(-1).mean()/loss_scales[0]
    rotation = geodesic_radians(pred[...,3:],target_raw[...,3:]).square().mean()/loss_scales[1]
    return .5*(position+rotation)
