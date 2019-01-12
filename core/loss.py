import torch
import torch.nn.functional as F

def cross_entropy(input, target, weight=None, ignore_index=255):
    '''
    Use 255 to fill empty values when padding or doing any augmentation operations
    like rotation. 
    '''
    target = torch.squeeze(target,dim=1)
    loss = F.cross_entropy(input, target, weight, reduction='sum',  ignore_index=255)
    return loss
