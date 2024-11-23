from operator import imod
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from .Multi_Loss import MultiLossFunc

def get_criterion(args):
    if args.Loss.name == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.Loss.name == 'Multi_branch':
        criterion = MultiLossFunc(args)
    else:
        raise NotImplementedError
    return criterion
