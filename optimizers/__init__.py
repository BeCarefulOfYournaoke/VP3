import torch
import torch.optim as optim
# from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import  _LRScheduler
import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_optimizer(args, model, criterion):
    model_params = [param for name, param in model.named_parameters() if param.requires_grad == True]
    criterion_params = [param for name, param in criterion.named_parameters() if param.requires_grad == True]


    if args.train.irScheduler.name == 'warmup':
        init_lr = args.train.init_lr/args.train.warmup.multiplier
        weight_lr = args.train.weight_lr/args.train.warmup.multiplier
    else:
        init_lr = args.train.init_lr
        weight_lr = args.train.weight_lr

    if args.train.optimizer.name == 'SGD':
        optimizer = optim.SGD([
                        {'params': model_params},
                        {'params': criterion_params, 'lr':weight_lr}],
                        lr = init_lr,
                        momentum = args.train.optimizer.momentum,
                        weight_decay = args.train.optimizer.weight_decay,
                        )
    elif args.train.optimizer.name == 'Adam':
        optimizer = optim.Adam([
                        {'params': model_params},
                        {'params': criterion_params, 'lr':weight_lr}],
                        lr = init_lr,
                        weight_decay = args.train.optimizer.weight_decay)

    else:
        raise NotImplementedError
        
    if args.train.irScheduler.name == 'warmup':
        gamma = args.train.irScheduler.gamma
        exp_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        warmup_Scheduler = GradualWarmupScheduler(optimizer, 
                                                args.train.warmup.multiplier,
                                                args.train.warmup.total_epoch, 
                                                exp_lr_scheduler)
        return optimizer, warmup_Scheduler
    elif args.train.irScheduler.name == 'exp_lr_scheduler':
        gamma = args.train.irScheduler.gamma
        exp_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        return optimizer, exp_lr_scheduler
    else:
        raise NotImplementedError
        return 


class  GradualWarmupScheduler(_LRScheduler):
    '''
         Args:
         optimizer (Optimizer): Wrapped optimizer.
         multiplier: target learning rate = base lr * multiplier
         total_epoch: target learning rate is reached at total_epoch, gradually
         after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    '''
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
         self.multiplier = multiplier
         if self.multiplier <=  1.:
             raise  ValueError('multiplier should be greater than 1.')
         self.total_epoch = total_epoch
         self.after_scheduler = after_scheduler
         self.finished =  False
         super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if  not self.finished:
                     self.after_scheduler.base_lrs =  [base_lr * self.multiplier for base_lr in self.base_lrs]
                     self.finished =  True
                # return self.after_scheduler.get_lr()
                return self.get_last_lr()
            return  [base_lr * self.multiplier for base_lr in self.base_lrs]

        return  [base_lr *  ((self.multiplier -  1.)  * self.last_epoch / self.total_epoch +  1.)  for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)