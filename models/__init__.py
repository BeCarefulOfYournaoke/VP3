# from .singleNetwork import singleNet
# from .triNet import triNet
# from .multiComNet import multiNet
from torchvision import models
from .multi_forClsCon import multiCom_VGG19, multiCom_res50




def get_model(args):
    if args.name == 'Multi_branch':
        if args.model.backbone == 'vgg19':
            net = multiCom_VGG19( args.model.pretrain_path,
                                    models.vgg19(pretrained = False), 
                                    args.model.embed_size, 
                                    args.model.fea_size
                                    )
        elif args.model.backbone == 'res50':
            net = multiCom_res50(args.model.pretrain_path,
                                    models.resnet50(pretrained = False),
                                    args.model.embed_size, 
                                    args.model.fea_size
                                    )
    else:
        # assert True, 'error name of Multi_branch at model.__init__.py'
        raise
    return net


