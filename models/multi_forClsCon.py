from __future__ import print_function
from __future__ import division
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F


class multiCom_res50(nn.Module):
    def __init__(self, pretrain_path, backbone, embed_size, fea_size) -> None:
        super(multiCom_res50, self).__init__()
        #load pth
        state_dict = torch.load(pretrain_path)
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        backbone.load_state_dict(state_dict, strict=False)

        self.encoder = nn.Sequential(*list(backbone.children())[:9])
        self.fc_1 = nn.Sequential(
                                    nn.Flatten(),
                                    nn.Linear(in_features=backbone.fc.in_features, out_features=embed_size),
                                    nn.ReLU(inplace = True),
                                    # nn.Dropout(p=0.5),
                                )

        self.fc_2 = nn.Sequential(
                                    nn.Linear(in_features=embed_size, out_features=fea_size),
                                )
        
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        fc_1 = self.fc_1(x)
        fc_2 = self.fc_2(fc_1)
        fc_2_norm = F.normalize(fc_2, p=2, dim=1)
        return fc_1, fc_2_norm
        # return fc_2_norm

       

class multiCom_VGG19(nn.Module):
    def __init__(self, pretrain_path, backbone, embed_size, fea_size):
        super(multiCom_VGG19, self).__init__()

        #load pth
        state_dict = torch.load(pretrain_path)
        del state_dict['classifier.0.weight']
        del state_dict['classifier.0.bias']
        del state_dict['classifier.3.weight']
        del state_dict['classifier.3.bias']
        del state_dict['classifier.6.weight']
        del state_dict['classifier.6.bias']
        backbone.load_state_dict(state_dict, strict=False)

        #print(model.features)
        self.vgg_layer_before = nn.Sequential(*list(backbone.features.children())[:36],)
        self.vgg_layer_after = nn.Sequential(
                                             nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                                            )

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=512, out_features=fea_size)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.vgg_layer_before(x)
        x = self.vgg_layer_after(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), 512)

        midfea = x

        x = self.linear_layer(x)
        x = F.normalize(x, p=2, dim=1)
        return midfea, x



