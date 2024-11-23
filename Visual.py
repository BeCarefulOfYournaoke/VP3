import os
cuda_id = '2,3'
os.environ['CUDA_VISIBLE_DEVICES']=cuda_id

import numpy as np
import torch
import torch.nn.functional as F

from arguments import get_args
from datasets import get_dataset
from models import get_model
from myLoss import get_criterion

from sklearn.model_selection import train_test_split
import seaborn as sns
from utils.Visual_Val_Phase import Visual_Find_Top_K



def main(device, args):
    trainval_dataset = get_dataset(args.dataset.train_path,
                                    args.dataset.name, 
                                    image_size = args.dataset.image_size, 
                                    train=False, 
                                    pattern=args.dataset.pattern,
                                    )
    
    indices1, indices2 = train_test_split(range(len(trainval_dataset)), test_size=args.dataset.val_ratio , random_state=0, shuffle=True)
    train_dataset = torch.utils.data.Subset(trainval_dataset, indices1)


    train_visual_loader = torch.utils.data.DataLoader(
            dataset = train_dataset,
            num_workers=args.dataset.val_num_worker,
            shuffle=False,
            batch_size = args.val.batch_size, 
        )
    
    

    model = get_model(args)
    if torch.cuda.is_available():
        if len(cuda_id.split(','))>1:
            # model = torch.nn.DataParallel(model,device_ids=list(map(int, cuda_id.split(','))))
            model = torch.nn.DataParallel(model, device_ids=[i for i in range(len(cuda_id.split(',')))])
        model.cuda()
    criterionMulti = get_criterion(args).cuda()


    modle_save_path = r'./11-Results/118-CIFAR/ckpt/CIFAR20/Multi_branch_59_20241119_2031_model.pth'
    weight_save_path = r'./11-Results/118-CIFAR/ckpt/CIFAR20/Multi_branch_59_20241119_2031_weight.pth'

    checkpoint_model = torch.load(modle_save_path)
    model.load_state_dict(checkpoint_model['state_dict'])
    checkpoint_weight = torch.load(weight_save_path)
    criterionMulti.load_state_dict(checkpoint_weight['state_dict'])

    color_map = np.array(sns.color_palette('hls', args.dataset.class_num))

    Visual_Find_Top_K(args, model, criterionMulti, train_visual_loader, color_map)





# python Visual.py -c ./config/multiCom_Cls_Con_Cifar.yaml
if __name__ == '__main__':

    print('==================:', torch.cuda.is_available(), ' :===================')
    args = get_args(False)

    for k,v in sorted(vars(args).items()):
        print(k,'=',v)
    print()

    main(device=args.device, args=args)



