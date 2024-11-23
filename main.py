# from cProfile import label
import os
cuda_id = '2,3'
os.environ['CUDA_VISIBLE_DEVICES']=cuda_id
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
import myLoss
from tqdm import tqdm
import sys
import seaborn as sns
from arguments import get_args
#from augmentations import get_aug
from models import get_model
from datasets import get_dataset
from optimizers import get_optimizer
from datetime import datetime
from datasets.getData import MyDataSet
from myLoss import get_criterion
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import KernelPCA
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from utils import ops
# from utils.angularNorm import angularNorm
# from utils.getWeight import getweight
# from utils.divAvg import divAvg

np.set_printoptions(threshold=np.inf)



def main(device, args):
    
    trainval_dataset = get_dataset(args.dataset.train_path,
                                    args.dataset.name, 
                                    image_size = args.dataset.image_size, 
                                    train=True, 
                                    pattern=args.dataset.pattern,
                                    )
    
    visual_dataset = get_dataset(args.dataset.train_path,
                                    args.dataset.name, 
                                    image_size = args.dataset.image_size, 
                                    train=False, 
                                    pattern=args.dataset.pattern,
                                    )


    test_dataset = get_dataset(args.dataset.test_path,
                                    args.dataset.name, 
                                    image_size = args.dataset.image_size, 
                                    train=False, 
                                    pattern=args.dataset.pattern,
                                    )
                                    
    indices1, indices2 = train_test_split(range(len(trainval_dataset)), test_size=args.dataset.val_ratio , random_state=0, shuffle=True)
    # train_dataset = torch.utils.data.Subset(trainval_dataset, indices1)
    train_dataset = trainval_dataset
    val_dataset = torch.utils.data.Subset(trainval_dataset, indices2)

    visual_dataset = torch.utils.data.Subset(visual_dataset, indices1)
    visual_val_dataset = torch.utils.data.Subset(visual_dataset, indices2)


    train_Loader = torch.utils.data.DataLoader(
            dataset = train_dataset,
            num_workers=args.dataset.num_workers,
            shuffle=True,
            batch_size = args.train.batch_size,
            drop_last = True,
        )
    
    val_Loader = torch.utils.data.DataLoader(
            dataset = val_dataset,
            num_workers=args.dataset.val_num_worker,
            shuffle=False,
            batch_size = args.val.batch_size
        )
    
    visual_loader = torch.utils.data.DataLoader(
            dataset = visual_dataset,
            num_workers=args.dataset.val_num_worker,
            shuffle=False,
            batch_size = args.val.batch_size,
        )

    cal_var_loader = torch.utils.data.DataLoader(
            dataset = visual_val_dataset,
            num_workers=args.dataset.val_num_worker,
            shuffle=False,
            batch_size = args.val.batch_size,
        )

    test_Loader = torch.utils.data.DataLoader(
            dataset = test_dataset,
            num_workers=args.dataset.val_num_worker,
            shuffle=False,
            batch_size = args.val.batch_size
        )
    

    #define model
    model = get_model(args)
    if torch.cuda.is_available():
        if len(cuda_id.split(','))>1:
            model = torch.nn.DataParallel(model, device_ids=[i for i in range(len(cuda_id.split(',')))])
        model.cuda()
    
    # print(model)
    # return

    #def loss
    criterionMulti = get_criterion(args).cuda()
    
    #define optimizer
    optimizer, lr_Scheduler = get_optimizer(args, model, criterionMulti)
    
    #tensorboardX
    writer = SummaryWriter(comment=(args.model.name))
    
    #def DictFea
    print()
    print(f'Total Train num is {len(train_dataset)} - val num is {len(val_dataset)} - test num is {len(test_dataset)}')
    print()
    color_map = np.array(sns.color_palette('hls', args.dataset.class_num))

    #===============================star training===================
    global_progress = tqdm(range(args.train.stop_at_epoch), desc=f'Training', file=sys.stdout)
    for epoch in global_progress:
        tqdm.write(f'==========================Epoch:{epoch} start ============================')
        #===================================train phase=========================
        local_progress_train = tqdm(train_Loader, file=sys.stdout)
        model.train()
        classify_loss_sum = 0
        sup_con_loss_sum = 0
        reg_loss_sum = 0
        reg_loss_diff_sum = 0

        for idx, (images, labels, paths) in enumerate(local_progress_train):
            local_progress_train.set_description(f'train Phase: {idx+1}/{len(train_Loader)}')
            optimizer.zero_grad()

            images = torch.cat([images[0], images[1]], dim=0)
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            _, fea_norm = model(images)
            fea_norm_weak, fea_norm_strong = torch.split(fea_norm, [labels.shape[0], labels.shape[0]], dim=0)

            classfy_loss, sup_con_loss, reg_loss, reg_loss_diff, _ = \
                        criterionMulti(fea_norm_weak, fea_norm_strong, labels, epoch, True)
            
            (classfy_loss + args.Loss.alpha_SupCon*sup_con_loss + \
                args.Loss.beta_reg_1*reg_loss + args.Loss.beta_reg_2*reg_loss_diff).backward()


            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                classify_loss_sum = classify_loss_sum + classfy_loss.item()
                sup_con_loss_sum = sup_con_loss_sum + sup_con_loss.item()
                reg_loss_sum = reg_loss_sum + reg_loss.item()
                reg_loss_diff_sum = reg_loss_diff_sum + reg_loss_diff.item()
                local_progress_train.set_postfix(classfy_loss_ = classfy_loss.cpu().item(), 
                                                    sup_con_loss_ = sup_con_loss.cpu().item(),
                                                    reg_loss_ = reg_loss.cpu().item(),
                                                    reg_loss_diff_ = reg_loss_diff.cpu().item())
        lr_Scheduler.step()

        classify_loss_avg = torch.true_divide(classify_loss_sum, len(train_Loader))
        sup_con_loss_avg = torch.true_divide(sup_con_loss_sum, len(train_Loader))
        reg_loss_avg = torch.true_divide(reg_loss_sum, len(train_Loader))
        reg_loss_diff_avg = torch.true_divide(reg_loss_diff_sum, len(train_Loader))
        writer.add_scalar('classfy_loss_avg', classify_loss_avg, global_step=epoch)
        writer.add_scalar('sup_con_loss_avg', sup_con_loss_avg, global_step=epoch)
        writer.add_scalar('reg_loss_avg', reg_loss_avg, global_step=epoch)
        writer.add_scalar('reg_loss_diff_avg', reg_loss_diff_avg, global_step=epoch)

        tqdm.write(' ')
        tqdm.write(f' Train: Epoch-{epoch}--classify_loss_avg:{classify_loss_avg}--   sup_con_loss_avg:{sup_con_loss_avg}==\
            reg_loss_avg:{reg_loss_avg}--    reg_loss_diff_avg:{reg_loss_diff_avg}')



        if (epoch <= 4):
            ops.update_var(args, model, criterionMulti, visual_loader)  
        tqdm.write(' ')
        print('class_var', criterionMulti.class_var.view(-1))
        tqdm.write(' ')


        #=====================================================================
        if (epoch>= args.train.stop_at_epoch-11) & ((epoch+1)%5 == 0) :
            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)
            model_path = os.path.join(args.ckpt_dir, f"{args.name}_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M')}_model.pth")
            torch.save({
                'epoch': epoch+1,
                'state_dict':model.state_dict()}, 
                model_path)
            # tqdm.write(f"Model saved to {model_path}")
            with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
                f.write(f'{model_path}')
            
            weight_path = os.path.join(args.ckpt_dir, f"{args.name}_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M')}_weight.pth")
            torch.save({
                'epoch': epoch+1,
                'state_dict':criterionMulti.state_dict()}, 
                weight_path)
        
        if ((epoch+1) % 5 == 0):
            # ==========================================test ==============================
            tqdm.write('============================ start val==========================')
            correct_num, val_classify_loss_sum, val_sup_con_loss_sum= \
                        ops.calculate_val_accu_loss(args, model, criterionMulti, val_Loader, 'Val')
            val_classify_lossAVG = torch.true_divide(val_classify_loss_sum, len(val_Loader))
            val_sup_con_lossAVG = torch.true_divide(val_sup_con_loss_sum, len(val_Loader))
            val_accuracy = torch.true_divide(correct_num, len(val_dataset))
        
            writer.add_scalar('val_classfy_lossAVG', val_classify_lossAVG, global_step=epoch)
            writer.add_scalar('val_sup_con_lossAVG', val_sup_con_lossAVG, global_step=epoch)
            writer.add_scalar('val_accu', val_accuracy, global_step=epoch)

            # tqdm.write(f'Val Phase: Epoch-{epoch} :val_lossAVG: {val_lossAVG.item()}')
            tqdm.write(f'Val accuracy is {val_accuracy.item()}')
        
        # start visualization TSNE
        if ((epoch+1) % 5 == 0):
            ops.visual_T_SNE(epoch, args, model, criterionMulti, visual_loader, color_map, False)
        # tqdm.write('*****************end val phase**********************')
        tqdm.write(' ')
    
    
    
# python main.py -c ./config/multiCom_Cls_Con_Cifar.yaml  
# python main.py -c ./config/multiCom_Cls_Con_Travel.yaml  
# python main.py -c ./config/multiCom_Cls_Con_Place.yaml 
# python main.py -c ./config/multiCom_Cls_Con_ILSVRC.yaml

if __name__ == '__main__':
    
    print('==================:', torch.cuda.is_available(), ' :===================')
    args = get_args()

    for k,v in sorted(vars(args).items()):
        print(k,'=',v)
    print()
    
    main(device=args.device, args=args)
        
