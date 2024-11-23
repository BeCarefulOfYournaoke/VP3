from cmath import sqrt
import torch
from tqdm import tqdm
import sys
import numpy as np
import os
import shutil
import torch.nn.functional as F
from utils.TSNE_softTri import TSNE_softTri



def calculate_val_accu_loss(args, model, criterion, val_Loader, Phase = 'Val'):
    model.eval()
    #calculate val accuracy
    local_progress_val = tqdm(val_Loader, file=sys.stdout)
    with torch.no_grad():
        correct_num = 0
        classify_loss_sum = 0
        sup_con_loss_sum = 0
        for idx, (images, labels, _) in enumerate(local_progress_val):
            local_progress_val.set_description(Phase + f' Phase: {idx+1}/{len(val_Loader)}')
            
            images = torch.cat([images[0], images[1]], dim=0)
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            _, fea_norm = model(images)
            fea_norm_weak, fea_norm_strong = torch.split(fea_norm, [labels.shape[0], labels.shape[0]], dim=0)
            classfy_loss, sup_con_loss, _, _, predict_label = \
                            criterion(fea_norm_weak, fea_norm_strong, labels, cur_epoch=30)


            correct_num = correct_num + (predict_label == labels).sum().cpu().item()
            classify_loss_sum = classify_loss_sum + classfy_loss.cpu().item()
            sup_con_loss_sum = sup_con_loss_sum + sup_con_loss.cpu().item()
            local_progress_val.set_postfix(classfy_loss_ = classfy_loss.cpu().item(), 
                                                    sup_con_loss_ = sup_con_loss.cpu().item())
            
    return correct_num, classify_loss_sum, sup_con_loss_sum

def update_var(args, model, criterion, train_visual_loader):
    model.eval()
    local_progress_visual = tqdm(train_visual_loader, file=sys.stdout)
    with torch.no_grad():
        for idx, (images, labels, paths) in enumerate(local_progress_visual):
            local_progress_visual.set_description(f'update var Phase: {idx+1}/{len(local_progress_visual)}')
            
            images = images[0]
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            _, fea_norm = model(images)

            fea_norm_tmp = fea_norm.detach().cpu().numpy()       
            labels_tmp = labels.detach().cpu().numpy()
            paths_tmp = np.array(paths)

            if idx == 0:
                fea_norm_list = fea_norm_tmp
                labels_list = labels_tmp
                paths_list = paths_tmp
            else:
                fea_norm_list = np.append(fea_norm_list, fea_norm_tmp, axis=0)
                labels_list = np.append(labels_list, labels_tmp, axis=0)
                paths_list = np.append(paths_list, paths_tmp, axis=0)
            
            if idx>args.dataset.visual_ratio*len(train_visual_loader):
                break
        
        class_num = args.dataset.class_num
        var = np.zeros((class_num, 1), dtype=float)
        for i in range(class_num):
            select_idx = np.where(labels_list == i)
            select_fea_norm = fea_norm_list[select_idx]

            center = np.mean(select_fea_norm, axis=0, keepdims=True)    #1-128
            center = center/np.linalg.norm(center, ord=2,axis=1)

            diff_square = np.sum((select_fea_norm - center)**2, axis=1)
            var[i] = np.mean(diff_square)
        
        criterion.update_var(var)
    return


def visual_T_SNE(epoch, args, model, criterion, train_visual_loader, color_map, is_cal_512=False):
    model.eval()
    local_progress_visual = tqdm(train_visual_loader, file=sys.stdout)
    with torch.no_grad():
        for idx, (images, labels, paths) in enumerate(local_progress_visual):
            local_progress_visual.set_description(f'Visual T-SNE Phase: {idx+1}/{len(local_progress_visual)}')
            
            images = images[0]
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            embedding, fea_norm = model(images)

            embedding_tmp = embedding.detach().cpu().numpy() 
            fea_norm_tmp = fea_norm.detach().cpu().numpy()       
            labels_tmp = labels.detach().cpu().numpy()
            paths_tmp = np.array(paths)

            if idx == 0:
                embedding_list = embedding_tmp
                fea_norm_list = fea_norm_tmp
                labels_list = labels_tmp
                paths_list = paths_tmp
            else:
                embedding_list = np.append(embedding_list, embedding_tmp, axis=0)
                fea_norm_list = np.append(fea_norm_list, fea_norm_tmp, axis=0)
                labels_list = np.append(labels_list, labels_tmp, axis=0)
                paths_list = np.append(paths_list, paths_tmp, axis=0)
            
            if idx>args.dataset.visual_ratio*len(train_visual_loader):
                break
        TSNE_softTri(epoch, args, embedding_list, fea_norm_list, labels_list, paths_list, criterion.weight, color_map, is_cal_512)
    return


def find_top_K_images(epoch, args, model, criterion, train_Loader):
    model.eval()
    local_progress_find_K = tqdm(train_Loader, file=sys.stdout)
    with torch.no_grad():
        for idx, (images, labels, paths) in enumerate(local_progress_find_K):
            local_progress_find_K.set_description(f'Find Top-K Images Phase: {idx+1}/{len(local_progress_find_K)}')
            
            images = images[0]
            images = images.cuda()
            labels = labels.cuda()

            _, fea_norm = model(images)

            fea_norm_tmp = fea_norm.detach().cpu().numpy()       
            labels_tmp = labels.detach().cpu().numpy()
            paths_tmp = np.array(paths)

            if idx == 0:
                fea_norm_list = fea_norm_tmp
                labels_list = labels_tmp
                paths_list = paths_tmp
            else:
                fea_norm_list = np.append(fea_norm_list, fea_norm_tmp, axis=0)
                labels_list = np.append(labels_list, labels_tmp, axis=0)
                paths_list = np.append(paths_list, paths_tmp, axis=0)

        #计算Top-K图像
        class_num = args.dataset.class_num
        pattern_num = args.model.pattern_num   
        Top_K_num = args.model.Top_K_num
        class_list = os.listdir(args.dataset.train_path)
        class_list.sort(key=lambda x:(x[0].zfill(2) if (x[0].isdigit() and x[1].isalpha()) else x[:2]))
        #权重归一化 weight: dim, cN*K
        weight_norm  = F.normalize(criterion.weight, p=2, dim=0).detach().cpu()
        weight_norm_3d = weight_norm.reshape(-1, class_num, pattern_num)


        for i in range(class_num):
            select_idx = np.where(labels_list == i)
            select_fea_norm = fea_norm_list[select_idx]
            select_feature = torch.tensor(select_fea_norm)
            select_feature = F.normalize(select_feature, p=2, dim=1)

            weight_select = weight_norm_3d[:, i, :].permute(1, 0)
            cos_match_mat = F.cosine_similarity( weight_select.unsqueeze(1), select_feature.unsqueeze(0), dim=-1)

            _, index = cos_match_mat.topk(Top_K_num, dim=1)
            index = index.cpu().numpy()
            assert pattern_num==len(index)
            assert args.model.Top_K_num==len(index[0])

            for j in range(args.model.pattern_num):
                save_dir = f'./T-SNE/{args.dataset.name}-Retrieval/{class_list[i]}-Pattern-{j}'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                for k in range(args.model.Top_K_num):
                    img_path = paths_list[select_idx[0][index[j][k]]]
                    if '\\' in img_path:
                        img_name =img_path.split('\\')[-1]
                    else:
                        img_name =img_path.split('/')[-1]
                    shutil.copyfile(img_path, os.path.join(save_dir, str(epoch).zfill(3) + '-' + img_name))
    return


                    











