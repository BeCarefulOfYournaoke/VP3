import numpy as np
import matplotlib.pyplot as plt 
import os
import shutil
import torch
import torch.nn.functional as F
import warnings

from sklearn.manifold import TSNE
from tqdm import tqdm





def Visual_Find_Top_K(args, model, criterion, train_visual_loader, color_map):
    model.eval()
    local_progress_find_K = tqdm(train_visual_loader)
    with torch.no_grad():
        for idx, (images, labels, paths) in enumerate(local_progress_find_K):
            local_progress_find_K.set_description(f'Find Top-K Images Phase: {idx+1}/{len(local_progress_find_K)}')
            
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
            
            if idx==int(args.dataset.visual_ratio*len(train_visual_loader)):
                embedding_list_TSNE = embedding_list[:]
                fea_norm_list_TSNE = fea_norm_list[:]
                labels_list_TSNE = labels_list[:]
                # paths_list_TSNE = paths_list[:]

        weight_dict = Find_Top_K(args, fea_norm_list, labels_list, 
                                    paths_list, criterion.weight, num_ratio=args.Loss.num_ratio)
        visual_T_SNE(args, embedding_list_TSNE, fea_norm_list_TSNE, labels_list_TSNE, 
                        weight_dict, color_map, False)
        return 

        


def Find_Top_K(args, fea_norm_list, labels_list, paths_list, weight, num_ratio=0.1):
    class_num = args.dataset.class_num
    pattern_num = args.model.pattern_num
    Top_K_num = args.model.Top_K_num

    classes_list = os.listdir(args.dataset.train_path)
    classes_list.sort(key=lambda x:(x[0].zfill(2) if (x[0].isdigit() and x[1].isalpha()) else x[:2]))

    #============================================================================
    featureList = torch.tensor(fea_norm_list).cpu()
    featureList = F.normalize(featureList, p=2, dim=1)      #train_num, dim

    # weight: dim, cN*K
    weight_norm  = F.normalize(weight, p=2, dim=0).detach().cpu()
    weight_norm_3d = weight_norm.reshape(-1, class_num, pattern_num)
    #============================================================================

    #==========================================================================
    pattern_dict = {}
    for i,each_class in enumerate(classes_list):
        select_idx = np.where(labels_list == i)
        select_feature = featureList[select_idx]
        num_each_class = len(select_feature)
        select_weight = weight_norm_3d[:, i, :]     #dim, K

        sim_Mat = select_feature.matmul(select_weight)       #num, K
        prob_Mat = F.softmax(torch.div(sim_Mat, args.Loss.tau), dim=1)      #num, K

        prob, pattern_idx = prob_Mat.max(dim=1)   #num,
        pattern_set = []
        for j in range(pattern_num):
            prob_select = torch.masked_select(prob, pattern_idx==j)
            yes_pattern_num = prob_select.greater_equal(args.Loss.visual_prob_thresh).sum()
            # yes_pattern_num = prob_select.greater_equal(0.7).sum()
            if ((yes_pattern_num/num_each_class)>=num_ratio):
                pattern_set.append(j)
        pattern_dict[f'class-{i}'] = select_weight.t().index_select(dim=0, index=torch.tensor(pattern_set, dtype=torch.int32)).numpy()
        print(f'{each_class} valid pattern num is {len(pattern_set)}')
        #==========================================================================

        #
        #==========================================================================
        cos_match_mat = F.cosine_similarity( select_weight.t()[pattern_set].unsqueeze(1), select_feature.unsqueeze(0), dim=-1)
        _, index = cos_match_mat.topk(Top_K_num, dim=1)

        index = index.numpy()
        assert len(pattern_set)==len(index)
        # assert Top_K_num==len(index[0])

        for j in range(len(pattern_set)):
            save_dir = f'./Visual/{args.dataset.name}-Retrieval/{classes_list[i]}-Pattern-{j}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for k in range(Top_K_num):
                img_path = paths_list[select_idx[0][index[j][k]]]
                if '\\' in img_path:
                    img_name =img_path.split('\\')[-1]
                else:
                    img_name =img_path.split('/')[-1]
                shutil.copyfile(img_path, os.path.join(save_dir, img_name))
                with open('./Visual/pattern_label_annotate.txt', 'a') as f:
                    f.write(img_path + ' ')
                    f.write(each_class + ' ')
                    f.write(str(j))
                    f.write('\n')
        #==========================================================================
    return pattern_dict



def visual_T_SNE(args, embedding_list, fea_norm_list, labels_list, weight_dict, color_map, is_cal_512):
    class_num = args.dataset.class_num
    pattern_num = args.model.pattern_num
    classes_list = os.listdir(args.dataset.train_path)
    classes_list.sort(key=lambda x:(x[0].zfill(2) if (x[0].isdigit() and x[1].isalpha()) else x[:2]))

    #=======================================================================
    if is_cal_512:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # tsne_embedding = TSNE(n_components=2, init='pca')
            tsne_embedding = TSNE(n_components=2)
            embed = tsne_embedding.fit_transform(embedding_list)
        fig = plt.figure(figsize=(8, 8))
        for i,each_class in enumerate(classes_list):
            select_idx = np.where(labels_list == i)
            select_samples = embed[select_idx]
            # print(f'{each_class} has {len(select_samples)} samples')
            plt.scatter(select_samples[:, 0], select_samples[:, 1], lw=0, s=10, c=color_map[i].reshape(1,3))
        # plt.axis('off')
        plt.axis('tight')

        if not os.path.exists(f'./Visual/{args.dataset.name}-512'):
            os.makedirs(f'./Visual/{args.dataset.name}-512')
        figdir = os.path.join(f'./Visual/{args.dataset.name}-512', args.name + '.png')
        plt.savefig(figdir, dpi=600)
        plt.close()
        del embed
    del embedding_list
    #=======================================================================
    
    #=======================================================================
    featureList = torch.tensor(fea_norm_list)
    featureList = F.normalize(featureList, p=2, dim=1)

    #eight: dim, cN*K
    weight_norm_3d_split = torch.cat([torch.tensor(weight_dict[f'class-{i}']) for i in range(class_num)], dim=0)
    total_pattern_num = len(weight_norm_3d_split)
    feature = torch.cat([weight_norm_3d_split, featureList], dim=0).numpy()
    #=======================================================================

    #====================================TSNE===================================
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne = TSNE(n_components=2, init='pca')
        tsne = TSNE(n_components=2)
        feature_encode = tsne.fit_transform(feature)
    
    #画总的分布图
    fig = plt.figure(figsize=(8, 8))
    for i,each_class in enumerate(classes_list):
        select_idx = tuple(np.array(np.where(labels_list == i)) + total_pattern_num)
        select_samples = feature_encode[select_idx]
        plt.scatter(select_samples[:, 0], select_samples[:, 1], lw=0, s=10, c=color_map[i].reshape(1,3))

    select_weight = feature_encode[: total_pattern_num, :]
    plt.scatter(select_weight[:, 0], select_weight[:, 1], lw=0, s=80, marker = '*', c='black')
    # plt.axis('off')
    plt.axis('tight')

    if not os.path.exists(f'./Visual/{args.dataset.name}-inter'):
        os.makedirs(f'./Visual/{args.dataset.name}-inter')
    figdir = os.path.join(f'./Visual/{args.dataset.name}-inter', args.name + '.png')
    plt.savefig(figdir, dpi=600)
    plt.close()
    #====================================TSNE===================================

    #====================================TSNE===================================

    for i,each_class in enumerate(classes_list):
        fig = plt.figure(figsize=(4, 4))
        select_idx = np.where(labels_list == i)
        select_feature = featureList[select_idx]
        select_weight = weight_dict[f'class-{i}']
        num_select_weight = len(select_weight)
        concat_fea = np.concatenate([select_weight, select_feature], axis=0)
    
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tsne = TSNE(n_components=2)
            concat_fea_encode = tsne.fit_transform(concat_fea)
        
        plt.scatter(concat_fea_encode[num_select_weight:, 0], concat_fea_encode[num_select_weight:, 1], lw=0, s=10, c=color_map[i].reshape(1,3))
        plt.scatter(concat_fea_encode[:num_select_weight, 0], concat_fea_encode[:num_select_weight, 1], lw=0, s=80, marker = '*', c='black')
        # plt.axis('off')
        plt.axis('tight')

        if not os.path.exists(f'./Visual/{args.dataset.name}-intra'):
            os.makedirs(f'./Visual/{args.dataset.name}-intra')
        figdir = os.path.join(f'./Visual/{args.dataset.name}-intra', args.name + str(each_class) +'.png')
        plt.savefig(figdir, dpi=500)
        plt.close()
    #====================================TSNE===================================
    
    return 


