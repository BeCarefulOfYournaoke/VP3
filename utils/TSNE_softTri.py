import pathlib
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt 
import os
import matplotlib
import torch
import torch.nn.functional as F
import warnings
from sklearn.preprocessing import StandardScaler



def TSNE_softTri (epoch, args, embedding_list, fea_norm_list, labels_list, paths_list, weight, color_map, is_cal_512):
    class_num = args.dataset.class_num
    pattern_num = args.model.pattern_num

    #读取数据
    datadir = args.dataset.train_path
    classes_list = os.listdir(datadir)
    classes_list.sort(key=lambda x:(x[0].zfill(2) if (x[0].isdigit() and x[1].isalpha()) else x[:2]))


    #计算512维度的特征分布
    if is_cal_512:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # tsne_embedding = TSNE(n_components=2, init='pca')
            tsne_embedding = TSNE(n_components=2)
            embed = tsne_embedding.fit_transform(embedding_list)
        fig = plt.figure(figsize=(8, 8))
        for i,each_class in enumerate(classes_list):
            # select_idx = np.where(paths_list == each_class)
            select_idx = np.where(labels_list == i)
            select_samples = embed[select_idx]
            # print(f'{each_class} has {len(select_samples)} samples')
            plt.scatter(select_samples[:, 0], select_samples[:, 1], lw=0, s=10, c=color_map[i].reshape(1,3))
        plt.axis('off')
        plt.axis('tight')

        if not os.path.exists(f'./T-SNE/{args.dataset.name}-512'):
            os.makedirs(f'./T-SNE/{args.dataset.name}-512')
        figdir = os.path.join(f'./T-SNE/{args.dataset.name}-512', args.name + '-' + str(epoch).zfill(3)  +'.png')
        plt.savefig(figdir, dpi=600)
        plt.close()
        del embed
    del embedding_list

    #处理归一化的特征
    featureList = torch.tensor(fea_norm_list)
    featureList = F.normalize(featureList, p=2, dim=1)

    #权重归一化 weight: dim, cN*K
    weight_norm  = F.normalize(weight, p=2, dim=0).detach().cpu()
    weight_norm_3d = weight_norm.reshape(-1, class_num, pattern_num)
    
    # 计算
    weight_norm_3d_split = torch.cat([weight_norm_3d[:, i, :].permute(1, 0) for i in range(class_num)], dim=0)
    feature = torch.cat([weight_norm_3d_split, featureList], dim=0).cpu().numpy()


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # tsne = TSNE(n_components=2, init='pca', metric='cosine')
        # tsne = TSNE(n_components=2, metric='cosine')
        # tsne = TSNE(n_components=2, metric='cosine')
        tsne = TSNE(n_components=2)
        feature_encode = tsne.fit_transform(feature)
    
    #画总的分布图
    fig = plt.figure(figsize=(8, 8))
    # ax = plt.subplot(aspect = 'equal')
    for i,each_class in enumerate(classes_list):
        select_idx = tuple(np.array(np.where(labels_list == i)) + class_num*pattern_num)
        select_samples = feature_encode[select_idx]
        select_weight = feature_encode[pattern_num*i: pattern_num*(i+1), :]

        plt.scatter(select_samples[:, 0], select_samples[:, 1], lw=0, s=10, c=color_map[i].reshape(1,3))
        # plt.scatter(select_weight[:, 0], select_weight[:, 1], lw=0, s=80, marker = '*', c=color_map[i].reshape(1,3)*1.1)
        plt.scatter(select_weight[:, 0], select_weight[:, 1], lw=0, s=80, marker = '*', c='black')
    plt.axis('off')
    plt.axis('tight')

    if not os.path.exists(f'./T-SNE/{args.dataset.name}-inter'):
        os.makedirs(f'./T-SNE/{args.dataset.name}-inter')
    figdir = os.path.join(f'./T-SNE/{args.dataset.name}-inter', args.name + '-' + str(epoch).zfill(3)  +'.png')
    plt.savefig(figdir, dpi=600)
    plt.close()




    if (epoch == (args.train.stop_at_epoch - 1) ):
        for i,each_class in enumerate(classes_list):
            fig = plt.figure(figsize=(4, 4))
            select_idx = np.where(labels_list == i)
            select_feature = featureList[select_idx]
            select_weight = weight_norm_3d_split[pattern_num*i: pattern_num*(i+1), :]
            concat_fea = np.concatenate([select_weight, select_feature], axis=0)
        
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tsne = TSNE(n_components=2, metric='cosine')
                concat_fea_encode = tsne.fit_transform(concat_fea)
            
            plt.scatter(concat_fea_encode[pattern_num:, 0], concat_fea_encode[pattern_num:, 1], lw=0, s=10, c=color_map[i].reshape(1,3))
            plt.scatter(concat_fea_encode[:pattern_num, 0], concat_fea_encode[:pattern_num, 1], lw=0, s=80, marker = '*', c='black')
            plt.axis('off')
            plt.axis('tight')

            if not os.path.exists(f'./T-SNE/{args.dataset.name}-intra'):
                os.makedirs(f'./T-SNE/{args.dataset.name}-intra')
            figdir = os.path.join(f'./T-SNE/{args.dataset.name}-intra', args.name + '-' + str(epoch).zfill(3) + '-' + str(each_class) +'.png')
            plt.savefig(figdir, dpi=500)
            plt.close()
    

    return 







