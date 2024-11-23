import os
cuda_id = '0,1'
os.environ['CUDA_VISIBLE_DEVICES']=cuda_id

import numpy as np
import torch
import torch.nn.functional as F

from arguments import get_args
from datasets import get_dataset
from models import get_model
from myLoss import get_criterion

from sklearn.model_selection import train_test_split

from tqdm import tqdm


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
            model = torch.nn.DataParallel(model,device_ids=[0,1])
        model.cuda()
    criterionMulti = get_criterion(args).cuda()


    modle_save_path = r'./01-Result/03-Travel/ckpt/Travel20/Multi_branch_59_20221014_2235_model.pth'
    weight_save_path = r'./01-Result/03-Travel/ckpt/Travel20/Multi_branch_59_20221014_2235_weight.pth'

    checkpoint_model = torch.load(modle_save_path)
    model.load_state_dict(checkpoint_model['state_dict'])
    checkpoint_weight = torch.load(weight_save_path)
    criterionMulti.load_state_dict(checkpoint_weight['state_dict'])


    #==========================计算每个模式之间的相似度=====================
    class_num = args.dataset.class_num
    pattern_num = args.model.pattern_num

    centers = F.normalize(criterionMulti.weight, p=2, dim=0)      # dim-cN*K
    sim_centers = centers.t().matmul(centers)
    selectt_sim_centers = sim_centers.masked_select(criterionMulti.select_idx).view(class_num, -1)
    


    num_data = len(train_dataset)
    #============================计算模式平均概率=========================
    model.eval()
    local_progress_find_K = tqdm(train_visual_loader)
    with torch.no_grad():
        for idx, (images, labels, paths) in enumerate(local_progress_find_K):
            local_progress_find_K.set_description(f'calculat Images Phase: {idx+1}/{len(local_progress_find_K)}')
            
            images = images[0]
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            _, fea_norm = model(images)

            weak_simMat = fea_norm.matmul(centers)                      # B-cN*K
            weak_simMat_3d = weak_simMat.reshape(-1, class_num, pattern_num)      # B-cN-K
            weak_select_sim_Mat = torch.gather(weak_simMat_3d, dim=1, 
                                    index=labels.view(-1,1,1).expand(-1,-1,pattern_num)).squeeze()   # B-K

            batch_prob = F.softmax(torch.div(weak_select_sim_Mat, args.Loss.tau), dim=1)      # B-K     

            fea_norm_tmp = weak_select_sim_Mat.detach().cpu().numpy()       
            labels_tmp = labels.detach().cpu().numpy()

            if idx == 0:
                fea_norm_list = fea_norm_tmp
                labels_list = labels_tmp
            else:
                fea_norm_list = np.append(fea_norm_list, fea_norm_tmp, axis=0)
                labels_list = np.append(labels_list, labels_tmp, axis=0)
        
        classes_list = os.listdir(args.dataset.train_path)
        classes_list.sort(key=lambda x:(x[0].zfill(2) if (x[0].isdigit() and x[1].isalpha()) else x[:2]))

        def cal_vec_var(vec):
            mean_vec = np.mean(vec, axis=0)
            l2_norm = np.sqrt(np.sum(mean_vec**2))
            mean_vec = mean_vec/l2_norm
            dist_square = [np.sum((v-mean_vec)**2) for v in vec]
            return np.mean(dist_square)

        for i,each_class in enumerate(classes_list):
            select_idx = np.where(labels_list == i)
        
            print(f'calss:{each_class} :')

            array = fea_norm_list[select_idx]
            var = cal_vec_var(array)

            print(var)

            print()


# python cal_sth.py -c ./01-Result/03-Travel/log/Travel20/in-progress_2022-1014-17-28_Multi_branch/multiCom_Cls_Con_Travel.yaml


if __name__ == '__main__':

    print('==================:', torch.cuda.is_available(), ' :===================')
    args = get_args(False)

    for k,v in sorted(vars(args).items()):
        print(k,'=',v)
    print()

    main(device=args.device, args=args)



