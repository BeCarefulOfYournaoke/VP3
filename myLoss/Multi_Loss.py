# Implementation of SoftTriple Loss
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MultiLossFunc(nn.Module):
    def __init__(self, args):
        super(MultiLossFunc, self).__init__()
        
        self.temperature1 = args.Loss.temperature1
        self.temperature2 = args.Loss.temperature2
        self.tau = args.Loss.tau
        self.weight_detach = args.Loss.weight_detach
        self.total_epoch = args.train.stop_at_epoch

        self.cN = args.dataset.class_num
        self.K = args.model.pattern_num
        self.dim = args.model.fea_size

        self.prob_threshold = args.Loss.prob_threshold

        self.select_idx = torch.zeros(self.cN*self.K, self.cN*self.K, dtype=torch.bool).cuda() # cN*K-cN*K  
        for i in range(0, self.cN):
            for j in range(0, self.K):
                self.select_idx[i*self.K+j, i*self.K+j+1:(i+1)*self.K] = 1
        
        # calculate the variance in each class, upadte parameter by Moving Acerage to reduce computing complexity
        self.momentum_class_p = args.Loss.momentum_class_p
        self.class_var = torch.ones((self.cN, 1), dtype=torch.float).cuda()

        # initalize K pattern prototype in each class
        self.weight = Parameter(torch.FloatTensor(self.dim, self.cN*self.K))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(self.K))
        return

    # #=============================================================================
    def forward(self, weak_aug, strong_aug , labels, cur_epoch=0, isTrain=False):
        centers = F.normalize(self.weight, p=2, dim=0)      # dim-cN*K

        # claculate L_DF
        classify_loss, predict_labels = self.cal_classify_loss(weak_aug, centers, labels)

        #========================================
        weak_simMat = weak_aug.matmul(centers)                      # B-cN*K
        weak_simMat_3d = weak_simMat.reshape(-1, self.cN, self.K)      # B-cN-K
        weak_select_sim_Mat = torch.gather(weak_simMat_3d, dim=1, 
                                index=labels.view(-1,1,1).expand(-1,-1,self.K)).squeeze()   # B-K
        #========================================

        #calculate tow reg terms, reg1 and reg2
        reg_loss = self.cal_reg_loss(weak_select_sim_Mat, labels, cur_epoch)    
        reg_loss_diff = self.cal_reg_loss_diff(centers)

        # calculate L_w2ps
        # first 5 epoch for warming up by contrastive loss
        if cur_epoch<5:
            #等价计算 method == 'SimCLR'
            features = torch.cat([weak_aug.unsqueeze(1), strong_aug.unsqueeze(1)], dim=1)
            sup_con_loss = self.cal_con_loss(features, self.temperature2)
            return classify_loss, sup_con_loss, reg_loss, reg_loss_diff, predict_labels
        # L_w2ps loss
        else:
            features = torch.cat([weak_aug.unsqueeze(1), strong_aug.unsqueeze(1)], dim=1)
            pseudo_pattern = self.cal_pseudoPattern(weak_select_sim_Mat, labels)
            sup_con_loss = self.cal_con_loss(features, self.temperature2, labels=pseudo_pattern) 

            return classify_loss, sup_con_loss, reg_loss, reg_loss_diff, predict_labels
        
    # #=============================================================================


    # #=============================================================================
    @torch.no_grad()
    def cal_pseudoPattern(self, select_sim_Mat, labels):
        label_flat = labels.view(-1)
        batch_prob = F.softmax(torch.div(select_sim_Mat, self.tau), dim=1)      # B-K
        batch_max_prob, pattern_idx = batch_prob.max(dim=1)


        batch_pattern_mask = batch_max_prob.greater_equal(self.prob_threshold)
        pseudo_pattern_labels = torch.where(batch_pattern_mask, label_flat*self.K + pattern_idx, 
                                    (torch.arange(select_sim_Mat.shape[0])+self.cN*self.K).cuda())

        return pseudo_pattern_labels.view(-1,1).detach()
    # #=============================================================================


    # #=============================================================================
    # calculate reg1
    def cal_reg_loss(self, select_sim_Mat, labels, cur_epoch=10):

        batch_prob = F.softmax(torch.div(select_sim_Mat, self.tau), dim=1)      # B-K                       
        class_pat_prob = torch.cat([batch_prob.masked_select(labels.eq(i).view(-1,1)).view(-1, self.K).mean(dim=0,keepdim=True) 
                                                    for i in labels.cpu().unique().numpy()],dim=0)     # cN-K

        if cur_epoch==0:
            scale = self.class_var.index_select(dim=0, index=labels.unique()).view(-1).detach()
        else:
            var = self.class_var - self.class_var.min()
            var = var/(var.max())
            scale = (((var - 1.0)*1.5).exp()).index_select(dim=0, index=labels.unique()).view(-1).detach()

        # avoid 0 in vector
        alpha = 0.90
        ave_class_pat_prob = alpha*class_pat_prob + (1.0-alpha)*(1.0 / self.K)

       

        reg_loss_entropy = ((math.log(self.K) + \
                (ave_class_pat_prob*torch.log(ave_class_pat_prob)).sum(dim=1))*scale).mean()
        return reg_loss_entropy 


    # #=============================================================================


    # #=============================================================================
    @torch.no_grad()
    def update_var(self, var):
        self.class_var.mul_(0.0).add_(torch.tensor(var).cuda())
        return
    # #=============================================================================


    # #=============================================================================
    def cal_reg_loss_diff(self, centers):
        sim_centers = centers.t().matmul(centers)
        selectt_sim_centers = sim_centers.masked_select(self.select_idx)
        reg_loss_diff = selectt_sim_centers.abs().mean()

        return reg_loss_diff
    # #=============================================================================


    def cal_classify_loss(self, weak_aug, centers, labels):
        device = (torch.device('cuda')
                  if weak_aug.is_cuda
                  else torch.device('cpu'))

        simMat = weak_aug.matmul(centers)                      # B-cN*K
        simMat_3d = simMat.reshape(-1, self.cN, self.K)      # B-cN-K

        # max similarity
        simMat_Max = torch.max(simMat_3d, dim=2)[0]         # B-cN
        simMat_Max_expand = torch.unsqueeze(simMat_Max, dim=2).expand(-1, -1, self.K)
        # select samples according similarity
        condition =  torch.zeros(weak_aug.shape[0], self.cN, self.K, dtype=torch.bool).to(device) # dim-cN-K
        for i in range(weak_aug.shape[0]):
            for k in range(self.K):
                condition[i][labels.view(-1)[i]][k] = 1
        simMax_select = torch.where(condition, simMat_3d, simMat_Max_expand).to(device)    # B-cN-K
        # calculate weight
        w_from_simMat = F.softmax(torch.div(simMat_3d, self.tau), dim=2)      # B-cN-K
        if self.weight_detach:
            w_from_simMat = w_from_simMat.detach()


        one_hot = torch.zeros(weak_aug.shape[0], self.cN).to(device).scatter_(1, labels.view(-1, 1), 1)   # B-cN
        one_hot_expand = one_hot.unsqueeze(2).expand(-1, -1, self.K)                         # B-cN-K
        log_soft = F.log_softmax(torch.div(simMax_select, self.temperature1), dim=1)     # B-cN-K

        weight_loss = ((one_hot_expand*w_from_simMat)*(one_hot_expand*(-log_soft))).sum(dim=2) # B-cN
        classify_loss = weight_loss.sum(dim=1).mean()
        # calculated the pseudo pattern label for calculate the L_w2ps
        predict_labels = simMat_Max.argmax(dim=1)
        return classify_loss, predict_labels
    


    def cal_con_loss(self, features, temperature=0.1, labels=None,
                                             mask=None, contrast_mode='all'):
        """
        Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)      # bsz*nview - dim
        if contrast_mode == 'one':
            anchor_feature = features[:, 0]     # bsz - dim
            anchor_count = 1
        elif contrast_mode == 'all':
            anchor_feature = contrast_feature       # bsz*nview - dim
            anchor_count = contrast_count           # nview 
        else:
            raise ValueError('Unknown mode: {}'.format(contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperature)                                   # bsz*nview - bsz*nview
        

        ##################################################################
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        logits = anchor_dot_contrast


        if labels is None:
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        else:
            exp_logits = torch.exp(logits) * logits_mask
            batch_batch_simMat = torch.matmul(anchor_feature, contrast_feature.T)

            same_img_mask = (torch.eye(batch_size).repeat(anchor_count, contrast_count).to(device))
            weight = torch.where(same_img_mask.bool(), same_img_mask, batch_batch_simMat).detach_()

            log_prob = weight*logits - torch.log(exp_logits.sum(1, keepdim=True))
            
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss
