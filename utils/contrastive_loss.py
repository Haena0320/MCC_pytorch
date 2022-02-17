import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CrossModal_CL(nn.Module):   # SemanticCL(cross modal contrastive learning)
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(CrossModal_CL, self).__init__()
        self.temperature = temperature # scaling hyper-parameter
        self.base_temperature = base_temperature

    def forward(self, anchor_feature, features, label):
        # anchor/fusion(Q_V)_features [bs, dim]
        # features /answer_features [bs, 4, dim]
        # label [bs]
        anchor_feature = anchor_feature.unsqueeze(1) # (bs, 1, dim)
        features = features.transpose(2,1) # (bs, 4, dim)

        logits = torch.div(torch.matmul(anchor_feature, features), self.temperature) # [bs, 4]
        logits_max = logits.max(dim=-1, keepdim=True)[0]
        logits = logits - logits_max.detach() # overflow 방지
        loss = F.cross_entropy(logits.squeeze(1), label)
        return loss

class CL_feat(nn.Module):
    def __init__(self, temperature=0.2):
        super(CL_feat, self).__init__()
        self.temperature = temperature
        self.contrastive_loss = CrossModal_CL(temperature=temperature)

    def forward(self, anchor, pos, neg):
        # anchor [bs, 4, 512]
        # anchor [bs, 4, 512] ???
        # neg [bs, x, 4, 512]
        if neg.dim() == 3:
            neg = neg.unsqueeze(1) # [bs, 1, O, 512]
        features = torch.cat([pos.unsqueeze(1), neg], dim=1) # [bs, 1+x, 4, 512]
        label = torch.zeros(anchor.shape[0]).long().cuda() # [0,1,2,..x] 중 첫번째(0) 위치가 정답(pos)
        loss = []
        for i in range(4):
            loss.append(self.contrastive_loss(anchor[:, i, :], features[:, :, i, :], label))

        mean_loss = sum(loss)/len(loss)
        return mean_loss








