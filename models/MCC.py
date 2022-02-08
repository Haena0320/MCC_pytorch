from typing import Dict, List, Any
import torch
import torch.nn as nn
from torchvision.models import resnet
from torch.nn.modules import BatchNorm2d,BatchNorm1d
from utils.pytorch_misc import Flattener
import torch.nn.functional as F
import torch.nn.parallel
from utils.mca import AttFlat, LayerNorm, AttFlat_nofc #???
from utils import contrastive_loss
import random
import numpy as np

class MultiLevelCC(torch.nn.Module):
    def __init__(self,config, input_dropout=0.3, vector_dim=1024):
        super(MultiLevelCC, self).__init__()
        self.vector_dim = vector_dim
        self.rnn_input_dropout = nn.Dropout(input_dropout) if (input_dropout > 0) else None

        # object feature 2048에서 1024 로 압축
        self.obj_downsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2048, self.vector_dim),
            torch.nn.ReLU(inplace=True)
        )
        # box feature + box coordinate 융합
        self.boxes_fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(self.vector_dim + 525*2, self.vector_dim),
            torch.nn.ReLU(inplace=True))

        # rationale, question grounding
        self.grounding_LSTM = nn.LSTM(config.input_size, config.hidden_size, config.num_layers, bidirectional=config.bidirectional)
        # softmax attention in each modality
        self.attFlat_image = AttFlat(in_size=self.vector_dim,  out_size=self.vector_dim)
        self.attFlat_option = AttFlat_nofc(in_size=self.vector_dim, out_size=self.vector_dim)
        self.attFlat_query = AttFlat(in_size=self.vector_dim, out_size=self.vector_dim)

        self.image_BN = BatchNorm1d(self.vector_dim)
        self.option_BN = torch.nn.Sequential(BatchNorm1d(self.vector_dim))
        self.query_BN = torch.nn.Sequential(BatchNorm1d(self.vector_dim))

        self.final_mlp = torch.nn.Sequential(torch.nn.Linear(self.vector_dim*2, 512), torch.nn.ReLU(inplace=True))
        self.final_BN = torch.nn.Sequential(BatchNorm1d(512))
        self.final_mlp_linear = torch.nn.Sequential(torch.nn.Linear(512,1))

        self.cal_loss = torch.nn.CrossEntropyLoss()
        self.proj_norm = LayerNorm(size=self.vector_dim)
        self.fusion_BN = torch.nn.Sequential(BatchNorm1d(self.vector_dim))
        self.CL_answer_feat = contrastive_loss.CrossModal_CL(temperature=0.1)

        self.initializer()

    def initializer(self):
        for layer in self.grounding_LSTM.all_weights:
            for weight in layer:
                if weight.ndim == 2:
                    weight.data.uniform_(-0.1, 0.1)
                else:
                    weight.data.fill_(0)

        nn.init.xavier_uniform_(self.final_mlp[0].weight)
        self.final_mlp[0].bias.data.fill_(0)

        nn.init.xavier_uniform_(self.final_mlp_linear[0].weight)
        self.final_mlp_linear[0].bias.data.fill_(0)




    def forward(self,objects_feat,boxes,boxes_feat,box_masks,
                question, question_tags, question_mask, answers,
                answer_tags, answer_mask, v_mask, neg, neg_img,label):
        # chop off boxes
        max_len = int(box_masks.sum(1).max())
        box_mask = box_masks[:, :max_len]
        boxes = boxes[:, :max_len]
        boxes_feat = boxes_feat[:, :max_len]
        object_features = objects_feat[:, :max_len]

        obj_reps_ = self.obj_downsample(object_features)
        boxes_feat = boxes_feat.repeat(1,1,105*2) # normalized 좌표를 옆으로 죽 나열함... 왜??
        obj_reps = torch.cat([obj_reps_, boxes_feat], dim=-1) # [bs, max_object_num, 1024(object 좌표) + 105*2*5(normalized 좌표)]
        obj_reps = self.boxes_fc(obj_reps) # 다시 1024로, [bs, max_object_num, 1024]

        """text encoder"""
        # make rationale feature : [bs, 4, max_len, dimension=1024]
        option_rep = self.embed_span(answers, answer_tags, obj_reps)

        B, O, M, D = option_rep.shape
        # use soft attention instead of mean : [4, bs, dimension=1024]
        option_features = torch.ones([option_rep.shape[1], option_rep.shape[0], option_rep.shape[3]], dtype=torch.float)
        # option_rep
        for i in range(4):
            option_features[i] = self.attFlat_option(option_rep[:, i, :, :], answer_mask[:, i, :])
        option_features = option_features.transpose(1,0).cuda()  # [bs, 4, dimension=512]

        option_features = option_features.contiguous().view(B*O, -1)
        option_features = self.option_BN(option_features)
        option_features = option_features.contiguous().view(B, O, -1)

        # query
        query_rep = self.embed_span(question, question_tags, obj_reps)
        B, O, M, D = query_rep.shape
        query_features = torch.ones([query_rep.shape[1], query_rep.shape[0], query_rep.shape[3]], dtype=torch.float)
        for i in range(4):
            query_features[i] = self.attFlat_query(query_rep[:, i, :, :], question_mask[:, i, :])
        query_features = query_features.transpose(1,0).cuda()
        query_features = query_features.contiguous().view(B*O, -1)
        query_features = self.query_BN(query_features)

        query_features = query_features.contiguous().view(B, O, -1)

        """image encoder"""
        images = obj_reps[:, 1:, :] # 전체 이미지 제외한 object 들  [bs, max_object_num-1, 1024]
        # positive samples
        if neg == False:
            if v_mask is None: # origin samples
                box_mask_counterfactual = box_mask[:, 1:] # [bs, max_obj-1] 전체 object 에 대해 soft attention 계산
            else: # positive object samples or positive img samples
                v_mask = v_mask[:, 1:] # 전체 이미지 고려안함
                box_mask_counterfactual = box_mask[:, 1:]*v_mask.long()

            images_features = self.Flat_img(images, box_mask_counterfactual, query_features.shape) # [bs, 4, dimension=512] # softmax attention 계산
            fusion_qv = self.fusion_QV(query_features, images_features) # fusion Q & V
        # negative samples
        else:
            if neg_img == False: # negative object sample
                fusion_qv = torch.ones([v_mask.shape[0], B, O, self.vector_dim]).float().cuda() # [top_neg, bs, obj_num, 512]
                for idx, mask in enumerate(v_mask):# v_mask : [topn, batch_size, obj_num]
                    mask = mask[:, 1:] # except whole img [bs, obj_num]
                    box_mask_counterfactual = box_mask[:, 1:] * mask.long() # [bs, obj_num]
                    images_features = self.Flat_img(images, box_mask_counterfactual, query_features.shape)
                    fusion_qv[idx] = self.fusion_QV(query_features, images_features)
                fusion_qv = fusion_qv.transpose(1,0) # [bs, x, 4, 512]
            else: # negative image samples
                img_sample_num = 3
                fusion_qv = torch.zeros([img_sample_num, B, O, self.vector_dim]).cuda()
                for k in range(img_sample_num):
                    images_neg = torch.zeros(images.shape).float().cuda() # [bs, object_num, 1024]
                    box_mask_neg = torch.zeros(box_mask.shape).long().cuda() # [bs, object_num]
                    for i in range(B):
                        rand_idx = random.randint(0, B-1)
                        while rand_idx == i:
                            rand_idx = random.randint(0, B-1)
                        images_neg[i] = images[rand_idx]
                        box_mask_neg[i] = box_mask[rand_idx]
                    image_features = self.Flat_img(images_neg, box_mask_neg[:, 1:], query_features.shape) # 한 배치 내 다른 샘플의 objects,box_mask
                    fusion_qv[k] = self.fusion_QV(query_features, image_features)
                fusion_qv = fusion_qv.transpose(1,0) # [bs, x, 4, 512]

        """answer level"""
        if label is not None and neg == False: # SemanticCL (cross-modal contrastive learning)
            fusion_qv_norm = F.normalize(fusion_qv, dim=-1)
            option_features_norm = F.normalize(option_features, dim=-1)
            loss_answer_feat = self.CL_answer_feat(fusion_qv_norm.mean(dim=1), option_features_norm, label.long().view(-1)) # anchor_feature(BS,1024), features(BS, 4, 1024), label (BS)
        else:
            loss_answer_feat = -1

        if fusion_qv.dim() == 4: # neg [bs, top_neg, object_num, dimension]
            query_option_image_cat = torch.cat((option_features, fusion_qv[:, 0,:,:]), -1)
        else:
            query_option_image_cat = torch.cat((option_features, fusion_qv), -1)
        assert (query_option_image_cat.shape == (B, O, self.vector_dim*2))

        query_option_image_cat = self.final_mlp(query_option_image_cat) # 2048 -> 512
        query_option_image_cat = query_option_image_cat.contiguous().view(B*O, 512)
        query_option_image_cat = self.final_BN(query_option_image_cat)
        query_option_image_cat = query_option_image_cat.contiguous().view(B, O, 512)
        logits = self.final_mlp_linear(query_option_image_cat) # 512 -> 1
        logits = logits.squeeze(2)
        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {"label_logits": logits, "label_probs": class_probabilities}

        if label is not None:
            loss = self.cal_loss(logits, label.long().view(-1))
            accuracy = self.cal_accuracy(logits, label)
            output_dict["loss"] = loss[None]
            output_dict["accuracy"] = accuracy
            output_dict["loss_answer_feat"] = loss_answer_feat # SemCL (cross-modal contrastive learning)
        output_dict["QV"] = fusion_qv
        return output_dict

    def Flat_img(self, images, box_mask_counterfactual, q_shape): # q_shape [bs, 4, dimension=512]
        images_features = self.attFlat_image(images, box_mask_counterfactual) # images_features [bs, dimension=512
        images_features = self.image_BN(images_features)
        images_features = images_features.unsqueeze(1).expand(q_shape) # image_features [bs, 4, dimension=512]
        return images_features

    def fusion_QV(self, q, v):
        fusion_qv = v + q
        B, O,_  = q.shape
        fusion_qv = self.proj_norm(fusion_qv)
        fusion_qv = fusion_qv.contiguous().view(B*O, -1)
        fusion_qv = self.fusion_BN(fusion_qv)
        fusion_qv = fusion_qv.contiguous().view(B, O, -1)
        return fusion_qv

    def embed_span(self, span, span_tags, object_reps):
        features = self._collect_obj_reps(span_tags, object_reps)
        span_rep = torch.cat((span, features), -1)
        B_, N, K, D = span_rep.shape
        if self.rnn_input_dropout:
            span_rep = self.rnn_input_dropout(span_rep)
        reps, _ = self.grounding_LSTM(span_rep.view(B_*N, K, D))
        B, N, D = reps.shape
        return reps.view(B_, -1, N, D)

    def _collect_obj_reps(self, span_tags, object_reps):
        span_tags_fixed = torch.clamp(span_tags, min=0)
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def cal_accuracy(self, logits, labels):
        logits_ = logits.cpu().detach().numpy()
        labels_ = labels.cpu().detach().numpy()
        preds = np.argmax(logits_, axis=-1)
        return np.sum(preds==labels_)


if __name__ == "__main__":
    from dataloaders.vcr_attribute_box import VCR, VCRLoader
    from utils.pytorch_misc import load_params
    train, val = VCR.splits()
    val_loader = VCRLoader.from_dataset(val, 10)
    for batch in val_loader:
        print(".")
        break

    config = load_params("/mnt/data/user8/MCC/MCC_pytorch/models/params.json")
    model = MultiLevelCC(config["model"]["option_encoder"])

    batch["v_mask"] = None
    batch["neg"] = None
    batch["neg_img"] = False

    output = model(batch["objects_feat"],batch["boxes"],batch["boxes_feat"],batch["box_masks"],
                batch["question"], batch["question_tags"], batch["question_mask"], batch["answers"],
                batch["answer_tags"], batch["answer_mask"], batch["v_mask"], batch["neg"], batch["neg_img"],batch["label"])




