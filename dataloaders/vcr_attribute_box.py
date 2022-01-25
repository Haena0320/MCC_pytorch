import json
import jsonlines as jsnl
from tqdm import tqdm
import os
import base64
import csv
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataloaders.box_utils import load_image, resize_image, to_tensor_and_normalize
from dataloaders.mask_utils import make_mask
import h5py
from copy import deepcopy
from config import VCR_DIR
from path import Path
csv.field_size_limit(sys.maxsize)

GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']

def tokenization(tokenized_sent,bert_embs,old_object_to_new_ind, object_to_type, pad_id=-1):
    new_tokenization_with_tags = []
    for tok in tokenized_sent:
        if isinstance(tok, list):
            for int_name in tok:
                obj_type = object_to_type[int_name]
                new_ind = old_object_to_new_ind[int_name]
                if new_ind < 0:
                    raise ValueError("invalid object index ! ")
                text_to_use = GENDER_NEUTRAL_NAMES[new_ind % len(GENDER_NEUTRAL_NAMES)] if obj_type == "person" else obj_type
                new_tokenization_with_tags.append((text_to_use, new_ind))
        else:
            new_tokenization_with_tags.append((tok, pad_id))

    tags = [tag for token, tag in new_tokenization_with_tags]

    assert bert_embs.shape[0] == len(tags)
    return bert_embs, tags

class VCR(Dataset):
    def __init__(self, split, mode, vcr_dir=VCR_DIR, only_use_relevant_dets=True, add_image_as_a_box=True, embs_to_load="bert_da",
                 conditioned_answer_choice=0, expand2obj36=False):
        self.split=split
        self.mode=mode
        self.vcr_dir = Path(vcr_dir)
        self.only_use_relevant_dets = only_use_relevant_dets
        self.expand2obj36 = expand2obj36
        print('Only relevant dets' if only_use_relevant_dets else "Using all detections", flush=True)

        self.add_image_as_a_box = add_image_as_a_box
        self.conditioned_answer_choice = conditioned_answer_choice

        items_path = self.vcr_dir / "{}.jsonl".format(split)
        self.items = [k for k in  tqdm(jsnl.Reader(items_path.open("r")).iter())]

        if split not in ("test", "train", "val"):
            raise ValueError("split must be in test, train, or val. Supplied {}".format(split))

        if mode not in ('answer', "rationale"):
            raise ValueError("split must be answer or rationale")

        coco_path = self.vcr_dir / "cocoontology.json"
        coco = json.load(coco_path.open("r"))
        self.coco_objects = ['__background__'] + [x["name"] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))]
        self.coco_obj_to_ind  = {o:i for i, o in enumerate(self.coco_objects)}

        self.embs_to_load = embs_to_load
        self.h5fn = self.vcr_dir / "bert_embeds" / f'{self.embs_to_load}_{self.mode}_{self.split}.h5'
        print("Loading embeddings from {}".format(self.h5fn), flush=True)
        self.tag_feature_path = self.vcr_dir / "butd_feats" / f'attribute_features_{self.split}.h5'

    @property
    def is_train(self):
        return self.split =="train"

    @classmethod
    def splits(cls, **kwargs):
        """Helper method to generate splits of the dataset"""
        kwargs_copy = {x:y for x, y in kwargs.items()}
        if "mode" not in kwargs:
            kwargs_copy["mode"] = 'answer'
        train = cls(split='train', **kwargs_copy)
        val = cls(split='val',**kwargs_copy)
        return train, val

    @classmethod
    def eval_splits(cls, **kwargs):
        for forbidden_key in ['mode', "split", "conditioned_answer_choice"]:
            if forbidden_key in kwargs:
                raise ValueError(f"don't supply {forbidden_key} to eval_splits()")

        stuff_to_return = [cls(split='test', mode="answer", **kwargs)] + [cls(split="test", mode="rationale", conditioned_answer_choice=i, **kwargs) for i in range(4)]
        return tuple(stuff_to_return)

    def __len__(self):
        return len(self.items)

    def _get_dets_to_use(self, item):
        question = item['question']
        answer_choices = item["{}_choices".format(self.mode)]

        if self.only_use_relevant_dets:
            dets2use = np.zeros(len(item["objects"]), dtype=bool)
            people = np.array([x=="person" for x in item["objects"]], dtype=bool)
            for sent in answer_choices + [question]:
                for possibly_det_list in sent:
                    if isinstance(possibly_det_list, list):
                        for tag in possibly_det_list:
                            if tag >= 0 and tag < len(item["objects"]):
                                dets2use[tag] = True
                    elif possibly_det_list.lower() in ("everyone", "everyones"):
                        dets2use |= people
            if not dets2use.any():
                dets2use |= people

            if self.expand2obj36:
                if np.sum(dets2use) >= 36:
                    pass
                else:
                    dets_num = np.sum(dets2use)
                    for i in range(len(item["objects"])):
                        if dets2use[i] ==0:
                            dets2use[i] = 1
                            dets_num += 1
                            if dets_num >= 36:
                                break
        else:
            dets2use = np.ones(len(item["objects"]), dtype=bool)

        dets2use = np.where(dets2use)[0] #object 1 인 인덱스 반환

        old_det_to_new_ind = np.zeros(len(item["objects"]), dtype=int)-1
        old_det_to_new_ind[dets2use] = np.arange(dets2use.shape[0], dtype=int)

        if self.add_image_as_a_box:
            old_det_to_new_ind[dets2use] += 1
        old_det_to_new_ind = old_det_to_new_ind.tolist()
        return dets2use, old_det_to_new_ind

    def __getitem__(self, index):
        item = deepcopy(self.items[index])
        image_id = int(item["img_id"].split('-')[-1])
        ######
        with h5py.File(self.tag_feature_path, "r") as h5:
            tag_features = np.array(h5[str(image_id)]['features'], dtype=np.float32)

        if self.mode =="rationale":
            conditioned_label = item["answer_label"] if self.split != "test" else self.conditioned_answer_choice
            item["question"] += item['answer_choices'][conditioned_label]

        answer_choices = item["{}_choices".format(self.mode)]
        dets2use, old_object_to_new_ind = self._get_dets_to_use(item)

        with h5py.File(self.h5fn, "r") as h5:
            bert_embeddings = {k:np.array(v, dtype=np.float16) for k, v in h5[str(index)].items()}

        condition_key = self.conditioned_answer_choice if self.split =='test' and self.mode =="rationale" else ""

        instance_dict = {}
        question_bert_embs, question_tags = zip(*[tokenization(item["question"],
                                                                     bert_embeddings[f"ctx_{self.mode}{condition_key}{i}"],
                                                                     old_object_to_new_ind,
                                                                     item["objects"],
                                                                     pad_id=0 if self.add_image_as_a_box else -1)
                                                   for i in range(4)])
        answer_bert_embs, answer_tags = zip(*[tokenization(answer,
                                                                bert_embeddings[f"answer_{self.mode}{condition_key}{i}"],
                                                                old_object_to_new_ind,
                                                                item["objects"],
                                                                pad_id=0 if self.add_image_as_a_box else -1)
                                              for i, answer in enumerate(answer_choices)])

        instance_dict['question'] = question_bert_embs
        instance_dict["question_tags"] = question_tags
        instance_dict["question_len"] = len(question_tags[0])
        instance_dict["question_mask"] = [len(q)*[1] for q in question_tags]

        instance_dict["answers"] = answer_bert_embs
        instance_dict["answer_tags"] = answer_tags
        instance_dict["answer_len"] = max([len(a) for a in answer_tags])
        instance_dict["answer_mask"] = [len(a)*[1] for a in answer_tags]

        if self.split != "test":
            instance_dict["label"] = item["{}_label".format(self.mode)]

        image = load_image(str(self.vcr_dir / "vcr1images" / item["img_fn"]))

        h = image.height
        w = image.width

        img_meta = self.vcr_dir / 'vcr1images' / item["metadata_fn"]
        metadata = json.load(img_meta.open("r"))

        boxes = np.array(metadata['boxes'])[dets2use, :-1]

        if self.add_image_as_a_box:
            boxes = np.row_stack(([1,1,w,h], boxes))

        # coordinate feature
        boxes_feat = np.eye(boxes.shape[0], 5)
        boxes_feat[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        boxes_feat[:, 1] = (boxes[:, 1] + boxes[:,3]) / 2
        boxes_feat[:, 2] = boxes[:, 2] - boxes[:,0]
        boxes_feat[:, 3] = boxes[:, 3] - boxes[:, 1]
        boxes_feat[:, 4] = boxes_feat[:, 2] * boxes_feat[:, 3]

        max_r = np.max(boxes_feat, axis=0)
        min_r = np.min(boxes_feat, axis=0)
        r_range = max_r - min_r
        boxes_feat = (boxes_feat - min_r) / r_range

        instance_dict["boxes_feat"] = boxes_feat
        instance_dict['boxes'] = boxes

        if self.add_image_as_a_box:
            dets2use += 1
            dets2use = np.insert(dets2use, 0, 0)
        instance_dict["objects_feat"] = tag_features[dets2use] # padding 0
        assert (tag_features[dets2use].shape[0] == boxes.shape[0])
        return instance_dict

def make_batch(instances, to_gpu=False):
    batch = dict()

    question_batch = [i["question_len"] for i in instances]
    answer_batch = [i["answer_len"] for i in instances]

    object_batch =[len(i["boxes"]) for i in instances]
    answer_each_batch = [[len(i["answer_mask"][j]) for j in range(4)] for i in instances]

    max_quest_len = max(question_batch)
    max_answer_len = max(answer_batch)
    max_object_num = max(object_batch)
    batch_n = len(question_batch)

    batch["question_mask"] = torch.zeros((batch_n, 4, max_quest_len)).long()
    batch["question_tags"] = torch.ones((batch_n, 4, max_quest_len)).long() * (-2)
    batch["question"] = torch.zeros((batch_n, 4, max_quest_len, 768)).float()

    batch["answer_mask"] = torch.zeros((batch_n, 4, max_answer_len)).long()
    batch["answer_tags"] = torch.ones((batch_n, 4, max_answer_len)).long() * (-2)
    batch["answers"] = torch.zeros((batch_n, 4, max_answer_len, 768)).float()

    batch["boxes"] = torch.ones(batch_n, max_object_num, 4)*(-1)
    batch["boxes_feat"] = torch.zeros(batch_n, max_object_num, 5).long()
    batch["objects_feat"] = torch.zeros(batch_n, max_object_num, 2048)

    for i in range(batch_n):
        batch["question_mask"][i, :, :question_batch[i]] = torch.tensor(instances[i]["question_mask"]).long()
        batch["question_tags"][i, :, :question_batch[i]] = torch.tensor(instances[i]["question_tags"]).long()
        batch["question"][i, :, :question_batch[i], :] = torch.tensor(instances[i]["question"])

        batch["boxes"][i, :object_batch[i], :] = torch.tensor(instances[i]["boxes"])
        batch["boxes_feat"][i, :object_batch[i], :] = torch.tensor(instances[i]["boxes_feat"])
        batch["objects_feat"][i, :object_batch[i],:] = torch.tensor(instances[i]["objects_feat"])

        a_batch = answer_each_batch[i]
        for j in range(4):
            batch["answer_mask"][i, j, :a_batch[j]] = torch.tensor(instances[i]["answer_mask"][j]).long()
            batch["answer_tags"][i, j, :a_batch[j]] = torch.tensor(instances[i]["answer_tags"][j]).long()
            batch["answers"][i, j, :a_batch[j], :] = torch.tensor(instances[i]["answers"][j])

    batch["box_masks"] = torch.all(batch["boxes"] >= 0, -1).long()
    batch["label"] = torch.tensor([i["label"] for i in instances])
    return batch

class VCRLoader(DataLoader):
    @classmethod
    def from_dataset(cls, data, batch_size=1, num_workers=1, num_gpus=1,**kwargs):
        loader = cls(
            dataset=data,
            batch_size=batch_size * num_gpus,
            shuffle=data.is_train,
            num_workers=num_workers,
            collate_fn=make_batch,
            drop_last=data.is_train,
            pin_memory=False,
            **kwargs
        )
        return loader

if __name__ == "__main__":
    train, val = VCR.splits()
    data_0 = val.__getitem__(0)
    data_1 = val.__getitem__(1)
    data_2 = val.__getitem__(2)
    batch = make_batch([data_0, data_1, data_2])

    loader = VCRLoader.from_dataset(val, 3)
    for item in loader:
        print(item)
        break

    torch.save(item, "vcr_attribute_box_sample.pkl")

import torch
data = torch.load('/mnt/data/user8/MCC/MCC_pytorch/vcr_attribute_box_sample.pkl')








