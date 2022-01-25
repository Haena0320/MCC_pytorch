import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataloaders.box_utils import load_image, resize_image, to_tensor_and_normalize
from dataloaders.mask_utils import make_mask

import h5py
from copy import deepcopy
from config import VCR_DIR
from path import Path
import jsonlines as jsnl
from tqdm import tqdm
GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']

# Here's an example jsonl
# {
# "movie": "3015_CHARLIE_ST_CLOUD",
# "objects": ["person", "person", "person", "car"],
# "interesting_scores": [0],
# "answer_likelihood": "possible",
# "img_fn": "lsmdc_3015_CHARLIE_ST_CLOUD/3015_CHARLIE_ST_CLOUD_00.23.57.935-00.24.00.783@0.jpg",
# "metadata_fn": "lsmdc_3015_CHARLIE_ST_CLOUD/3015_CHARLIE_ST_CLOUD_00.23.57.935-00.24.00.783@0.json",
# "answer_orig": "No she does not",
# "question_orig": "Does 3 feel comfortable?",
# "rationale_orig": "She is standing with her arms crossed and looks disturbed",
# "question": ["Does", [2], "feel", "comfortable", "?"],
# "answer_match_iter": [3, 0, 2, 1],
# "answer_sources": [3287, 0, 10184, 2260],
# "answer_choices": [
#     ["Yes", "because", "the", "person", "sitting", "next", "to", "her", "is", "smiling", "."],
#     ["No", "she", "does", "not", "."],
#     ["Yes", ",", "she", "is", "wearing", "something", "with", "thin", "straps", "."],
#     ["Yes", ",", "she", "is", "cold", "."]],
# "answer_label": 1,
# "rationale_choices": [
#     ["There", "is", "snow", "on", "the", "ground", ",", "and",
#         "she", "is", "wearing", "a", "coat", "and", "hate", "."],
#     ["She", "is", "standing", "with", "her", "arms", "crossed", "and", "looks", "disturbed", "."],
#     ["She", "is", "sitting", "very", "rigidly", "and", "tensely", "on", "the", "edge", "of", "the",
#         "bed", ".", "her", "posture", "is", "not", "relaxed", "and", "her", "face", "looks", "serious", "."],
#     [[2], "is", "laying", "in", "bed", "but", "not", "sleeping", ".",
#         "she", "looks", "sad", "and", "is", "curled", "into", "a", "ball", "."]],
# "rationale_sources": [1921, 0, 9750, 25743],
# "rationale_match_iter": [3, 0, 2, 1],
# "rationale_label": 1,
# "img_id": "train-0",
# "question_number": 0,
# "annot_id": "train-0",
# "match_fold": "train-0",
# "match_index": 0,
# }

def tokenization(tokenized_sent, bert_embs, old_object_to_new_ind, object_to_type, pad_id=-1):
    new_tokenization_with_tags = []
    for tok in tokenized_sent:
        if isinstance(tok, list):
            for int_name in tok:
                obj_type= object_to_type[int_name]
                new_ind = old_object_to_new_ind[int_name]
                if new_ind < 0:
                    raise ValueError("invalid object index ! ")
                #txt_to_use = GENDER_NEUTRAL_NAMES[new_ind % len(GENDER_NEUTRAL_NAMES)] if obj_type =="person" else obj_typee
                new_tokenization_with_tags.append(new_ind)
        else:
            new_tokenization_with_tags.append(pad_id)
    assert bert_embs.shape[0] == len(new_tokenization_with_tags)
    return bert_embs, new_tokenization_with_tags


class VCR(Dataset):
    def __init__(self, split, mode, vcr_dir=VCR_DIR, only_use_relevant_dets=True, add_image_as_a_box=True, embs_to_load='bert_da', conditioned_answer_choice=0, expand2obj36=False):
        self.split = split
        self.mode = mode
        self.vcr_dir = Path(vcr_dir)
        self.only_use_relevant_dets=only_use_relevant_dets
        self.expand2obj36 = expand2obj36
        print("Only relevant dets" if only_use_relevant_dets else "Using all detections", flush=True)

        self.add_image_as_a_box = add_image_as_a_box
        self.embs_to_load = embs_to_load
        self.conditioned_answer_choice = conditioned_answer_choice

        vcr_annots = self.vcr_dir / "{}.jsonl".format(split)
        vcr_annots_reader = jsnl.Reader(vcr_annots.open("r"))
        self.items = [k for k in tqdm(vcr_annots_reader.iter())]

        if split not in ("test", "train", "val"):
            raise ValueError("Mode must be in test, train, or val. Supplied {}".format(mode))

        if mode not in ("answer", "rationale"):
            raise ValueError("split must be answer or rationale")

        coco = self.vcr_dir / "cocoontology.json"
        coco_data = json.load(coco.open("r"))

        self.coco_objects = ["__background__"] + [x["name"] for k, x in sorted(coco_data.items(), key=lambda x: int(x[0]))]
        self.coco_obj_to_ind = {o:i for i, o in enumerate(self.coco_objects)}

        self.h5fn = self.vcr_dir / "bert_embeds" / f'{self.embs_to_load}_{self.mode}_{self.split}.h5'
        print("Loading embeddings from {}".format(self.h5fn), flush=True)

    @property
    def is_train(self):
        return self.split == "train"

    @classmethod
    def splits(cls, **kwargs):
        kwargs_copy = {x:y for x, y in kwargs.items()}
        if 'mode' not in kwargs:
            kwargs_copy["mode"] ="answer"
        train = cls(split="train", **kwargs_copy)
        val = cls(split="val", **kwargs_copy)
        return train, val

    @classmethod
    def eval_splits(cls, **kwargs):
        for forbidden_key in ["mode", "split", "conditioned_answer_choice"]:
            if forbidden_key in kwargs:
                raise ValueError(f"don't supply {forbidden_key} to eval_splits()")

        stuff_to_return = [cls(split="test", mode="answer", **kwargs)] + [cls(split='test', mode='rationale', conditioned_answer_choice=i, **kwargs) for i in range(4)]
        return tuple(stuff_to_return)

    def __len__(self):
        return len(self.items)

    def _get_dets_to_use(self, item):
        question = item["question"]
        answer_choices = item['{}_choices'.format(self.mode)]

        if self.only_use_relevant_dets:
            object2use = np.zeros(len(item["objects"]), dtype=bool)
            people = np.array([x=="person" for x in item["objects"]], dtype=bool)
            for sent in answer_choices + [question]:
                for possibly_det_list in sent:
                    if isinstance(possibly_det_list, list):
                        for tag in possibly_det_list:
                            if tag >=0 and tag < len(item["objects"]):
                                object2use[tag] = True
                    elif possibly_det_list.lower() in ("everyone", "everyones"):
                        object2use |= people
            if not object2use.any():
                object2use |= people
            # expand obj to 36 items
            if self.expand2obj36:
                if np.sum(object2use) >= 36:
                    pass
                else:
                    object_num = np.sum(object2use)
                    for i in range(len(item["objects"])):
                        if object2use[i] == 0:
                            object2use[i] = 1
                            object_num += 1
                            if object_num >= 36:
                                break

        else:
            object2use = np.ones(len(item["objects"]), dtype=bool)

        object2use = np.where(object2use)[0]

        old_det_to_new_ind = np.zeros(len(item["objects"]), dtype=np.int32) - 1   # [-1, -1, -1]
        old_det_to_new_ind[object2use] = np.arange(object2use.shape[0], dtype=np.int32) # [-1, -1, 0, -1, 1, 2, -1, -1, 3]
        if self.add_image_as_a_box:
            old_det_to_new_ind[object2use] += 1
        old_det_to_new_ind = old_det_to_new_ind.tolist()
        return object2use, old_det_to_new_ind

    def __getitem__(self, index):
        item = deepcopy(self.items[index])

        if self.mode == 'rationale':
            conditioned_label = item["answer_label"] if self.split != "test" else self.conditioned_answer_choice
            item["question"] += item["answer_choices"][conditioned_label]

        answer_choices = item[f'{self.mode}_choices']
        object2use, old_object_to_new_ind = self._get_dets_to_use(item)

        with h5py.File(self.h5fn, "r") as h5:
            bert_embeddings = {k: np.array(v, dtype=np.float16) for k, v in h5[str(index)].items()}

        condition_key = self.conditioned_answer_choice if self.split =="test" and self.mode =="rationale" else ""

        instance_dict = {}
        if 'endingonly' not in self.embs_to_load:
            question_bert_embs, question_tags = zip(*[tokenization(item["question"],
                                                                    bert_embeddings[f"ctx_{self.mode}{condition_key}{i}"],
                                                                    old_object_to_new_ind,
                                                                    item["objects"],
                                                                    pad_id=0 if self.add_image_as_a_box else -1) for i in range(4)]) # padding 위치에 전체 이미지 넣어짐 --> 어차피 나중에 다 사라짐

        answer_bert_embs, answer_tags = zip(*[tokenization(answer,
                                                           bert_embeddings[f"answer_{self.mode}{condition_key}{i}"],
                                                           old_object_to_new_ind,
                                                           item["objects"],
                                                           pad_id=0 if self.add_image_as_a_box else -1)
                                              for i, answer in enumerate(answer_choices)])

        ## mask
        if 'endingonly' not in self.embs_to_load:
            instance_dict['question'] = question_bert_embs
            instance_dict["question_mask"] = [len(q)*[1] for q in question_tags]
            instance_dict["question_len"] = len(question_tags[0])
            instance_dict["question_tags"] = question_tags
            assert len(question_bert_embs) == len(question_tags)

        instance_dict["answers"] = answer_bert_embs
        instance_dict["answer_mask"] = [len(a)*[1] for a in answer_tags]
        instance_dict["answer_len"] = len(answer_tags[0])
        instance_dict["answer_tags"] = answer_tags
        assert len(answer_bert_embs) == len(answer_tags)

        if self.split != "test":
            instance_dict["label"] = item["{}_label".format(self.mode)]


        image = load_image(self.vcr_dir / "vcr1images" / item["img_fn"])
        h_box = image.height
        w_box = image.width
        # load image
        images, window, img_scale, padding = resize_image(image, random_pad=self.is_train)  # padding
        images = to_tensor_and_normalize(images)
        c, h, w = images.shape
        # load bounding boxes, object segms, object names
        meta_path = self.vcr_dir / "vcr1images" / item["metadata_fn"]
        metadata = json.load(meta_path.open("r"))
        # [4, 14, 14]
        segms = np.stack(
            [make_mask(mask_size=14, box=metadata['boxes'][i], polygons_list=metadata["segms"][i]) for i in object2use])
        boxes = np.array(metadata["boxes"])[object2use, :-1]
        boxes_box = np.array(metadata["boxes"])[object2use, :-1]

        boxes *= img_scale
        boxes[:, :2] += np.array(padding[:2])[None]
        boxes[:, 2:] += np.array(padding[:2])[None]

        obj_labels = [self.coco_obj_to_ind[item["objects"][i]] for i in object2use.tolist()]
        if self.add_image_as_a_box:
            boxes = np.row_stack((window, boxes))
            segms = np.concatenate((np.ones((1, 14, 14), dtype=np.float32), segms), 0)
            obj_labels = [self.coco_obj_to_ind["__background__"]] + obj_labels

            boxes_box = np.row_stack((([1,1,w_box, h_box], boxes_box)))

        assert np.all((boxes[:, 1] >= 0.) & (boxes[:, 1] < boxes[:, 3]))
        assert np.all((boxes[:, 2] <= w))
        assert np.all((boxes[:, 3] <= h))

        instance_dict["segms"] = segms
        instance_dict["objects"] = obj_labels
        instance_dict["boxes"] = boxes

        #coordinate feature
        boxes_feat = np.eye(boxes_box.shape[0], 5)
        boxes_feat[:, 0] = (boxes_box[:, 0] + boxes_box[:, 2]) / 2   # center x
        boxes_feat[:, 1] = (boxes_box[:, 1]+boxes_box[:, 3]) / 2   # center y
        boxes_feat[:, 2] = (boxes_box[:, 2] - boxes_box[:, 0])  # w
        boxes_feat[:, 3] = (boxes_box[:, 3] - boxes_box[:, 1]) # h
        boxes_feat[:, 4] = boxes_feat[:, 2] * boxes_feat[:, 3] # w * h (너비)

        # coordinate feature normalization
        max_r = np.max(boxes_feat, axis=0)
        min_r = np.min(boxes_feat, axis=0)
        r_range = max_r - min_r
        boxes_feat = (boxes_feat - min_r) / r_range
        instance_dict["boxes_feat"] = boxes_feat
        return images, instance_dict

def make_batch(data, to_gpu=False):
    batch = dict()
    images, instances = zip(*data)
    batch["images"] = torch.stack(images, 0)

    question_batch = [i["question_len"] for i in instances]
    answer_batch = [i["answer_len"] for i in instances]
    object_batch =[len(i["objects"]) for i in instances]
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

    batch["objects"] = torch.ones(batch_n, max_object_num).long()*(-1)

    batch["boxes"] = torch.ones(batch_n, max_object_num, 4)*(-1)
    batch["segms"] = torch.zeros(batch_n, max_object_num, 14, 14)
    batch["boxes_feat"] = torch.zeros(batch_n, max_object_num, 5).long()

    for i in range(batch_n):
        batch["question_mask"][i, :, :question_batch[i]] = torch.tensor(instances[i]["question_mask"]).long()
        batch["question_tags"][i, :, :question_batch[i]] = torch.tensor(instances[i]["question_tags"]).long()
        batch["question"][i, :, :question_batch[i], :] = torch.tensor(instances[i]["question"])

        batch["objects"][i, :object_batch[i]] = torch.tensor(instances[i]["objects"])
        batch["boxes"][i, :object_batch[i], :] = torch.tensor(instances[i]["boxes"])
        batch["boxes_feat"][i, :object_batch[i], :] = torch.tensor(instances[i]["boxes_feat"])
        batch["segms"][i, :object_batch[i], :, :] = torch.tensor(instances[i]["segms"])

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
    def from_dataset(cls, data, batch_size=1, num_workers=3, num_gpus=1, **kwargs):
        loader = cls(dataset=data,
                     batch_size=batch_size*num_gpus,
                     shuffle=data.is_train,
                     num_workers=num_workers,
                     collate_fn=make_batch,
                     drop_last=data.is_train,
                     pin_memory=False,
                     **kwargs)
        return loader

if __name__ =="__main__":
    train, val = VCR.splits()
    data1 = val.__getitem__(0)
    data2 = val.__getitem__(1)

    batch_ = make_batch([data1, data2])
    torch.save(batch_, "data_sample.pkl")
    loader = VCRLoader.from_dataset(val)
    for batch in loader:
        print(batch)
        break














