import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataloaders.box_utils import load_image, resize_image, to_tensor_and_normalize
from dataloaders.mask_utils import make_mask
import jsonlines as jsnl
import h5py
import multiprocessing
from copy import deepcopy
from config import VCR_DIR
from path import Path
GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']

class VCRImage(Dataset):
    def __init__(self, split, vcr_dir=VCR_DIR, add_image_as_a_box=True):
        self.split = split
        self.vcr_dir = Path(vcr_dir)
        self.add_image_as_a_box = add_image_as_a_box
        self.img_id_2_meta_folder = {}
        self.img_id_2_image_folder = {}
        data_split = self.vcr_dir / "{}.jsonl".format(split)
        data_split = jsnl.Reader(data_split.open("r"))
        for item in data_split:
            self.img_id_2_meta_folder[item["img_id"]] = str(self.vcr_dir / 'vcr1images' / item["metadata_fn"])
            self.img_id_2_image_folder[item['img_id']] = str(self.vcr_dir / 'vcr1images' / item["img_fn"])
        self.img_ids = list(self.img_id_2_image_folder.keys())

        if split not in ("test", "train", "val"):
            raise ValueError("split must be in test, train, or val. but, got {}".format(split))

        coco_path = self.vcr_dir / "cocoontology.json"
        coco = json.load(coco_path.open('r'))
        self.coco_objects = ["__background__"] + [x['name'] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))]
        self.coco_obj_to_ind = {v: k for k, v in enumerate(self.coco_objects)}

    @classmethod
    def splits(cls, **kwargs):
        kwargs_copy = {k:v for k, v in kwargs.items()}
        train = cls(split="train", **kwargs_copy)
        val = cls(split="val", **kwargs_copy)
        return train, val

    @property
    def is_train(self):
        return self.split =='train'

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        instance_dict = {}
        #print(self.img_id_2_image_folder[img_id])
        image = load_image(self.img_id_2_image_folder[img_id])

        image, window, img_scale, padding = resize_image(image, random_pad=False)
        image = to_tensor_and_normalize(image)
        c, h, w = image.shape

        metadata_path = Path(self.img_id_2_meta_folder[img_id])
        metadata = json.load(metadata_path.open("r"))

        boxes = np.array(metadata["boxes"])[:,:-1]
        boxes *= img_scale
        boxes[:, :2] += np.array(padding[:2])[None]
        boxes[:, 2:] += np.array(padding[2:])[None]

        if self.add_image_as_a_box:
            boxes = np.row_stack((window, boxes))

        assert np.all((boxes[:, 1]>=0.) & (boxes[:, 1] < boxes[:,3]))
        assert np.all((boxes[:, 2] <= w))
        assert np.all((boxes[:, 3] <= h))
        instance_dict["boxes"] = boxes
        if int(img_id.split("-")[-1]) == 53716:
            print(" find ")
        return image, instance_dict, int(img_id.split("-")[-1])

def collate_fn(data, to_gpu=False):
    images, instances, img_ids = zip(*data)
    batch = dict()
    batch["images"] = torch.stack(images, 0)

    object_num = [len(instance["boxes"]) for instance in instances]
    max_object_num = max(object_num)
    batch_n = len(object_num)

    batch['boxes'] = torch.ones(batch_n, max_object_num, 4)* (-1)
    for i in range(batch_n):
        batch['boxes'][i, :object_num[i], :] = instances[i]['boxes']

    batch['box_masks'] = torch.all(batch["boxes"]>=0 , -1).long()
    batch["img_ids"] = torch.LongTensor(list(img_ids))

    return batch

class VCRImageLoader(DataLoader):
    @classmethod
    def from_dataset(cls, data, batch_size=1, num_workers=3, num_gpus=1, **kwargs):
        loader = cls(
            dataset=data,
            batch_size=batch_size * num_gpus,
            shuffle=False,
            num_workers=num_workers,
            collate_fn = collate_fn,
            drop_last = False,
            pin_memory = False,
            **kwargs
        )
        return loader


if __name__ == "__main__":
    train, val = VCRImage.splits()
    data0 = val.__getitem__(0)
    data1 = val.__getitem__(1)
    loader = VCRImageLoader.from_dataset(val)

    for i in loader:
        print(i)
        break
