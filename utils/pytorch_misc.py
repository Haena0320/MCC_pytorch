import logging
import os
import json
import glob
import torch
from torch.nn import DataParallel
from path import Path
import shutil
import re
import time
def time_batch(gen, reset_every=100):
    """
    Gets timing info for a batch
    :param gen:
    :param reset_every: How often we'll reset
    :return:
    """
    start = time.time()
    start_t = 0
    for i, item in enumerate(gen):
        time_per_batch = (time.time() - start) / (i + 1 - start_t)
        yield time_per_batch, item
        if i % reset_every == 0:
            start = time.time()
            start_t = i

def print_para(model):
    """
    Prints parameters of a model
    :param opt:
    :return:
    """
    st = {}
    total_params = 0
    total_params_training = 0
    for p_name, p in model.named_parameters():
        # if not ('bias' in p_name.split('.')[-1] or 'bn' in p_name.split('.')[-1]):
        st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
        if p.requires_grad:
            total_params_training += np.prod(p.size())
    pd.set_option('display.max_columns', None)
    shapes_df = pd.DataFrame([(p_name, '[{}]'.format(','.join(size)), prod, p_req_grad)
                              for p_name, (size, prod, p_req_grad) in sorted(st.items(), key=lambda x: -x[1][1])],
                             columns=['name', 'shape', 'size', 'requires_grad']).set_index('name')

    print('\n {:.1f}M total parameters. {:.1f}M training \n ----- \n {} \n ----'.format(total_params / 1000000.0,
                                                                                        total_params_training / 1000000.0,
                                                                                        shapes_df.to_string()),
          flush=True)
    return shapes_df


class Flattener(torch.nn.Module):
    def __init__(self):
        super(Flattener).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def Select_obj_new_topn(v_grad, box_mask, topn, batch_size, obj_num): #visual_grad_cam, batch["box_mask"], top_num, loader_params["batch_size"], batch["object_features"].shape[1]
    topn = min(topn, obj_num) # 3

    v_grad_ind = v_grad.sort(1, descending=True)[1] # (bs, max_object_num) --> index 정렬

    v_mask_top_pos = torch.zeros(batch_size, obj_num).cuda().long()
    v_mask_top_neg = torch.zeros(batch_size, obj_num).cuda().long()
    top_pos = torch.zeros(batch_size, topn).cuda().long() # [96, 3]
    top_neg = torch.zeros(batch_size, topn).cuda().long() # [96, 3]

    for j in range(batch_size):
        top_pos[j] = v_grad_ind[j][:topn] # [2,3,8]

        if 0 in top_pos[j]: # do not contain whole img
            image_idx = list(top_pos[j]).index(0)
            top_pos[j][image_idx] = v_grad_ind[j][topn]

        badest = box_mask[j].sum() - 1 # object num (image 제외)

        if badest < topn:
            top_neg[j][:badest] = v_grad_ind[j][:badest] ## ??? positive 랑 같은디

            if 0 in top_neg[j]:
                image_idx = list(top_neg[j]).index(0)
                top_neg[j][image_idx] = v_grad_ind[j][badest-3-1]

        else:
            top_neg[j] = v_grad_ind[j][badest-topn:badest]

            if 0 in top_neg[j]:
                image_idx = list(top_neg[j]).index(0)
                top_neg[j][image_idx] = v_grad_ind[j][badest-topn-1]

    v_mask_pos = v_mask_top_pos.scatter_(1, top_pos.long(), 1)
    v_mask_neg = torch.zeros(topn, batch_size, obj_num).cuda()
    for i in range(topn):
        v_mask_neg[i] = v_mask_top_neg.scatter_(1, top_neg[:, i:].long(), 1)
    return v_mask_pos, v_mask_neg








def find_checkpoint(save_dir):
    checkpoint_files = os.listdir(save_dir)
    have_checkpoint = (save_dir is not None and any("model_state_epoch_" in x for x in checkpoint_files))
    if not have_checkpoint:
        print("there is no checkpoint ! please train model")
        return None

    model_checkpoints = [x for x in checkpoint_files if "model_state_epoch" in x]
    found_epochs = [re.search("model_state_epoch_([0-9\.\-]+)\.th", x).group(1) for x in model_checkpoints] # [0,1,2,3,4,...]
    int_epochs = []
    for epoch in found_epochs:
        pieces = epoch.split(".")
        if len(pieces) == 1:
            int_epochs.append([int(pieces[0]), 0])
        else:
            int_epochs.append([int(pieces[0]), int(pieces[1])])
    last_epoch = sorted(int_epochs, reverse=True)[0]
    if last_epoch[1] ==0:
        epoch_to_load = str(last_epoch[0])
    else:
        epoch_to_load = "{0}.{1}".format(last_epoch[0], last_epoch[1])
    save_dir_path = Path(save_dir)
    model_path = save_dir_path / "model_state_epoch_{}.th".format(epoch_to_load)
    training_state_path = save_dir_path / "training_state_epoch_{}.th".format(epoch_to_load)
    return str(model_path), str(training_state_path)


def save_checkpoint(model, optimizer, save_dir, epoch, val_metric_per_epoch, is_best=None, learning_rate_scheduler=None):
    if save_dir is not None:
        model_path = Path(save_dir) / "model_state_epoch_{}.th".format(epoch)
        model_state = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
        torch.save(model_state, model_path)

        training_state = {"epoch":epoch,
                          "val_metric_per_epoch":val_metric_per_epoch,
                          "optimizer":optimizer.state_dict()}
        if learning_rate_scheduler is not None:
            training_state["learning_rate_schedule"] = learning_rate_scheduler.state_dict()
        training_path = Path(save_dir) / "training_state_epoch_{}.th".format(epoch)
        torch.save(training_state, training_path)

    if is_best:
        print("Best validation performance so far. Copying weights to '{}/best.th'".format(save_dir))
        shutil.copyfile(str(model_path), Path(save_dir) / "best.th")

def restore_checkpoint(model, optimizer, save_dir, learning_rate_scheduler=None):

    checkpoint = find_checkpoint(save_dir)
    if checkpoint is None:
        return 0, []


    model_path, training_state_path = checkpoint
    model_state = torch.load(model_path, map_location='cuda:0')
    training_state = torch.load(training_state_path, map_location='cuda:0')

    if isinstance(model, DataParallel):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)
    optimizer.load_state_dict(training_state["optimizer"])

    if learning_rate_scheduler is not None and "learning_rate_scheduler" in training_state:
        learning_rate_scheduler.lr_scheduler.load_state_dict(training_state['learning_rate_schedule'])

    if "val_metric_per_epoch" not in training_state:
        print("trainer state 'val_metric_per_epoch' is not found, using empty list")
        val_metric_per_epoch = []
    else:
        val_metric_per_epoch = training_state['val_metric_per_epoch']

    if isinstance(training_state["epoch"], int):
        epoch_to_return = training_state['epoch'] + 1
    else:
        epoch_to_return = int(training_state["epoch"].split(".")[0]) + 1
    return epoch_to_return, val_metric_per_epoch

def restore_best_checkpoint(model, save_dir):
    fn = os.path.join(save_dir, "best.th")
    model_state = torch.load(fn, map_location="cuda:0")
    assert os.path.exists(fn)
    if isinstance(model, DataParallel):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

def clip_grad_norm(named_parameters, max_norm, clip=True, verbose=False): # global minima로 방향은 유지하되, 적게 이동함.
    max_norm = float(max_norm)
    parameters = [(n,p) for n, p in named_parameters if p.grad is not None]
    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}

    for n, p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
        param_to_norm[n] = param_norm
        param_to_shape[n] = tuple(p.size())
        if np.isnan(param_norm.item()):
            raise ValueError("the param {} was null.".format(n))

    total_norm = total_nom **(1/2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef.item() < 1 and clip:
        for n, p in parameters:
            p.grad.data.mul_(clip_coef)

    if verbose:
        print('---Total norm {:.3f} clip coef {:.3f}-----------------'.format(total_norm, clip_coef))
        for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
            print("{:<60s}: {:.3f}, ({}: {})".format(name, norm, np.prod(param_to_shape[name]), param_to_shape[name]))
        print('-------------------------------', flush=True)

    return pd.Series({name: norm.item() for name, norm in param_to_norm.items()})


class Dictobj(object):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def __getitem__(self, key):
        return getattr(self, key)

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Dictobj(value) if isinstance(value, dict) else value

def load_params(conf):
    with open(conf, "r") as f:
        config = json.load(f)
    return Dictobj(config)

