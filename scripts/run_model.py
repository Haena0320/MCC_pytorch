import argparse
import os
import shutil
from pathlib import Path
import random
import multiprocessing
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
sys.path.append("/mnt/data/user8/MCC/MCC_pytorch")
from models.MCC import MultiLevelCC
from dataloaders.vcr_attribute_box import VCR, VCRLoader
from utils.pytorch_misc import time_batch, save_checkpoint, clip_grad_norm, restore_checkpoint, print_para, \
    restore_checkpoint, print_para, restore_best_checkpoint, Select_obj_new_topn, load_params

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

from utils import contrastive_loss
import torch.nn.functional as F

def random_control(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def set_processing(num_threads=2):
    torch.set_num_threads(num_threads)
    num_gpus = torch.cuda.device_count()
    print("num_gpus : {}".format(num_gpus))
    num_cpus = multiprocessing.cpu_count()
    if num_cpus == 0:
        raise ValueError('you need gpus ! ')
    return num_gpus, num_cpus

def _to_gpu(td, num_gpus=1):
    if num_gpus > 1:
        return td
    for k in td:
        if k != "metadata":
            td[k] = {k2:v.cuda(non_blocking=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(non_blocking=True)
    return td

def setup_opt(args, model):
    sch_a = args["learning_rate_scheduler"]
    del sch_a.type

    optimizer_args= args["optimizer"]
    del optimizer_args.type

    no_decay = ["bias", "BatchNorm.weight", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': optimizer_args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    adam_betas = (0.9, 0.999)

    optimizer = Adam(optimizer_grouped_parameters, lr=optimizer_args.lr, betas=adam_betas)
    scheduler = ReduceLROnPlateau(optimizer, sch_a.mode, sch_a.factor, sch_a.patience, cooldown=sch_a.cooldown, verbose=sch_a.verbose)
    return model, optimizer, scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("-params", dest='params', default="/mnt/data/user8/MCC/MCC_pytorch/models/params.json", type=str)
    parser.add_argument("-rationale", action='store_true', help="use rationale")
    parser.add_argument("-folder", help="folder location", default="./saves" , type=str)
    parser.add_argument("-no_tqdm", dest='no_tqdm', action="store_true")
    parser.add_argument("-seed", dest="seed", help='set seed nums',default=0, type=int)
    parser.add_argument("-threads", default=2, type=int)
    parser.add_argument("-num_workers", default=4, type=int)
    parser.add_argument("-reset_every",default=50, type=int)
    args = parser.parse_args()

    params = load_params(args.params)

    train, val = VCR.splits(mode="rationale" if args.rationale else "answer",
                            embs_to_load=params["dataset_reader"]["embs"],
                            only_use_relevant_dets=True,
                            expand2obj36=True)

    #environment settings
    random_control(args.seed)
    NUM_GPUS,NUM_CPUS = set_processing(args.threads)
    ARGS_RESET_EVERY = args.reset_every

    loader_params = {'batch_size':96 // NUM_GPUS, 'num_gpus': 1, 'num_workers':args.num_workers}
    train_loader = VCRLoader.from_dataset(train, **loader_params)
    val_loader = VCRLoader.from_dataset(val, **loader_params)

    print("Loading model params for {}".format( 'rationales' if args.rationale else 'answer'),flush=True)
    model = MultiLevelCC(params["model"]["option_encoder"])

    if hasattr(model, "detector"):
        for submodule in model.detector.backbone.modules():
            for p in submodule.parameters():
                p.requires_grad = False

    model = DataParallel(model).cuda() if NUM_GPUS > 1 else model.cuda()
    model, optimizer, scheduler = setup_opt(params["trainer"], model)

    save_folder = Path(args.folder)
    log_folder = save_folder / 'log'
    if save_folder.exists():
        print("Found folder! restoring", flush=True)
        start_epoch, val_metric_per_epoch = restore_checkpoint(model, optimizer, args.folder,
                                                               learning_rate_scheduler=scheduler)
    else:
        print("Making directories")
        save_folder.mkdir()
        log_folder.mkdir()
        start_epoch, val_metric_per_epoch = 0, []
        shutil.copy2(args.params, args.folder)

    writer = SummaryWriter(str(log_folder))

    num_batches = 0
    global_train_loss = []
    global_train_acc = []
    global_val_loss = []
    global_val_acc = []

    # instance CL
    CL_obj_feat = contrastive_loss.CL_feat(temperature=params["trainer"]["obj_feat_temp"])
    # Image CL
    CL_img_feat = contrastive_loss.CL_feat(temperature=params["trainer"]["img_feat_temp"])

    for epoch_num in tqdm(range(start_epoch, params["trainer"]["num_epochs"]+start_epoch)):
        train_results_1 = []
        train_results_2 = []
        norms = []
        model.train()
        for b, (time_per_batch, batch) in enumerate(time_batch(tqdm(train_loader), reset_every=args.reset_every)):
            batch = _to_gpu(batch)

            """origin loss, SemanticCL(image+question, answer)"""
            batch['objects_feat'] = batch["objects_feat"].requires_grad_()
            batch["v_mask"] = None
            batch["neg"] = False
            batch["neg_img"] = False
            optimizer.zero_grad()

            output_dict = model(**batch) # origin loss & SemanticCL_loss 생성

            label = batch["label"].long()
            label = torch.zeros(loader_params["batch_size"], 4).cuda().scatter_(1, label.unsqueeze(1), 1)# one hot , scatter(dim, index, src)

            visual_grad = torch.autograd.grad((output_dict["label_logits"] * (label.unsqueeze(1)>0).float()).sum(), batch["objects_feat"],create_graph=True)[0] # 정답 라벨의 확률에 영향을 준 probabilities
            # visual_grad shape : (bs, max_object_num, 2048)

            loss_origin = output_dict["loss"].mean()
            if 'loss_answer_feat' in output_dict:
                loss_answer_feat_origin = output_dict["loss_answer_feat"].mean()

            QV_anchor = output_dict["QV"] # fusion feature
            train_results_1.append(pd.Series({'total_loss': loss_origin.item(),
                                              "VCR_loss": output_dict['loss'].mean().item(),
                                              "accuracy": output_dict["accuracy"],
                                              "sec_per_batch": time_per_batch,
                                              "hr_per_epoch": len(train_loader) * time_per_batch / 3600}))

            writer.add_scalar("VCR_accuracy_per_batch/train", output_dict["accuracy"], num_batches)
            writer.add_scalar("VCR_loss_per_batch/train", output_dict['loss'].mean().item(), num_batches)

            v_mask = torch.zeros(loader_params["batch_size"], batch["objects_feat"].shape[1]).cuda() # [bs max_object_num]
            visual_grad_cam = visual_grad.sum(2) # (bs, max_object_num)
            visual_mask = (batch["box_masks"]==0).bool() # [bs, max_object_num]
            visual_grad_cam = visual_grad_cam.masked_fill(visual_mask, -1e9)
            top_num = params['trainer']['sample_num']
            # choose critical object
            v_mask_pos, v_mask_neg = Select_obj_new_topn(visual_grad_cam, batch["box_masks"], top_num, loader_params["batch_size"], batch["objects_feat"].shape[1])

            # v_mask_pos (batch_size, obj_num)
            # v_mask_neg (topn, batch_size, obj_num)
            """instanceCL (positive obect, negative object, anchor object) """
            # --------------------- positive instance V+ ---------------------------
            batch["v_mask"] = v_mask_pos # (bs,
            batch["neg"] = False
            batch["neg_img"] = False
            output_dict_pos = model(**batch)
            QV_pos = output_dict_pos["QV"]

            loss_pos = output_dict_pos["loss"].mean()
            if "loss_answer_feat" in output_dict_pos:
                loss_answer_feat_pos = output_dict_pos["loss_answer_feat"].mean()

            # ------------------- negative instance V- -----------------------------
            if v_mask_neg.dim() == 3:
                v_mask = v_mask_neg
            else:
                v_mask = v_mask_neg.unsqueeze(0)
            batch["v_mask"] = v_mask
            batch['neg'] = True
            batch['neg_img'] = False
            output_dict_neg = model(**batch)
            QV_neg = output_dict_neg["QV"]
            loss_neg = output_dict_neg['loss'].mean()

            # Auxilary instance loss
            loss_obj_VCR = (loss_pos + max(0, params["trainer"]["margin_obj_VCR"]-loss_neg))
            loss_obj_VCR = loss_obj_VCR * params["trainer"]["lambda_obj_VCR"]

            # L2-normalization
            QV_anchor = F.normalize(QV_anchor, p=2, dim=-1)
            QV_pos = F.normalize(QV_pos, p=2, dim=-1)
            QV_neg = F.normalize(QV_neg, p=2, dim=-1)
            #Instance contrastive loss
            loss_obj_feat = CL_obj_feat(QV_anchor, QV_pos, QV_neg)
            """ImageCL """
            if params['trainer']['img_level'] == True:
                # ---------- pos whole image -------------
                random_mask = torch.cuda.FloatTensor(loader_params["batch_size"], batch["box_masks"].shape[1]).uniform_() > 0.5
                v_mask = batch["box_masks"] * random_mask.long() # [bs, max_object_num]
                v_mask = v_mask + v_mask_pos.long() # the most important object should exists
                v_mask = (v_mask > 0).long().cuda()

                batch["v_mask"] = v_mask
                batch["neg"] = False
                batch['neg_img'] = False

                output_dict_pos_img = model(**batch)
                QV_pos_img = output_dict_pos_img["QV"]
                loss_pos_img = output_dict_pos_img["loss"].mean()
                if 'loss_answer_feat' in output_dict_pos_img:
                    loss_answer_feat_pos_img = output_dict_pos_img["loss_answer_feat"].mean()

                # ------------- neg whole img -----------------
                batch['v_mask'] = None
                batch['neg'] = True
                batch["neg_img"] = True
                output_dict_neg_img = model(**batch)
                QV_neg_img = output_dict_neg_img["QV"] # [bs, 4, 512]

                loss_neg_img = output_dict_neg_img["loss"].mean()

                loss_img_VCR = loss_pos_img + max(0, params["trainer"]["margin_img_VCR"]-loss_neg_img)
                loss_img_VCR *= params["trainer"]["lambda_img_VCR"]

                # L2-normalization
                QV_pos_img = F.normalize(QV_pos_img, p=2, dim=-1)
                QV_neg_img = F.normalize(QV_neg_img, p=2, dim=-1)
                loss_img_feat = CL_img_feat(QV_anchor, QV_pos_img, QV_neg_img)
                loss_img_feat *= params["trainer"]["lambda_answer_feat"]

                loss_semantic = loss_answer_feat_origin + loss_answer_feat_pos + loss_answer_feat_pos_img # negative 는 학습할 필요가 없음, semantic CL
                loss_semantic = loss_semantic * params["trainer"]["lambda_answer_feat"]
                loss_auxilary = loss_obj_VCR + loss_img_VCR
                loss_total = loss_origin + loss_semantic + loss_auxilary + loss_obj_feat + loss_img_feat
            else:
                loss_semantic = loss_answer_feat_origin + loss_answer_feat_pos
                loss_semantic = loss_semantic * params["trainer"]["lambda_answer_feat"]
                loss_auxilary = loss_obj_VCR
                loss_total = loss_origin + loss_auxilary + loss_obj_feat + loss_semantic

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            optimizer.zero_grad()

            train_results_2.append(pd.Series({'total_loss': loss_total.item(),
                                              'origin_loss': loss_origin.item(),
                                              'loss_obj_VCR': loss_obj_VCR.item(),
                                              'loss_obj_feat': loss_obj_feat.item(),
                                              'loss_img_VCR': loss_img_VCR.item() if params['trainer']['img_level'] else -1,
                                              'loss_img_feat': loss_img_feat.item() if params['trainer']['img_level'] else -1,
                                              'loss_answer_feat': loss_semantic.item(),
                                              }))
            writer.add_scalar("loss_origin/train", loss_origin.item(), num_batches)
            writer.add_scalar("loss_obj_VCR/train", loss_obj_VCR.item(), num_batches)
            writer.add_scalar("loss_obj_feat/train", loss_obj_feat.item(), num_batches)
            writer.add_scalar("loss_img_VCR/train", loss_img_VCR.item(), num_batches)
            writer.add_scalar("loss_img_feat/train", loss_img_feat.item(), num_batches)
            writer.add_scalar("loss_semantic_feat/train", loss_semantic.item(), num_batches)

            if b % ARGS_RESET_EVERY == 0 and b > 0:
                print("\ne{:2d}b{:5d}/{:5d}. ---- \nsumm:\n{}\n   ~~~~~~~~~~~~~~~~~~\n".format(
                    epoch_num, b, len(train_loader),
                    pd.DataFrame(train_results_2[-ARGS_RESET_EVERY:]).mean(),
                ), flush=True)

            num_batches += 1
        epoch_stats = pd.DataFrame(train_results_1).mean()
        train_loss = epoch_stats['total_loss']
        train_acc = epoch_stats['accuracy']
        writer.add_scalar('VCR_loss_per_epoch/train', train_loss, epoch_num)
        writer.add_scalar('VCR_accuracy_per_epoch/train', train_acc, epoch_num)
        global_train_loss.append(train_loss)
        global_train_acc.append(train_acc)
        print("---\nTRAIN EPOCH {:2d}: -- origin--\n{}\n--------".format(epoch_num, pd.DataFrame(train_results_1).mean()))

        val_probs = []
        val_labels = []
        val_loss_sum = 0.0
        model.eval()
        for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
            with torch.no_grad():
                batch = _to_gpu(batch)
                batch['v_mask'] = None
                batch['neg'] = False
                batch['neg_img'] = False
                output_dict = model(**batch)
                val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
                val_labels.append(batch['label'].detach().cpu().numpy())
                val_loss_sum += output_dict['loss'].mean().item() * batch['label'].shape[0]
        val_labels = np.concatenate(val_labels, 0)
        val_probs = np.concatenate(val_probs, 0)
        val_loss_avg = val_loss_sum / val_labels.shape[0]

        val_metric_per_epoch.append(float(np.mean(val_labels == val_probs.argmax(1))))
        if scheduler:
            scheduler.step(val_loss_sum)

        print("Val epoch {} has acc {:.3f} and loss {:.3f}".format(epoch_num, val_metric_per_epoch[-1], val_loss_avg),
              flush=True)
        writer.add_scalar('VCR_loss_per_epoch/validation', val_loss_avg, epoch_num)
        writer.add_scalar('VCR_accuracy_per_epoch/validation', val_metric_per_epoch[-1], epoch_num)
        global_val_loss.append(val_loss_avg)
        global_val_acc.append(val_metric_per_epoch[-1])
        # if int(np.argmax(val_metric_per_epoch)) < (len(val_metric_per_epoch) - 1 - params['trainer']['patience']):
        #     print("Stopping at epoch {:2d}".format(epoch_num))
        #     break
        if scheduler:
            save_checkpoint(model, optimizer, args.folder, epoch_num, val_metric_per_epoch,
                            is_best=int(np.argmax(val_metric_per_epoch)) == (len(val_metric_per_epoch) - 1),
                            learning_rate_scheduler=scheduler)
        else:
            save_checkpoint(model, optimizer, args.folder, epoch_num, val_metric_per_epoch,
                            is_best=int(np.argmax(val_metric_per_epoch)) == (len(val_metric_per_epoch) - 1))

    writer.close()
    print("STOPPING. now running the best model on the validation set", flush=True)
    # Load best
    restore_best_checkpoint(model, args.folder)
    model.eval()
    val_probs = []
    val_labels = []
    for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
        with torch.no_grad():
            batch = _to_gpu(batch)
            batch['v_mask'] = None
            batch['neg'] = False
            batch['neg_img'] = False
            output_dict = model(**batch)
            val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
            val_labels.append(batch['label'].detach().cpu().numpy())
    val_labels = np.concatenate(val_labels, 0)
    val_probs = np.concatenate(val_probs, 0)
    acc = float(np.mean(val_labels == val_probs.argmax(1)))

    print("Final val accuracy is {:.3f}".format(acc))
    np.save(os.path.join(args.folder, f'valpreds.npy'), val_probs)
    np.save(os.path.join(args.folder, f'global_val_loss.npy'), global_val_loss)
    np.save(os.path.join(args.folder, f'global_val_acc.npy'), global_val_acc)
    np.save(os.path.join(args.folder, f'global_train_loss.npy'), global_train_loss)
    np.save(os.path.join(args.folder, f'global_train_acc.npy'), global_train_acc)

#
#
# import numpy as np
# preds = np.load("/mnt/data/user8/MCC/MCC_pytorch/saves/valpreds.npy")






































