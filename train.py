import os
import glob
import gc
from copy import copy
import numpy as np
import pandas as pd
import importlib
import sys
from tqdm import tqdm, notebook
import argparse
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import transformers
# from decouple import Config, RepositoryEnv
import neptune
import random
from neptune.utils import stringify_unsupported
from utils import calc_grad_norm, set_seed


BASEDIR= './'#'../input/asl-fingerspelling-config'
for DIRNAME in 'configs data models postprocess metrics'.split():
    sys.path.append(f'{BASEDIR}/{DIRNAME}/')


parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename", default="cfg_ch_38")
parser.add_argument("-G", "--gpu_id", default="", help="GPU ID")
parser_args, other_args = parser.parse_known_args(sys.argv)
cfg = copy(importlib.import_module(parser_args.config).cfg)
if parser_args.gpu_id != "":
    os.environ['CUDA_VISIBLE_DEVICES'] = str(parser_args.gpu_id)

# overwrite params in config with additional args
if len(other_args) > 1:
    other_args = {k.replace('-',''):v for k, v in zip(other_args[1::2], other_args[2::2])}

    for key in other_args:
        if key in cfg.__dict__:

            print(f'overwriting cfg.{key}: {cfg.__dict__[key]} -> {other_args[key]}')
            cfg_type = type(cfg.__dict__[key])
            if cfg_type == bool:
                cfg.__dict__[key] = other_args[key] == 'True'
            elif cfg_type == type(None):
                cfg.__dict__[key] = other_args[key]
            else:
                cfg.__dict__[key] = cfg_type(other_args[key])

                
if cfg.seed < 0:
    cfg.seed = np.random.randint(1_000_000)
print("seed", cfg.seed)
set_seed(cfg.seed)



                
# Import experiment modules
post_process_pipeline = importlib.import_module(cfg.post_process_pipeline).post_process_pipeline
calc_metric = importlib.import_module(cfg.metric).calc_metric
Net = importlib.import_module(cfg.model).Net
CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
tr_collate_fn = importlib.import_module(cfg.dataset).tr_collate_fn
val_collate_fn = importlib.import_module(cfg.dataset).val_collate_fn
batch_to_device = importlib.import_module(cfg.dataset).batch_to_device

# Start neptune
fns = [parser_args.config] + [getattr(cfg, s) for s in 'dataset model metric post_process_pipeline'.split()]
fns = sum([glob.glob(f"{BASEDIR }/*/{fn}.py") for fn in  fns], [])

if cfg.neptune_project == "common/quickstarts":
    neptune_api_token=neptune.ANONYMOUS_API_TOKEN
else:
    neptune_api_token=os.environ['NEPTUNE_API_TOKEN']
    
neptune_run = neptune.init_run(
        project=cfg.neptune_project,
        tags="demo",
        mode="async",
        api_token=neptune_api_token,
        capture_stdout=False,
        capture_stderr=False,
        source_files=fns
    )
print(f"Neptune system id : {neptune_run._sys_id}")
print(f"Neptune URL       : {neptune_run.get_url()}")
neptune_run["cfg"] = stringify_unsupported(cfg.__dict__)


# Read our training data
df = pd.read_csv(cfg.train_df)
train_df = df[df["fold"] != cfg.fold].copy()
if cfg.fold == -1:
    val_df = df[df["fold"] == 0].copy()
else:
    val_df = df[df["fold"] == cfg.fold].copy()

# cfg.data_folder = "datamount/train_landmarks_v3_mount/"
cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set up the dataset and dataloader
train_dataset = CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train")
val_dataset = CustomDataset(val_df, cfg, aug=cfg.train_aug, mode="val")

train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,#//4,
        pin_memory=cfg.pin_memory,
        collate_fn=tr_collate_fn,
    )
val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,#//4,
        pin_memory=cfg.pin_memory,
        collate_fn=val_collate_fn,
    )

# Set up the model
model = Net(cfg).to(cfg.device)

total_steps = len(train_dataset)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size),
            num_training_steps=cfg.epochs * (total_steps // cfg.batch_size),
            num_cycles=0.5
        )
scaler = GradScaler()


# Start the training and validation loop
cfg.curr_step = 0
optimizer.zero_grad()
total_grad_norm = None    
total_grad_norm_after_clip = None
i = 0 

if not os.path.exists(f"{cfg.output_dir}/fold{cfg.fold}/"): 
    os.makedirs(f"{cfg.output_dir}/fold{cfg.fold}/")

for epoch in range(cfg.epochs):
    
    cfg.curr_epoch = epoch
    progress_bar = tqdm(range(len(train_dataloader))[:], desc=f'Train epoch {epoch}')
    tr_it = iter(train_dataloader)
    losses = []
    gc.collect()

    model.train()
    for itr in progress_bar:
        i += 1
        cfg.curr_step += cfg.batch_size
        data = next(tr_it)
        torch.set_grad_enabled(True)
        batch = batch_to_device(data, cfg.device)
        if cfg.mixed_precision:
            with autocast():
                output_dict = model(batch)
        else:
            output_dict = model(batch)
        loss = output_dict["loss"]
        losses.append(loss.item())

        if cfg.grad_accumulation >1:
            loss /= cfg.grad_accumulation

            
        if cfg.mixed_precision:
            scaler.scale(loss).backward()

            if i % cfg.grad_accumulation == 0:
                if (cfg.track_grad_norm) or (cfg.clip_grad > 0):
                    scaler.unscale_(optimizer)                          
                if cfg.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:

            loss.backward()
            if i % cfg.grad_accumulation == 0:
                if cfg.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                optimizer.step()
                optimizer.zero_grad()
            

        if scheduler is not None:
            scheduler.step()

        loss_names = [key for key in output_dict if 'loss' in key]
        for l in loss_names:
            neptune_run[f"train/{l}"].log(value=output_dict[l].item(), step=cfg.curr_step)

        neptune_run["lr"].log(
                value=optimizer.param_groups[0]["lr"], step=cfg.curr_step
            )
        if total_grad_norm is not None:
            neptune_run["total_grad_norm"].log(value=total_grad_norm.item(), step=cfg.curr_step)
            neptune_run["total_grad_norm_after_clip"].log(value=total_grad_norm_after_clip.item(), step=cfg.curr_step)
    

    if (epoch + 1) % cfg.eval_epochs == 0 or (epoch + 1) == cfg.epochs:
        model.eval()
        torch.set_grad_enabled(False)
        val_data = defaultdict(list)
        val_score =0
        for ind_, data in enumerate(tqdm(val_dataloader, desc=f'Val epoch {epoch}')):
            batch = batch_to_device(data, cfg.device)
            if cfg.mixed_precision:
                with autocast():
                    output = model(batch)
            else:
                output = model(batch)
            for key, val in output.items():
                val_data[key] += [output[key]]
        for key, val in output.items():
            value = val_data[key]
            if isinstance(value[0], list):
                val_data[key] = [item for sublist in value for item in sublist]
            else:
                if len(value[0].shape) == 0:
                    val_data[key] = torch.stack(value)
                else:
                    val_data[key] = torch.cat(value, dim=0)
                    
        if cfg.save_val_data:
            torch.save(val_data, f"{cfg.output_dir}/fold{cfg.fold}/val_data_seed{cfg.seed}.pth")
            
        loss_names = [key for key in output if 'loss' in key]
        loss_names += [key for key in output if 'score' in key]

        val_df = val_dataloader.dataset.df
        pp_out = post_process_pipeline(cfg, val_data, val_df)

        val_score = calc_metric(cfg, pp_out, val_df, "val")
        if type(val_score)!=dict:
            val_score = {f'score':val_score}

        for k, v in val_score.items():
            print(f"val_{k}: {v:.3f}")
            if neptune_run:
                neptune_run[f"val/{k}"].log(v, step=cfg.curr_step)
    

        
    if not cfg.save_only_last_ckpt:
        torch.save({"model": model.state_dict()}, f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth")
        
torch.save({"model": model.state_dict()}, f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth")
print(f"Checkpoint save : " +  f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth")
