import json
from pathlib import Path
from dataset import *
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from model import *
from tqdm import tqdm
import sys
import os
from metrics import *
import torch
import argparse
import bitsandbytes as bnb

parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--model_name_or_path', type=str,
                    default='microsoft/graphcodebert-base')
parser.add_argument('--train_mark_path', type=str,
                    default='./data/train_mark.csv')
parser.add_argument('--train_features_path', type=str,
                    default='./data/train_fts.json')
parser.add_argument('--val_mark_path', type=str, default='./data/val_mark.csv')
parser.add_argument('--val_features_path', type=str,
                    default='./data/val_fts.json')
parser.add_argument('--val_path', type=str, default="./data/val.csv")

parser.add_argument('--md_max_len', type=int, default=64)
parser.add_argument('--total_max_len', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--accumulation_steps', type=int, default=4)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--n_workers', type=int, default=8)

args = parser.parse_args()
os.mkdir("./outputs_graphcodebert")
data_dir = Path('./input/')

train_df_mark = pd.read_csv(args.train_mark_path).drop(
    "parent_id", axis=1).dropna().reset_index(drop=True)
train_fts = json.load(open(args.train_features_path))
val_df_mark = pd.read_csv(args.val_mark_path).drop(
    "parent_id", axis=1).dropna().reset_index(drop=True)
val_fts = json.load(open(args.val_features_path))
val_df = pd.read_csv(args.val_path)

order_df = pd.read_csv("./input/train_orders.csv").set_index("id")
df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

train_ds = MarkdownDataset(train_df_mark, model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
                           total_max_len=args.total_max_len, fts=train_fts)
val_ds = MarkdownDataset(val_df_mark, model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
                         total_max_len=args.total_max_len, fts=val_fts)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
                          pin_memory=False, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                        pin_memory=False, drop_last=False)


def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


def validate(model, val_loader):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)

# Cài đặt bitsandbytes-cuda110, chọn version phù hợp
#!pip install -q bitsandbytes-cuda110


def set_embedding_parameters_bits(embeddings_path, optim_bits=32):
    """
    https://github.com/huggingface/transformers/issues/14819#issuecomment-1003427930
    """

    embedding_types = ("word", "position", "token_type")
    for embedding_type in embedding_types:
        attr_name = f"{embedding_type}_embeddings"

        if hasattr(embeddings_path, attr_name):
            bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                getattr(embeddings_path, attr_name), 'weight', {
                    'optim_bits': optim_bits}
            )


def train(model, train_loader, val_loader, epochs):
    np.random.seed(0)
    # Định nghĩa max_norm mặc định 1.5
    max_norm = 1.5

    # Creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Sử dụng optimizer Adam 8bit
    optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters, lr=3e-5)

    # Sử dựng optimizer AdamW 8bit
    #optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=3e-5)


    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                module, 'weight', {'optim_bits': 32}
            )

    num_train_optimization_steps = int(args.epochs * len(train_loader) / args.accumulation_steps)
    # optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5,
    #                   correct_bias=False)
    # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    criterion = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()

    for e in range(epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)

                # computing loss
                loss = criterion(pred, target)

            # scale gradint and perform backward pass
            scaler.scale(loss).backward()

            if idx % args.accumulation_steps == 0 or idx == len(tbar) - 1:
                # before gradient clipping the optimizer parameters must be unscaled.
                scaler.unscale_(optimizer)

                # perform optimization step
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()
                scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(
                f"Epoch {e + 1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}")

        y_val, y_pred = validate(model, val_loader)
        val_df["pred"] = val_df.groupby(["id", "cell_type"])[
            "rank"].rank(pct=True)
        val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred
        y_dummy = val_df.sort_values("pred").groupby('id')[
            'cell_id'].apply(list)
        print("Preds score", kendall_tau(
            df_orders.loc[y_dummy.index], y_dummy))
        torch.save(model.state_dict(), "./outputs_graphcodebert/model.bin")

    return model, y_pred


model = MarkdownModelCodeBERT(args.model_name_or_path)

model = model.cuda()
model, y_pred = train(model, train_loader, val_loader, epochs=args.epochs)
