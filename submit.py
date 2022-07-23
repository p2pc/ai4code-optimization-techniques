import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

pd.options.display.width = 180
pd.options.display.max_colwidth = 120
data_dir = Path('../input/AI4Code')

def read_notebook(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
        .assign(id=path.stem)
        .rename_axis('cell_id')
    )

paths_test = list((data_dir / 'test').glob('*.json'))
notebooks_test = [
    read_notebook(path) for path in tqdm(paths_test, desc='Test NBs')
]
test_df = (
    pd.concat(notebooks_test)
    .set_index('id', append=True)
    .swaplevel()
    .sort_index(level='id', sort_remaining=False)
).reset_index()
test_df["rank"] = test_df.groupby(["id", "cell_type"]).cumcount()
test_df["pred"] = test_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)

# Additional code cells
def clean_code(cell):
    return str(cell).replace("\\n", "\n")


def sample_cells(cells, n):
    cells = [clean_code(cell) for cell in cells]
    if n >= len(cells):
        return [cell[:200] for cell in cells]
    else:
        results = []
        step = len(cells) / n
        idx = 0
        while int(np.round(idx)) < len(cells):
            results.append(cells[int(np.round(idx))])
            idx += step
        assert cells[0] in results
        if cells[-1] not in results:
            results[-1] = cells[-1]
        return results


def get_features(df):
    features = dict()
    df = df.sort_values("rank").reset_index(drop=True)
    for idx, sub_df in tqdm(df.groupby("id")):
        features[idx] = dict()
        total_md = sub_df[sub_df.cell_type == "markdown"].shape[0]
        code_sub_df = sub_df[sub_df.cell_type == "code"]
        total_code = code_sub_df.shape[0]
        codes = sample_cells(code_sub_df.source.values, 20)
        features[idx]["total_code"] = total_code
        features[idx]["total_md"] = total_md
        features[idx]["codes"] = codes
    return features

test_fts = get_features(test_df)

from tqdm import tqdm
import sys, os
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import torch.nn as nn
import torch

class MarkdownModel(nn.Module):
    def __init__(self, model_path):
        super(MarkdownModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        layers = []
        in_channels = 769
        hidden_channels = [300, 200, 100]
        out_channels = 1
        layers.append(nn.Linear(in_channels, hidden_channels[0]))
        layers.append(nn.Sigmoid())
        for i in range(1, len(hidden_channels)):
            layers.append(nn.Linear(hidden_channels[i-1], hidden_channels[i]))
            layers.append(nn.Sigmoid())
        layers.append(nn.Linear(hidden_channels[-1], out_channels))
        self.net = nn.Sequential(*layers)
        #self.top = nn.Linear(769, 1)

    def forward(self, ids, mask, fts):
        x = self.model(ids, mask)[0]
        x = torch.cat((x[:, 0, :], fts), 1)
        #x = self.top(x)
        x = self.net(x)
        return x

from torch.utils.data import DataLoader, Dataset
class MarkdownDataset(Dataset):

    def __init__(self, df, model_name_or_path, total_max_len, md_max_len, fts):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.fts = fts

    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = self.tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=self.md_max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        code_inputs = self.tokenizer.batch_encode_plus(
            [str(x) for x in self.fts[row.id]["codes"]],
            add_special_tokens=True,
            max_length=23,
            padding="max_length",
            truncation=True
        )
        n_md = self.fts[row.id]["total_md"]
        n_code = self.fts[row.id]["total_md"]
        if n_md + n_code == 0:
            fts = torch.FloatTensor([0])
        else:
            fts = torch.FloatTensor([n_md / (n_md + n_code)])

        ids = inputs['input_ids']
        for x in code_inputs['input_ids']:
            ids.extend(x[:-1])
        ids = ids[:self.total_max_len]
        if len(ids) != self.total_max_len:
            ids = ids + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(ids))
        ids = torch.LongTensor(ids)

        mask = inputs['attention_mask']
        for x in code_inputs['attention_mask']:
            mask.extend(x[:-1])
        mask = mask[:self.total_max_len]
        if len(mask) != self.total_max_len:
            mask = mask + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(mask))
        mask = torch.LongTensor(mask)

        assert len(ids) == self.total_max_len

        return ids, mask, fts, torch.FloatTensor([row.pct_rank])

    def __len__(self):
        return self.df.shape[0]

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

            pred = model(*inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)

def predict(model_path, ckpt_path):
    model = MarkdownModel(model_path)
    model = model.cuda()
    model.eval()
    model.load_state_dict(torch.load(ckpt_path))
    BS = 32
    NW = 8
    MAX_LEN = 64
    test_df["pct_rank"] = 0
    test_ds = MarkdownDataset(test_df[test_df["cell_type"] == "markdown"].reset_index(drop=True), md_max_len=64,total_max_len=512, model_name_or_path=model_path, fts=test_fts)
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=NW,
                              pin_memory=False, drop_last=False)
    _, y_test = validate(model, test_loader)
    return y_test

model_path = "../input/huggingface-code-models/codebert-base"
ckpt_path = "../input/optim-transformer/model_optim1.bin"
y_test_1 = predict(model_path, ckpt_path)

model_path = "../input/huggingface-code-models/graphcodebert-base"
ckpt_path = "../input/modelgraphcodebert/model_graphcodebert.bin"
y_test_2 = predict(model_path, ckpt_path)

# y_test = (y_test_1 + y_test_2) / 2
y_test = (y_test_1 * 0.3 + y_test_2 * 0.7)
# y_test = y_test_2

# y_test = y_test_2

sub_df = test_df.sort_values("pred").groupby("id")["cell_id"].apply(lambda x: " ".join(x)).reset_index()
sub_df.rename(columns={"cell_id": "cell_order"}, inplace=True)
sub_df.head()

sub_df.to_csv("submission.csv", index=False)