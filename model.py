import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup


class MarkdownModelCodeBERT(nn.Module):
    """Mô hình sử dụng Code-Bert, GraphCodeBert pretrain

    Args:
        nn (_type_): _description_
    """
    def __init__(self, model_path):
        super(MarkdownModelCodeBERT, self).__init__()
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


class MarkdownModelDistllBERT(nn.Module):
    """Mô hình sử dụng Distill Bert pretrain

    Args:
        nn (_type_): _description_
    """
    def __init__(self, model_path):
        super(MarkdownModelDistllBERT, self).__init__()
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


class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(2, 1)

    def forward(self, ids, mask, fts):
        x1 = self.modelA(ids, mask, fts)
        x2 = self.modelB(ids, mask, fts)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        return x


class MarkdownModel(nn.Module):
    def __init__(self, model_path_1, model_path_2):
        super(MarkdownModel, self).__init__()
        self.model_1 = AutoModel.from_pretrained(model_path_1)
        self.model_2 = AutoModel.from_pretrained(model_path_2)
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
        self.fc = nn.Linear(768+769, 769)
        self.act = nn.Sigmoid()

    def forward(self, ids, mask, fts):
        x = self.model_1(ids, mask)[0]
        y = self.model_2(ids, mask)[0]
        y = y[:, 0, :]
        x = torch.cat((x[:, 0, :], fts), 1)
        z = torch.cat([x, y], dim=1)
        z = self.fc(z)
        z = self.act(z)
        #x = self.top(x)
        x = self.net(x)
        return x
