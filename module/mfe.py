from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
from module.digit_caps import DigitCaps
from module.multi_attn import MultiAttn
import torch.nn.functional as F
from einops import rearrange
from torch.optim.lr_scheduler import ReduceLROnPlateau

class focal_loss(nn.Module):
    def __init__(self, gamma=2):
        super(focal_loss, self).__init__()

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim() == 2 and labels.dim()==1
        labels = labels.view(-1, 1)  # [B * S, 1]
        preds = preds.view(-1, preds.size(-1))  # [B * S, C]

        preds_logsoft = F.log_softmax(preds, dim=1)  # 先softmax, 然后取log
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels)  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels)

        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = loss.mean()

        return loss


class loss1(nn.Module):
    def __init__(self, calssific_loss_weight):
        super(loss1, self).__init__()

        self.calssific_loss_weight = calssific_loss_weight
        self.criteria1 = torch.nn.CrossEntropyLoss()
        self.criteria2 = torch.nn.MSELoss()

    def forward(self, x, target):
        loss = self.calssific_loss_weight * self.criteria1(x, target)
        return loss


class loss2(nn.Module):
    def __init__(self, calssific_loss_weight):
        super(loss2, self).__init__()

        self.calssific_loss_weight = calssific_loss_weight

        self.criteria1 = focal_loss()
        self.criteria2 = torch.nn.MSELoss()

    def forward(self, x, target):
        loss = self.calssific_loss_weight * self.criteria1(x, target)
        return loss


class Mfe(pl.LightningModule):
    def __init__(self,
                 output_dim,
                 drop_out_rating,
                 learn_rating,
                 calssific_loss_weight,
                 epoch_changeloss,
                 attn_heads,
                 attn_heads_dim,
                 attn_times,
                 digit_dim,
                 num_routing,
                 embedding_dim_other,
                 embedding_dim_struct,
                 other_len,
                 struct_len):
        super().__init__()

        self.output_dim = output_dim
        self.learn_rating = learn_rating
        self.calssific_loss_wight = calssific_loss_weight
        self.epoch_changeloss = epoch_changeloss
        self.digit_dim = digit_dim
        self.num_routing = num_routing
        self.struct_len = struct_len
        self.other_len = other_len
        self.embedding_dim_other = embedding_dim_other  # embedding_dim
        self.embedding_dim_struct = embedding_dim_struct  # embedding_dim

        self.embedding_other = nn.Parameter(torch.randn(other_len, embedding_dim_other), requires_grad=True)
        self.embedding_struct = nn.Parameter(torch.randn(struct_len, embedding_dim_struct),
                                             requires_grad=True)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool1d(1)

        self.multi_attn = []
        for _ in range(attn_times):
            self.multi_attn.append(
                [MultiAttn(embedding_dim_struct, embedding_dim_struct, heads=attn_heads, dim_head=attn_heads_dim), \
                 MultiAttn(embedding_dim_struct, embedding_dim_struct, heads=attn_heads, dim_head=attn_heads_dim)])

        self.digit_caps = DigitCaps(other_len, 2, 2, digit_dim, num_routing).to(self.device)
        self.re1 = nn.ReLU()
        self.re2 = nn.ReLU()

        # 2*digit_dim+2*struct_len
        self.AN = torch.nn.LayerNorm(2 * digit_dim + 2 * struct_len)

        self.l1 = torch.nn.Linear(2 * digit_dim + 2 * struct_len, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)

        self.l2 = torch.nn.Linear(256, output_dim)

        self.ac = nn.GELU()
        self.dr = torch.nn.Dropout(drop_out_rating)

    def multiple_one_hot(self, x,dim):
        # x -> [b,feature_len]
        new_x = []
        for i in x:
            j = np.tile(i.cpu(), (dim, 1))
            new_x.append(np.array(j))
        # new_x -> [b,embed_dim,feature_len]
        new_x = np.array(new_x)
        new_x = torch.from_numpy(new_x)
        new_x = torch.tensor(new_x, dtype=torch.float)
        # trans-> [b,feature_len, embed_dim]
        new_x = torch.transpose(new_x, -1, -2)
        new_x = new_x.to(x.device)
        return new_x

    def embedding_layer_other(self, x):
        return x * self.embedding_other

    def embedding_layer_struct(self, x):
        return x * self.embedding_struct

    def avgpool_layer(self, x):
        # avg -> [b, feature_len,1]
        x = self.avg_pool(x)
        # trans -> [b, 1, feature_len]
        return torch.transpose(x, -1, -2)

    def avgpool_layer2(self, x):
        # avg -> [b, feature_len,1]
        x = self.avg_pool2(x)
        # trans -> [b, 1, feature_len]
        return torch.transpose(x, -1, -2)

    def other_part(self, other_A, other_B):
        # [b,feature_len, embed_dim]
        other_A, other_B = self.multiple_one_hot(other_A,self.embedding_dim_other), self.multiple_one_hot(other_B,self.embedding_dim_other)
        # embedding part ->  [b,feature_len,embed_dim]
        embed_other_A, embed_other_B = self.embedding_layer_other(other_A), self.embedding_layer_other(other_B)
        # average pooling part -> [b,1,feature_len]
        embed_other_A, embed_other_B = self.avgpool_layer(embed_other_A), self.avgpool_layer(embed_other_B)
        # hstack->[b, 2, feature_len]
        embed_other = torch.hstack((embed_other_A, embed_other_B))
        # multi_interest -> [b ,2, digit_dim]
        multi_interest = self.digit_caps(embed_other)
        # multi_interest -> [b ,2*digit_dim]
        multi_interest = rearrange(multi_interest, 'b n d -> b (n d)')
        return multi_interest

    def struct_part(self, struct_A, struct_B):
        # [b,feature_len, embed_dim]
        struct_A, struct_B = self.multiple_one_hot(struct_A,self.embedding_dim_struct), self.multiple_one_hot(struct_B,self.embedding_dim_struct)
        # embedding part ->  [b,feature_len,embed_dim]
        embed_struct_A, embed_struct_B = self.embedding_layer_struct(struct_A), self.embedding_layer_struct(struct_B)

        while len(self.multi_attn) > 0:
            attnA, attnB = self.multi_attn.pop(0)
            embed_struct_A = attnA(embed_struct_A, embed_struct_B)
            embed_struct_B = attnB(embed_struct_B, embed_struct_A)

        # hstack->[b, 1, feature_len]
        embed_struct_A, embed_struct_B = self.avg_pool2(embed_struct_A), self.avg_pool2(embed_struct_B)
        # hstack->[b, 2, feature_len]
        embed_struct = torch.hstack((embed_struct_A, embed_struct_B))

        multi_attn_feat = rearrange(embed_struct, 'b n d -> b (n d)')
        return multi_attn_feat

    def forward(self, x):
        feature_len = x.shape[1] // 2
        drugA, drugB = x[:, :feature_len], x[:, feature_len:]
        other_A, other_B = drugA[:, :feature_len - self.struct_len], drugB[:, :feature_len - self.struct_len]
        struct_A, struct_B = drugA[:, feature_len - self.struct_len:], drugB[:, feature_len - self.struct_len:]

        multi_interest = self.other_part(other_A, other_B)

        multi_attn_feat = self.struct_part(struct_A, struct_B)

        x = torch.cat((multi_interest, multi_attn_feat), 1)

        x = self.re1(x)
        x = self.re2(x)

        x = self.dr(self.bn1(self.ac(self.l1(x))))

        x = self.l2(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        targets = y
        pred = self(x.float())

        curr_epoch = self.current_epoch
        if curr_epoch > self.epoch_changeloss:
            loss = loss1(self.calssific_loss_wight)
        else:
            loss = loss2(self.calssific_loss_wight)

        train_loss = loss(pred, targets)

        self.log("train_loss", train_loss)
        return train_loss

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=self.learn_rating)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": ReduceLROnPlateau(optimizer),
             "monitor": "train_loss",
            },
        }

    def validation_step(self, batch, batch_idx):
        inputs, target = batch

        x = self(inputs.float())

        curr_epoch = self.current_epoch
        if curr_epoch < self.epoch_changeloss:
            loss = loss1(self.calssific_loss_wight)
        else:
            loss = loss2(self.calssific_loss_wight)

        train_loss = loss(x, target)

        self.log("val_loss", train_loss)
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, target = batch
        x = self(inputs.float())
        return x
