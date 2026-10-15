import torch
import torch.nn as nn


class ConvEmbedding(nn.Module):
    def __init__(self, dim, hidden_dim=64, reshape_H=20, filter_size=(1, 3), margin = 1.0):
        super(ConvEmbedding, self).__init__()
        self.margin = margin
        self.channels_in = 1
        self.channels_out = hidden_dim
        # 嵌入重塑后的高和宽，即行数和列数
        self.reshape_H = reshape_H
        self.reshape_W = dim // self.reshape_H
        self.conv = nn.Conv2d(self.channels_in, self.channels_out, filter_size, stride=(1, 1), padding=0, bias=True)
        self.out_h = (self.reshape_H * 2 - filter_size[0]) // 1 + 1
        self.out_w = (self.reshape_W - filter_size[1]) // 1 + 1

        self.fc_length = self.conv.out_channels * self.out_h * self.out_w
        self.fc = nn.Linear(self.fc_length, dim)
        self.drop_1 = nn.Dropout(0.2)
        self.drop_2 = nn.Dropout(0.5)

        self.bn1 = nn.BatchNorm2d(self.channels_in)
        self.bn2 = nn.BatchNorm2d(self.channels_out)
        self.bn3 = nn.BatchNorm1d(dim)

    def forward(self, h_emb, r_emb, t_emb):
        # (batch_size, dim)→(batch_size, 1, H, W)
        emb_head = h_emb.view(-1, 1, self.reshape_H, self.reshape_W)
        emd_rel = r_emb.view(-1, 1, self.reshape_H, self.reshape_W)

        # 堆叠合并，第3个维度相加，即h
        # (batch_size, 1, 2H, W)
        x_input = torch.cat([emb_head, emd_rel], 2)
        x = self.bn1(x_input)
        x = self.drop_1(x)

        # 卷积
        # (batch_size, 1, 2H, W) → (batch_size, channels_out, 2H-filter_h+1, W-filter_w+1)
        x = self.conv(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.drop_2(x)

        # 全连接层，投影到d维
        # (batch_size, channels_out, 2H-filter_h+1, W-filter_w+1) -→ (batch_size, ?)
        x = x.view(x.shape[0], -1)
        # (batch_size, ?) -→ (batch_size, d)
        x = self.fc(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.drop_2(x)
        # (batch_size, d) * (batch_size, d) -→ (batch_size, d) -sum→ (batch_size, )
        t_emb = self.drop_1(t_emb)
        score = torch.sum(x * t_emb, dim=1)
        # return torch.sigmoid(score)
        return score

    def loss(self, h_mix, r_mix, t_mix):
        pos_h, neg_h = torch.chunk(h_mix, chunks = 2, dim=0)
        pos_r, neg_r = torch.chunk(r_mix, chunks = 2, dim=0)
        pos_t, neg_t = torch.chunk(t_mix, chunks = 2, dim=0)
        pos_score = self(pos_h, pos_r, pos_t)
        neg_score = self(neg_h, neg_r, neg_t)

        return nn.functional.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )


class DistMultEmbedding(nn.Module):
    def __init__(self, margin=1.0):
        super(DistMultEmbedding, self).__init__()
        self.margin = margin

    def forward(self, h_emb, r_emb, t_emb):

        return (h_emb * r_emb * t_emb).sum(dim=-1)

    def loss(self, h_mix, r_mix, t_mix):
        pos_h, neg_h = torch.chunk(h_mix, chunks = 2, dim=0)
        pos_r, neg_r = torch.chunk(r_mix, chunks = 2, dim=0)
        pos_t, neg_t = torch.chunk(t_mix, chunks = 2, dim=0)
        pos_score = self(pos_h, pos_r, pos_t)
        neg_score = self(neg_h, neg_r, neg_t)

        return nn.functional.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )


class TransEmbedding(nn.Module):
    def __init__(self, margin=1.0, p_norm=1.0):
        super(TransEmbedding, self).__init__()
        self.p_norm = p_norm
        self.margin = margin

    def forward(self, h_emb, r_emb, t_emb):
        # (batch_size, dim)
        head = nn.functional.normalize(h_emb, p=self.p_norm, dim=-1)
        tail = nn.functional.normalize(t_emb, p=self.p_norm, dim=-1)
        # (batch_size, )
        return -((head + r_emb) - tail).norm(p=self.p_norm, dim=-1)

    def loss(self, h_mix, r_mix, t_mix):
        pos_h, neg_h = torch.chunk(h_mix, chunks = 2, dim=0)
        pos_r, neg_r = torch.chunk(r_mix, chunks = 2, dim=0)
        pos_t, neg_t = torch.chunk(t_mix, chunks = 2, dim=0)
        pos_score = self(pos_h, pos_r, pos_t)
        neg_score = self(neg_h, neg_r, neg_t)

        return nn.functional.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )