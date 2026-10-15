import numpy as np
import math
import torch
from decoder import *
from torch.fft import irfft2
from torch.fft import rfft2
import dgl
from dgl.nn.pytorch import edge_softmax

def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

def rfft(x, d):
    t = rfft2(x, dim=-d)
    return torch.stack((t.real, t.imag), -1)

def irfft(x, d, signal_sizes):
    return irfft2(torch.complex(x[:, :, 0], x[:, :, 1]), s=signal_sizes, dim=-d)

def conj(a):
    a[..., 1] = -a[..., 1]
    return a

def ccorr(a, b):
    return irfft(com_mult(conj(rfft(a, 1)), rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


class MPKGE_path_representation(nn.Module):
    def __init__(self, dim_embed, dropout, comp_op='sub', slope=0.2):
        super(MPKGE_path_representation, self).__init__()
        self.dim_embed = dim_embed
        self.leaky_relu = nn.LeakyReLU(slope)
        self.dropout = nn.Dropout(dropout)
        self.comp_op = comp_op
        self.W_beta = nn.Linear(dim_embed, 1, bias=False)
        self.W_p = nn.Linear(dim_embed, dim_embed, bias=False)
        self.W_mlp = nn.Linear(dim_embed*2, dim_embed, bias=False)
        self.W_e = nn.Linear(dim_embed, dim_embed, bias=False)
        self.W_r = nn.Linear(dim_embed, dim_embed, bias=False)
        # nn.init.xavier_normal_(self.qT.weight)
        nn.init.xavier_normal_(self.W_beta.weight)
        nn.init.xavier_normal_(self.W_p.weight)
        nn.init.xavier_normal_(self.W_mlp.weight)
        nn.init.xavier_normal_(self.W_e.weight)
        nn.init.xavier_normal_(self.W_r.weight)

    def forward(self, instance, feature, feature_relation):
        length = instance.shape[1]
        list_h_t_im = []
        list_beta_t_im = []
        for t in range(0, length-1, 2):
            # instance: num_instance × (id_h, id_r, id_t)
            # feature: embedding(num_entity, dim)
            # 通过id从feature中检索得到num_instance × dim
            h_r = self.W_r(feature_relation[instance[:, t+1]])
            h_t = self.W_e(feature[instance[:, t+2]])
            # equation (2)
            # (num_instance, dim)
            if self.comp_op == 'sub':
                h_t_im = h_t - h_r
            elif self.comp_op == 'mul':
                h_t_im = h_t * h_r
            elif self.comp_op == 'corr':
                h_t_im = ccorr(h_t, h_r)
            elif self.comp_op == 'mlp':
                h_t_im = self.W_mlp(torch.cat([h_t, h_r], dim=1))
                h_t_im = self.dropout(h_t_im)
            else:
                raise NotImplementedError
            # equation (3)
            # (num_instance, dim) → (num_instance, dim) → (num_instance, 1)
            # beta_t_im = self.qT(torch.tanh(self.W_beta(h_t_im)))
            # beta_t_im = self.W_beta(self.bn(h_t_im))
            beta_t_im = self.leaky_relu(self.W_beta(h_t_im))
            list_h_t_im.append(h_t_im)
            list_beta_t_im.append(beta_t_im)
        # [(num_instance, 1)×t] → (t, num_instance, 1)
        beta_t_im = torch.stack(list_beta_t_im, dim=0)
        # [(num_instance, dim)×t] → (t, num_instance, dim)
        h_t_im = torch.stack(list_h_t_im, dim=0)
        # equation (4)
        # (t, num_instance, 1) → (t, num_instance, 1)
        alpha_t_im = nn.functional.softmax(beta_t_im, dim=0)
        # equation (5)
        # (t, num_instance, 1) * (t, num_instance, dim) = (t, num_instance, dim) -sum→ (num_instance, dim)
        p_im = torch.sum(alpha_t_im * h_t_im, dim=0)
        return p_im


class MPKGE_path_specific_agg(nn.Module):
    def __init__(self, dim_embed, num_head: int, dropout, comp_op, slope = 0.2):
        super(MPKGE_path_specific_agg, self).__init__()
        self.dim_embed = dim_embed
        self.num_head = num_head
        self.leaky_relu = nn.LeakyReLU(slope)
        self.dropout = nn.Dropout(dropout)

        self.W_ij = nn.Linear(dim_embed, dim_embed, bias=False)
        self.attn1 = nn.Linear(dim_embed, num_head, bias=False)
        self.attn2 = nn.Parameter(torch.empty(size=(1, num_head, dim_embed)))

        nn.init.xavier_normal_(self.W_ij.weight)
        nn.init.xavier_normal_(self.attn1.weight)
        nn.init.xavier_normal_(self.attn2.data)

        self.layer_instance_embedding = MPKGE_path_representation(dim_embed, dropout, comp_op)

    def message_passing(self, edges):
        h_piv = edges.data['gamma_v_im'] * edges.data['p_im']
        return {'h_piv': h_piv}

    def forward(self, g, instance, idx_target, feature, feature_relation):
        # (num_instance, dim)
        p_im = self.layer_instance_embedding(instance, feature, feature_relation)
        # p_im = self.W_n(p_im)
        # (num_instance, head×dim) → (1, num_instance, head×dim)
        p_im = torch.cat([p_im] * self.num_head, dim=1).unsqueeze(dim=0)
        # (1, num_instance, dim×head) → (num_instance, 1, dim×head) → (num_instance, head, dim)
        p_im_head = p_im.permute(1, 0, 2).view(-1, self.num_head, self.dim_embed)

        # equation (11)
        h_h = feature[instance[:, 0]] # head of the instance, including source and target
        # (num_instance, dim)
        h_iv = self.W_ij(h_h)
        # h_iv = self.dropout(h_iv)

        # equation (12)
        # (num_instance, dim) → (num_instance, head)
        a1 = self.attn1(h_iv)
        # (num_instance, head, dim) * (1, head, dim) → (num_instance, head, dim) → (num_instance, head)
        a2 = (p_im_head * self.attn2).sum(dim=-1)
        # (num_instance, head) → (num_instance, head, 1)
        a = (a1 + a2).unsqueeze(dim=-1)
        gamma_v_is = self.leaky_relu(a)

        g.edata.update({'p_im': p_im_head, 'gamma_v_is': gamma_v_is})
        g.edata['gamma_v_im'] = edge_softmax(g, g.edata.pop('gamma_v_is'))  # pop表示取出后删除

        # equation (13)
        g.update_all(self.message_passing, dgl.function.sum('h_piv', 'h_piv'))
        # (num_node, head, dim)
        h_embedding = g.ndata['h_piv']
        # (batch_size*2, head, dim)
        return h_embedding[idx_target]


class MPKGE_path_based_agg(nn.Module):
    def __init__(self, num_path, dim_embed, dim_w, num_head, dropout, comp_op):
        super(MPKGE_path_based_agg, self).__init__()

        self.dim_embed = dim_embed
        self.num_head = num_head

        self.layer_metapaths = nn.ModuleList()
        for _ in range(num_path):
            self.layer_metapaths.append(MPKGE_path_specific_agg(dim_embed, num_head, dropout, comp_op))

        self.W_i = nn.Linear(dim_embed, dim_w, bias=True)
        self.wT = nn.Linear(dim_w, 1, bias=False)
        nn.init.xavier_normal_(self.W_i.weight)
        nn.init.xavier_normal_(self.wT.weight)

    def forward(self, list_g, list_instance, list_idx_mapped, feature, feature_relation, device):
        list_h_piv = []
        b = len(list_idx_mapped[0])

        for g, instance, idx_mapped, layer_metapath in zip(list_g, list_instance, list_idx_mapped, self.layer_metapaths):
            if instance.shape[0]:
                # (batch_size*2, head, dim)
                h_piv = layer_metapath(g, instance, idx_mapped, feature, feature_relation)
                # equation (14) 多头求平均
                # (batch_size*2, head, dim) → (batch_size*2, dim)
                # list_h_piv.append(torch.relu(torch.mean(h_piv,dim=1)))
                list_h_piv.append(torch.mean(h_piv,dim=1))
            else:
                list_h_piv.append(torch.zeros(b, self.dim_embed).to(device))

        list_w = []
        list_mask = []
        for h_piv in list_h_piv:
            h_piv = h_piv[torch.any(h_piv != 0, dim=1)] # 筛选出不全为0的行，即该元路径下有实例的节点的嵌入
            if h_piv.numel() == 0:
                list_w.append(torch.tensor([0]).to(device))
                list_mask.append(False)
            else:
                # (batch_size*2, dim) → (batch_size*2, dim_q)
                w_p = torch.tanh(self.W_i(h_piv))
                # (batch_size*2, dim_q) → (dim_q, )
                w_pi = torch.mean(w_p, dim=0)
                # (dim_q, ) → (1,)
                w_ij = self.wT(w_pi)
                list_w.append(w_ij)
                list_mask.append(True)

        # [(1, ) × num_path] → (num_path, )
        tensor_w = torch.cat(list_w, dim=0)
        # equation (14)
        # (num_path, )
        tensor_w[list_mask] = nn.functional.softmax(tensor_w[list_mask], dim=0)
        # (num_path,) → (num_path, 1)
        tensor_tau = torch.unsqueeze(tensor_w, dim=-1)
        # (num_path, 1) → (num_path, 1, 1)
        tensor_tau = torch.unsqueeze(tensor_tau, dim=-1)

        # (batch_size*2, dim) → (1, batch_size*2, dim)
        list_h_pv_temp = [torch.unsqueeze(h_piv, dim=0) for h_piv in list_h_piv]
        # [(1, batch_size*2, dim) × num_path] → (num_path, batch_size*2, dim)
        tensor_h_pv_temp = torch.cat(list_h_pv_temp, dim=0)
        # equation (16)
        # (num_path, 1, 1)*(num_path, batch_size*2, dim) = (num_path, batch_size*2, dim) → (batch_size*2, dim)
        h_pv = torch.sum(tensor_tau * tensor_h_pv_temp, dim=0)
        # (batch_size*2, dim)
        return h_pv


class MPKGE_LP_layer(nn.Module):
    def __init__(self, dim_embed, dim_hidden, num_path, num_head, lambda_self, dropout, comp_op):
        super(MPKGE_LP_layer, self).__init__()
        self.dim_embed = dim_embed
        self.num_head = num_head
        self.lambda_self = lambda_self

        self.layer_entity = MPKGE_path_based_agg(num_path, dim_embed, dim_hidden, num_head, dropout, comp_op)

        self.W_ent = nn.Linear(dim_embed, dim_embed, bias=False)
        nn.init.xavier_normal_(self.W_ent.weight)

    def forward(self, list_g, list_instance, list_idx, array_train, feature, feature_relation, device):
        # (batch_size*2, dim)
        h_pv_entity = self.layer_entity(list_g, list_instance, list_idx, feature, feature_relation, device)
        array_entity = np.concatenate([array_train[:,0], array_train[:,2]])
        # (batch_size*2, dim)
        h_entity = self.W_ent(feature[array_entity].squeeze() * (1 - self.lambda_self) + h_pv_entity * self.lambda_self)
        # # (num_ent, dim)
        feature[array_entity] = h_entity

        return feature

class MPKGE_LP(nn.Module):
    def __init__(self, dim_embed, dim_hidden, num_rel, list_dim_feature, num_layer, num_path, num_head, lambda_self, dropout, comp_op, score_model):
        super(MPKGE_LP,self).__init__()

        self.dim_embed = dim_embed
        self.n_layer = num_layer

        self.feature_relation = nn.Embedding(num_rel, dim_embed)
        self.W_rel = nn.Linear(dim_embed, dim_embed, bias=False)
        nn.init.xavier_normal_(self.feature_relation.weight)
        nn.init.xavier_normal_(self.W_rel.weight)

        self.drop_feature = nn.Dropout(dropout)

        self.list_fc = nn.ModuleList()
        for dim_feature in list_dim_feature:
            fc = nn.Linear(dim_feature, dim_embed, bias=False)
            nn.init.xavier_normal_(fc.weight)
            self.list_fc.append(fc)

        self.layer1 = MPKGE_LP_layer(dim_embed, dim_hidden, num_path, num_head, lambda_self, dropout, comp_op)

        if num_layer>1:
            self.layers = nn.ModuleList()
            for _ in range(num_layer):
                self.layers.append(MPKGE_LP_layer(dim_embed, dim_hidden, num_path, num_head, lambda_self, dropout, comp_op))

        if score_model == 'transe':
            self.decoder = TransEmbedding()
        elif score_model == 'distmult':
            self.decoder = DistMultEmbedding()
        elif score_model == 'conve':
            self.decoder = ConvEmbedding(dim_embed)
        else:
            raise NotImplementedError

        self.bce_loss = nn.BCELoss()

    def forward(self, list_g, list_instance, list_idx, type_mask, list_feature, array_train, device):
        # (num_ent, dim)
        transformed_features = torch.zeros(type_mask.shape[0], self.dim_embed, device=device)
        for i, fc in enumerate(self.list_fc):
            # node_indices = np.where(type_mask == i)[0]
            node_indices = torch.where(torch.from_numpy(type_mask) == i)[0]
            # 将one-hot稀疏特征转换为指定维度的稠密特征
            transformed_features[node_indices] = fc(list_feature[i])
        transformed_features = self.drop_feature(transformed_features)

        # (num_ent, dim)
        feature_entity = self.layer1(list_g, list_instance, list_idx, array_train, transformed_features, self.feature_relation.weight, device)
        feature_relation = self.W_rel(self.feature_relation.weight)
        if self.n_layer > 1:
            for layer in self.layers:
                feature_entity = layer(list_g, list_instance, list_idx, array_train, feature_entity, feature_relation, device)
                feature_relation = self.W_rel(feature_relation)

        return feature_entity, feature_relation

    def cal_score(self, emb_e, emb_r, array_batch):

        h_emb = emb_e[array_batch[:, 0]]
        r_emb = emb_r[array_batch[:, 1]]
        t_emb = emb_e[array_batch[:, 2]]

        # (num_batch, )
        return self.decoder(h_emb, r_emb, t_emb)

    def cal_loss(self, h_mix, r_mix, t_mix):
        return self.decoder.loss(h_mix, r_mix, t_mix)