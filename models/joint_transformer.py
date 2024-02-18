import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_sum
from models.common import GaussianSmearing, MLP

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=15, edge_types=4):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x + bias
        x = x.expand(-1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

class BaseX2HAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='mish', norm=True, ew_net_type='r', out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        # attention key func
        kv_input_dim = input_dim * 2 + r_feat_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index,e_w=None):
        N = h.size(0)#num_nodes
        src, dst = edge_index#2*num_edges
        hi, hj = h[dst], h[src]#num_edges*128   r_feat:num_edges*80

        # multi-head attention
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:#执行
            kv_input = torch.cat([edge_feat, kv_input], -1)#num_edges*340

        # compute k
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)#num_edges*16*8
        # compute v
        v = self.hv_func(kv_input)#num_edges*16*8

        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = self.ew_net(v[..., :self.hidden_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1)#执行 edge权重
        else:
            e_w = 1.
        v = v * e_w
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute q
        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)#num_nodes*16*8

        # compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0,
                                dim_size=N)  # [num_edges, n_heads]

        # perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (num_edges, heads, H_per_head)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (total_nodes, heads, H_per_head)
        output = output.view(-1, self.output_dim)#num_nodes,128

        if self.out_fc:#false
            output = self.node_output(torch.cat([output, h], -1))
        return output

class BaseH2XAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='mish', norm=True, ew_net_type='r'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + r_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index,e_w=None):
        N = h.size(0)
        src, dst = edge_index[0],edge_index[1]
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)#num_edges*336
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)#num_edges*340

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(kv_input)
        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = 1.
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (xi - xj) [n_edges, n_heads, 3]
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # Compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)  # (num_edges, heads:16)

        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, dst, dim=0, dim_size=N).mean(1)  # (num_nodes, heads, 3)
        return output  # [num_nodes, heads, 3]

class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='mish', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        # self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)
        # self.pro_pos_embed = PositionalEncoding(1,3,128,64,128,4)
        # self.lig_pos_embed = PositionalEncoding(1,3,128,64,128,4)
        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=15,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )

        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=15,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type)
            )

    def forward(self, h, x, edge_attr, edge_index, mask_ligand,dist_feat,interaction=None,e_w=None, fix_x=False):
        src, dst = edge_index[0],edge_index[1]#num_edges,num_edges
        # pro_pos_emb = self.pro_pos_embed(x[~mask_ligand])
        # lig_pos_emb = self.lig_pos_embed(x[mask_ligand])
        rel_x = x[dst] - x[src]#num_edges*3
        # dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)#num_edges*1
        #edge_attr: num_edges*4
        h_in = h
        # h_in[~mask_ligand] = h_in[~mask_ligand] + pro_pos_emb + interaction
        h_in[~mask_ligand] = h_in[~mask_ligand] + interaction
        # h_in[mask_ligand] = h_in[mask_ligand] + lig_pos_emb
        # dist_feat = self.distance_expansion(dist)
        for i in range(self.num_x2h):#1
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_attr, edge_index,e_w=e_w)
            h_out = h_in + h_out
            h_in = h_out
        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out#sync_twoup False
        for i in range(self.num_h2x):#1
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_attr, edge_index, e_w=e_w)
            if not fix_x:#执行
                x = x + delta_x
            rel_x = x[dst] - x[src]#num_edges*3

        return x2h_out, x

class PromptTransformer(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, k=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='mish', norm=True,
                 cutoff_mode='radius', ew_net_type='r',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        # Build the network
        self.num_blocks = num_blocks
        self.num_layers = num_layers#9
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        # radius graph / knn graph
        self.cutoff_mode = cutoff_mode  # [radius, none]
        self.k = k
        self.ew_net_type = ew_net_type  # [r, m, none]
        self.edge_cutoff = 7
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        # self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)
        self.distance_expansion = GaussianLayer()
        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)

        self.base_block = self._build_share_blocks()

    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_share_blocks(self):
        # Equivariant layers
        base_block = []
        for l_idx in range(self.num_layers):
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)
    def get_edges(self, x, batch_mask):

        adj = batch_mask[:, None] == batch_mask[None, :]
        dist_mat = torch.cdist(x, x)
        if self.edge_cutoff is not None:
            adj = adj & (dist_mat <= self.edge_cutoff)
        edges = torch.stack(torch.where(adj), dim=0)#返回的是为true的元素的坐标，以tuple形式
        radial = dist_mat[edges[0],edges[1]].unsqueeze(1)

        return edges, radial

    def build_edge_type(self,edge_index, mask_ligand):
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        # edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type
    def forward(self, h, x, mask_ligand, batch,protein_len,ligand_len, interaction,return_all=False, fix_x=False,return_attention=False):

        all_x = [x]#x坐标 total_nodes*3
        all_h = [h]#h原子类型 total_nodes*128
        # self.get_edges(x,batch)

        atts = []
        for b_idx in range(self.num_blocks):
            edge_index, edge_distance = self.get_edges(x, batch)
            edge_type = self.build_edge_type(edge_index,mask_ligand)
            dist_feat = self.distance_expansion(edge_distance,edge_type)
            if self.ew_net_type == 'global':
                # dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
                
                logits = self.edge_pred_layer(dist_feat)#num_edges,1
                e_w = torch.sigmoid(logits)#edge weights
            else:
                e_w = None
            #h,x:num_nodes*128 num_nodes*3
            for l_idx, layer in enumerate(self.base_block):#9 layers
                if not return_attention:
                    h, x = layer(h, x, None, edge_index, mask_ligand,protein_len,ligand_len, dist_feat,interaction,e_w=e_w, fix_x=fix_x)
                else:
                    h, x, att = layer(h, x, None, edge_index, mask_ligand,protein_len,ligand_len, dist_feat,interaction,e_w=e_w, fix_x=fix_x,return_attention=return_attention)
                    atts.append(att)
            all_x.append(x)
            all_h.append(h)

        outputs = {'x': x, 'h': h,'atts':atts}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h,'atts':atts})
        return outputs
