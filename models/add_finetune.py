import torch, math
import torch.nn as nn
# import torch.nn.functional as F
from transformers.activations import ACT2FN
from models.prompt_transformer import AttentionLayerO2TwoUpdateNodeGeneral
# import numpy as np
# from models.common import MLP
# from torch_scatter import scatter_mean
from einops import rearrange
from torch import einsum
import torch


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def stable_softmax(t, dim=-1):
    t = t - t.amax(dim=dim, keepdim=True)
    return t.softmax(dim=dim)


class PositionalEncoding(nn.Module):
    def __init__(self, M: int, H_dim: int, D: int, fix_sigma=False):

        super().__init__()
        self.M = M
        self.H_dim = H_dim
        self.D = D
        if not fix_sigma:
            self.sigma = nn.Parameter(torch.tensor([2.]))
        else:
            self.sigma = 2
        self.mlp = nn.Sequential(
            nn.Linear(self.M, self.H_dim, bias=True),
            nn.Mish(),
            nn.Linear(self.H_dim, self.D)
        )

    def forward(self, x):
        x = torch.exp(-0.5 * x / self.sigma ** 2)
        projected = self.mlp(x)

        return projected


class CrossAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads=32,
            dim_head=32,
            context_dim=None,
            dropout=0.,
            prenorm=False,
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()
        self.attn_mlp = nn.Sequential(
            nn.Linear(heads * 2, heads * 4, bias=False),
            nn.Mish(),
            nn.Linear(heads * 4, heads * 2)
        )
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias=False)
        """
        self.to_out = nn.Sequential(nn.Linear(inner_dim, inner_dim*2),
                                    nn.Mish(),
                                    nn.Linear(inner_dim*2, dim)
                                    )
        """
        self.to_out = nn.Linear(inner_dim, dim)
        self.dis_to_out = nn.Sequential(
            nn.Linear(heads, heads * 2, bias=True),
            nn.Mish(),
            nn.Linear(heads * 2, heads)
        )
        self.context_to_out = nn.Linear(inner_dim, context_dim)

    def forward(
            self,
            x,
            context, batch_dis, batch_dis_emb, protein_len, ligand_len,
            return_attn=False,
            computer_context=False,
    ):
        h, dtype = self.heads, x.dtype

        x = self.norm(x)
        context = self.context_norm(context)

        # get shared query/keys and values for sequence and context
        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)

        # split out head
        qk, context_qk, v, context_v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h=h), (qk, context_qk, v, context_v))

        # get similarities
        sims = list(map(lambda x: einsum('h i d, h j d -> h i j', x[0], x[1]) * self.scale, zip(qk.split(ligand_len, dim=1), context_qk.split(protein_len, dim=1))))
        sims = list(map(lambda x: torch.cat((x[0], x[1].permute(2, 1, 0)), dim=0), zip(sims, batch_dis_emb)))
        sims = list(map(lambda x: self.attn_mlp(x.permute(1, 2, 0)), sims))
        # sim = einsum('h i d, h j d -> h i j', qk, context_qk) * self.scale
        # get attention along both sequence length and context length dimensions
        # shared similarity matrix
        attns = list(map(lambda x: stable_softmax(x.permute(2, 0, 1), dim=-1), sims))
        # attn = stable_softmax(sim, dim = -1)
        if computer_context:
            context_attns = list(map(lambda x: stable_softmax(x, dim=-2), sims))

        attns = list(map(self.dropout, attns))
        if computer_context:
            context_attns = list(map(self.context_dropout, attns))

        # src sequence aggregates values from context, context aggregates values from src sequence
        # out = einsum('h i j, h j d -> h i d', attn, context_v)
        out = torch.cat(list(map(lambda x: einsum('h i j, h j d -> h i d', x[0].to(context_v.dtype)[:self.heads, :, :], x[1]), zip(attns, context_v.split(protein_len, dim=1)))), dim=1)
        out_dis = torch.cat(list(map(lambda x: einsum('h i j, j i d -> i d h', x[0][self.heads:, :, :], x[1]), zip(attns, batch_dis))), dim=0)
        if computer_context:
            # context_out = einsum('h j i, h j d -> h i d', context_attn, v)
            context_out = torch.cat(list(map(lambda x: einsum('h j i, h j d -> h i d', x[0], x[1]), zip(context_attns, v.split(ligand_len, dim=1)))), dim=1)

        # merge heads and combine out
        out = rearrange(out, 'h n d -> n (h d)')
        if computer_context:
            context_out = rearrange(context_out, 'h n d -> n (h d)')

        out = self.to_out(out).to(dtype)
        out_dis = self.dis_to_out(out_dis)
        if computer_context:
            context_out = self.context_to_out(context_out)

        if return_attn:
            if computer_context:
                return out, context_out, attns, context_attns
            else:
                return out, out_dis, attns
        else:
            if computer_context:
                return out, context_out
            else:
                return out, out_dis


class GateFFNDense(nn.Module):
    def __init__(self, model_dim, dropout_rate, hidden_unit=256):
        super(GateFFNDense, self).__init__()

        self.W = nn.Linear(model_dim, hidden_unit, bias=False)
        self.V = nn.Linear(model_dim, hidden_unit, bias=False)
        self.W2 = nn.Linear(hidden_unit, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.W(hidden_states))
        hidden_linear = self.V(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.W2(hidden_states)
        return hidden_states


class GateFFNLayer(nn.Module):
    def __init__(self, model_dim, dropout_rate):
        super(GateFFNLayer, self).__init__()

        self.DenseReluDense = GateFFNDense(model_dim, dropout_rate)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states).to(dtype)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


peft_config = {
    'init_weights': 'bert',
    'target_modules': 'AttentionLayerO2TwoUpdateNodeGeneral'
}


class BottleneckModel(torch.nn.Module):

    def __init__(self, model, config=peft_config):
        super().__init__()
        self.model = model
        self.peft_config = config
        self._find_and_replace()
        self.forward = self.model.forward

    def _find_and_replace(self):

        is_target_modules_in_base_model = False
        key_dict = {key: value for key, value in self.model.named_modules()}
        for key in key_dict.keys():

            if isinstance(key_dict[key], AttentionLayerO2TwoUpdateNodeGeneral):
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                # determine the type of adapter to be used, this will effect the forward pass

                if isinstance(key_dict[key], AttentionLayerO2TwoUpdateNodeGeneral):
                    new_module = wrap_atte(target, self.peft_config['init_weights'])

                setattr(parent, target_name, new_module)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config['target_modules']} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def __getattr__(self, name: str):

        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def mark_embedding_as_fix(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "cross" not in n and "prompt" not in n and "inference" not in n:
            p.requires_grad = False


class wrap_atte(nn.Module):
    def __init__(
            self,
            target,
            init_weights: str):
        super().__init__()
        self.n_heads = 16
        self.cross_atte = CrossAttention(dim=128, heads=self.n_heads, context_dim=256)
        self.pos_emb = PositionalEncoding(3, 64, self.n_heads)
        # self.gatedFFN = GateFFNLayer(131,0.2)
        self.model = target
        self.init_weights = init_weights

        # 初始化提示参数
        self.reset_parameters()

    def reset_parameters(self):

        if hasattr(self, "cross_atte"):
            if self.init_weights == "bert":
                self.cross_atte.apply(self.init_bert_weights)
            elif self.init_weights == "mam_adapter":
                nn.init.kaiming_uniform_(self.cross_atte.weight, a=math.sqrt(5))
            else:
                raise ValueError("Unknown init_weights type: {}".format(peft_config["init_weights"]))

    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def train(self, mode: bool = True):
        self.cross_atte.train(mode)

    def eval(self):
        self.cross_atte.eval()

    def forward(self, h, x, edge_type, edge_index, mask_ligand, protein_len, ligand_len, dist_feat, interaction, e_w, fix_x, return_attention=False):

        h, x = self.model(h, x, edge_type, edge_index, mask_ligand, dist_feat, interaction, e_w=e_w, fix_x=fix_x)

        h_protein, h_ligand = h[~mask_ligand], h[mask_ligand]
        x_ligand, x_protein = x[mask_ligand], x[~mask_ligand]
        # print(x_protein.shape, protein_len, x_ligand.shape, ligand_len)
        batch_dis = list(
            map(lambda x: x[0][:, None, :] - x[1], zip(x_protein.split(protein_len, 0), x_ligand.split(ligand_len, 0)))
        )
        # batch_dis_emb = list(map(lambda x:self.pos_emb(torch.norm(x, p=2, dim=-1).unsqueeze(2)),batch_dis))
        batch_dis_emb = list(map(lambda x: self.pos_emb(x), batch_dis))
        context = torch.cat((h_protein, interaction), 1)
        if not return_attention:
            input, batch_dis = self.cross_atte(h_ligand, context, batch_dis, batch_dis_emb, protein_len, ligand_len)  # 修改,不同batch之间不应该有信息交换
        else:
            input, batch_dis, att = self.cross_atte(h_ligand, context, batch_dis, batch_dis_emb, protein_len, ligand_len, return_attn=return_attention)
        # input = self.gatedFFN(input)
        h_added = torch.zeros_like(h)
        x_added = torch.zeros_like(x)
        h_added[mask_ligand] = input
        x_added[mask_ligand] = batch_dis.mean(-1)
        h = h + h_added.clone()
        x = x + x_added.clone()
        if not return_attention:
            return h, x
        else:
            return h, x, att
