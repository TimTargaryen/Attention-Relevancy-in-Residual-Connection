'''
    This is the modifying CLIP components for our works, which
    change from the original program code, mainly, we change the
    part of 'forward', thanks to openai's CLIP, it's really a brilliant
    and impressive work which enlight me a lot!
                                                Yiming Huang, 2022.09.26.
'''

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .auxliary import multi_head_attention_forwardX

class CLIPimageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, image_resolution: int = 224, vision_width: int = 768,
                 vision_patch_size: int = 32, vision_layers: int = 12):
        super().__init__()
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

    def forward(self, image: torch.Tensor, Imask = None):
        return self.visual(image.type(self.visual.conv1.weight.dtype), Imask)


class CLIPtextEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512,  vocab_size: int = 49408, transformer_width: int = 512,
                 transformer_heads: int = 8, transformer_layers: int = 12
                 ):
        super().__init__()

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(77, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.dtype = self.positional_embedding.dtype

    def build_attention_mask(self):
        mask = torch.empty(77, 77)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, text: torch.Tensor, keyPad: torch.Tensor = None):
        x = self.token_embedding(text).type(self.dtype)

        x = x + self.positional_embedding.type(self.dtype)[:x.size(1)]
        x = x.permute(1, 0, 2)
        x = self.transformer(x, keyPad)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        return x


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module): # we do some little changes about it
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, bf = False):
        super().__init__()
        self.bf = bf

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=bf)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.exp = False
        self.attn_probs = None
        self.attn_grad = None

    def set_attn_probs(self, attn_probs):
        self.attn_probs = attn_probs

    def set_attn_grad(self, attn_grad):
        self.attn_grad = attn_grad

    def set_x(self, x):
        self.x = x

    def set_xx(self, xx):
        self.xx = xx

    def attention(self, x: torch.Tensor, keyPad: torch.Tensor = None): # not fixed length mask, but max length still 77.
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        dim = 1 if self.bf else 0
        mask = self.attn_mask[:x.size(dim), :x.size(dim)] if self.attn_mask is not None else None

        if not self.exp:
            return self.attn(x, x, x, attn_mask=mask, key_padding_mask=keyPad)[0]

        else:
            if self.bf:
                x = x.permute(1, 0, 2)
            x =  multi_head_attention_forwardX(
                x, x, x, self.attn.embed_dim, self.attn.num_heads,
                self.attn.in_proj_weight, self.attn.in_proj_bias, self.attn.bias_k, self.attn.bias_v,
                add_zero_attn=self.attn.add_zero_attn,
                dropout_p=self.attn.dropout, out_proj_weight=self.attn.out_proj.weight,
                out_proj_bias=self.attn.out_proj.bias,
                key_padding_mask=keyPad,
                need_weights=False,
                attn_mask=mask,
                attention_probs_forward_hook=self.set_attn_probs,
                attention_probs_backwards_hook=self.set_attn_grad)[0]
            if self.bf:
                x = x.permute(1, 0, 2)

            return x


    def forward(self, x: torch.Tensor, keyPad: torch.Tensor = None):
        if self.exp:
            self.set_x(x)

        a = self.attention(self.ln_1(x), keyPad)

        if self.exp:
            self.set_xx(a)

        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x


class CrossAttentionBlock(nn.Module): # we do some little changes about it
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, bf = False):
        super().__init__()
        self.bf = bf

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=bf)
        self.coattn = nn.MultiheadAttention(d_model, n_head, batch_first=bf)
        self.ln_1 = LayerNorm(d_model)
        self.ln_2Q = LayerNorm(d_model)
        self.ln_2KV = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_3 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.exp = False
        self.attn_probs = None
        self.attn_grad = None
        self.co_attn_probs = None
        self.co_attn_grad = None

    def set_attn_probs(self, attn_probs):
        self.attn_probs = attn_probs

    def set_attn_grad(self, attn_grad):
        self.attn_grad = attn_grad

    def set_co_attn_probs(self, co_attn_probs):
        self.co_attn_probs = co_attn_probs

    def set_co_attn_grad(self, co_attn_grad):
        self.co_attn_grad = co_attn_grad

    def set_x(self, x):
        self.x = x

    def set_xx(self, xx):
        self.xx = xx

    def set_xy(self, xy):
        self.xy = xy

    def set_xxy(self, xxy):
        self.xxy = xxy

    def attention(self, x: torch.Tensor, keyPad: torch.Tensor = None): # not fixed length mask, but max length still 77.
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        dim = 0 if self.bf is not True else 1
        mask = self.attn_mask[:x.size(dim), :x.size(dim)] if self.attn_mask is not None else None

        if self.exp is not True:
            outcome = self.attn(x, x, x, attn_mask=mask, key_padding_mask=keyPad)[0]
            return outcome
        else:
            x = x.permute(1, 0, 2)
            return multi_head_attention_forwardX(
                x, x, x, self.attn.embed_dim, self.attn.num_heads,
                self.attn.in_proj_weight, self.attn.in_proj_bias, self.attn.bias_k, self.attn.bias_v,
                add_zero_attn=self.attn.add_zero_attn,
                dropout_p=self.attn.dropout, out_proj_weight=self.attn.out_proj.weight,
                out_proj_bias=self.attn.out_proj.bias,
                key_padding_mask=keyPad,
                need_weights=False,
                attn_mask=mask,
                attention_probs_forward_hook=self.set_attn_probs,
                attention_probs_backwards_hook=self.set_attn_grad)[0].permute(1, 0, 2)

    def CrossAttention(self, x: torch.Tensor, y:torch.Tensor):
        if self.exp is not True:
            return self.coattn(x, y, y, attn_mask=None)[0]
        else:
            x = x.permute(1, 0, 2)
            y = y.permute(1, 0, 2)
            return multi_head_attention_forwardX(
                x, y, y, self.coattn.embed_dim, self.coattn.num_heads,
                self.coattn.in_proj_weight, self.coattn.in_proj_bias, self.coattn.bias_k, self.coattn.bias_v,
                add_zero_attn=self.coattn.add_zero_attn,
                dropout_p=self.coattn.dropout, out_proj_weight=self.coattn.out_proj.weight,
                out_proj_bias=self.coattn.out_proj.bias,
                need_weights=False,
                attention_probs_forward_hook=self.set_co_attn_probs,
                attention_probs_backwards_hook=self.set_co_attn_grad)[0].permute(1, 0, 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor, keyPad: torch.Tensor = None):
        if self.exp:
            self.set_x(x)

        a = self.attention(self.ln_1(x), keyPad)

        if self.exp:
            self.set_xx(a)

        x = x + a

        if self.exp:
            self.set_xy(x)

        a2 = self.CrossAttention(self.ln_2Q(x), self.ln_2KV(y))

        if self.exp:
            self.set_xxy(a2)

        x = x + a2
        x = x + self.mlp(self.ln_3(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, keyPad: torch.Tensor = None):
        for i in range(len(self.resblocks)):
            x = self.resblocks[i](x, keyPad)

        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, Imask=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, Imask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        return x



