from .basic import CLIPimageEncoder, CLIPtextEncoder
from .block import blockA, blockB
from torch import nn
import torch
from copy import deepcopy

class Mapper(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, layers=6, type='A', Vlen=50, isCap=True):
        super().__init__()
        self.Type = type
        self.Vlen = Vlen
        self.isCap = isCap
        self.dim = embed_dim

        if type == 'A':
            self.blocks = nn.ModuleList([blockA(embed_dim, num_heads, Vlen, isCap) for _ in range(layers)])
        elif type == 'B':
            self.blocks = nn.ModuleList([blockB(embed_dim, num_heads, Vlen, isCap) for _ in range(layers)])

    def initalize(self):
        proj_std = (self.dim ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.dim ** -0.5
        fc_std = (2 * self.dim) ** -0.5

        for blk in self.blocks:
            if type == 'A':
                nn.init.normal_(blk.IRelativeAtt.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(blk.IRelativeAtt.attn.out_proj_weight, std=proj_std)
                nn.init.normal_(blk.IRelativeAtt.coattn.in_proj_weight, std=attn_std)
                nn.init.normal_(blk.IRelativeAtt.coattn.out_proj_weight, std=proj_std)
                nn.init.normal_(blk.IRelativeAtt.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(blk.IRelativeAtt.mlp.c_proj.weight, std=proj_std)

                nn.init.normal_(blk.TRelativeAtt.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(blk.TRelativeAtt.attn.out_proj_weight, std=proj_std)
                nn.init.normal_(blk.TRelativeAtt.coattn.in_proj_weight, std=attn_std)
                nn.init.normal_(blk.TRelativeAtt.coattn.out_proj_weight, std=proj_std)
                nn.init.normal_(blk.TRelativeAtt.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(blk.TRelativeAtt.mlp.c_proj.weight, std=proj_std)

            elif type == 'B':
                nn.init.normal_(blk.RelativeAtt.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(blk.RelativeAtt.attn.out_proj_weight, std=proj_std)
                nn.init.normal_(blk.RelativeAtt.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(blk.RelativeAtt.mlp.c_proj.weight, std=proj_std)

    def forward(self, I, T, keyPad, Imask):
        M = torch.concat([I, T], dim=1)

        for i in range(len(self.blocks)):
            M = self.blocks[i](M, keyPad, Imask)

        return M


class CaptionHead(nn.Module):  # Nature Language Generation
    def __init__(self):
        super().__init__()
        self.ToVocab = nn.Linear(512, 49408)
        self.soft = nn.Softmax(dim=-1)

    def initalize(self):
        std = 512 ** (-0.5)
        nn.init.normal_(self.ToVocab.weight, std=std)

    def forward(self, M):
        T = self.ToVocab(M)
        return self.soft(T)


class VqaHead(nn.Module): # MultiModal Classification
    def __init__(self):
        super().__init__()
        self.ToCLS = nn.Sequential( 
                nn.Linear(512, 512 * 2),
                nn.LayerNorm(512 * 2),
                nn.GELU(),
                nn.Linear(512 * 2, 3129)
        )
        self.soft = nn.Softmax(dim=-1)

    def initalize(self):
        std = 512 ** (-0.5)
        nn.init.normal_(self.ToCLS[0].weight, std=std)
        nn.init.normal_(self.ToCLS[3].weight, std=std)

    def forward(self, M):
        cls = self.ToCLS(M)
        return self.soft(cls), cls


class CLIPmapper(nn.Module):
    def __init__(self, Visualbackbone, Lingualbackbone, mapper, TaskHead, Vdim=768):
        super().__init__()
        self.Visual = Visualbackbone
        self.Lingual = Lingualbackbone
        self.mapper = mapper
        self.TaskHead = TaskHead
        self.Vproj = nn.Parameter(torch.empty(Vdim, 512))
        self.Tproj = nn.Parameter(torch.empty(512, 512))
        self.isFreeze = False


    def initialize(self):
        self.mapper.initalize()
        self.TaskHead.initalize()
        std = 512 ** (-0.5)
        nn.init.normal_(self.Vproj, std=std)
        nn.init.normal_(self.Tproj, std=std)

    def freeze(self):
        for name, para in self.Visual.named_parameters():
            para.requires_grad_(False)
        for name, para in self.Lingual.named_parameters():
            para.requires_grad_(False)
        for name, para in self.mapper.named_parameters():
            para.requires_grad_(True)
        for name, para in self.TaskHead.named_parameters():
            para.requires_grad_(True)
        self.Vproj.requires_grad_(True)
        self.Tproj.requires_grad_(True)
        self.isFreeze = True

    def unfreeze(self):
        for name, para in self.Visual.named_parameters():
            para.requires_grad_(True)
        for name, para in self.Lingual.named_parameters():
            para.requires_grad_(True)
        for name, para in self.mapper.named_parameters():
            para.requires_grad_(True)
        for name, para in self.TaskHead.named_parameters():
            para.requires_grad_(True)
        self.Vproj.requires_grad_(True)
        self.Tproj.requires_grad_(True)
        self.isFreeze = False

    def forward(self, I, T, keyPad = None, Imask = None):
        if self.isFreeze is not True:
            I = self.Visual(I, Imask)
            T = self.Lingual(T, keyPad)

        I = I @ self.Vproj
        T = T @ self.Tproj

        M = self.mapper(I, T, keyPad, Imask)

        if isinstance(self.TaskHead, CaptionHead):
            return self.TaskHead(M[:, self.mapper.Vlen:])
        
        if isinstance(self.TaskHead, VqaHead):
            return self.TaskHead(M[:, 0])
