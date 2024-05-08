import torch
from torch import nn
from .basic import ResidualAttentionBlock, QuickGELU, CrossAttentionBlock


class blockA(nn.Module): # Mapper A: Dual Flow with CrossAttention
    def __init__(self, embed_dim=512, num_heads=8, Vlen=50, isCap=True):
        super().__init__()
        self.Vlen = Vlen
        self.isCap = isCap
        
        mask = torch.empty(77, 77)
        mask.fill_(float("-inf"))
        mask.triu_(1)

        self.IRelativeAtt = CrossAttentionBlock(embed_dim, num_heads, bf=True)
        self.TRelativeAtt = CrossAttentionBlock(embed_dim, num_heads, attn_mask=mask if isCap else None, bf=True)

    def forward(self, M, keyPad, Imask=None):
        I = M[:, :self.Vlen]
        T = M[:, self.Vlen:]

        if self.isCap:
            I_ = self.IRelativeAtt(I, I, Imask)
            T_ = self.TRelativeAtt(T, I, keyPad)


            return torch.concat([I_, T_], dim=1)

        else:
            I_ = self.IRelativeAtt(I, T, Imask)
            T_ = self.TRelativeAtt(T, I, keyPad)

            return torch.concat([I_, T_], dim=1)



class blockB(nn.Module):  # Mapper B: Single Flow
    def __init__(self, embed_dim=512, num_heads=8, Vlen=50, isCap=True):
        super().__init__()
        self.Vlen = Vlen
        self.dim = embed_dim

        if isCap:
            mask = torch.zeros((Vlen + 77, Vlen + 77))

            mask2 = torch.empty(77, 77)
            mask2.fill_(float("-inf"))
            mask2.triu_(1)

            mask3 = torch.empty(Vlen, 77)
            mask3.fill_(float("-inf"))

            mask[Vlen:, Vlen:] = mask2
            mask[:Vlen, Vlen:] = mask3

            self.RelativeAtt = ResidualAttentionBlock(embed_dim, num_heads, attn_mask=mask, bf=True)

        else:
            self.RelativeAtt = ResidualAttentionBlock(embed_dim, num_heads, bf=True)

    def forward(self, M, keyPad, Imask=None):
        if keyPad is not None and Imask is None:
            keyPad = torch.concat([torch.zeros(len(keyPad), self.Vlen).to(keyPad.device).to(torch.bool), keyPad], dim=1)
        if keyPad is not None and Imask is not None:
            keyPad = torch.concat([Imask, keyPad], dim=1)

        
        return self.RelativeAtt(M, keyPad)


