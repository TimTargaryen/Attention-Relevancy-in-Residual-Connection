import torch
from torch.autograd import grad

def selfMeetRssx(answer, probs, Rss):
    probsGrad = grad(answer, probs, retain_graph=True, allow_unused=True)[0]
    AttMap = torch.mean(torch.clamp(probs * probsGrad, min=0), dim=0)
    Rss += AttMap @ Rss
    return Rss

def selfMeetRsqx(answer, probs, Rsq):
    probsGrad = grad(answer, probs, retain_graph=True, allow_unused=True)[0]
    AttMap = torch.mean(torch.clamp(probs * probsGrad, min=0), dim=0)
    Rsq += AttMap @ Rsq
    return Rsq

def crossMeetRssx(answer, probs, Rss, Rqs):
    probsGrad = grad(answer, probs, retain_graph=True, allow_unused=True)[0]
    AttMap = torch.mean(torch.clamp(probs * probsGrad, min=0), dim=0)
    Rss += AttMap @ Rqs
    return Rss

def crossMeetRsqx(answer, probs, Rsq, Rss, Rqq):
    probsGrad = grad(answer, probs, retain_graph=True, allow_unused=True)[0]
    AttMap = torch.mean(torch.clamp(probs * probsGrad, min=0), dim=0)
    Rss_ = Rss - torch.eye(Rss.shape[0], Rss.shape[0], dtype=Rss.dtype)
    Rqq_ = Rqq - torch.eye(Rqq.shape[0], Rqq.shape[0], dtype=Rqq.dtype)
    RssAvg = Rss_ / Rss_.sum(dim=-1, keepdim=True) + torch.eye(Rss.shape[0], Rss.shape[0], dtype=Rss.dtype)
    RqqAvg = Rqq_ / Rqq_.sum(dim=-1, keepdim=True) + torch.eye(Rqq.shape[0], Rqq.shape[0], dtype=Rqq.dtype)
    Rsq += RssAvg.T @ AttMap @ RqqAvg
    return Rsq

def getScoresx(model, l, p, answer):
    Rii = torch.eye(model.mapper.Vlen, model.mapper.Vlen, dtype=answer.dtype)
    Rit = torch.zeros(model.mapper.Vlen, l, dtype=answer.dtype)
    Rtt = torch.eye(l, l, dtype=answer.dtype)
    Rti = torch.zeros(l, model.mapper.Vlen, dtype=answer.dtype)

    Rmm = torch.eye(model.mapper.Vlen + l, model.mapper.Vlen + l, dtype=answer.dtype)
    first = True

    for blk in model.Visual.visual.transformer.resblocks:
        Rii = selfMeetRssx(answer, blk.attn_probs, Rii)

    for blk in model.Lingual.transformer.resblocks:
        Rtt = selfMeetRssx(answer, blk.attn_probs, Rtt)

    cnt = 0

    for blk in model.mapper.blocks:
        if model.mapper.Type == "A":
            Rii = selfMeetRssx(answer, blk.IRelativeAtt.attn_probs, Rii)
            Rit = selfMeetRsqx(answer, blk.IRelativeAtt.attn_probs, Rit)
            Rtt = selfMeetRssx(answer, blk.TRelativeAtt.attn_probs, Rtt)
            Rti = selfMeetRsqx(answer, blk.TRelativeAtt.attn_probs, Rti)

            Rii_ = crossMeetRssx(answer, blk.IRelativeAtt.co_attn_probs, Rii, Rti).to(Rii.device)
            Rit_ = crossMeetRsqx(answer, blk.IRelativeAtt.co_attn_probs, Rit, Rii, Rtt).to(Rii.device)
            if cnt > 3:
                Rtt = crossMeetRssx(answer, blk.TRelativeAtt.co_attn_probs, Rtt, Rit)
                Rti = crossMeetRsqx(answer, blk.TRelativeAtt.co_attn_probs, Rti, Rtt, Rii)
            Rii = Rii_
            Rit = Rit_

        if model.mapper.Type == "B":
            if first:
                Rmm[:model.mapper.Vlen, :model.mapper.Vlen] = Rii
                Rmm[model.mapper.Vlen:, model.mapper.Vlen:] = Rtt
                first = False

            Rmm = selfMeetRssx(answer, blk.RelativeAtt.attn_probs, Rmm)

    if first:
        Rmm[:model.mapper.Vlen, :model.mapper.Vlen] = Rii
        Rmm[:model.mapper.Vlen, model.mapper.Vlen:] = Rit
        Rmm[model.mapper.Vlen:, model.mapper.Vlen:] = Rtt
        Rmm[model.mapper.Vlen:, :model.mapper.Vlen] = Rti
        first = False

    scores = Rmm[0] + Rmm[-1]
    Iscores = scores[1:model.mapper.Vlen].unsqueeze(0).view(p, p).detach().numpy()
    Tscores = scores[model.mapper.Vlen + 1:-1].detach().numpy()

    return Iscores, Tscores