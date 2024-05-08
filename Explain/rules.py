import torch
from torch.autograd import grad


#All of rules was consicdered about the batch situation!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def selfMeetRss(answer, probs, Rss):
    probsGrad = grad(answer, probs, retain_graph=True, grad_outputs=torch.ones_like(answer))[0]\
        .reshape(Rss.shape[0], probs.shape[0] // Rss.shape[0], probs.shape[1], probs.shape[2]).to(device)
    probs = probs.reshape(Rss.shape[0], probs.shape[0] // Rss.shape[0], probs.shape[1], probs.shape[2]).to(device)
    AttMap = torch.mean(torch.clamp(probs * probsGrad, min=0), dim=1).to(device)
    Rss += torch.bmm(AttMap, Rss).to(device)
    return Rss

def selfMeetRsq(answer, probs, Rsq):
    probsGrad = grad(answer, probs, retain_graph=True, grad_outputs=torch.ones_like(answer))[0] \
        .reshape(Rsq.shape[0], probs.shape[0] // Rsq.shape[0], probs.shape[1], probs.shape[2]).to(device)
    probs = probs.reshape(Rsq.shape[0], probs.shape[0] // Rsq.shape[0], probs.shape[1], probs.shape[2]).to(device)
    AttMap = torch.mean(torch.clamp(probs * probsGrad, min=0), dim=1).to(device)
    Rsq += torch.bmm(AttMap, Rsq).to(device)
    return Rsq

def crossMeetRss(answer, probs, Rss, Rqs):
    probsGrad = grad(answer, probs, retain_graph=True, grad_outputs=torch.ones_like(answer))[0] \
        .reshape(Rss.shape[0], probs.shape[0] // Rss.shape[0], probs.shape[1], probs.shape[2]).to(device)
    probs = probs.reshape(Rss.shape[0], probs.shape[0] // Rss.shape[0], probs.shape[1], probs.shape[2]).to(device)
    AttMap = torch.mean(torch.clamp(probs * probsGrad, min=0), dim=1).to(device)
    Rss += torch.bmm(AttMap, Rqs).to(device)
    return Rss

def crossMeetRsq(answer, probs, Rsq, Rss, Rqq):
    probsGrad = grad(answer, probs, retain_graph=True, grad_outputs=torch.ones_like(answer))[0] \
        .reshape(Rss.shape[0], probs.shape[0] // Rss.shape[0], probs.shape[1], probs.shape[2]).to(device)
    probs = probs.reshape(Rss.shape[0], probs.shape[0] // Rss.shape[0], probs.shape[1], probs.shape[2]).to(device)
    AttMap = torch.mean(torch.clamp(probs * probsGrad, min=0), dim=1).to(device)
    Rss_ = Rss - torch.eye(Rss.shape[1], Rss.shape[2], dtype=Rss.dtype).to(device)  #broadcast
    Rqq_ = Rqq - torch.eye(Rqq.shape[1], Rqq.shape[2], dtype=Rqq.dtype).to(device)
    RssAvg = Rss_ / Rss_.sum(dim=-1, keepdim=True) + torch.eye(Rss.shape[1], Rss.shape[2], dtype=Rss.dtype).to(device)
    RqqAvg = Rqq_ / Rqq_.sum(dim=-1, keepdim=True) + torch.eye(Rqq.shape[1], Rqq.shape[2], dtype=Rqq.dtype).to(device)
    Rsq += torch.bmm(torch.bmm(RssAvg.transpose(1, 2), AttMap), RqqAvg).to(device)
    return Rsq

def getScoresR(model, l, answer):
    Rii = torch.stack([torch.eye(model.mapper.Vlen, model.mapper.Vlen, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)
    Rit = torch.stack([torch.zeros(model.mapper.Vlen, l, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)
    Rtt = torch.stack([torch.eye(l, l, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)
    Rti = torch.stack([torch.zeros(l, model.mapper.Vlen, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)

    Rmm = torch.stack([torch.eye(model.mapper.Vlen + l, model.mapper.Vlen + l, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)
    first = True

    for blk in model.Visual.visual.transformer.resblocks:
        Rii = selfMeetRss(answer, blk.attn_probs, Rii)

    for blk in model.Lingual.transformer.resblocks:
        Rtt = selfMeetRss(answer, blk.attn_probs, Rtt)

    cnt = 0

    for blk in model.mapper.blocks:
        if model.mapper.Type == "A":#dual-tower, cross-attention
            cnt += 1

            Rii = selfMeetRss(answer, blk.IRelativeAtt.attn_probs, Rii)
            Rit = selfMeetRsq(answer, blk.IRelativeAtt.attn_probs, Rit)
            Rtt = selfMeetRss(answer, blk.TRelativeAtt.attn_probs, Rtt)
            Rti = selfMeetRsq(answer, blk.TRelativeAtt.attn_probs, Rti)
            if model.mapper.isCap:
                Rii_ = selfMeetRss(answer, blk.IRelativeAtt.co_attn_probs, Rii).to(Rii.device)
            else:
                Rii_ = crossMeetRss(answer, blk.IRelativeAtt.co_attn_probs, Rii, Rti).to(Rii.device)

            Rit_ = crossMeetRsq(answer, blk.IRelativeAtt.co_attn_probs, Rit, Rii, Rtt).to(Rii.device)
            if cnt > 3:
                Rtt = crossMeetRss(answer, blk.TRelativeAtt.co_attn_probs, Rtt, Rit)
                Rti = crossMeetRsq(answer, blk.TRelativeAtt.co_attn_probs, Rti, Rtt, Rii)
            Rii = Rii_
            Rit = Rit_

        if model.mapper.Type == "B":#single flow, self-attention
            if first:
                Rmm[:, :model.mapper.Vlen, :model.mapper.Vlen] = Rii
                Rmm[:, model.mapper.Vlen:, model.mapper.Vlen:] = Rtt
                first = False

            Rmm = selfMeetRss(answer, blk.RelativeAtt.attn_probs, Rmm)

    if first:
        Rmm[:, :model.mapper.Vlen, :model.mapper.Vlen] = Rii
        Rmm[:, :model.mapper.Vlen, model.mapper.Vlen:] = Rit
        Rmm[:, model.mapper.Vlen:, model.mapper.Vlen:] = Rtt
        Rmm[:, model.mapper.Vlen:, :model.mapper.Vlen] = Rti
        first = False

    return Rmm.detach()


def getScoresRaw(model, l, answer):
    Rmm = torch.stack([torch.zeros(model.mapper.Vlen + l, model.mapper.Vlen + l, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)

    if model.mapper.Type == "A":
        Rmm[:, :model.mapper.Vlen, :model.mapper.Vlen] = model.mapper.blocks[2].IRelativeAtt.attn_probs.reshape(answer.shape[0], -1, model.mapper.Vlen, model.mapper.Vlen).mean(dim=1).to(device)
        if model.mapper.isCap:
            Rmm[:, :model.mapper.Vlen, :model.mapper.Vlen] = model.mapper.blocks[2].IRelativeAtt.co_attn_probs.reshape(
                answer.shape[0], -1, model.mapper.Vlen, model.mapper.Vlen).mean(dim=1).to(device)
        else:
            Rmm[:, :model.mapper.Vlen, model.mapper.Vlen:] = model.mapper.blocks[2].IRelativeAtt.co_attn_probs.reshape(answer.shape[0], -1, model.mapper.Vlen, l).mean(dim=1).to(device)
        Rmm[:, model.mapper.Vlen:, model.mapper.Vlen:] = model.mapper.blocks[2].TRelativeAtt.attn_probs.reshape(answer.shape[0], -1, l, l).mean(dim=1).to(device)
        Rmm[:, model.mapper.Vlen:, :model.mapper.Vlen] = model.mapper.blocks[1].TRelativeAtt.co_attn_probs.reshape(answer.shape[0], -1, l, model.mapper.Vlen).mean(dim=1).to(device)
    else:
        Rmm = model.mapper.blocks[7].RelativeAtt.attn_probs.reshape(answer.shape[0], -1, model.mapper.Vlen + l, model.mapper.Vlen + l).mean(dim=1).to(device)

    return Rmm.detach()


def updateRollout(probs, Rii):
    probs = probs.reshape(Rii.shape[0], probs.shape[0] // Rii.shape[0], probs.shape[1], probs.shape[2]).to(device)
    probs = probs.mean(dim=1) + torch.eye(Rii.shape[1], probs.shape[2]).to(device)
    return torch.bmm(Rii,  probs).to(device)


def getScoresRoll(model, l, answer):
    Rii = torch.stack(
        [torch.eye(model.mapper.Vlen, model.mapper.Vlen, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)
    Rit = torch.stack([torch.zeros(model.mapper.Vlen, l, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)
    Rtt = torch.stack([torch.eye(l, l, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)
    Rti = torch.stack([torch.zeros(l, model.mapper.Vlen, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)

    Rmm = torch.stack(
        [torch.eye(model.mapper.Vlen + l, model.mapper.Vlen + l, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)
    first = True

    for blk in model.Visual.visual.transformer.resblocks:
        Rii = updateRollout(blk.attn_probs, Rii)

    for blk in model.Lingual.transformer.resblocks:
        Rtt = updateRollout(blk.attn_probs, Rtt)

    cnt = 0

    for blk in model.mapper.blocks:
        if model.mapper.Type == "A":
            cnt += 1

            Rii = updateRollout(blk.IRelativeAtt.attn_probs, Rii)
            Rtt = updateRollout(blk.TRelativeAtt.attn_probs, Rtt)

            if cnt == 2:
                Rti = torch.bmm(Rtt.transpose(1, 2), torch.bmm(blk.TRelativeAtt.\
                co_attn_probs.reshape(Rti.shape[0], -1, Rti.shape[1], Rti.shape[2]).mean(dim=1), Rii))

            if cnt == 3:
                Rit = torch.bmm(Rii.transpose(1, 2), torch.bmm(blk.IRelativeAtt.\
                co_attn_probs.reshape(Rit.shape[0], -1, Rit.shape[1], Rit.shape[2]).mean(dim=1), Rtt))


        if model.mapper.Type == "B":
            if first:
                Rmm[:, :model.mapper.Vlen, :model.mapper.Vlen] = Rii
                Rmm[:, model.mapper.Vlen:, model.mapper.Vlen:] = Rtt
                first = False

            Rmm = updateRollout(answer, blk.RelativeAtt.attn_probs, Rmm)

    if first:
        Rmm[:, :model.mapper.Vlen, :model.mapper.Vlen] = Rii
        Rmm[:, :model.mapper.Vlen, model.mapper.Vlen:] = Rit
        Rmm[:, model.mapper.Vlen:, model.mapper.Vlen:] = Rtt
        Rmm[:, model.mapper.Vlen:, :model.mapper.Vlen] = Rti
        first = False

    return Rmm.detach()



def Cchange(answer, probs, x, xx, C, modal1=None, modal2=None): #ours
    Adummy = torch.stack([torch.eye(C.shape[1], C.shape[2], dtype=answer.dtype) for _ in range(int(C.shape[0]))]).to(answer.device)
    AlphaDummy = torch.stack([torch.eye(C.shape[1], C.shape[2], dtype=answer.dtype) for _ in range(int(C.shape[0]))]).to(answer.device)
    BetaDummy = torch.stack([torch.zeros(C.shape[1], C.shape[2], dtype=answer.dtype) for _ in range(int(C.shape[0]))]).to(answer.device)

    probsGrad = grad(answer, probs, retain_graph=True, grad_outputs=torch.ones_like(answer))[0]\
        .reshape(C.shape[0], probs.shape[0] // C.shape[0], probs.shape[1], probs.shape[2]).to(device)
    preGrad = grad(answer, x, retain_graph=True, grad_outputs=torch.ones_like(answer))[0].to(device)
    postGrad = grad(answer, xx, retain_graph=True, grad_outputs=torch.ones_like(answer))[0].to(device)

    preSen = torch.clamp(preGrad * x, min=0).to(answer.device)
    postSen = torch.clamp(postGrad * xx, min=0).to(answer.device)

    if x.shape[0] != C.shape[0]: # our mapper is batch frist, but clip isn't
        preSen = preSen.permute(1, 0, 2).to(answer.device)
        postSen = postSen.permute(1, 0, 2).to(answer.device)

    preSen = torch.clamp(preSen.mean(dim=-1).mean(dim=-1), min=1e-20) # make eps = 1e-6
    postSen = torch.clamp(postSen.mean(dim=-1).mean(dim=-1), min=0)

    alpha = preSen / (preSen + postSen)
    beta = 1 - alpha

    Alpha = torch.stack([torch.full((C.shape[1], C.shape[2]), float(alpha[i])) for i in range(C.shape[0])]).to(answer.device)
    Beta  = torch.stack([torch.full((C.shape[1], C.shape[2]), float(beta[i])) for i in range(C.shape[0])]).to(answer.device)
    '''
    Alpha = torch.stack([torch.diag(alpha[i]) for i in range(C.shape[0])]).to(answer.device)
    Beta = torch.stack([torch.diag(beta[i]) for i in range(C.shape[0])]).to(answer.device)
    '''
    if modal1 == 's':
        AlphaDummy[:, :Alpha.shape[1], :Alpha.shape[2]] = Alpha
        BetaDummy[:, :Beta.shape[1], :Beta.shape[2]] = Beta
    if modal1 == 'q':
        AlphaDummy[:, -Alpha.shape[1]:, -Alpha.shape[2]:] = Alpha
        BetaDummy[:, -Beta.shape[1]:, -Beta.shape[2]:] = Beta

    probs = probs.reshape(C.shape[0], probs.shape[0] // C.shape[0], probs.shape[1], probs.shape[2])
    AttMap = torch.nn.functional.normalize(torch.mean(torch.clamp(probs * probsGrad, min=0), dim=1), dim=-1, p=1).to(answer.device)
    #AttMap = torch.nn.functional.softmax(torch.mean(torch.clamp(probs * probsGrad, min=0), dim=1), dim=-1).to(answer.device)

    if modal1 == 's' and modal2 == 's':
        Adummy[:, :AttMap.shape[1], :AttMap.shape[2]] = AttMap
    elif modal1 == 'q' and modal2 == 'q':
        Adummy[:, - AttMap.shape[1]:, - AttMap.shape[2]:] = AttMap
    elif modal1 == 's' and modal2 == 'q':
        zeros = torch.stack(
            [torch.zeros(AttMap.shape[1], AttMap.shape[1], dtype=answer.dtype) for _ in range(C.shape[0])]).to(
            answer.device)
        Adummy[:, :AttMap.shape[1], :AttMap.shape[1]] =  zeros
        Adummy[:, :AttMap.shape[1], AttMap.shape[1]:] = AttMap
    elif modal1 == 'q' and modal2 == 's':
        zeros = torch.stack(
            [torch.zeros(AttMap.shape[1], AttMap.shape[1], dtype=answer.dtype) for _ in range(C.shape[0])]).to(
            answer.device)
        Adummy[:, AttMap.shape[2]:, AttMap.shape[2]:] = zeros
        Adummy[:, AttMap.shape[2]:, :AttMap.shape[2]] = AttMap
    else:
        Adummy = AttMap

    #C = torch.bmm(AlphaDummy, C).to(device) + torch.bmm(BetaDummy ,torch.bmm(Adummy, C).to(device)).to(device)
    C = Alpha * C + Beta * torch.bmm(Adummy, C)

    return C, C, torch.bmm(Adummy, C)



def getScoresC(model, l, answer):
    C = torch.stack([torch.eye(model.mapper.Vlen + l, model.mapper.Vlen + l, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)
    Cres = []
    Catt = []

    for blk in model.Visual.visual.transformer.resblocks:
        Call = Cchange(answer, blk.attn_probs, blk.x, blk.xx, C, modal1='s', modal2='s')
        C = Call[0]
        Cres.append(Call[1])
        Catt.append(Call[2])

    for blk in model.Lingual.transformer.resblocks:
        Call = Cchange(answer, blk.attn_probs, blk.x, blk.xx, C, modal1='q', modal2='q')
        C = Call[0]
        Cres.append(Call[1])
        Catt.append(Call[2])

    cnt = 0

    for blk in model.mapper.blocks:
        if model.mapper.Type == "A":
            cnt +=1
            Call = Cchange(answer, blk.IRelativeAtt.attn_probs, blk.IRelativeAtt.x, blk.IRelativeAtt.xx, C, modal1='s', modal2='s')
            C = Call[0]

            Call = Cchange(answer, blk.TRelativeAtt.attn_probs, blk.TRelativeAtt.x, blk.TRelativeAtt.xx, C, modal1='q', modal2='q')
            C = Call[0]


            if model.mapper.isCap:
                Call = Cchange(answer, blk.IRelativeAtt.co_attn_probs, blk.IRelativeAtt.xy, blk.IRelativeAtt.xxy, C, modal1='s', modal2='s')
                C = Call[0]
            else:
                Call = Cchange(answer, blk.IRelativeAtt.co_attn_probs, blk.IRelativeAtt.xy, blk.IRelativeAtt.xxy, C, modal1='s', modal2='q')
                C = Call[0]
            if cnt < 3:
                Call = Cchange(answer, blk.TRelativeAtt.co_attn_probs, blk.TRelativeAtt.xy, blk.TRelativeAtt.xxy, C, modal1='q', modal2='s')
                C = Call[0]
        if model.mapper.Type == "B":
            Call = Cchange(answer, blk.RelativeAtt.attn_probs, blk.RelativeAtt.x, blk.RelativeAtt.xx, C)
            C = Call[0]
    return C.detach(), Cres, Catt


def CchangeX(answer, probs, x, xx, C, modal1=None, modal2=None): #ours
    Adummy = torch.stack([torch.eye(C.shape[1], C.shape[2], dtype=answer.dtype) for _ in range(int(C.shape[0]))]).to(answer.device)


    probs = probs.reshape(C.shape[0], probs.shape[0] // C.shape[0], probs.shape[1], probs.shape[2])
    AttMap = torch.nn.functional.normalize(torch.mean(probs, dim=1), dim=-1, p=1).to(answer.device)
    #AttMap = torch.nn.functional.softmax(torch.mean(torch.clamp(probs * probsGrad, min=0), dim=1), dim=-1).to(answer.device)

    if modal1 == 's' and modal2 == 's':
        Adummy[:, :AttMap.shape[1], :AttMap.shape[2]] = AttMap
    elif modal1 == 'q' and modal2 == 'q':
        Adummy[:, - AttMap.shape[1]:, - AttMap.shape[2]:] = AttMap
    elif modal1 == 's' and modal2 == 'q':
        zeros = torch.stack(
            [torch.zeros(AttMap.shape[1], AttMap.shape[1], dtype=answer.dtype) for _ in range(C.shape[0])]).to(
            answer.device)
        Adummy[:, :AttMap.shape[1], :AttMap.shape[1]] =  zeros
        Adummy[:, :AttMap.shape[1], AttMap.shape[1]:] = AttMap
    elif modal1 == 'q' and modal2 == 's':
        zeros = torch.stack(
            [torch.zeros(AttMap.shape[1], AttMap.shape[1], dtype=answer.dtype) for _ in range(C.shape[0])]).to(
            answer.device)
        Adummy[:, AttMap.shape[2]:, AttMap.shape[2]:] = zeros
        Adummy[:, AttMap.shape[2]:, :AttMap.shape[2]] = AttMap
    else:
        Adummy = AttMap

    #C = torch.bmm(AlphaDummy, C).to(device) + torch.bmm(BetaDummy ,torch.bmm(Adummy, C).to(device)).to(device)
    C = 0.5 * C + 0.5 * torch.bmm(Adummy, C)

    return C, C, torch.bmm(Adummy, C)


def getScoresCX(model, l, answer):
    C = torch.stack([torch.eye(model.mapper.Vlen + l, model.mapper.Vlen + l, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)
    Cres = []
    Catt = []

    for blk in model.Visual.visual.transformer.resblocks:
        Call = CchangeX(answer, blk.attn_probs, blk.x, blk.xx, C, modal1='s', modal2='s')
        C = Call[0]
        Cres.append(Call[1])
        Catt.append(Call[2])

    for blk in model.Lingual.transformer.resblocks:
        Call = CchangeX(answer, blk.attn_probs, blk.x, blk.xx, C, modal1='q', modal2='q')
        C = Call[0]

    cnt = 0

    for blk in model.mapper.blocks:
        if model.mapper.Type == "A":
            cnt +=1
            Call = CchangeX(answer, blk.IRelativeAtt.attn_probs, blk.IRelativeAtt.x, blk.IRelativeAtt.xx, C, modal1='s', modal2='s')
            C = Call[0]
            Call = CchangeX(answer, blk.TRelativeAtt.attn_probs, blk.TRelativeAtt.x, blk.TRelativeAtt.xx, C, modal1='q', modal2='q')
            C = Call[0]

            if model.mapper.isCap:
                Call = CchangeX(answer, blk.IRelativeAtt.co_attn_probs, blk.IRelativeAtt.xy, blk.IRelativeAtt.xxy, C, modal1='s', modal2='s')
                C = Call[0]
            else:
                Call = CchangeX(answer, blk.IRelativeAtt.co_attn_probs, blk.IRelativeAtt.xy, blk.IRelativeAtt.xxy, C, modal1='s', modal2='q')
                C = Call[0]
            if cnt < 3:
                Call = CchangeX(answer, blk.TRelativeAtt.co_attn_probs, blk.TRelativeAtt.xy, blk.TRelativeAtt.xxy, C, modal1='q', modal2='s')
                C = Call[0]
        if model.mapper.Type == "B":
            Call = CchangeX(answer, blk.RelativeAtt.attn_probs, blk.RelativeAtt.x, blk.RelativeAtt.xx, C)
            C = Call[0]

    return C.detach(), Cres, Catt