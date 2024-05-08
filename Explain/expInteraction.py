from .rules import *

# for this level, we can do in-place op

def getInteraction(model, l, answer):
    C = torch.stack([torch.eye(model.mapper.Vlen + l, model.mapper.Vlen + l, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)
    Ait = torch.stack([torch.zeros(model.mapper.Vlen, l, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)
    Ati = torch.stack([torch.zeros(l, model.mapper.Vlen, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)

    for blk in model.Visual.visual.transformer.resblocks:
        C = Cchange(answer, blk.attn_probs, blk.x, blk.xx, C, modal1='s', modal2='s')

    for blk in model.Lingual.transformer.resblocks:
        C = Cchange(answer, blk.attn_probs, blk.x, blk.xx, C, modal1='q', modal2='q')

    cnt = 0

    for blk in model.mapper.blocks:
        if model.mapper.Type == "A":
            cnt +=1
            C = Cchange(answer, blk.IRelativeAtt.attn_probs, blk.IRelativeAtt.x, blk.IRelativeAtt.xx, C, modal1='s', modal2='s')
            C = Cchange(answer, blk.TRelativeAtt.attn_probs, blk.TRelativeAtt.x, blk.TRelativeAtt.xx, C, modal1='q', modal2='q')

            Ci_ = 1 * C[:, :model.mapper.Vlen, :model.mapper.Vlen]
            Ct_ = 1 * C[:, model.mapper.Vlen:, model.mapper.Vlen:]  # for new a tensor

            probsit = blk.IRelativeAtt.co_attn_probs
            probsGradit = grad(answer, probsit, retain_graph=True, grad_outputs=torch.ones_like(answer))[0] \
                .reshape(C.shape[0], probsit.shape[0] // C.shape[0], probsit.shape[1], probsit.shape[2]).to(device)
            AttMapit = torch.mean(torch.clamp(probsit * probsGradit, min=0), dim=1)

            probsti = blk.TRelativeAtt.co_attn_probs
            probsGradti = grad(answer, probsti, retain_graph=True, grad_outputs=torch.ones_like(answer))[0] \
                .reshape(C.shape[0], probsti.shape[0] // C.shape[0], probsti.shape[1], probsti.shape[2]).to(device)
            AttMapti = torch.mean(torch.clamp(probsti * probsGradti, min=0), dim=1)

            C = Cchange(answer, blk.IRelativeAtt.co_attn_probs, blk.IRelativeAtt.xy, blk.IRelativeAtt.xxy, C, modal1='s', modal2='q')
            Ait += torch.bmm(Ci_, torch.bmm(AttMapit, Ct_)) # preivous composition!

            if cnt < 3:
                C = Cchange(answer, blk.TRelativeAtt.co_attn_probs, blk.TRelativeAtt.xy, blk.TRelativeAtt.xxy, C, modal1='q', modal2='s')
                Ati += torch.bmm(Ct_, torch.bmm(AttMapti, Ci_))


        if model.mapper.Type == "B":
            Ci_ = 1 * C[:, :model.mapper.Vlen, :model.mapper.Vlen]
            Ct_ = 1 * C[:, model.mapper.Vlen:, model.mapper.Vlen:]  # for new a tensor

            probsit = blk.RelativeAtt.attn_probs[:, :model.mapper.Vlen, model.mapper.Vlen:]
            probsGradit = grad(answer, probsit, retain_graph=True, grad_outputs=torch.ones_like(answer))[0] \
                .reshape(C.shape[0], probsit.shape[0] // C.shape[0], probsit.shape[1], probsit.shape[2]).to(device)
            AttMapit = torch.mean(torch.clamp(probsit * probsGradit, min=0), dim=1)

            probsti = blk.RelativeAtt.attn_probs[:, model.mapper.Vlen:, :model.mapper.Vlen]
            probsGradti = grad(answer, probsti, retain_graph=True, grad_outputs=torch.ones_like(answer))[0] \
                .reshape(C.shape[0], probsti.shape[0] // C.shape[0], probsti.shape[1], probsti.shape[2]).to(device)
            AttMapti = torch.mean(torch.clamp(probsti * probsGradti, min=0), dim=1)

            C = Cchange(answer, blk.IRelativeAtt.attn_probs, blk.RelativeAtt.x, blk.IRelativeAtt.xx, C)
            Ait += torch.bmm(C[:, :model.mapper.Vlen, :model.mapper.Vlen],
                             torch.bmm(AttMapit, Ct_))  # formal composition!
            Ati += torch.bmm(C[:, model.mapper.Vlen:, model.mapper.Vlen:], torch.bmm(AttMapti, Ci_))


    return Ait, Ati, Ait.transpose(1, 2) * Ati


def getInteractionX(model, l, answer):
    C = torch.stack([torch.eye(model.mapper.Vlen + l, model.mapper.Vlen + l, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)
    Ait = torch.stack([torch.zeros(model.mapper.Vlen, l, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)
    Ati = torch.stack([torch.zeros(l, model.mapper.Vlen, dtype=answer.dtype) for _ in range(answer.shape[0])]).to(device)

    for blk in model.Visual.visual.transformer.resblocks:
        C = CchangeX(answer, blk.attn_probs, blk.x, blk.xx, C, modal1='s', modal2='s')

    for blk in model.Lingual.transformer.resblocks:
        C = CchangeX(answer, blk.attn_probs, blk.x, blk.xx, C, modal1='q', modal2='q')

    cnt = 0

    for blk in model.mapper.blocks:
        if model.mapper.Type == "A":
            cnt += 1
            C = CchangeX(answer, blk.IRelativeAtt.attn_probs, blk.IRelativeAtt.x, blk.IRelativeAtt.xx, C, modal1='s', modal2='s')
            C = CchangeX(answer, blk.TRelativeAtt.attn_probs, blk.TRelativeAtt.x, blk.TRelativeAtt.xx, C, modal1='q', modal2='q')

            Ci_ = 1 * C[:, :model.mapper.Vlen, :model.mapper.Vlen]
            Ct_ = 1 * C[:, model.mapper.Vlen:, model.mapper.Vlen:]  # for new a tensor

            probsit = blk.IRelativeAtt.co_attn_probs
            AttMapit = torch.mean(probsit, dim=0, keepdim=True)

            probsti = blk.TRelativeAtt.co_attn_probs
            AttMapti = torch.mean(probsti, dim=0, keepdim=True)

            C = CchangeX(answer, blk.IRelativeAtt.co_attn_probs, blk.IRelativeAtt.xy, blk.IRelativeAtt.xxy, C, modal1='s', modal2='q')
            Ait += torch.bmm(Ci_, torch.bmm(AttMapit, Ct_)) # preivous composition!

            if cnt < 3:
                C = Cchange(answer, blk.TRelativeAtt.co_attn_probs, blk.TRelativeAtt.xy, blk.TRelativeAtt.xxy, C, modal1='q', modal2='s')
                Ati += torch.bmm(Ct_, torch.bmm(AttMapti, Ci_))


        if model.mapper.Type == "B":
            Ci_ = 1 * C[:, :model.mapper.Vlen, :model.mapper.Vlen]
            Ct_ = 1 * C[:, model.mapper.Vlen:, model.mapper.Vlen:]  # for new a tensor

            probsit = blk.RelativeAtt.attn_probs[:, :model.mapper.Vlen, model.mapper.Vlen:]
            AttMapit = torch.mean(probsit, dim=1)

            probsti = blk.RelativeAtt.attn_probs[:, model.mapper.Vlen:, :model.mapper.Vlen]
            AttMapti = torch.mean(probsti, dim=1)

            C = Cchange(answer, blk.IRelativeAtt.attn_probs, blk.RelativeAtt.x, blk.IRelativeAtt.xx, C)
            Ait += torch.bmm(C[:, :model.mapper.Vlen, :model.mapper.Vlen],
                             torch.bmm(AttMapit, Ct_))  # formal composition!
            Ati += torch.bmm(C[:, model.mapper.Vlen:, model.mapper.Vlen:], torch.bmm(AttMapti, Ci_))


    return Ait, Ati, Ait.transpose(1, 2) * Ati










