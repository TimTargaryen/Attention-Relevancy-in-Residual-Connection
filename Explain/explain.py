import torch
from torch.autograd import grad
from Dataset import transform, Tokenizer
from .rules import getScoresR
from PIL import Image
from tools import BeamSearch2


def getVQAexp(model, picPath, question, device, transform, tokenizer):
    for blk in model.Visual.visual.transformer.resblocks:
        blk.exp = True
    for blk in model.Lingual.transformer.resblocks:
        blk.exp = True
    for blk in model.mapper.blocks:
        if model.mapper.type == "A":
            blk.IRelative.exp = True
            blk.TRelative.exp = True
        if model.mapper.type == "B":
            blk.Relative.exp = True

    img = Image.open(picPath)
    w, h = img.size
    img = torch.stack([transform(img).to(device)])
    txt = tokenizer.encode(question)
    txt.insert(0, 49406)
    txt.append(49407)
    txt = torch.LongTensor([txt]).to(device)
    l = txt.shape[1]
    p =  model.Visual.visual.input_resolution // model.Visual.visual.conv1.kernel_size[0]
    pad = torch.LongTensor([[0 for _ in range(l)]]).to(device)

    model = model.to(device)
    answer = model(img, txt, pad)
    answer = answer[torch.arange(answer.shape[0]), answer.argmax(dim=-1)]
    model.zero_grad()

    return getScoresR(model, l, p, answer)


"""def getCaptionExp(model, picPath, device, transform, tokenizer):
    for blk in model.Visual.visual.transformer.resblocks:
        blk.exp = True
    for blk in model.Lingual.transformer.resblocks:
        blk.exp = True
    for blk in model.mapper.blocks:
        if model.mapper.type == "A":
            blk.IRelative.exp = True
            blk.TRelative.exp = True
        if model.mapper.type == "B":
            blk.Relative.exp = True

    img = Image.open(picPath)
    w, h = img.size
    img = transform(img).to(device)

    p =  model.Visual.visual.input_resolution // model.Visual.visual.conv1.kernel_size
    pad = torch.LongTensor([[0 for _ in range(l)]]).to(device)

    model = model.to(device)
    caption = BeamSearch2(img, model, device)

    txt = tokenizer(question).to(device)
    l = txt.shape[1]
    answer = answer[torch.arange(answer.shape[0]), answer.argmax(dim=-1)]

    return getScores(model, l, p, answer)"""










