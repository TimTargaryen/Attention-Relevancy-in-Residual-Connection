import torch
from torch import nn
from tools import Cider
from tools.tokenizer import Tokenizer
from . import *
import nltk
from tools import PTBTokenizer
from .rules import getScoresC, getScoresRaw, getScoresR
from Dataset import transform
#from pkg_resources import packaging
from .model import CLIP
from .clip import tokenize, load

tokenizer = Tokenizer()

cider = Cider()
Clip, _ = load('ViT-B-32.pt')

dummyGT = ['Monuments of unageing intellect.', 'To the holy city of Byzantium.', 'Into the artifice of eternity.', 'Of what is past, or passing, or to come.']
dummyRef = ['Soul, not only God-given.']



def next(subs, Cs, i, word):
    subs[i] += subs[i + 1]
    subs.pop(i + 1)
    Cs[i] = Cs[i] + Cs[i + 1]
    Cs.pop(i + 1)

    if word != subs[i]:
        next(subs, Cs, i, word)

    return subs, Cs

#merge the subword
def merge(Cs, subs, words):
    for i, word in enumerate(words):
        if subs[i] == word:
            continue
        else:
            subs, Cs = next(subs, Cs, i, word)

    Cs = [torch.nn.functional.normalize(C.reshape(49), p=1, dim=-1) for C in Cs]
    Cs = [C.reshape(7, 7) for C in Cs]
    assert len(Cs) == len(words)

    return Cs, subs


def lack(words, j):
    s = ""

    for i in range(len(words)):
        if i != j:
            s += " " + words[i]

    return s


# not batched!
def computeRefCLIPscore(img, refs, gts):
    R = Clip.encode_text(tokenize(gts))
    v = Clip.encode_text(tokenize(refs))
    c = Clip.encode_image(img)

    R /= R.norm(dim=-1, keepdim=True)
    v /= v.norm(dim=-1, keepdim=True)
    c /= c.norm(dim=-1, keepdim=True)

    CLIPscore = 2.5 * torch.clamp(c @ v.T, min=0)
    maxRef = torch.max(torch.clamp(c @ R.T, min=0), dim=-1)[0]
    refCLIPscore = 2 * (CLIPscore * maxRef) / (CLIPscore + maxRef)
    return float(refCLIPscore[0])


# not batch!
def getCapScoreC(img, gt, model, device, max_length=30, mode='avg'):
    img = torch.stack([img])
    seq = torch.LongTensor([[49406]]).to(device)
    Cavg = torch.zeros(7, 7).to(device)
    f = open("stopwords.txt")
    s = f.readlines()
    s = [w.strip('\n') for w in s]
    Cs = []

    for i in range(max_length - 1):
        probs = model(img, seq)[:, -1:]
        token_id = int(probs[0].argmax(dim=-1))

        if token_id == 49406:
            break

        Cs.append(
            torch.nn.functional.normalize(
                getScoresC(model, seq.shape[1], torch.max(probs, dim=-1)[0])[:, -1, 1:50], p=1, dim=-1
            ).reshape(7, 7)
        )

        seq = torch.cat([seq, torch.LongTensor([[token_id]]).to(device)], dim=1)

    ref = tokenizer.decode([int(i) for i in seq[0][1:]])

    gts = {"1": gt, "2": dummyGT}
    refs = PTBTokenizer.tokenize({"1": [ref], "2": dummyRef})
    CIDEr = cider.compute_score(gts, refs)[0]
    refCLIPscore = computeRefCLIPscore(img, ref, gt)

    subs = [tokenizer.decode([sub]).strip() for sub in tokenizer.encode(ref)]
    words = nltk.word_tokenize(ref)
    Cs, subs = merge(Cs, subs, words)
    Alphas = []

    for i in range(len(Cs)):
        if mode == 'avg':
            if words[i].lower() in s or words[i].upper() in s:
                Alphas.append(0)
            else:
                Alphas.append(1)
                Cavg += Cs[i]

        if mode == 'cider':
            lackRefs = {"1": [lack(words, i)], "2": dummyRef}
            alpha = CIDEr - cider.compute_score(gts, lackRefs)[0]
            Alphas.append(alpha)
            Cavg += alpha * Cs[i]

        if model == 'clip':
            lackRefs = lack(words, i)
            alpha = refCLIPscore - computeRefCLIPscore(img, lackRefs, gt)
            Alphas.append(alpha)
            Cavg += alpha * Cs[i]

    return Cs, Alphas, Cavg, refs['1'], words

'''
def getCapScoreRaw(img, gt, model, device, max_length=30, mode='avg'):
    img = torch.stack([img])
    seq = torch.LongTensor([[49406]]).to(device)
    Cavg = torch.zeros(7, 7).to(device)
    Cs = []

    for i in range(max_length - 1):
        probs = model(img, seq)[:, -1:]
        token_id = int(probs[0].argmax(dim=-1))

        if token_id == 49406:
            break

        Cs.append(
            torch.nn.functional.normalize(
                getScoresRaw(model, seq.shape[1], torch.max(probs, dim=-1)[0])[:, -1, 1:50], p=1, dim=-1
            ).reshape(7, 7)
        )

        seq = torch.cat([seq, torch.LongTensor([[token_id]]).to(device)], dim=1)

    ref = tokenizer.decode([int(i) for i in seq[0][1:]])

    gts = {"1": gt, "2": dummyGT}
    refs = PTBTokenizer.tokenize({"1": [ref], "2": dummyRef})
    CIDEr = cider.compute_score(gts, refs)[0]
    refCLIPscore = computeRefCLIPscore(img, ref, gt)

    subs = [tokenizer.decode([sub]).strip() for sub in tokenizer.encode(ref)]
    words = nltk.word_tokenize(ref)
    Cs, subs = merge(Cs, subs, words)
    Alphas = []

    for i in range(len(Cs)):
        if mode == 'avg':
            Cavg += Cs[i]

        if mode == 'cider':
            lackRefs = {"1": [lack(words, i)], "2": dummyRef}
            alpha = CIDEr - cider.compute_score(gts, lackRefs)[0]
            Alphas.append(alpha)
            Cavg += alpha * Cs[i]

        if mode == 'clip':
            lackRefs = lack(words, i)
            alpha = refCLIPscore - computeRefCLIPscore(img, lackRefs, gt)
            Alphas.append(alpha)
            Cavg += alpha * Cs[i]

    return Cs, Alphas, Cavg, refs['1']

def getCapScoreR(img, gt, model, device, max_length=30, mode='avg'):
    img = torch.stack([img])
    seq = torch.LongTensor([[49406]]).to(device)
    Cavg = torch.zeros(7, 7).to(device)
    Cs = []

    for i in range(max_length - 1):
        probs = model(img, seq)[:, -1:]
        token_id = int(probs[0].argmax(dim=-1))

        if token_id == 49406:
            break

        Cs.append(
            torch.nn.functional.normalize(
                getScoresR(model, seq.shape[1], torch.max(probs, dim=-1)[0])[:, -1, 1:50], p=1, dim=-1
            ).reshape(7, 7)
        )

        seq = torch.cat([seq, torch.LongTensor([[token_id]]).to(device)], dim=1)

    ref = tokenizer.decode([int(i) for i in seq[0][1:]])

    gts = {"1": gt, "2": dummyGT}
    refs = PTBTokenizer.tokenize({"1": [ref], "2": dummyRef})
    CIDEr = cider.compute_score(gts, refs)[0]
    refCLIPscore = computeRefCLIPscore(img, ref, gt)

    subs = [tokenizer.decode([sub]).strip() for sub in tokenizer.encode(ref)]
    words = nltk.word_tokenize(ref)
    Cs, subs = merge(Cs, subs, words)
    Alphas = []

    for i in range(len(Cs)):
        if mode == 'avg':
            Cavg += Cs[i]

        if mode == 'cider':
            lackRefs = {"1": [lack(words, i)], "2": dummyRef}
            alpha = CIDEr - cider.compute_score(gts, lackRefs)[0]
            Alphas.append(alpha)
            Cavg += alpha * Cs[i]

        if model == 'clip':
            lackRefs = lack(words, i)
            alpha = refCLIPscore - computeRefCLIPscore(img, lackRefs, gt)
            Alphas.append(alpha)
            Cavg += alpha * Cs[i]

    return Cs, Alphas, Cavg, refs['1']
'''

