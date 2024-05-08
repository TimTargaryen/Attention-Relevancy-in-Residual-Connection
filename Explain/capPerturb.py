import sys

sys.path.append("..")
import torch
from torch import nn
import os
import json
from Explain import getCapScore
from tools import compute_scores, GreedySearch
from tools import PTBTokenizer
from tools import tokenizer

def line(c):
    for i in range(50):
        print(c, end="")
    print()


def GreedySearchMASK(img, imask, model, device, max_length=77):
    img = torch.stack([img])
    seq = torch.LongTensor([[49406]]).to(device)

    for i in range(max_length - 1):
        probs = model(img, seq, Imask=imask)[:, -1:]
        token_id = int(probs[0].argmax(dim=-1))

        if token_id == 49406:
            break

        seq = torch.cat([seq, torch.LongTensor([[token_id]]).to(device)], dim=1)

    realIds = [int(i) for i in seq[0][1:]]
    return tokenizer.decode(realIds)


def capPerturbImg(model, valLoader, bNum, positive=True):
    print("cap perturbation!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    steps = [0.05, 0.10, 0.15, 0.20, 0.25, 0.5, 0.75, 1]
    avgScores = {'b4': [0 for _ in range(len(steps) + 1)], 'm': [0 for _ in range(len(steps) + 1)], 's': [0 for _ in range(len(steps) + 1)]}
    ciderScores = {'b4': [0 for _ in range(len(steps) + 1)], 'm': [0 for _ in range(len(steps) + 1)], 's': [0 for _ in range(len(steps) + 1)]}
    clipScores = {'b4': [0 for _ in range(len(steps) + 1)], 'm': [0 for _ in range(len(steps) + 1)], 's': [0 for _ in range(len(steps) + 1)]}

    for i, (img, gts, _) in enumerate(valLoader):
        if i >= bNum:
            break

        img = img.to(device)
        gts = [gt[0] for gt in gts]
        imask = torch.zeros((1, 50))
        ref = GreedySearch(img, model, device)
        Gts = PTBTokenizer.tokenize({'1': gts})
        Refs = PTBTokenizer.tokenize({'1': [ref]})

        score, scores = compute_scores(gts, Refs)

        b, m, s =score['BLEU'][3], score['METEOR'], score["SPICE"]
        avgScores['b4'][0] += b; avgScores['m'][0] += m; avgScores['s'][0] += s;
        ciderScores['b4'][0] += b; ciderScores['m'][0] += m; ciderScores['s'][0] += s;
        clipScores['b4'][0] += b; clipScores['m'][0] += m; clipScores['s'][0] += s;

        avg = getCapScore(img.squeeze(0), gts, model, device, mode='avg')
        cider = getCapScore(img.squeeze(0), gts, model, device, mode='cider')
        clip = getCapScore(img.squeeze(0), gts, model, device, mode='clip')


        line("*")
        line("*")
        for s in range(len(steps)):
            avGmask = torch.concat([imask], dim=1)
            cideRmask = torch.concat([imask], dim=1)
            cliPmask = torch.concat([imask], dim=1)

            _, kavg = torch.topk(torch.from_numpy(avg[:, 0, :50]), k=int((steps[s]) * (avGmask.shape[1])), dim=-1,
                               largest=positive)
            _, kcider = torch.topk(torch.from_numpy(cider[:, 0, :50]), k=int((steps[s]) * (cideRmask.shape[1])), dim=-1,
                               largest=positive)
            _, kclip = torch.topk(torch.from_numpy(clip[:, 0, :50]), k=int((steps[s]) * (cliPmask.shape[1])), dim=-1,
                                  largest=positive)


            for b in range(1):
                avGmask[b, kavg[b]] = 1
                cideRmask[b, kcider[b]] = 1
                cliPmask[b, kclip[b]] = 1


            avGmask = avGmask.to(torch.long)
            cideRmask = cideRmask.to(torch.long)
            cliPmask = cliPmask.to(torch.long)


            avGref = GreedySearchMASK(img, avGmask, model, device, max_length=25)
            cideRref = GreedySearchMASK(img, cideRmask, model, device, max_length=25)
            cliPref = GreedySearchMASK(img, cliPmask, model, device, max_length=25)


            avGRefs = PTBTokenizer.tokenize({'1': [avGref]})
            cideRefs = PTBTokenizer.tokenize({'1': [cideRref]})
            cliPRefs = PTBTokenizer.tokenize({'1': [cliPref]})

            avGscore, _ = compute_scores(gts, avGRefs)
            cideRscore, _ = compute_scores(gts, cideRefs)
            cliPscore, _ = compute_scores(gts, cliPRefs)

            avGb, avGm, avGs = avGscore['BLEU'][3], avGscore['METEOR'], avGscore["SPICE"]
            cideRb, cideRm, cideRs = cideRscore['BLEU'][3], cideRscore['METEOR'], cideRscore["SPICE"]
            cliPb, cliPm, cliPs = cliPscore['BLEU'][3], cliPscore['METEOR'], cliPscore["SPICE"]

            avgScores['b4'][0] += avGb; avgScores['m'][0] += avGm; avgScores['s'][0] += avGs;
            ciderScores['b4'][0] += cideRb; ciderScores['m'][0] += cideRm; ciderScores['s'][0] += cideRs;
            clipScores['b4'][0] += cliPb; clipScores['m'][0] += cliPm; clipScores['s'][0] += cliPs;

            line("-")
            print("AVG avg b4 perturbation:", [c / (i + 1) for c in avgScores['b4']])
            print("CIDER avg b4 perturbation:", [r / (i + 1) for r in ciderScores['b4']])
            print("CLIP avg b4 perturbation:", [roll / (i + 1) for roll in clipScores['b4']])

    avgScores['b4'] = [b4 / bNum for b4 in avgScores['b4']]; avgScores['m'] = [m / bNum for m in avgScores['m']]; avgScores['s'] = [s / bNum for s in avgScores['s']];
    ciderScores['b4'] = [b4 / bNum for b4 in ciderScores['b4']]; ciderScores['b4'] = [m / bNum for m in ciderScores['m']]; ciderScores['s'] = [s / bNum for s in ciderScores['s']];
    clipScores['b4'] = [b4 / bNum for b4 in clipScores['b4']]; clipScores['b4'] = [m / bNum for m in clipScores['m']]; clipScores['s'] = [s / bNum for s in clipScores['s']];

    return avgScores, ciderScores, clipScores
