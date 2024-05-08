import sys

sys.path.append("..")
import torch
from torch import nn
import os
import json
from Explain import getScoresR, getScoresC, getScoresRaw, getScoresRoll


def line(c):
    for i in range(50):
        print(c, end="")
    print()


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    logits.to('cpu')
    one_hots = torch.zeros(*labels.size()).to('cpu')
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def VqaPerturbImg(model, valLoader, bNum, positive=True):
    print("perturbation!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    steps = [0.05, 0.10, 0.15, 0.20, 0.25, 0.5, 0.75, 1]
    correctRatesC = [0 for _ in range(len(steps) + 1)]
    correctRatesR = [0 for _ in range(len(steps) + 1)]
    correctRatesRoll = [0 for _ in range(len(steps) + 1)]
    correctRatesRaw = [0 for _ in range(len(steps) + 1)]


    for i, (img, seq, pad, label) in enumerate(valLoader):
        if i >= bNum:
            break

        img = img.to(device)
        seq = seq.to(device)
        pad = pad.to(device)
        imask = torch.zeros((pad.shape[0], 50))
        label = label.to(device)
        label[:, 0] = 0
        l = pad.shape[1]

        predict = model(img, seq, pad)

        C = getScoresC(model, l, torch.max(predict[1], dim=-1)[0])
        R = getScoresR(model, l, torch.max(predict[1], dim=-1)[0])
        Roll = getScoresRoll(model, l, torch.max(predict[1], dim=-1)[0])
        Raw = getScoresRaw(model, l, torch.max(predict[1], dim=-1)[0])

        correctRatesC[0] += (compute_score_with_logits(predict[0], label).sum() / len(label)).item()
        correctRatesR[0] += (compute_score_with_logits(predict[0], label).sum() / len(label)).item()
        correctRatesRoll[0] += (compute_score_with_logits(predict[0], label).sum() / len(label)).item()
        correctRatesRaw[0] += (compute_score_with_logits(predict[0], label).sum() / len(label)).item()

        line("*")
        line("*")
        for s in range(len(steps)):
            Cmask = torch.concat([imask], dim=1)
            Rmask = torch.concat([imask], dim=1)
            RollMask = torch.concat([imask], dim=1)
            RawMask = torch.concat([imask], dim=1)

            _, kc = torch.topk(torch.from_numpy(C[:, 0, :50]), k=int((steps[s]) * (Cmask.shape[1])), dim=-1,
                               largest=positive)
            _, kr = torch.topk(torch.from_numpy(R[:, 0, :50]), k=int((steps[s]) * (Rmask.shape[1])), dim=-1,
                               largest=positive)
            _, kroll = torch.topk(torch.from_numpy(Roll[:, 0, :50]), k=int((steps[s]) * (RollMask.shape[1])), dim=-1,
                                  largest=positive)
            _, kraw = torch.topk(torch.from_numpy(Raw[:, 0, :50]), k=int((steps[s]) * (RawMask.shape[1])), dim=-1,
                                 largest=positive)

            for b in range(Rmask.shape[0]):
                Cmask[b, kc[b]] = 1
                Rmask[b, kr[b]] = 1
                RollMask[b, kroll[b]] = 1
                RawMask[b, kraw[b]] = 1

            Cmask = Cmask.to(torch.long)
            Rmask = Rmask.to(torch.long)
            RollMask = RollMask.to(torch.long)
            RawMask = RawMask.to(torch.long)

            Cpred = model(img, seq, pad, Cmask[:, :50])[0]
            Rpred = model(img, seq, pad, Rmask[:, :50])[0]
            Rollpred = model(img, seq, pad, RollMask[:, :50])[0]
            Rawpred = model(img, seq, pad, RawMask[:, :50])[0]

            correctRatesC[s + 1] += (compute_score_with_logits(Cpred, label).sum() / len(label)).item()
            correctRatesR[s + 1] += (compute_score_with_logits(Rpred, label).sum() / len(label)).item()
            correctRatesRaw[s + 1] += (compute_score_with_logits(Rollpred, label).sum() / len(label)).item()
            correctRatesRoll[s + 1] += (compute_score_with_logits(Rawpred, label).sum() / len(label)).item()

            line("-")
            print("C avg perturbation:", [c / (i + 1) for c in correctRatesC])
            print("R avg perturbation:", [r / (i + 1) for r in correctRatesR])
            print("Roll avg perturbation:", [roll / (i + 1) for roll in correctRatesRoll])
            print("Raw avg perturbation:", [raw / (i + 1) for raw in correctRatesRaw])

    correctRatesC = [c / bNum for c in correctRatesC]
    correctRatesR = [r / bNum for r in correctRatesR]
    correctRatesRoll = [roll / bNum for roll in correctRatesRoll]
    correctRatesRaw = [raw / bNum for raw in correctRatesRaw]

    return correctRatesC, correctRatesR, correctRatesRoll, correctRatesRaw


def VqaPerturbTxt(model, valLoader, bNum, positive=True):
    print("perturbation!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    steps = [0.05, 0.10, 0.15, 0.20, 0.25, 0.5, 0.75, 1]
    correctRatesC = [0 for _ in range(len(steps) + 1)]
    correctRatesR = [0 for _ in range(len(steps) + 1)]
    correctRatesRoll = [0 for _ in range(len(steps) + 1)]
    correctRatesRaw = [0 for _ in range(len(steps) + 1)]

    for i, (img, seq, pad, label) in enumerate(valLoader):
        if i >= bNum:
            break

        img = img.to(device)
        seq = seq.to(device)
        pad = pad.to(device)
        label = label.to(device)
        label[:, 0] = 0
        l = pad.shape[1]

        predict = model(img, seq, pad)

        C = getScoresC(model, l, torch.max(predict[1], dim=-1)[0])
        R = getScoresR(model, l, torch.max(predict[1], dim=-1)[0])
        Roll = getScoresRoll(model, l, torch.max(predict[1], dim=-1)[0])
        Raw = getScoresRaw(model, l, torch.max(predict[1], dim=-1)[0])

        correctRatesC[0] += (compute_score_with_logits(predict[0], label).sum() / len(label)).item()
        correctRatesR[0] += (compute_score_with_logits(predict[0], label).sum() / len(label)).item()
        correctRatesRoll[0] += (compute_score_with_logits(predict[0], label).sum() / len(label)).item()
        correctRatesRaw[0] += (compute_score_with_logits(predict[0], label).sum() / len(label)).item()

        line("*")
        line("*")
        for s in range(len(steps)):
            Cmask = torch.concat([pad], dim=1)
            Rmask = torch.concat([pad], dim=1)
            RollMask = torch.concat([pad], dim=1)
            RawMask = torch.concat([pad], dim=1)

            _, kc = torch.topk(torch.from_numpy(C[:, 0, 50:]), k=int((steps[s]) * (Cmask.shape[1])), dim=-1,
                               largest=positive)
            _, kr = torch.topk(torch.from_numpy(R[:, 0, 50:]), k=int((steps[s]) * (Rmask.shape[1])), dim=-1,
                               largest=positive)
            _, kroll = torch.topk(torch.from_numpy(Roll[:, 0, 50:]), k=int((steps[s]) * (RollMask.shape[1])), dim=-1,
                                  largest=positive)
            _, kraw = torch.topk(torch.from_numpy(Raw[:, 0, 50:]), k=int((steps[s]) * (RawMask.shape[1])), dim=-1,
                                 largest=positive)

            for b in range(Rmask.shape[0]):
                Cmask[b, kc[b]] = 1
                Rmask[b, kr[b]] = 1
                RollMask[b, kroll[b]] = 1
                RawMask[b, kraw[b]] = 1

            Cmask = Cmask.to(torch.long)
            Rmask = Rmask.to(torch.long)
            RollMask = RollMask.to(torch.long)
            RawMask = RawMask.to(torch.long)

            Cpred = model(img, seq, Cmask[:, 50:])[0]
            Rpred = model(img, seq, Rmask[:, 50:])[0]
            Rollpred = model(img, seq, RollMask[:, 50:])[0]
            Rawpred = model(img, seq, RawMask[:, 50:])[0]

            correctRatesC[s + 1] += (compute_score_with_logits(Cpred, label).sum() / len(label)).item()
            correctRatesR[s + 1] += (compute_score_with_logits(Rpred, label).sum() / len(label)).item()
            correctRatesRaw[s + 1] += (compute_score_with_logits(Rollpred, label).sum() / len(label)).item()
            correctRatesRoll[s + 1] += (compute_score_with_logits(Rawpred, label).sum() / len(label)).item()

            line("-")
            print("C avg perturbation:", [c / (i + 1) for c in correctRatesC])
            print("R avg perturbation:", [r / (i + 1) for r in correctRatesR])
            print("Roll avg perturbation:", [roll / (i + 1) for roll in correctRatesRoll])
            print("Raw avg perturbation:", [raw / (i + 1) for raw in correctRatesRaw])

    correctRatesC = [c / bNum for c in correctRatesC]
    correctRatesR = [r / bNum for r in correctRatesR]
    correctRatesRoll = [roll / bNum for roll in correctRatesRoll]
    correctRatesRaw = [raw / bNum for raw in correctRatesRaw]

    return correctRatesC, correctRatesR, correctRatesRoll, correctRatesRaw