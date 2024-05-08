import sys

sys.path.append("..")
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import yaml
import datetime
import os

config = yaml.load(open("config.yaml"), yaml.FullLoader)
beginTime = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")


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


def VqaTrain(model, trainLoader, epochs, LR):
    writer = SummaryWriter(os.path.join(config['saveDir'], beginTime))
    interval = config['Interval']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    weight = torch.ones(3129).to(device)
    weight[0] = 0
    citerition = nn.BCELoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(epochs):

        cnt = 0
        Loss = 0.0
        correct = 0.0

        model.train()
        for i, (img, seq, pad, label) in enumerate(trainLoader):
            cnt += 1

            img = img.to(device)
            seq = seq.to(device)
            pad = pad.to(device)
            label = label.to(device)

            predict = model(img, seq, pad)

            loss = citerition(predict, label)
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()
            Loss += loss.item()

            predict[:, 0] = 0
            correct += (compute_score_with_logits(predict, label).sum() / len(label)).item()

            if loss.item() > 20000 or Loss != Loss:
                print(loss)
                print("sth wrong")
                exit(1)

            if cnt % interval == 0 and cnt != 0:
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print("epoch:{}/{}, step:{}/{}, avgloss:{}, correctRate:{}, lr:{}"
                      .format(epoch + 1, epochs, cnt,
                              trainLoader.__len__(), Loss / cnt, correct / cnt, lr))

                writer.add_scalar('train/loss', Loss / cnt, epoch * trainLoader.__len__() + cnt)
                writer.add_scalar('train/correctRate', correct / cnt, epoch * trainLoader.__len__() + cnt)


        torch.save(model, os.path.join(config['saveDir'], beginTime, str(epoch + 1) + ".pth"))