import sys

sys.path.append("..")
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tools import Cider
#from pycocoevalcap.cider.cider import Cider
from tools import Bleu
from tools import Meteor

cider = Cider()
bleu = Bleu()
meteor = Meteor()

from tools import compute_scores
from tools.beamsearch import BeamSearch2
from tools.tokenizer import Tokenizer
from tools import PTBTokenizer
import yaml
import datetime
import os


config = yaml.load(open("config.yaml"), yaml.FullLoader)
tokenizer = Tokenizer()
beginTime = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")


def line(c):
    for i in range(50):
        print(c, end="")
    print()


def CaptionTrain(model, trainLoader, valLoader, epochs, LR):
    writer = SummaryWriter(os.path.join(config['saveDir'], beginTime))
    citerition = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    interval = (config['Interval'] / trainLoader.batch_size)
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    show = 1

    for epoch in range(epochs):

        cnt = 0
        Loss = 0.0
        B, C, M, S = 0.0, 0.0, 0.0, 0.0

        model.train()
        model.unfreeze()
        for i, (img, cap, gt, keyPad) in enumerate(trainLoader):
            break
            cnt += 1

            img = img.to(device)
            cap = cap.to(device)
            gt = gt.to(device)
            keyPad = keyPad.to(device)

            predict = model(img, cap, keyPad)
            predict = predict.view(-1, predict.size(-1))
            gt = gt.view(-1)

            loss = citerition(predict, gt)
            loss = loss.mean()
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            Loss += loss.item()
            
            if loss.item() > 20000 or Loss != Loss:
                print(loss)
                print("sth wrong")
                exit(1)

            if cnt % interval == 0 and cnt != 0:
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print("epoch:{}/{}, step:{}/{}, avgloss:{}, lr:{}"
                      .format(epoch + 1, epochs, cnt, trainLoader.__len__(), Loss / cnt, lr))
                writer.add_scalar('avgloss', Loss / cnt, epoch * trainLoader.__len__() + cnt)

        if (epoch + 1) % show == 0 and epoch > 0:

            model.unfreeze()
            with torch.no_grad():
                for i, (imgs, gt) in enumerate(valLoader):
                    imgs = imgs.to(device)
                    pred = BeamSearch2(imgs, model, device, max_length=5)

                    predicts, gts = {}, {}
                    for j in range(1, len(imgs) + 1):
                        predicts[str(j)] = [pred[j - 1]]
                        gts[str(j)] = [gt[k][j - 1] for k in range(len(gt))]

                        if i % 50 == 0 and j == 1:
                            print("<Ground Truth>:", gts[str(j)])
                            print("<Predict>:", predicts[str(j)])

                    line("-")
                    predicts = PTBTokenizer.tokenize(predicts)
                    gts = PTBTokenizer.tokenize(gts)
                    '''
                    score, scores = compute_scores(gts, predicts)
                    c, b, m, s = score[0], score[1][3], score[2], score[3]
                    C += c * (len(imgs) / 5000)
                    B += b * (len(imgs) / 5000)
                    M += m * (len(imgs) / 5000)
                    S += s * (len(imgs) / 5000)
                    '''
                    print(predicts, gts)
                    c, b, m = cider.compute_score(gts, predicts)[0], bleu.compute_score(gts, predicts)[0][3], meteor.compute_score(gts, predicts)[0]
                    C += c * (len(imgs) / 5000)
                    B += b * (len(imgs) / 5000)
                    M += m * (len(imgs) / 5000)
                    #S += s * (len(imgs) / 5000)
                    print("{}th validate sample, bleu4:".format(i + 1), b, " cider:", c, " meteor:", m, " spice:", s)
                    
                    if c < 0.1:
                        print(c)
                        print("sth wrong")
                        exit(1)
                    
                line("*")
                line("*")
                print("epoch:{}/{}, bleu4:{}, cider:{}, spice:{}, meteor:{}".format(epoch + 1, epochs, B, C, S, M))
                line("*")
                line("*")

            writer.add_scalar('performance/cider', C, epoch + 1)
            writer.add_scalar('performance/bleu4', B, epoch + 1)
            writer.add_scalar('performance/spice', S, epoch + 1)
            writer.add_scalar('performance/meteor', M, epoch + 1)

            torch.save(model, os.path.join(config['saveDir'], beginTime, str(epoch + 1) + ".pth"))




