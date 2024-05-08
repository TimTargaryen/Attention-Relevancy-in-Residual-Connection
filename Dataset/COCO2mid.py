from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append("..")
from Model.basic import CLIPimageEncoder, CLIPtextEncoder
from tools.tokenizer import Tokenizer
from Dataset.COCO import CocoTrain, CocoValTest, transform
import torch
import os
import numpy as np
import yaml
import time
import datetime
import os

config = yaml.load(open("../config.yaml"), yaml.FullLoader)


def existSave(cap, gt, name, dest, num):
    if os.path.exists(os.path.join(dest, "Tmid", name + "-" + str(num) + ".npy")) is not True:
        np.save(arr=cap.detach().cpu().numpy().astype(np.float16),
                file=os.path.join(dest, "Tmid", name + "-" + str(num) + ".npy"))
        np.save(arr=gt.detach().numpy().astype(np.int32),
                file=os.path.join(dest, "Gmid", name + "-" + str(num) + ".npy"))

        return True

    else:
        return False

def ToMid(dest, Loader, imageEnc, textEnc): # batchsize=1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imageEnc.to(device)
    textEnc.to(device)

    for i, (img, cap, gt, name) in enumerate(Loader):
        name = name[0]

        cap = cap.to(device)
        cap = textEnc(cap)
        cap = cap.squeeze(0)
        gt = gt.squeeze(0)

        for j in range(5):
            if existSave(cap, gt, name, dest, j):
                break

        if os.path.exists(os.path.join(dest, "Imid", name + ".npy")) is not True:
            img = img.to(device)
            img = imageEnc(img)
            img = img.squeeze(0)

            np.save(arr=img.detach().cpu().numpy().astype(np.float16), file=os.path.join(dest, "Imid", name + ".npy"))

        print("{}th is done:".format(i), name)
        del img, cap, gt, name

if __name__ == "__main__":
    imgEnc = CLIPimageEncoder()
    txtEnc = CLIPtextEncoder()

    imgEnc.load_state_dict(torch.load("../CLIPimage_vitb32.pt"))
    txtEnc.load_state_dict(torch.load("../CLIPtext_vitb32.pt"))

    coco_train = CocoTrain("", config['cocoTrain'], config['cocoVal'], transform(224), Tokenizer())
    trainLoader = DataLoader(coco_train, batch_size=1)

    ToMid(config['cocoMid'], trainLoader, imgEnc, txtEnc)



