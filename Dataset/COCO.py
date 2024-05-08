import copy

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from torchvision.transforms import *
import os
import PIL.Image as Image
from tools.tokenizer import Tokenizer

import os
from os.path import join

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class CocoTrain(Dataset):
    def __init__(self, ann, rootA, rootB, transform=None, tokenizer=None):
        super().__init__()
        self.sourceA = COCO(join(ann, 'captions_train2014.json'))
        self.idsA = np.load(join(ann, 'coco_train.npy'))
        self.sourceB = COCO(join(ann, 'captions_val2014.json'))
        self.idsB = np.load(join(ann, 'coco_restval.npy'))
        self.rootA = rootA
        self.rootB = rootB
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.idsA) * 5 + len(self.idsB) * 5

    def __getitem__(self, idx):
        source, ID, root = None, None, None

        if idx < len(self.idsA) * 5:
            ID = int(self.idsA[idx // 5])
            source = self.sourceA
            root = self.rootA

        else:
            idx -= len(self.idsA) * 5
            ID = int(self.idsB[idx // 5])
            source = self.sourceB
            root = self.rootB

        path = source.loadImgs(ID)[0]['file_name']
        img = self.transform(Image.open(os.path.join(root, path)))

        cap = source.loadAnns(source.getAnnIds(ID))
        cap = cap[idx % 5]['caption']

        caption = self.tokenizer.encode(cap)
        caption.insert(0, 49406)
        caption.append(49407)
        caption = torch.LongTensor(caption)

        gt = self.tokenizer.encode(cap)
        gt.extend([49406, 0])
        gt = torch.LongTensor(gt)

        return img, caption, gt, path


def CocoTrainCollate(batchData):
    maxLen = max(map(lambda x: len(x[1]), batchData))
    imgs, captions, gts, keyPads = [], [], [], []

    for data in batchData:
        pad1, pad2 = torch.LongTensor([0 for _ in range(maxLen - len(data[1]))]), \
                     torch.LongTensor([0 for _ in range(maxLen - len(data[2]))])
        caption, gt = torch.concat([data[1], pad1]), torch.concat([data[2], pad2])
        keyPad = torch.concat([torch.LongTensor([0 for _ in range(len(data[1]))]), torch.LongTensor([1 for _ in range(len(pad1))])]).to(torch.bool)

        imgs.append(data[0])
        captions.append(caption)
        gts.append(gt)
        keyPads.append(keyPad)

    return torch.stack(imgs), torch.stack(captions), torch.stack(gts), torch.stack(keyPads)


class CocoValTest(Dataset):
    def __init__(self, ann, root, VorT, transform=None):
        super().__init__()
        self.source = COCO(join(ann, "captions_val2014.json"))
        self.ids = np.load(join(ann, "coco_" + VorT + ".npy"))
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ID = int(self.ids[idx])

        img = path = self.source.loadImgs(ID)[0]['file_name']
        img = self.transform(Image.open(os.path.join(self.root, path)))

        captions = self.source.loadAnns(self.source.getAnnIds(ID))
        captions = [caps['caption'] for caps in captions]
        gts = [cap for cap in captions]
        
        if len(gts) > 5:
            gts = gts[:5]

        return img, gts, path


if __name__ == "__main__":
    #coco_train = CocoTrain("", "F:\\backups\\datasets\\cocoCaption\\train2014", "F:\\backups\\datasets\\cocoCaption\\val2014", _transform(224), Tokenizer())

    #print(coco_train[550009])
    #trainLoader = DataLoader(coco_train, batch_size=4, collate_fn=CocoTrainCollate)

    #for i, (img, caption, gt, path) in enumerate(trainLoader):
    #    print(img.shape, caption.shape, gt.shape, path)
    #    break

    coco_val = CocoValTest("", "F:\\backups\\datasets\\cocoCaption\\val2014", "val", transform(224))
    print(coco_val[4999])

