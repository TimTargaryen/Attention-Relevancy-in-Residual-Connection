import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class COCOmid(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.list = [os.path.join(path, 'Tmid', name) for name in os.listdir(os.path.join(path, 'Tmid'))]

    def __len__(self):
        return len(os.listdir(os.path.join(self.path, "Tmid")))

    def __getitem__(self, idx):
        Tmid = torch.tensor(np.load(self.list[idx]), dtype=torch.float32)

        ImidName = self.list[idx].replace("Tmid", "Imid")
        ImidName = ImidName[:ImidName.find('-')] + ImidName[ImidName.find('-') + 2:]
        Imid = torch.tensor(np.load(ImidName), dtype=torch.float32)

        GTName = self.list[idx].replace("Tmid", "Gmid")
        GT = torch.LongTensor(np.load(GTName))

        return Imid, Tmid, GT

def COCOmidCollate(batchData):
    Imids, Tmids, GTs, keyPads = [], [], [], []
    maxLen = max(map(lambda x: len(x[1]), batchData))

    for data in batchData:
        Imids.append(data[0])
        Tmids.append(torch.concat([data[1], torch.zeros((maxLen -
                    len(data[1]), len(data[1][0])))]))
        GTs.append(torch.concat([data[2], torch.zeros(maxLen - len(data[2]))]).type(torch.int64))
        keyPads.append(torch.concat([torch.zeros(len(data[2])), torch.ones(maxLen - len(data[2]))]).type(torch.bool))
    

    return torch.stack(Imids), torch.stack(Tmids), torch.stack(GTs), torch.stack(keyPads)

if __name__ == "__main__":
    coco_mid = COCOmid("../../data/mid2014")
    midLoader = DataLoader(coco_mid, batch_size=4, collate_fn=COCOmidCollate)

    for i, (I, T, G, P) in enumerate(midLoader):
        print(I.shape, T, G, P)
        if i > 3:
            break



