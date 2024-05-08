import sys

sys.path.append("..")
import torch
from torch import nn
import os
import json

def VqaTrainMapper(model, valLoader, cls2ans, ans):
    print("VQA test begin!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    answers = []

    model.unfreeze()
    with torch.no_grad():
        for i, (img, seq, pad, ids) in enumerate(valLoader):

            img = img.to(device)
            seq = seq.to(device)
            pad = pad.to(device)
            predict = model(img, seq, pad)

            for j in range(len(img)):
                answers.append({"answer": cls2ans[str(int(predict[j].argmax(dim=-1)))],
                                "question_id": ids[j]})

    json.dump(answers, ans)
    print("VQA test over!")



