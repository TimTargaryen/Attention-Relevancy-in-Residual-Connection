import torch
from torch.utils.data import DataLoader
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='caption')
    parser.add_argument("--type", type=str, default='A')
    parser.add_argument("--patch", type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--batch', type=int, default=2)
    args = parser.parse_args()

    from Model.basic import CLIPimageEncoder, CLIPtextEncoder
    from Model.mapper import Mapper, CaptionHead, CLIPmapper, VqaHead

    imageEnc, textEnc = None, None
    vlen = 50

    #imageEnc.load_state_dict(torch.load("../CLIPimage_vitb32.pt"))
    #textEnc.load_state_dict(torch.load("../CLIPtext_vitb32.pt"))
    if args.patch == 32:
        imageEnc = CLIPimageEncoder()
        textEnc = CLIPtextEncoder()
    elif args.patch == 16:
        imageEnc = CLIPimageEncoder(vision_patch_size=16)
        textEnc = CLIPtextEncoder()
        vlen = 197
    else:
        imageEnc = CLIPimageEncoder(vision_patch_size=14, vision_width=1024)
        textEnc = CLIPtextEncoder()

    myMapper2 = Mapper(type=args.type, Vlen=vlen, layers=8 if args.type == "B" else 3,
                       isCap=False if args.task == 'vqa' else True)
    myHead = VqaHead() if args.task == 'vqa' else CaptionHead()
    myCLIPmapper = CLIPmapper(imageEnc, textEnc, myMapper2, myHead)

    if args.task == 'vqa':
        from Dataset.VQAv2 import VQAv2, VQAv2collate, VQAv2All
        from Train import VqaTrain

        v2 = VQAv2("F:\\datasets\\MultiModal\\VQAv2")
        v2v = VQAv2("F:\\datasets\\MultiModal\\VQAv2", mode='val')
        VQAall = DataLoader(VQAv2All(v2, v2v), batch_size=args.batch, collate_fn=VQAv2collate)
        VqaTrain(myCLIPmapper, VQAall, args.epoch, args.lr)

    else :
        from Model.basic import CLIPimageEncoder, CLIPtextEncoder
        from tools.tokenizer import Tokenizer
        from Dataset.COCO import CocoTrain, CocoValTest, transform, CocoTrainCollate
        from Train import CaptionTrain

        coco_train = CocoTrain("Dataset", "F:\\backups\\datasets\\cocoCaption\\train2014",
                               "F:\\backups\\datasets\\cocoCaption\\val2014", transform(224), Tokenizer())
        coco_val = CocoValTest("Dataset", "F:\\backups\\datasets\\cocoCaption\\val2014", 'val', transform(224))
        cocoTrain = DataLoader(coco_train, batch_size=args.batch, shuffle=True, collate_fn=CocoTrainCollate)
        cocoVal = DataLoader(coco_val, batch_size=args.batch)
        CaptionTrain(myCLIPmapper, cocoTrain, cocoVal, args.epoch, args.lr)






'''
if __name__ == "__main__":
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)

    from Model.basic import CLIPimageEncoder, CLIPtextEncoder
    from Model.mapper import Mapper, CaptionHead, CLIPmapper

    imageEnc = CLIPimageEncoder()
    textEnc = CLIPtextEncoder()
    myMapper2 = Mapper(type='B', Vlen=50, layers=12)
    myHead = CaptionHead()
    imageEnc.load_state_dict(torch.load("../CLIPimage_vitb32.pt"))
    textEnc.load_state_dict(torch.load("../CLIPtext_vitb32.pt"))
    myCLIPmapper = CLIPmapper(imageEnc, textEnc, myMapper2, myHead)

    from Dataset.COCO import CocoTrain, transform, CocoValTest, CocoTrainCollate
    from tools.tokenizer import Tokenizer

    tokenizer = Tokenizer()
    coco_val = CocoValTest("../Dataset", config['cocoVal'], 'val', transform(224))
    valLoader = DataLoader(coco_val, batch_size=32, shuffle=True)

    from Dataset.COCOmid import COCOmid, COCOmidCollate

    coco_train2 = COCOmid("../../data/mid2014")
    trainLoader2 = DataLoader(coco_train2, batch_size=32, collate_fn=COCOmidCollate, shuffle=True)

    myCLIPmapper.initialize()
    myCLIPmapper.freeze()
    CaptionTrain(myCLIPmapper, trainLoader2, valLoader, 30, 1)


if __name__ == "__main__":
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)

    from Model.basic import CLIPimageEncoder, CLIPtextEncoder
    from Model.mapper import Mapper, VqaHead, CLIPmapper

    imageEnc = CLIPimageEncoder()
    textEnc = CLIPtextEncoder()
    myMapper2 = Mapper(type='B', Vlen=50, layers=6)
    myHead = VqaHead()
    imageEnc.load_state_dict(torch.load("../CLIPimage_vitb32.pt"))
    textEnc.load_state_dict(torch.load("../CLIPtext_vitb32.pt"))
    myCLIPmapper = CLIPmapper(imageEnc, textEnc, myMapper2, myHead)

    from Dataset.VQAv2 import VQAv2, VQAv2collate

    v2 = VQAv2("../../data")
    v2v = VQAv2("../../data", mode='val')
    VQAtrain = DataLoader(v2, batch_size=128, collate_fn=VQAv2collate)
    VQAval = DataLoader(v2v, batch_size=128, collate_fn=VQAv2collate)

    VqaTrainMapper(myCLIPmapper, VQAtrain, VQAval, 10, 5e-4)
'''