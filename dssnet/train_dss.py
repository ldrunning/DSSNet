from optparse import OptionParser

import imgaug.augmenters as iaa
import torch
import torchvision
from torch.utils import data as data_
from tqdm import tqdm
from utils import transform
from data import CellSeg
from model.dss_trainer import DSSTrainer
from utils.iou_loss import iou
from utils.utils import ColorJitterHED, ColorJitterHRD
from utils.vis_tool import Visualizer


def eval_net(net, test_dataloader):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    acc = 0
    num = 0
    for i, (img, true_mask, label) in enumerate(test_dataloader):
        for j in range(1, 3, 1):
            img = img.cuda()
            batchnum = int(true_mask.size(0))
            num += batchnum
            true_mask = torch.squeeze(true_mask).cuda()
            with torch.no_grad():
                mask_pred, sr, boundary = net(img)
            true_mask0 = (true_mask == j)
            true_mask2 = torch.unsqueeze(true_mask0, dim=1)
            true_mask2 = true_mask2.cuda().float()
            mask_pred = mask_pred.max(1)[1]
            # mask_pred2 = torch.stack([mask_pred == 1, mask_pred == 2], dim=1)
            mask_pred0 = (mask_pred == j)
            mask_pred2 = torch.unsqueeze(mask_pred0, dim=1)
            mask_pred2 = mask_pred2.cuda().float()
            # print(mask_pred2.shape)
            tot += iou(true_mask2, mask_pred2).item() * batchnum
            acc += float(
                (torch.sum(((true_mask == mask_pred) & (true_mask > 0)).float()))
                / torch.sum((true_mask > 0).float().cuda())) * batchnum
    return {"IOU": tot / num, "acc": acc / num}


def train_net():
    args = get_args()
    model = DSSTrainer(args, isTrain=True)
    # model.load_networks("unetdeep/10130811_0.9286")
    color_jitter = torchvision.transforms.ColorJitter(brightness=(0.9, 1.2),
                                                      contrast=(0.9, 1.2),
                                                      saturation=0.2,
                                                      hue=0.2)
    sometimes = lambda aug: iaa.Sometimes(0.4, aug)
    seq = iaa.Sequential([
        iaa.Affine(
            rotate=(-180, 180),
        ),
        sometimes(iaa.OneOf([iaa.AverageBlur((1, 3)),
                             iaa.GaussianBlur((1, 3)),
                             iaa.MedianBlur((1, 3)),
                             ])),

        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
    ])
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    train_transform = transform.Compose([
        transform.RandRotate([-90, 90], padding=mean, ignore_label=0),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.RandomVerticalFlip()])
    augment = torchvision.transforms.RandomChoice([
        color_jitter
    ])
    train_data = CellSeg('D:/larger/train_dir', transform=train_transform, augment=augment)
    test_data = CellSeg('D:/larger/eval_dir')
    train_dataloader = data_.DataLoader(train_data,
                                        batch_size=args.batchsize,
                                        shuffle=True,
                                        num_workers=4,
                                        drop_last=True
                                        )
    test_dataloader = data_.DataLoader(test_data,
                                       batch_size=args.batchsize,
                                       shuffle=True,
                                       num_workers=4,
                                       drop_last=False
                                       )
    best_iou = 0
    print(len(train_dataloader))
    val_iou = eval_net(model.netDSS, test_dataloader)
    print(val_iou)
    for epoch in range(args.epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        model.train()
        for i, (imgs, true_masks, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            model.set_input(imgs, true_masks, labels)
            model.optimize_parameters()
        val_iou = eval_net(model, test_dataloader)
        print('Validation IOU: {}'.format(val_iou))
        # vis.plot_many(val_dice)
        model.update_learning_rate()
        if best_iou < val_iou["IOU"]:
            best_iou = val_iou["IOU"]
            model.save_networks(best_iou)
            print('Checkpoint {} saved !'.format(epoch + 1))


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=400, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batchsize', dest='batchsize', default=4,
                      type='int', help='batch size')
    parser.add_option('--G_lr', dest='G_lr', default=1e-3, type='float',
                      help='learning rate')
    parser.add_option('--plot_every', dest='plot_every', default=20,
                      type='int')
    parser.add_option('--inchannel', dest='inchannel', default=3,
                      type='int')
    parser.add_option('--outchannel', dest='outchannel', default=3,
                      type='int')
    parser.add_option('--gpu_ids', dest='gpu_ids',
                      default=[0, 1])

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    train_net()
