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
from model.dss import DSS


def eval_net(net, test_dataloader):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    acc = 0
    num = 0
    for i, (img, true_mask) in enumerate(test_dataloader):
        for j in range(1, 3, 1):
            img = img.cuda()
            batchnum = int(true_mask.size(0))
            num += batchnum
            true_mask = true_mask.cuda()
            # img = F.interpolate(img, scale_factor=2)
            with torch.no_grad():
                mask_pred, A, B = net(img)
            true_mask0 = (true_mask == j)
            true_mask2 = torch.unsqueeze(true_mask0, dim=1)
            # true_mask2 = torch.stack(
            #     [true_mask == 1, true_mask == 2, true_mask == 3, true_mask == 4, true_mask == 5, true_mask == 6,
            #      true_mask == 7, true_mask == 8, true_mask == 9, true_mask == 10, true_mask == 11, true_mask == 12,
            #      true_mask == 13, true_mask == 14, true_mask == 15, true_mask == 16, true_mask == 17, true_mask == 18,
            #      true_mask == 19], dim=1)
            # for k in range(3, 20):
            #     ture_mask0 = (true_mask == k)
            #     ture_mask1 = torch.unsqueeze(ture_mask0, dim=1)
            # true_mask2 = torch.stack([true_mask == 1, true_mask == 2], dim=1)
            true_mask2 = true_mask2.cuda().float()
            mask_pred = mask_pred.max(1)[1]
            # mask_pred2 = torch.stack([mask_pred == 1, mask_pred == 2], dim=1)
            mask_pred0 = (mask_pred == j)
            mask_pred2 = torch.unsqueeze(mask_pred0, dim=1)
            # mask_pred2 = torch.stack(
            #     [mask_pred == 1, mask_pred == 2, mask_pred == 3, mask_pred == 4, mask_pred == 5, mask_pred == 6,
            #      mask_pred == 7, mask_pred == 8, mask_pred == 9, mask_pred == 10, mask_pred == 11, mask_pred == 12,
            #      mask_pred == 13, mask_pred == 14, mask_pred == 15, mask_pred == 16, mask_pred == 17, mask_pred == 18,
            #      mask_pred == 19], dim=1)
            # for k in range(2, 20):
            #     mask_pred2 = torch.stack([mask_pred2, mask_pred == k], dim=1)
            mask_pred2 = mask_pred2.cuda().float()
            tot += iou(true_mask2, mask_pred2).item() * batchnum
            acc += float(
                (torch.sum(((true_mask == mask_pred) & (true_mask > 0)).float()))
                / torch.sum((true_mask > 0).float().cuda())) * batchnum
    return {"IOU": tot / num, "acc": acc / num}


def train_net():
    args = get_args()
    model_path = "12142035_0.7749/12142035_0.7749_netUNet.pth"
    # model = UNetTrainer(args, isTrain=False)
    model = DSS(3, 3).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # model.load_networks("unetdeep/10130811_0.9286")

    test_data = CellSeg('/home/luodie/larger/wsi_test')

    test_dataloader = data_.DataLoader(test_data,
                                       batch_size=args.batchsize,
                                       shuffle=True,
                                       num_workers=4,
                                       drop_last=False
                                       )

    val_iou = eval_net(model, test_dataloader)
    print('Validation IOU: {}'.format(val_iou))


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

    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    train_net()
