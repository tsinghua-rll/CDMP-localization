import argparse
import os
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter

from augmentations import SSDAugmentation 
from voc0712 import *
from coco import *
from model import CDMP_Localization

parser = argparse.ArgumentParser(description='cdmp-localization')
parser.add_argument('--tag', type=str, default='default')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--mode', choices=['train', 'test'], required=True)
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--model-path', type=str, default='./assets/learned_models', 
                   help='pre-trained model path')
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--save-interval', type=int, default=10)

args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available else False

if args.cuda:
    torch.cuda.manual_seed(1)

logger = SummaryWriter(os.path.join('./assets/log/', args.tag))
np.random.seed(int(time.time()))
device_id = [0,1,2]

# dataset hyper params
min_dim = 224
object_size = 224
means = (104, 117, 123)

train_loader = torch.utils.data.DataLoader(
    VOCLocalization(root='/home1/mxj/workspace/dataset/voc/VOCdevkit',
                 transform=SSDAugmentation(min_dim, means),
                 image_sets=[('2012', 'trainval')],
                 object_size=object_size,
    ),
    batch_size=args.batch_size,
    num_workers=64,
    pin_memory=True,
    shuffle=True,
    collate_fn=localization_collate
)


test_loader = torch.utils.data.DataLoader(
    VOCLocalization(root='/home1/mxj/workspace/dataset/voc/VOCdevkit',
                 transform=SSDAugmentation(min_dim, means),
                 image_sets=[('2007', 'trainval')],
                 object_size=object_size,
    ),
    batch_size=args.batch_size,
    num_workers=64,
    pin_memory=True,
    shuffle=True,
    collate_fn=localization_collate
)


def train(model, loader, epoch, optimizer):
    model.train()
    for batch_idx, (img, object_img, target) in enumerate(loader):
        # img: (N, C, H, W)
        # object_img: (N, C, H, W)
        # target: (N, 3) [x, y, id]
        if args.cuda:
            img, object_img, target = img.cuda(), object_img.cuda(), target.cuda()
        img, object_img, target = Variable(img), Variable(object_img), Variable(target)
        optimizer.zero_grad()
        output = model(img, object_img)
        loss = F.mse_loss(output, target[:, :2]).mean() # because you've already using log_softmax as output
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * args.batch_size, len(loader.dataset),
            100. * batch_idx * args.batch_size / len(loader.dataset), loss.data[0]))
            logger.add_scalar('train_loss', loss.cpu().data[0]/args.batch_size, 
                    batch_idx + epoch * len(loader))
            # visualize gt 
            x, y = (output[0]*min_dim).data.cpu().numpy().astype(np.int32)
            gt_x, gt_y = (target[0]*min_dim).data.cpu().numpy().astype(np.int32)[:2]
            label_id = int(target[0, -1])
            log_img = img[0].permute(1,2,0).data.cpu().numpy().astype(np.uint8)
            log_img += np.array(means).astype(np.uint8)
            log_img = cv2.circle(log_img, (x, y), 10, (255,0,0), 5)
            cv2.putText(log_img, VOC_CLASSES[label_id], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                    (255,0,0))
            log_img = cv2.circle(log_img, (gt_x, gt_y), 10, (0,255,0), 5)
            cv2.putText(log_img, VOC_CLASSES[label_id], (gt_x, gt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                    (0,255,0))
            obj_img = object_img[0].permute(1,2,0).data.cpu().numpy().astype(np.uint8)
            obj_img += np.array(means).astype(np.uint8)
            logger.add_image('train_log_img', log_img, batch_idx + epoch * len(loader))
            logger.add_image('train_obj_img', obj_img, batch_idx + epoch * len(loader))


def test(model, loader, epoch):
    model.eval()
    test_loss = 0
    for batch_idx, (img, object_img, target) in enumerate(loader):
        # img: (N, C, H, W)
        # object_img: (N, C, H, W)
        # target: (N, 3) [x, y, id]
        if args.cuda:
            img, object_img, target = img.cuda(), object_img.cuda(), target.cuda()
        img, object_img, target = Variable(img), Variable(object_img), Variable(target)
        output = model(img, object_img)
        loss = F.mse_loss(output, target[:, :2]).mean() # because you've already using log_softmax as output
        test_loss += loss.data.cpu()
        if batch_idx % args.log_interval == 0:
            # visualize gt 
            x, y = (output[0]*min_dim).data.cpu().numpy().astype(np.int32)
            gt_x, gt_y, label_id = (target[0]*min_dim).data.cpu().numpy().astype(np.int32)
            label_id = int(target[0, -1])
            log_img = img[0].permute(1,2,0).data.cpu().numpy().astype(np.uint8)
            log_img = cv2.circle(log_img, (x, y), 10, (255,0,0), 5)
            log_img += np.array(means).astype(np.uint8)
            cv2.putText(log_img, VOC_CLASSES[label_id], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                    (255,0,0))
            log_img = cv2.circle(log_img, (gt_x, gt_y), 10, (0,255,0), 5)
            cv2.putText(log_img, VOC_CLASSES[label_id], (gt_x, gt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                    (0,255,0))
            obj_img = object_img[0].permute(1,2,0).data.cpu().numpy().astype(np.uint8)
            obj_img += np.array(means).astype(np.uint8)
            logger.add_image('test_log_img', log_img, batch_idx + epoch * len(loader))
            logger.add_image('test_obj_img', obj_img, batch_idx + epoch * len(loader))


    test_loss /= len(loader.dataset) 
    # visualize gt
    # TBD
    return test_loss


def main():
    if args.mode == 'train':
        model = CDMP_Localization(input_size=min_dim, object_size=object_size)
        if args.cuda:
            model = nn.DataParallel(model, device_ids=device_id).cuda()
            # model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(args.epoch):
            train(model, train_loader, epoch, optimizer)       
            loss = test(model, test_loader, epoch)
            logger.add_scalar('test_loss', loss, epoch)
            if epoch % args.save_interval == 0:
                torch.save(model, os.path.join(args.model_path, 
                    args.tag + '_{}.model'.format(epoch)))
    else:
        model = torch.load(os.path.join(args.model_path, args.tag + '.model'))
        if args.cuda:
            model = nn.DataParallel(model, device_ids=device_id).cuda()
            # model = model.cuda()
        loss = test(model, test_loader, 0)
        print('Test done, loss={}'.format(loss))
              
if __name__ == "__main__":
    main()
