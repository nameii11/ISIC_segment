# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
from tqdm import tqdm


#from sklearn.externals import joblib
import joblib


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
from dataset import Dataset

from metrics import batch_iou, mean_iou, iou_score,accuracy
import losses
from utils import str2bool, count_params
import pandas as pd
from model import ModelBuilder
import unet
arch_names = list(unet.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEDiceLoss')#BCEWithLogitsLoss
tmp_path = 'tmp_see'
device_ids = [0, 1]



def onehot_to_binary_edges(mask, radius=1, num_classes=1):
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    #print(mask_pad.size())
    edgemap = np.empty((1, 128, 128))

    for i in range(num_classes):

        dist = distance_transform_edt(mask_pad[i, :, :]) + distance_transform_edt(1.0 - mask_pad[i, :, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap[i, :, :] = dist
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap

def mask_to_edges(mask):
    _edge = mask
    _edge = onehot_to_binary_edges(_edge)
    return torch.from_numpy(_edge).float()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Dilated_Unet_lr3e-4_512c_nodown',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: NestedUNet)')
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--dataset', default="isic",
                        help='dataset name')
    parser.add_argument('--input-channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='jpg',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=None, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=12, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-3, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    accs = AverageMeter()

    model.train()
    #, edge
    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()#device=device_ids[0]
        target = target.cuda()
        # edge = edge.cuda()
        # mask = target.permute((2, 0, 1))
        # compute output
        if args.deepsupervision:
            outputs = model(input)
            loss = 0

            for output in outputs:
                loss += criterion(output, target)#
            loss /= len(outputs)
            acc = accuracy(outputs[-1], target)
        else:
            output = model(input)#, output_edge
            loss = criterion(output, target)#,output_edge, edge
            acc = accuracy(output, target)

        losses.update(loss.item(), input.size(0))
        accs.update(acc, input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()
    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', accs.avg),
    ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluate mode
    model.eval()
    #, edge
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):

            input = input.cuda()
            target = target.cuda()
            # edge = edge.cuda()
            # mask = target.permute((2, 0, 1))
            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for (output) in (outputs):
                    loss += criterion(output, target)                                                                                                              #
                loss /= len(outputs)
                acc = accuracy(outputs[-1], target)
            else:
                output = model(input)#, output_edge
                loss = criterion(output, target)#,output_edge, edge
                acc = accuracy(output, target)

            losses.update(loss.item(), input.size(0))
            accs.update(acc, input.size(0))
            '''
            if i % 20 == 0:
                vutils.save_image(torch.sigmoid(output).data, tmp_path + '/iter%d-sal-0.jpg' % i, normalize=True,
                                  padding=0)
                vutils.save_image(input.data, tmp_path + '/iter%d-sal-data.jpg' % i)#
                vutils.save_image(target.data, tmp_path + '/iter%d-sal-target.jpg' % i, padding=0)
            '''
    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', accs.avg),
    ])

    return log




def main():
    args = parse_args()
    # np.random.seed(44)
    #torch.manual_seed(44)
    torch.cuda.manual_seed_all(44)
    torch.backends.cudnn.deterministic = True
    #args.dataset = "datasets"


    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_wDS' %(args.dataset, args.arch)
        else:
            args.name = '%s_%s_woDS' %(args.dataset, args.arch)
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # define loss function (criterion)

    criterion = losses.__dict__[args.loss]().cuda()


    # Data loading code
    #train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
    #    train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
    #print("train_num:%s"%str(len(train_img_paths)))
    #print("val_num:%s"%str(len(val_img_paths)))
    train_paths = '../ISIC/train.txt'
    valid_paths = '../ISIC/valid.txt'
    # k折交叉验证

    # k = 4
    #img_list = os.listdir(r'/home/yld/pwd/ISIC/1/train')
    # fold_size = 2074 // k#len(img_list)
    # create model
    print("=> creating model %s" % args.arch)
    model = ModelBuilder().build_unet()
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    # model = model.cuda(device=device_ids[0])#device_ids[0]
    model.to("cuda:0")
    print(count_params(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,mode='max', factor=0.1, patience=10, verbose=True, threshold=0.0001,
    #                                           threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


    # for i in range(k):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0

    fh = open(train_paths, 'r')
    fh2 = open(valid_paths, 'r')
    # imgs = []
    # masks = []
    train_img = []
    train_mask = []
    valid_img = []
    valid_mask = []
    for line in fh:
        line = line.rstrip()
        words = line.split()
        train_img.append(words[0])
        train_mask.append(words[1])

    for line in fh2:
        line = line.rstrip()
        words = line.split()
        valid_img.append(words[0])
        valid_mask.append(words[1])

    # for j in range(k):
    #     idx = slice(j * fold_size, (j + 1) * fold_size)
    #     img, mask = imgs[idx], masks[idx]
    #     if j == i:
    #         valid_img, valid_mask = img, mask
    #     else:

    train_dataset = Dataset(args, train_img, train_mask, aug_data=True)
    val_dataset = Dataset(args, valid_img, valid_mask)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size * len(device_ids),
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=0)  #
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size * len(device_ids),
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=0)  #



    log = pd.DataFrame(index=[], columns=[
        'epoch', 'loss', 'acc', 'val_loss', 'val_acc'
    ])

    best_acc = 0
    trigger = 0
    for epoch in range(args.epochs):
        train_loss_sum, valid_loss_sum = 0, 0
        train_acc_sum, valid_acc_sum = 0, 0
        # print('Epoch %s' %epoch)
        # for i in range(k):

        print('Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch, scheduler=scheduler)
        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion)
        #iou = val_log['iou']
        scheduler.step()
        #lr = scheduler.get_last_lr()
        # print('*' * 25, '第', i + 1, '折', '*' * 25)
        print('loss %.4f - acc %.4f - val_loss %.4f - val_acc %.4f'
              %(train_log['loss'], train_log['acc'], val_log['loss'], val_log['acc']))
        torch.cuda.empty_cache()
        train_loss_sum += train_log['loss']
        train_acc_sum += train_log['acc']
        valid_loss_sum += val_log['loss']
        valid_acc_sum += val_log['acc']


        tmp = pd.Series([
            epoch,
            #lr,
            train_log['loss'],
            train_log['acc'],
            val_log['loss'],
            val_log['acc'],
        ], index=['epoch', 'loss', 'acc', 'val_loss', 'val_acc'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv'%args.name, index=False)

        trigger += 1
        if trigger == 60:
            torch.save(model.state_dict(), 'models/%s/model-60epoch.pth' %args.name)

        if trigger == 70:
            torch.save(model.state_dict(), 'models/%s/model-70epoch.pth' %args.name)

        if trigger == 80:
            torch.save(model.state_dict(), 'models/%s/model-80epoch.pth' %args.name)

        if trigger == 90:
            torch.save(model.state_dict(), 'models/%s/model-90epoch.pth' %args.name)

        if trigger == 100:
            torch.save(model.state_dict(), 'models/%s/model-100epoch.pth' %args.name)



if __name__ == '__main__':
    main()
