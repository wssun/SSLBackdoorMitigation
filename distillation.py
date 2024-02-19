import os
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from at import AT
from models import get_encoder_architecture
from datasets import get_pretraining_dataset
from evaluation import knn_predict
from KDistill_ZOO import *

# train for one epoch, we refer to the implementation from: https://github.com/leftthomas/SimCLR
def train(net1,net2,data_loader, train_optimizer, epoch, args):
    criterionAT = AT(2)
    if args.distill_way == 'CC':
        "feature map"
        args.opt1 = 1000
        args.opt2 = 1000
        args.opt3 = 1000
        args.opt4 = 1000
        args.opt5 = 0
        criterionAT = Correlation()
    elif args.distill_way == 'AT':
        "attention map"
        criterionAT = AT(2)
        args.opt1 = 1000
        args.opt2 = 1000
        args.opt3 = 1000
        args.opt4 = 1000
        args.opt5 = 0
    elif args.distill_way == 'FitNet':
        "feature map"
        criterionAT = HintLoss()
        args.opt1 = 1000
        args.opt2 = 1000
        args.opt3 = 1000
        args.opt4 = 1000
        args.opt5 = 0
    elif args.distill_way == 'KD':
        "the last layer"
        criterionAT = DistillKL(0.5)
        args.opt1 = 0
        args.opt2 = 0
        args.opt3 = 0
        args.opt4 = 1000
        args.opt5 = 0
    elif args.distill_way == 'SP':
        "the last layer"
        criterionAT = DistillKL(0.5)
        args.opt1 = 0
        args.opt2 = 0
        args.opt3 = 0
        args.opt4 = 0
        args.opt5 = 1
    
    elif args.distill_way == 'AFD':
        "attemtion map"
        
        args.opt1 = 1000
        args.opt2 = 1000
        args.opt3 = 1000
        args.opt4 = 1000
        args.opt5 = 0
        criterionAT1 = AFD(64)
        criterionAT2 = AFD(128)
        criterionAT3 = AFD(256)
        criterionAT4 = AFD(512)
        
    net1.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for im_1, im_2 in train_bar:
        im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)
        feature_1, out_1 = net1(im_1)
        feature_2, out_2 = net1(im_2)
        # [2*B, D]
        feature_3, out_3 = net2(im_1)
        # feature_3, out_3 = net2(im_1)
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.knn_t)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * args.batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * args.batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.knn_t)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        conloss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        # loss = net(im_1, im_2, args)
        atloss = -torch.sum(feature_3 * feature_1, dim=-1).mean()
        train_list1 = []
        for name, module in net2.f.f._modules.items():
            if name == '0':
                train_list1.append(module(im_1))
            else:
                train_list1.append(module(train_list1[int(name)-1]))
        train_list1[-1] = F.normalize(train_list1[-1], dim=-1)
        train_list2 = []
        for name, module in net1.f.f._modules.items():
            if name == '0':
                train_list2.append(module(im_1))
            else:
                train_list2.append(module(train_list2[int(name)-1]))
        train_list2[-1] = F.normalize(train_list2[-1], dim=-1)
       
        if args.distill_way == 'AFD':
            at4_loss = criterionAT4(train_list2[6], train_list1[6].detach()) * args.opt1
            at3_loss = criterionAT3(train_list2[5], train_list1[5].detach()) * args.opt2
            at2_loss = criterionAT2(train_list2[4], train_list1[4].detach()) * args.opt3
            at1_loss = criterionAT1(train_list2[3], train_list1[3].detach()) * args.opt4
        else:
            at4_loss = criterionAT(train_list2[6], train_list1[6].detach()) * args.opt1
            at3_loss = criterionAT(train_list2[5], train_list1[5].detach()) * args.opt2
            at2_loss = criterionAT(train_list2[4], train_list1[4].detach()) * args.opt3
            at1_loss = criterionAT(train_list2[3], train_list1[3].detach()) * args.opt4   
        atloss = atloss*args.opt5
        
        loss = conloss + at1_loss + at2_loss + at3_loss + at4_loss + atloss
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num


# we use a knn monitor to check the performance of the pre-trained image encoder by following the implementation: https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
def test(net, memory_data_loader, test_data_clean_loader, epoch, args):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_num, feature_bank = 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_clean_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--pretraining_dataset', type=str, default='cifar10')
    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save the results (default: none)')
    parser.add_argument('--opt1',default=1000,type=int,help='opt1')
    parser.add_argument('--opt2',default=1000,type=int,help='opt2')
    parser.add_argument('--opt3',default=1000,type=int,help='opt3')
    parser.add_argument('--opt4',default=1000,type=int,help='opt4')
    parser.add_argument('--opt5',default=1,type=int,help='opt5')
    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')
    parser.add_argument('--knn-t', default=0.5, type=float, help='softmax temperature in kNN monitor')
    parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
    parser.add_argument('--teacher',type=str,default='',metavar='PATH',help='bad teacher')
    parser.add_argument('--teacher1',type=str,default='',metavar='PATH',help='bad teacher')
    parser.add_argument('--student',type=str,default='',metavar='PATH',help='good student')
    parser.add_argument('--ratio',type=float,default=0.05,help='the ratio of clean sample')
    parser.add_argument('--distill_way',type=str,default='CC',help='distillation way')
    CUDA_LAUNCH_BLOCKING=1
    args = parser.parse_args()

    # Set the random seeds and GPU information
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



    # Specify the pre-training data directory
    args.data_dir = f'/home/jspi/data/HTX/DATA/BadEncoder/{args.pretraining_dataset}/'

    print(args)

    # Load the data and create the data loaders, note that the memory data and test_data_clean are only used to monitor the pre-training of the image encoder
    train_data, memory_data, test_data_clean = get_pretraining_dataset(args)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=5,
        pin_memory=True,
        drop_last=True
    )
    memory_loader = DataLoader(
        memory_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=5,
        pin_memory=True
    )
    test_loader_clean = DataLoader(
        test_data_clean,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=5,
        pin_memory=True
    )

    # Intialize the model
    teacher = get_encoder_architecture(args).cuda()
    checkpoint1 = torch.load(args.teacher,map_location='cuda:0')
    if 'clip' in args.teacher:
        teacher.visual.load_state_dict(checkpoint1['state_dict'])
    else:
        teacher.load_state_dict(checkpoint1['state_dict'])

    
    student = get_encoder_architecture(args).cuda()
    checkpoint3 = torch.load(args.student,map_location='cuda:0')
    student.load_state_dict(checkpoint3['state_dict'])
    
    
    
    # Define the optimizer
    teacher.eval()

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-6)


    epoch_start = 1


    # Logging
    # results = {'train_loss': [], 'test_acc@1': []}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    # Dump args
    # with open(args.results_dir + '/args.json', 'w') as fid:
      #   json.dump(args.__dict__, fid, indent=2)

    # Training loop
    for epoch in range(epoch_start, args.epochs + 1):
        print("=================================================")
        train_loss = train(student,teacher,train_loader, optimizer, epoch, args)

        if epoch % 500 == 0:
            torch.save({'epoch': epoch, 'state_dict': student.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + 'model' + str(epoch) + '.pth')
