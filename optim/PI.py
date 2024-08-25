# -*- coding: utf-8 -*-

import torch
import utils.transforms as transforms
from dataloader.ucr2018 import *
import torch.utils.data as data
from model.models import *
from model.architecture import TimeConv
from torch.utils.data.sampler import SubsetRandomSampler
from model.trainer import Model_SemiMean
import numpy as np
from model.TCNmodel import TCN, MTNet
from itertools import cycle
import torch.nn as nn
from optim.generalWay import *
from model.model_backbone import SimConv4

def trainPI(x_train, y_train, x_val, y_val, x_test, y_test, opt):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    ckpt_dir = opt.ckpt_dir
    prob = 0.2  # Transform Probability
    raw = transforms.Raw()
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)
    criterion_classification = nn.CrossEntropyLoss()
    criterion_forecasting = nn.MSELoss()

    transforms_list = {'jitter': [jitter],
                       'cutout': [cutout],
                       'scaling': [scaling],
                       'magnitude_warp': [magnitude_warp],
                       'time_warp': [time_warp],
                       'window_slice': [window_slice],
                       'window_warp': [window_warp],
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': [raw]}

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    train_transform = transforms.Compose(transforms_targets)
    train_transform_label = transforms.Compose(transforms_targets + [transforms.ToTensor()])
    tensor_transform = transforms.ToTensor()

    train_set_labeled = UCR2018(data=x_train, targets=y_train, transform=train_transform_label)
    val_set = UCR2018(data=x_val, targets=y_val, transform=tensor_transform)
    test_set = UCR2018(data=x_test, targets=y_test, transform=tensor_transform)
    train_set = MultiUCR2018_Forecast(data=x_train, targets=y_train, K=K,
                                      transform=train_transform,
                                      totensor_transform=tensor_transform)

    train_dataset_size = len(train_set_labeled)
    partial_size = int(opt.label_ratio * train_dataset_size)
    train_ids = list(range(train_dataset_size))
    np.random.shuffle(train_ids)
    train_sampler = SubsetRandomSampler(train_ids[:partial_size])

    train_loader_label = torch.utils.data.DataLoader(train_set_labeled, batch_size=batch_size, sampler=train_sampler)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    rampup = exp_rampup(opt.weight_rampup)
    patience = opt.patience
    early_stopping = EarlyStopping(patience, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

    backbone_lineval = SimConv4().cuda()
    linear_layer = torch.nn.Linear(opt.feature_size, opt.nb_class).cuda()
    # mtnet = MTNet(in_channels=x_train.shape[2], nb_classes=opt.nb_class).cuda()
    # optimizer = torch.optim.Adam(mtnet.parameters(), lr=opt.learning_rate)
    optimizer = torch.optim.Adam([{'params': backbone_lineval.parameters()},
                  {'params': linear_layer.parameters()}], lr=opt.learning_rate)

    train_max_epoch, train_best_acc, val_best_acc = 0, 0, 0

    for epoch in range(tot_epochs):
        # mtnet.train()
        backbone_lineval.train()
        linear_layer.train()
        acc_label, acc_unlabel, loss_label, loss_unlabel, alp = list(), list(),list(), list(), list()
        for i, (data_labeled, data_unlabel) in enumerate(zip(cycle(train_loader_label), train_loader)):
            x, targets = data_labeled
            aug1, aug2, targetAug = data_unlabel
            x, targets, aug1, aug2, targetAug = x.cuda(), targets.cuda(), \
                                                aug1.float().cuda(), aug2.float().cuda(), targetAug.cuda()
            # output = mtnet(x)
            # output1 = mtnet(aug1)
            # outputs2 = mtnet(aug2)
            output = backbone_lineval(x)
            output = linear_layer(output)
            output1 = linear_layer(backbone_lineval(aug1))
            output2 = linear_layer(backbone_lineval(aug2))

            loss = criterion_classification(output, targets)
            pi_loss = criterion_forecasting(output1, output2)

            loss_mtl = loss + rampup(epoch) * opt.usp_weight  * pi_loss
            optimizer.zero_grad()
            loss_mtl.backward()
            optimizer.step()

            prediction = output.argmax(-1)
            correct = prediction.eq(targets.view_as(prediction)).sum()
            loss_label.append(loss.item())
            acc_label.append(100.0 * correct / len(targets))

            prediction = output1.argmax(-1)
            correct = prediction.eq(targetAug).sum()
            loss_unlabel.append(pi_loss.item())
            acc_unlabel.append(100.0 * correct / len(targetAug))

        acc_epoch_label = sum(acc_label) / len(acc_label)
        acc_epoch_unlabel = sum(acc_unlabel) / len(acc_unlabel)
        loss_epoch_unlabel = sum(loss_unlabel) / len(loss_unlabel)

        if acc_epoch_unlabel > train_best_acc:
            train_best_acc = acc_epoch_unlabel
            train_max_epoch = epoch

        acc_vals, acc_tests = list(), list()
        backbone_lineval.eval()
        linear_layer.eval()

        with torch.no_grad():
            for i, (x, target) in enumerate(val_loader):
                x, target = x.cuda(), target.cuda()
                output = linear_layer(backbone_lineval(x))
                # estimate the accuracy
                prediction = output.argmax(-1)
                correct = prediction.eq(target.view_as(prediction)).sum()
                accuracy = (100.0 * correct / len(target))
                acc_vals.append(accuracy.item())
                val_acc = sum(acc_vals) / len(acc_vals)

                if val_acc >= val_best_acc:
                    val_best_acc = val_acc
                    val_best_epoch = epoch
                    for i, (x, target) in enumerate(test_loader):
                        x, target = x.cuda(), target.cuda()
                        # output = mtnet(x)
                        output = linear_layer(backbone_lineval(x))
                        prediction = output.argmax(-1)
                        correct = prediction.eq(target.view_as(prediction)).sum()
                        accuracy = (100.0 * correct / len(target))
                        acc_tests.append(accuracy.item())
                    test_acc = sum(acc_tests) / len(acc_tests)

                    print('[Test-{}] Val ACC:{:.2f}%, Best Test ACC.: {:.2f}% in Epoch {}'
                          .format(epoch, val_acc, test_acc, val_best_epoch))
                    opt.wb.log({'best_test_acc' : test_acc})

            # early_stopping(val_acc, mtnet)
            early_stopping(val_acc, backbone_lineval)
            if(early_stopping.early_stop):
                print("Early stopping")
                break

            if (epoch + 1) % opt.save_freq == 0:
                print("[INFO] save backbone at epoch {}!".format(epoch))
                torch.save(backbone_lineval.state_dict(), '{}/backbone_last.tar'.format(opt.ckpt_dir))

            print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch Label ACC.= {:.2f}%, UnLabel ACC.= {:.2f}%, '
                  'Train Unlabel Best ACC.= {:.1f}%, Train Max Epoch={}' \
                  .format(epoch + 1, opt.model_name, opt.dataset_name, loss_epoch_unlabel, acc_epoch_label,
                          acc_epoch_unlabel, train_best_acc, train_max_epoch))
            opt.wb.log({'loss_epoch_unlabel': loss_epoch_unlabel, 'acc_epoch_unlabel': acc_epoch_unlabel})

    # def interpret(test_loader, model, opt):
    # interpret(test_loader, mtnet, opt)
    # interpretloader = test_loader
    # quaInterpret(interpretloader, mtnet, opt)
    # createMasks(opt)
    # getMaskedAccuracy(interpretloader, mtnet, opt)
    # getAccuracyMetrics(mtnet, opt)

    return test_acc, acc_unlabel, val_best_epoch

def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper
