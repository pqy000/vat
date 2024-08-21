# -*- coding: utf-8 -*-
from datetime import datetime
import torch
import utils.transforms as transforms
from dataloader.ucr2018 import *
import torch.utils.data as data
from model.models import *
from model.architecture import TimeConv
from torch.utils.data.sampler import SubsetRandomSampler
from model.trainer import Model_SemiMean
import numpy as np
from model.TCNmodel import TCN
from optim.InterpretMethod import *
import os
import time
import pandas as pd
import torch.utils.data as data_utils
from torch.autograd import Variable
import torch.nn.functional as F
from optim.help import *
import timesynth as ts
import torch.utils.data as data_utils
percentageArray = [i for i in range(10, 91, 10)]
maskedPercentages = [ i for i in range(0, 101, 10)]

def train_SemiMean(x_train, y_train, x_val, y_val, x_test, y_test, opt):
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

    ### Different Types of train, validation and test loader.
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

    ##########################################################
    #########Various data augmentation transformation#########
    ##########################################################
    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    train_transform = transforms.Compose(transforms_targets)
    train_transform_label = transforms.Compose(transforms_targets + [transforms.ToTensor()])
    tensor_transform = transforms.ToTensor()
    trend_transform = transforms.Scaling(sigma=1.0, p=0.8)

    #########################################
    #########Different torch dataset#########
    #########################################
    train_set_labeled = UCR2018(data=x_train, targets=y_train, transform=train_transform_label)
    val_set = UCR2018(data=x_val, targets=y_val, transform=tensor_transform)
    test_set = UCR2018(data=x_test, targets=y_test, transform=tensor_transform)
    train_set = MultiUCR2018_Forecast(data=x_train, targets=y_train, K=K,
                                      transform=train_transform,
                                      totensor_transform=tensor_transform,
                                      trend_transform=trend_transform)

    #######################################
    #########Separate labeled data#########
    #######################################
    train_dataset_size = len(train_set_labeled)
    partial_size = int(opt.label_ratio * train_dataset_size)
    train_ids = list(range(train_dataset_size))
    np.random.shuffle(train_ids)
    train_sampler = SubsetRandomSampler(train_ids[:partial_size])

    #######################################
    #########Generate data loader##########
    #######################################
    train_loader_label = torch.utils.data.DataLoader(train_set_labeled, batch_size=batch_size, sampler=train_sampler)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    ##############################################
    #########Generate model architec##############
    ##############################################

    if opt.model_select == "TCN":
        channel_sizes = [opt.nhid] * opt.levels
        kernel_size = opt.ksize
        model = TCN(input_size=1, output_size=opt.nb_class,
                    num_channels=channel_sizes, kernel_size=kernel_size,
                    dropout=opt.dropout).cuda()
        ema_model = TCN(input_size=1, output_size=opt.nb_class,
                    num_channels=channel_sizes, kernel_size=kernel_size,
                    dropout=opt.dropout).cuda()

    trainer = Model_SemiMean(model, ema_model, opt).cuda()
    torch.save(trainer.model.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    test_acc, acc_unlabel, best_epoch = trainer.train(tot_epochs=tot_epochs, train_loader=train_loader,
                                                      train_loader_label=train_loader_label,
                                                      val_loader=val_loader,
                                                      test_loader=test_loader,
                                                      opt=opt)
    torch.save(trainer.model.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    return test_acc, acc_unlabel, best_epoch

def interpret(test_loader, model, opt):
    total_loss = 0
    n_samples = 0
    model.train()
    for i, (x, target) in enumerate(test_loader):
        sampleX, sampleY = x[0,:,:].cuda(), target[0].cuda()
        original_img = perturbation(sampleX, 'original', None)
        original_img_tensor = image_preprocessing(original_img)
        perturbed_img = perturbation(sampleX, 'blur', 5)
        perturbed_img_tensor = image_preprocessing(perturbed_img).squeeze(0)
        mask_Opt = Optimize(model, original_img_tensor, perturbed_img_tensor, sampleY, opt)
        gen_mask = mask_Opt.build()
        save(gen_mask)
        break

def quaInterpret(test_loader, model, opt):
    model.train()
    step = 0
    mask = None
    for i, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        original_img = perturbation(data, 'original', None)
        original_img_tensor = torch.from_numpy(original_img).cuda().unsqueeze(1)
        perturbed_img = perturbation(data, 'noise', 5)
        perturbed_img_tensor = torch.from_numpy(perturbed_img).cuda().unsqueeze(1)
        mask_Opt = Optimize(model, original_img_tensor, perturbed_img_tensor, target.cuda(), opt)
        gen_mask = mask_Opt.build()
        if mask is None:
            mask = gen_mask.clone().detach()
        else:
            mask = torch.cat((mask, gen_mask.clone().detach()))
        step += 1
    print(mask.shape)
    mask = RescaledSaliency(mask)
    if not os.path.exists(opt.Saliency_dir): os.makedirs(opt.Saliency_dir)
    path = opt.Saliency_dir + opt.model_name + "_" + opt.dataset_name + "_rescaled"
    np.save(path, mask)

def createMasks(opt):
    if not os.path.exists(opt.Mask_dir): os.makedirs(opt.Mask_dir)
    Saliency_Methods = ['ours']
    for saliency in Saliency_Methods:
        mask = np.load(opt.Saliency_dir + opt.model_name + "_" + opt.dataset_name + "_rescaled.npy")
        mask = mask.reshape(mask.shape[0], -1)
        indexGrid = np.zeros((mask.shape[0], mask.shape[1], len(percentageArray)))
        indexGrid[:,:,:] = np.nan
        for i in range(mask.shape[0]):
            indexs = getIndexHighest(mask[i,:], percentageArray)
            for l in range(len(indexs)):
                indexGrid[i,:len(indexs[l]),l]=indexs[l]
        file_name = opt.Mask_dir + opt.model_name
        for p, percentage in enumerate(percentageArray):
            np.save(file_name+"_"+str(percentage)+"_"+opt.dataset_name, indexGrid[:,:,p])
        print("Creating Masks for "+opt.model_name+" Dataset " + opt.dataset_name)

def maskData(mask, dataX):
    newData = np.zeros((dataX.shape))
    noiseSample = generateNewSample(dataX)
    for i in range(mask.shape[0]):
        newData[i, :] = dataX[i, :]
        cleanIndex = mask[i, :]
        cleanIndex = cleanIndex[np.logical_not(pd.isna(cleanIndex))]
        cleanIndex = cleanIndex.astype(np.int64)
        newData[i, cleanIndex] = noiseSample[cleanIndex]
    newData = newData.reshape(newData.shape[0],-1)
    return newData

def generateNewSample(sampleX):
    noise = ts.noise.GaussianNoise(std=0.4)
    sample = np.random.normal(0, 5, [sampleX.shape[1]])
    return sample

def getMaskedAccuracy(test_loader, model, opt):
    model.eval()
    total_loss = 0
    predict, test = None, None
    Y_DimOfGrid = len(maskedPercentages) + 1
    X_DimOfGrid = 1
    Grid = np.zeros((X_DimOfGrid, Y_DimOfGrid), dtype="object")
    Gridce = np.zeros((X_DimOfGrid, Y_DimOfGrid), dtype="object")
    Grid[:,0] = "SeriesSaliency"
    Gridce[:, 0] = "SeriesSaliency"
    columns = ["SaliencyMethod"]
    for m in maskedPercentages: columns.append(str(m))
    saliency = "ours"
    test_acc, test_ce = checkAccuracy(test_loader, model)

    for i, maskedPercentage in enumerate(maskedPercentages):
        if maskedPercentage == 0:
            Grid[0][i + 1] = test_acc
            Gridce[0][i + 1] = test_ce
        else:
            if(maskedPercentage != 100):
                path = opt.Mask_dir + opt.model_name + "_" + str(maskedPercentage) + "_" + opt.dataset_name + ".npy"
                mask = np.load(path, allow_pickle=True)
                tempX, dataY = test_loader.dataset.data, test_loader.dataset.targets
                dataX = tempX.reshape(tempX.shape[0], -1)
                newData = maskData(mask, dataX)
                newData = newData.reshape(-1, tempX.shape[1], tempX.shape[2])
                newData = torch.from_numpy(newData).float()
                Maskedtest_data = data_utils.TensorDataset(newData, torch.from_numpy(dataY))
                Maskedtest_loader = data_utils.DataLoader(Maskedtest_data, batch_size=opt.batch_size, shuffle=False)

                test_acc, test_ce = checkAccuracy(Maskedtest_loader, model)
                Grid[0][i + 1] = test_acc
                Gridce[0][i + 1] = test_ce

    file_name = opt.Mask_dir + opt.model_name + "_" + opt.dataset_name
    for percent in maskedPercentages: file_name += "_" + str(percent)
    time = datetime.now().strftime("%m-%d-%H-%M-%S"),
    resultACC = file_name + "_"+ str(time) +"_acc.csv"
    resultCE = file_name + "_"+ str(time) + "_ce.csv"
    save_intoCSV(Grid, resultACC, col=columns)
    # save_intoCSV(Gridce, resultCE, col=columns)

def checkAccuracy(test_loader, model):
    acc_tests, ce_tests  = list(), list()
    CEloss = torch.nn.CrossEntropyLoss()
    for i, (x, target) in enumerate(test_loader):
        x, target = x.cuda(), target.cuda()
        output = model(x)
        prediction = output.argmax(-1)
        correct = prediction.eq(target.view_as(prediction)).sum()
        celoss = CEloss(output, target)
        accuracy = (100.00*correct / len(target))
        acc_tests.append(accuracy.item())
        ce_tests.append(celoss.item())
    test_acc = sum(acc_tests) / len(acc_tests)
    test_ce = sum(ce_tests) / len(ce_tests)
    return test_acc, test_ce

def getAccuracyMetrics(model, opt):
    Saliency_Methods = ["Ours"]

    resultacc = opt.Mask_dir+opt.model_name+"_"+opt.dataset_name+"_0_10_20_30_40_50_60_70_80_90_100acc.csv"
    resultce = opt.Mask_dir + opt.model_name+"_"+opt.dataset_name + "_0_10_20_30_40_50_60_70_80_90_100ce.csv"
    acc = load_CSV(resultacc)[:,1:][0]
    ce = load_CSV(resultce)[:,1:][0]
    resultAcc, resultCE, index = list(), list(), list()
    b = 0 # 0 0.1 1.0
    index.append(b)
    for i in range(10):
        b += 0.1
        index.append(b)
    AUacc = np.trapz(acc, x=index)
    AUce = np.trapz(ce, x=index)
    print(AUacc, AUce)



