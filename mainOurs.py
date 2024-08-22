# -*- coding: utf-8 -*-

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from optim.pretrain import *
from optim.generalWay import *
from optim.train import supervised_train
from optim.PI import *
from optim.MTL import *
from optim.TapNet import *
from datetime import datetime
from dataloader.tempData import *
import math
import warnings
import wandb
warnings.filterwarnings("ignore")

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--save_freq', type=int, default=200,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--K', type=int, default=4, help='Number of augmentation for each sample')
    parser.add_argument('--alpha', type=float, default=0.5, help='Past-future split point')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--feature_size', type=int, default=64,
                        help='feature_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')
    parser.add_argument('--patience', type=int, default=200, help='training patience')
    parser.add_argument('--aug_type', type=str, default='none', help='Augmentation type')

    parser.add_argument('--class_type', type=str, default='3C', help='Classification type')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')

    parser.add_argument('--learning_rate', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--weight_rampup', type=int, default=30, help='weight rampup')
    # model dataset
    parser.add_argument('--dataset_name', type=str, default='EpilepticSeizure', help='dataset')
    parser.add_argument('--nb_class', type=int, default=3, help='class number')

    # ucr_path = '../datasets/UCRArchive_2018'
    parser.add_argument('--ucr_path', type=str, default='./datasets',help='Data root for dataset.')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                        help='Data path for checkpoint.')
    # method
    parser.add_argument('--backbone', type=str, default='SimConv4')
    parser.add_argument('--model_name', type=str, default='SemiTeacher',
                        choices=['SupCE', 'SemiTime','SemiTeacher', 'PI', 'MTL', 'TapNet'], help='choose method')
    parser.add_argument('--label_ratio', type=float, default=0.4, help='label ratio')
    parser.add_argument('--usp_weight', type=float, default=1, help='usp weight')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='weight')
    parser.add_argument('--model_select', type=str, default='TCN', help='Training model type')
    parser.add_argument('--nhid', type=int, default=128, help='feature_size')
    parser.add_argument('--levels', type=int, default=10, help='feature_size')
    parser.add_argument('--ksize', type=int, default=3, help='kernel size')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout applied to layers (default: 0.05)')
    parser.add_argument('--lip', type=bool, default=True, help='Whether to limit the lipisitz constant')
    parser.add_argument('--saliency', type=bool, default=False, help='Whether to use series saliency')
    parser.add_argument('--lambda_lp', type=float, default=1, help='lipisitz weight')
    parser.add_argument('--L', type=int, default=0, help='lipisitz constant')
    parser.add_argument('--iter', type=int, default=50, help='iteration')
    # cuda settings
    parser.add_argument('--no-cuda', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    parser.add_argument('--wd', type=float, default=1e-3,
                        help='Weight decay (L2 loss on parameters). default: 5e-3')
    parser.add_argument('--stop_thres', type=float, default=1e-9,
                        help='The stop threshold for the training error. If the difference between training losses '
                             'between epoches are less than the threshold, the training will be stopped. Default:1e-9')

    parser.add_argument('--use_cnn', type=bool, default=True,
                        help='whether to use CNN for feature extraction. Default:False')
    parser.add_argument('--use_lstm', type=bool, default=True,
                        help='whether to use LSTM for feature extraction. Default:False')
    parser.add_argument('--use_rp', type=bool, default=False,
                        help='Whether to use random projection')
    parser.add_argument('--rp_params', type=str, default='-1,3',
                        help='Parameters for random projection: number of random projection, '
                             'sub-dimension for each random projection')
    parser.add_argument('--use_metric', action='store_true', default=False,
                        help='whether to use the metric learning for class representation. Default:False')
    parser.add_argument('--metric_param', type=float, default=0.01,
                        help='Metric parameter for prototype distances between classes. Default:0.000001')
    parser.add_argument('--filters', type=str, default="256,256,128",
                        help='filters used for convolutional network. Default:256,256,128')
    parser.add_argument('--kernels', type=str, default="8,5,3",
                        help='kernels used for convolutional network. Default:8,5,3')
    parser.add_argument('--dilation', type=int, default=1,
                        help='the dilation used for the first convolutional layer. '
                             'If set to -1, use the automatic number. Default:-1')
    parser.add_argument('--layers', type=str, default="500,300",
                        help='layer settings of mapping function. [Default]: 500,300')
    parser.add_argument('--lstm_dim', type=int, default=128,
                        help='Dimension of LSTM Embedding.')
    parser.add_argument('--tv_coeff', type=float, default='2', help='Coefficient of TV')
    parser.add_argument('--tv_beta', type=float, default='1', help='TV beta value')
    parser.add_argument('--l1_coeff', type=float, default='1e-2', help='L1 regularization')
    parser.add_argument('--factor', type=int, default=7, help='Factor to upsampling')
    parser.add_argument('--img_path', type=str, default='examples/fl.jpg', help='image path')
    parser.add_argument('--lambda_Rm', type=float, default=1, help='lambda_Rm')
    parser.add_argument('--lambda_Rs', type=float, default=1, help='lambda_Rs')
    parser.add_argument('--lambda_reg', type=float, default=1, help='lambda_reg')
    parser.add_argument('--beta_reg', type=float, default=1e-3, help='beta_reg')
    parser.add_argument('--Saliency_dir', type=str, default='results/saliency/', help='saliency path')
    parser.add_argument('--Mask_dir', type=str, default='results/mask/', help='saliency path')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    import os
    import numpy as np

    opt = parse_option()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    opt.sparse = True
    opt.layers = [int(l) for l in opt.layers.split(",")]
    opt.kernels = [int(l) for l in opt.kernels.split(",")]
    opt.filters = [int(l) for l in opt.filters.split(",")]
    opt.rp_params = [float(l) for l in opt.rp_params.split(",")]
    opt.wb = wandb.init(project=opt.dataset_name+"_semitime", config=opt, mode="online", group=str(opt.label_ratio))
    exp = 'exp-cls'

    Seeds = [2000]
    Runs = range(0, 2, 1)

    aug1 = ['jitter','cutout']
    aug2 = ['G0']
    if opt.model_name == 'SemiTime':
        model_paras = 'label{}_{}'.format(opt.label_ratio, opt.alpha)
    if opt.model_name == "SemiTeacher":
        model_paras = 'label{}_{}'.format(opt.label_ratio, opt.saliency)
    else:
        model_paras = 'label{}'.format(opt.label_ratio)

    if aug1 == aug2:
        opt.aug_type = [aug1]
    elif type(aug1) is list:
        opt.aug_type = aug1 + aug2
    else:
        opt.aug_type = [aug1, aug2]

    log_dir = './results/{}/{}/{}/{}'.format(exp, opt.dataset_name, opt.model_name, model_paras)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file2print_detail_train = open("{}/train_detail.log".format(log_dir), 'a+')
    print(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), file=file2print_detail_train)
    print("Dataset  Train  Test  Dimension  Class  Seed  Acc_label  Acc_unlabel  Epoch_max",file=file2print_detail_train)
    file2print_detail_train.flush()

    file2print = open("{}/test.log".format(log_dir), 'a+')
    print(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), file=file2print)
    print("Dataset  Acc_mean   Acc_std  Epoch_max", file=file2print)
    file2print.flush()

    file2print_detail = open("{}/test_detail.log".format(log_dir), 'a+')
    print(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), file=file2print_detail)
    print("Dataset  Train  Test   Dimension  Class  Seed  Acc_max  Epoch_max", file=file2print_detail)
    file2print_detail.flush()

    ACCs = {}

    MAX_EPOCHs_seed = {}
    ACCs_seed = {}
    for seed in Seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        opt.ckpt_dir = './ckpt/{}/{}/{}/{}/{}/{}'.format(
            exp, opt.model_name, opt.dataset_name, '_'.join(opt.aug_type),
            model_paras, str(seed))

        if not os.path.exists(opt.ckpt_dir):
            os.makedirs(opt.ckpt_dir)

        print('[INFO] Running at:', opt.dataset_name)

        if opt.dataset_name == "CricketX" or opt.dataset_name == "UWaveGestureLibraryAll" \
                or opt.dataset_name == "InsectWingbeatSound" or opt.dataset_name == "EpilepticSeizure"\
                or opt.dataset_name == "MFPT" or opt.data_name == "XJTU":
            x_train, y_train, x_val, y_val, x_test, y_test, opt.nb_class, _ = load_ucr2018(opt.ucr_path, opt.dataset_name)
        elif opt.dataset_name == "Heartbeat" or opt.dataset_name == "NATOPS" \
                or opt.dataset_name == "SelfRegulationSCP2":
            x_train, y_train, x_val, y_val, x_test, y_test, opt.nb_class, idx = load_multi_ts(opt.ucr_path, opt.dataset_name)

        # update random permutation parameter
        if opt.rp_params[0] < 0:
            dim = x_train.shape[2]
            opt.rp_params = [3, math.floor(dim / (3 / 2))]
        else:
            dim = x_train.shape[1]
            opt.rp_params[1] = math.floor(dim)

        opt.rp_params = [int(l) for l in opt.rp_params]
        print("rp_params:", opt.rp_params)

        ACCs_run = {}
        MAX_EPOCHs_run = {}
        acc_test, acc_unlabel, epoch_max = 0, 0, 0
        for run in Runs:
            if opt.model_name == 'SupCE':
                acc_test, epoch_max = supervised_train(x_train, y_train, x_val, y_val, x_test, y_test, opt)
                acc_unlabel = 0

            elif 'SemiTime' in opt.model_name:
                acc_test, acc_unlabel, epoch_max = train_SemiTime(x_train, y_train, x_val, y_val, x_test, y_test, opt)

            elif 'SemiTeacher' in opt.model_name:
                acc_test, acc_unlabel, epoch_max = train_SemiMean(x_train, y_train, x_val, y_val, x_test, y_test, opt)

            elif 'PI' in opt.model_name:
                acc_test, acc_unlabel, epoch_max = trainPI(x_train, y_train, x_val, y_val, x_test, y_test, opt)

            elif 'TapNet' in opt.model_name:
                acc_test, acc_unlabel, epoch_max = trainTapNet(x_train, y_train, x_val, y_val, x_test, y_test, opt)

            print("{}  {}  {}  {}  {}  {}  {}  {}  {}".format(opt.dataset_name, x_train.shape[0], x_test.shape[0],
                x_train.shape[1], opt.nb_class, seed, acc_test, acc_unlabel, epoch_max),
                file=file2print_detail_train)
            file2print_detail_train.flush()

            ACCs_run[run] = acc_test
            MAX_EPOCHs_run[run] = epoch_max
            # opt.wb.log({'acc_test': acc_test, 'acc_unlabel': acc_unlabel})

        ACCs_seed[seed] = round(np.mean(list(ACCs_run.values())), 2)
        MAX_EPOCHs_seed[seed] = np.max(list(MAX_EPOCHs_run.values()))

        print("{}  {}  {}  {}  {}  {}  {}  {}".format(
            opt.dataset_name, x_train.shape[0], x_test.shape[0], x_train.shape[1], opt.nb_class,
            seed, ACCs_seed[seed], MAX_EPOCHs_seed[seed]),
            file=file2print_detail)

        file2print_detail.flush()

    ACCs_seed_mean = round(np.mean(list(ACCs_seed.values())), 2)
    ACCs_seed_std = round(np.std(list(ACCs_seed.values())), 2)
    MAX_EPOCHs_seed_max = np.max(list(MAX_EPOCHs_seed.values()))
    opt.wb.log({"acc_mean" : ACCs_seed_mean, "acc_std"  : ACCs_seed_std})
    print("{} {} {} {}".format(opt.dataset_name, ACCs_seed_mean,
                               ACCs_seed_std, MAX_EPOCHs_seed_max), file=file2print)
    file2print.flush()
