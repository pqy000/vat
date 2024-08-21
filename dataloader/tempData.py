import numpy as np
import torch

def load_multi_ts(dataset_path, dataset_name):
    path = dataset_path + "/raw/" + dataset_name + "/"
    print("[INFO] {}".format(dataset_name))
    x_train = np.load(path + 'X_train.npy')
    y_train = np.load(path + 'y_train.npy')
    x_test = np.load(path + 'X_test.npy')
    y_test = np.load(path + 'y_test.npy')
    x_train, y_train, x_test, y_test = x_train.astype(np.float32), y_train.astype(np.float32), x_test.astype(np.float32), y_test.astype(np.float32)
    y_train, y_test = np.squeeze(y_train, axis=1), np.squeeze(y_test, axis=1)

    x_all = np.concatenate((x_train, x_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    n_class = int(np.amax(y_all)) + 1
    ts_idx = list(range(x_all.shape[0]))
    np.random.shuffle(ts_idx)
    x_all = x_all[ts_idx]
    y_all = y_all[ts_idx]
    label_idxs = np.unique(y_all)
    class_stat_all = {}
    for idx  in label_idxs:
        class_stat_all[idx] = len(np.where(y_all == idx)[0])
    print("[Stat] All class: {}".format(class_stat_all))

    test_idx = []
    val_idx = []
    train_idx = []
    for idx in label_idxs:
        target = list(np.where(y_all == idx)[0])
        nb_samp = int(len(target))
        test_idx += target[:int(nb_samp * 0.2)]
        val_idx += target[int(nb_samp * 0.2):int(nb_samp * 0.4)]
        train_idx += target[int(nb_samp * 0.4):]

    x_test = x_all[test_idx]
    y_test = y_all[test_idx]
    x_val = x_all[val_idx]
    y_val = y_all[val_idx]
    x_train = x_all[train_idx]
    y_train = y_all[train_idx]

    return x_train, y_train, x_val, y_val, x_test, y_test, n_class, train_idx

def load_multi_for(dataset_path, dataset_name, opt):
    path = dataset_path + "/raw/" + dataset_name + "/"
    print("[INFO] {}".format(dataset_name))
    x_train = np.load(path + 'X_train.npy')
    y_train = np.load(path + 'y_train.npy')
    x_test = np.load(path + 'X_test.npy')
    y_test = np.load(path + 'y_test.npy')
    x_train, y_train, x_test, y_test = x_train.astype(np.float32), y_train.astype(np.float32),\
                                       x_test.astype(np.float32), y_test.astype(np.float32)
    y_train, y_test = np.squeeze(y_train, axis=1), np.squeeze(y_test, axis=1)

    x_all = np.concatenate((x_train, x_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    n_class = int(np.amax(y_all)) + 1
    ts_idx = list(range(x_all.shape[0]))
    np.random.shuffle(ts_idx)
    x_all = x_all[ts_idx]
    y_all = y_all[ts_idx]
    label_idxs = np.unique(y_all)
    class_stat_all = {}
    for idx  in label_idxs:
        class_stat_all[idx] = len(np.where(y_all == idx)[0])
    print("[Stat] All class: {}".format(class_stat_all))

    test_idx = []
    val_idx = []
    train_idx = []
    for idx in label_idxs:
        target = list(np.where(y_all == idx)[0])
        nb_samp = int(len(target))
        test_idx += target[:int(nb_samp * 0.2)]
        val_idx += target[int(nb_samp * 0.2):int(nb_samp * 0.4)]
        train_idx += target[int(nb_samp * 0.4):]

    x_test = x_all[test_idx]
    y_test = y_all[test_idx]
    x_val = x_all[val_idx]
    y_val = y_all[val_idx]
    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    xf, yf = [], []
    for i in range(0, x_train.shape[1], int(opt.stride * x_train.shape[1])):
        horizon1 = int(opt.horizon * x_train.shape[1])
        if(i + horizon1 + horizon1 <= x_train.shape[1]):
            xf.append(x_train[:, i: i+horizon1, :])
            yf.append(x_train[:, i+horizon1: i+horizon1+horizon1, :])

    xf = np.vstack(xf)
    yf = np.vstack(yf)

    xf_idx = list(range(xf.shape[0]))
    np.random.shuffle(xf_idx)
    xf = xf[xf_idx]
    yf = yf[xf_idx]

    return x_train, y_train, xf, yf,  x_val, y_val, x_test, y_test, n_class, (train_idx, val_idx, test_idx)





