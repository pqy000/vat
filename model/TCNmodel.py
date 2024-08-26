import torch.nn.functional as F
from torch import nn
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd.variable import Variable
import torchgeometry as tgm
import numpy as np
import scipy.sparse as sp
import sklearn
import sklearn.metrics
import random


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size,
                 dropout, factor = 3, series=True):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.K = 2
        self.factor = factor
        self.series = series

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        inputs_ = inputs.view(inputs.shape[0], 1, -1)
        y1 = self.tcn(inputs_)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)

def perturbation(X, method, std=0.2, mean=0.):
    img_shape = X.shape
    if method == 'noise':
        noise = torch.randn(img_shape) * std + mean
        noise = noise.cuda()
        X = X + noise
    elif method == 'blur':
        X = torch.unsqueeze(X, 1)
        X = tgm.image.gaussian_blur(X, (1, 3), (0.01, 0.3))
        X = torch.squeeze(X, 1)
    return X

class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MTNet(torch.nn.Module):
    def __init__(self, in_channels, nb_classes, dropout1=0.1):
        super(MTNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 128, 9, padding=(9 // 2))
        self.bnorm1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, 5, padding=(5 // 2))
        self.bnorm2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, 3, padding=(3 // 2))
        self.bnorm3 = nn.BatchNorm1d(128)
        self.classification_head = nn.Linear(128, nb_classes)
        self.drop = nn.Dropout(p=dropout1)

    def forward(self, x_class):
        x_class = x_class.view(-1, x_class.shape[2], x_class.shape[1])
        b1_c = self.drop(F.relu(self.bnorm1(self.conv1(x_class))))
        b2_c = self.drop(F.relu(self.bnorm2(self.conv2(b1_c))))
        b3_c = self.drop(F.relu(self.bnorm3(self.conv3(b2_c))))

        classification_features = torch.mean(b3_c,2)

        classification_out = self.classification_head(classification_features)
        return classification_out


class MTForNet(torch.nn.Module):
    def __init__(self, in_channel, nb_classes, horizon):
        super(MTForNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, 128, 9, padding=(9 // 2))
        self.bnorm1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, 5, padding=(5 // 2))
        self.bnorm2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, 3, padding=(3 // 2))
        self.bnorm3 = nn.BatchNorm1d(128)
        self.classification_head = nn.Linear(128, nb_classes)
        self.conv4 = nn.Conv1d(128, in_channel, 3, padding=(3 // 2))
        self.bnorm4 = nn.BatchNorm1d(in_channel)
        self.forecasting_head = nn.Linear(128, horizon)

    def forward(self, x_class, x_forecast):
        x_class = x_class.view(-1, x_class.shape[2], x_class.shape[1])
        x_forecast = x_forecast.view(-1, x_forecast.shape[2], x_forecast.shape[1])
        b1_c = F.relu(self.bnorm1(self.conv1(x_class)))
        b1_f = F.relu(self.bnorm1(self.conv1(x_forecast)))

        b2_c = F.relu(self.bnorm2(self.conv2(b1_c)))
        b2_f = F.relu(self.bnorm2(self.conv2(b1_f)))

        b3_c = F.relu(self.bnorm3(self.conv3(b2_c)))
        b3_f = F.relu(self.bnorm3(self.conv3(b2_f)))

        classification_features = torch.mean(b3_c,2)
        # forecasting_features = torch.mean(b3_f, 2)

        classification_out = self.classification_head(classification_features)
        forecasting_out = self.conv4(b3_f)
        forecasting_out = forecasting_out.view(-1, forecasting_out.shape[2], forecasting_out.shape[1])
        return classification_out, forecasting_out

    def forward_test(self, x_class):
        x_class = x_class.view(-1, x_class.shape[2], x_class.shape[1])
        b1_c = F.relu(self.bnorm1(self.conv1(x_class)))
        b2_c = F.relu(self.bnorm2(self.conv2(b1_c)))
        b3_c = F.relu(self.bnorm3(self.conv3(b2_c)))
        classification_features = torch.mean(b3_c, 2)
        # (64,128)#that is now we have global avg pooling, 1 feature from each conv channel
        classification_out = self.classification_head(classification_features)
        return classification_out

class TapNet(nn.Module):

    def __init__(self, nfeat, len_ts, nclass, dropout, filters, kernels, dilation, layers, use_rp, rp_params,
                 use_att=True, use_metric=False, use_lstm=False, use_cnn=True, lstm_dim=128):
        super(TapNet, self).__init__()
        self.nclass = nclass
        self.dropout = dropout
        self.use_metric = use_metric
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn

        # parameters for random projection
        self.use_rp = use_rp
        self.rp_group, self.rp_dim = rp_params

        if True:
            # LSTM
            self.channel = nfeat
            self.ts_length = len_ts

            self.lstm_dim = lstm_dim
            self.lstm = nn.LSTM(self.ts_length, self.lstm_dim)

            paddings = [0, 0, 0]
            if self.use_rp:
                self.conv_1_models = nn.ModuleList()
                self.idx = []
                for i in range(self.rp_group):
                    self.conv_1_models.append(
                        nn.Conv1d(self.rp_dim, filters[0], kernel_size=kernels[0], dilation=dilation, stride=1,
                                  padding=paddings[0]))
                    self.idx.append(np.random.permutation(nfeat)[0: self.rp_dim])
            else:
                self.conv_1 = nn.Conv1d(self.channel, filters[0], kernel_size=kernels[0], dilation=dilation, stride=1,
                                        padding=paddings[0])

            self.conv_bn_1 = nn.BatchNorm1d(filters[0])

            self.conv_2 = nn.Conv1d(filters[0], filters[1], kernel_size=kernels[1], stride=1, padding=paddings[1])

            self.conv_bn_2 = nn.BatchNorm1d(filters[1])

            self.conv_3 = nn.Conv1d(filters[1], filters[2], kernel_size=kernels[2], stride=1, padding=paddings[2])

            self.conv_bn_3 = nn.BatchNorm1d(filters[2])

            # compute the size of input for fully connected layers
            fc_input = 0
            if self.use_cnn:
                conv_size = len_ts
                for i in range(len(filters)):
                    conv_size = output_conv_size(conv_size, kernels[i], 1, paddings[i])
                fc_input += conv_size * filters[-1]
            if self.use_lstm:
                fc_input += self.lstm_dim

            if self.use_rp:
                fc_input = self.rp_group * filters[2] + self.lstm_dim

        # Representation mapping function
        layers = [fc_input] + layers
        print("Layers", layers)
        self.mapping = nn.Sequential()
        for i in range(len(layers) - 2):
            self.mapping.add_module("fc_" + str(i), nn.Linear(layers[i], layers[i + 1]))
            self.mapping.add_module("bn_" + str(i), nn.BatchNorm1d(layers[i + 1]))
            self.mapping.add_module("relu_" + str(i), nn.LeakyReLU())

        # add last layer
        self.mapping.add_module("fc_" + str(len(layers) - 2), nn.Linear(layers[-2], layers[-1]))
        if len(layers) == 2:  # if only one layer, add batch normalization
            self.mapping.add_module("bn_" + str(len(layers) - 2), nn.BatchNorm1d(layers[-1]))

        self.output = nn.Linear(layers[-1], self.nclass)

        #
        # # Attention
        # att_dim, semi_att_dim = 128, 128
        # self.use_att = use_att
        # if self.use_att:
        #     self.att_models = nn.ModuleList()
        #     for _ in range(nclass):
        #         att_model = nn.Sequential(
        #             nn.Linear(layers[-1], att_dim),
        #             nn.Tanh(),
        #             nn.Linear(att_dim, 1)
        #         )
        #         self.att_models.append(att_model)

    def forward(self, input):
        # x, labels, idx_train, idx_val, idx_test = input  # x is N * L, where L is the time-series feature dimension
        x = input.view(-1, input.shape[2], input.shape[1])
        if True:
            N = x.size(0)

            # LSTM
            if self.use_lstm:
                x_lstm = self.lstm(x)[0]
                x_lstm = x_lstm.mean(1)
                x_lstm = x_lstm.view(N, -1)

            if self.use_cnn:
                # Covolutional Network
                # input ts: # N * C * L
                if self.use_rp:
                    for i in range(len(self.conv_1_models)):
                        # x_conv = x
                        x_conv = self.conv_1_models[i](x[:, self.idx[i], :])
                        x_conv = self.conv_bn_1(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = self.conv_2(x_conv)
                        x_conv = self.conv_bn_2(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = self.conv_3(x_conv)
                        x_conv = self.conv_bn_3(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = torch.mean(x_conv, 2)

                        if i == 0:
                            x_conv_sum = x_conv
                        else:
                            x_conv_sum = torch.cat([x_conv_sum, x_conv], dim=1)

                    x_conv = x_conv_sum
                else:
                    x_conv = x
                    x_conv = self.conv_1(x_conv)  # N * C * L
                    x_conv = self.conv_bn_1(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_2(x_conv)
                    x_conv = self.conv_bn_2(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_3(x_conv)
                    x_conv = self.conv_bn_3(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = x_conv.view(N, -1) #36736 + 128

            if self.use_lstm and self.use_cnn:
                x = torch.cat([x_conv, x_lstm], dim=1)
            elif self.use_lstm:
                x = x_lstm
            elif self.use_cnn:
                x = x_conv
            #

        # linear mapping to low-dimensional space
        x = self.mapping(x)
        out = self.output(x)
        #
        # # generate the class protocal with dimension C * D (nclass * dim)
        # proto_list = []
        # for i in range(self.nclass):
        #     idx = (labels[idx_train].squeeze() == i).nonzero().squeeze(1)
        #     if self.use_att:
        #         A = self.att_models[i](x[idx_train][idx])  # N_k * 1
        #         A = torch.transpose(A, 1, 0)  # 1 * N_k
        #         A = F.softmax(A, dim=1)  # softmax over N_k
        #
        #         class_repr = torch.mm(A, x[idx_train][idx])  # 1 * L
        #         class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
        #     else:  # if do not use attention, simply use the mean of training samples with the same labels.
        #         class_repr = x[idx_train][idx].mean(0)  # L * 1
        #     proto_list.append(class_repr.view(1, -1))
        # x_proto = torch.cat(proto_list, dim=0)
        #
        # # prototype distance
        # proto_dists = euclidean_dist(x_proto, x_proto)
        # proto_dists = torch.exp(-0.5 * proto_dists)
        # num_proto_pairs = int(self.nclass * (self.nclass - 1) / 2)
        # proto_dist = torch.sum(proto_dists) / num_proto_pairs
        #
        # dists = euclidean_dist(x, x_proto)
        #
        # dump_embedding(x_proto, x, labels)
        return out

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def loaddata(filename):
    df = pd.read_csv(filename, header=None, delimiter=",")
    a = np.array(df.as_matrix())
    return a


def load_raw_ts(path, dataset, tensor_format=True):
    path = path + "raw/" + dataset + "/"
    x_train = np.load(path + 'X_train.npy')
    y_train = np.load(path + 'y_train.npy')
    x_test = np.load(path + 'X_test.npy')
    y_test = np.load(path + 'y_test.npy')
    ts = np.concatenate((x_train, x_test), axis=0)
    ts = np.transpose(ts, axes=(0, 2, 1))
    labels = np.concatenate((y_train, y_test), axis=0)
    nclass = int(np.amax(labels)) + 1
    # N * L * D  N * 1

    train_size = y_train.shape[0]

    total_size = labels.shape[0]
    idx_train = range(train_size)
    idx_val = range(train_size, total_size)
    idx_test = range(train_size, total_size)

    if tensor_format:
        # features = torch.FloatTensor(np.array(features))
        ts = torch.FloatTensor(np.array(ts))
        labels = torch.LongTensor(labels)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    return ts, labels, idx_train, idx_val, idx_test, nclass


def normalize(mx):
    """Row-normalize sparse matrix"""
    row_sums = mx.sum(axis=1)
    mx = mx.astype('float32')
    row_sums_inverse = 1 / row_sums
    f = mx.multiply(row_sums_inverse)
    return sp.csr_matrix(f).astype('float32')


def accuracy(output, labels):
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy_score = (sklearn.metrics.accuracy_score(labels, preds))

    return accuracy_score



def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def output_conv_size(in_size, kernel_size, stride, padding):

    output = int((in_size - kernel_size + 2 * padding) / stride) + 1

    return output

def dump_embedding(proto_embed, sample_embed, labels, dump_file='./plot/embeddings.txt'):
    proto_embed = proto_embed.cpu().detach().numpy()
    sample_embed = sample_embed.cpu().detach().numpy()
    embed = np.concatenate((proto_embed, sample_embed), axis=0)

    nclass = proto_embed.shape[0]
    labels = np.concatenate((np.asarray([i for i in range(nclass)]),
                             labels.squeeze().cpu().detach().numpy()), axis=0)

    with open(dump_file, 'w') as f:
        for i in range(len(embed)):
            label = str(labels[i])
            line = label + "," + ",".join(["%.4f" % j for j in embed[i].tolist()])
            f.write(line + '\n')

