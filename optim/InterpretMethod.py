from tqdm import tqdm
import torchvision.models as models
import cv2
import numpy as np
from torch.autograd import Variable
from PIL import Image,ImageFilter
import torch

from torch.nn import functional as F
import os
from scipy.ndimage.filters import gaussian_filter
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import time

import matplotlib.image as mpimg
from scipy import signal

def numpy_to_torch(img, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if cuda_available():
        output = output.cuda()
    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v

def perturbation(X, method,radius):
    img_shape=X.shape
    img = X.data.cpu().numpy()
    if method=='noise':
        noise=np.random.normal(0, 25.5, img_shape).astype("float32")
        img = img + noise
        #img=Image.fromarray(np.uint8(img))
    elif method=='blur':
        if radius==None:
            radius=10
        #img=img.filter(ImageFilter.GaussianBlur(radius))
        img = gaussian_filter(img, sigma=radius)
    elif method=='original':
        pass
    return img

def cuda_available():
    use_cuda = torch.cuda.is_available()
    return use_cuda

def image_preprocessing(img):
    if cuda_available():
        img = torch.from_numpy(img).cuda()
    else:
        img = torch.from_numpy(img)
    img.unsqueeze_(0)
    img.unsqueeze_(0)
    return img

def TV(img,tv_coeff,tv_beta):
    temp1, temp2 = img[:, :, :, :-1], img[:, :, :, 1:]

    tv_loss = tv_coeff * (
            torch.sum(torch.pow(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]),tv_beta)) +
            torch.sum(torch.pow(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]),tv_beta))
    )
    return tv_loss

class Optimize():
    def  __init__(self, model,original_img,perturbed_img, sample_Y, opt):
        self.model=model
        self.original_img=original_img
        self.perturbed_img=perturbed_img
        self.factor=opt.factor
        self.iter=opt.iter
        self.lr=opt.lr
        self.tv_coeff=opt.tv_coeff
        self.tv_beta=opt.tv_beta
        self.l1_coeff=opt.l1_coeff
        if len(sample_Y.shape) == 0:
            self.sampleY = sample_Y.unsqueeze_(0)
        else:
            self.sampleY = sample_Y
        self.opt = opt
        self.sampleY = sample_Y

    def upsample(self,img):
        if cuda_available():
            upsample=F.interpolate(img,size=(self.original_img.size(2), \
                                    self.original_img.size(3)),mode='bilinear'\
                                   ,align_corners=False).cuda()
        else:
            upsample = F.interpolate(img, size=(self.original_img.size(2), \
                                    self.original_img.size(3)),mode='bilinear'\
                                     ,align_corners=False)
        return upsample
    def build(self):
        mask_init=np.random.rand(int(self.original_img.size(2)/self.factor),\
                                  int(self.original_img.size(3)/self.factor))
        mask = numpy_to_torch(mask_init)
        optimizer=torch.optim.Adam([mask],self.lr)
        criterion = nn.L1Loss(size_average=False)
        if cuda_available():
            criterion.cuda()
        CE = torch.nn.CrossEntropyLoss()
        for i in tqdm(range(self.iter)):
            upsampled_mask = self.upsample(mask)
            mask_img = torch.mul(upsampled_mask, self.original_img) + \
                       torch.mul((1 - upsampled_mask), self.perturbed_img)
            mask_img = mask_img.squeeze(1)
            mask_output = self.model(mask_img) # batch_size * window * feature
            total_loss = CE(mask_output, self.sampleY)

            loss = self.l1_coeff * torch.mean(1 - torch.abs(mask))+\
                 TV(mask, self.tv_coeff, self.tv_beta) + 0.01*total_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mask.data.clamp_(0, 1)

        gen_mask = self.upsample(mask)
        gen_mask = gen_mask.expand_as(self.original_img)
        t = gen_mask.detach().cpu().numpy()[0,0]
        np.save('mask', t)
        return gen_mask

def save(gen_mask):
    font1 = {'family': 'Times New Roman',
             'size': 16}
    fontsize = 10
    mask = gen_mask.cpu().data.numpy()[0][0]
    mask = (mask - np.min(mask)) / np.max(mask)
    mask_color = 1 - mask
    timestep = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))
    mask_color_path = "log/heat_"+timestep+".png"
    mask_color = mask_color.transpose(1,0)
    fig, axis = plt.subplots()
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    hm = axis.pcolor(mask_color, cmap = plt.cm.jet)
    axis.set(xlim=[0, mask_color.shape[1]], ylim=[0, mask_color.shape[0]], aspect=1)
    axis.set_xlabel("Time", fontsize = fontsize)
    axis.set_ylabel("Heartbeat value", fontsize = fontsize)
    dirs = "results/"
    shrink_scale = 1.0
    aspect = mask_color.shape[0] / float(mask_color.shape[1])
    if aspect < 1.0:
        shrink_scale = aspect
    plt.rcParams['font.size'] = 13
    clb = plt.colorbar(hm, shrink = shrink_scale)
    clb.ax.set_title("Weight", fontsize=fontsize)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    plt.savefig(dirs + "SeriesSaliency_" + timestep + ".png", bbox_inches='tight')
    plt.show()




