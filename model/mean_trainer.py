# -*- coding: utf-8 -*-
import torch
from optim.pytorchtools import EarlyStopping
import torch.nn as nn
from itertools import cycle
from collections import defaultdict
from torch.nn import functional as F
import numpy as np
from copy import deepcopy
from torch.autograd.variable import Variable
import torchgeometry as tgm

class Model_Teacher(torch.nn.Module):

    def __init__(self, model, ema_model, opt, ip=1, xi=1,
                 eps_min=0.1, eps_max=5, L=0, K=2, factor=3):
        super(Model_Teacher, self).__init__()
        self.model = model
        self.ema_model = ema_model
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.global_step = 0
        self.epoch = 0
        self.usp_weight  = opt.usp_weight
        self.rampup = exp_rampup(opt.weight_rampup)
        ######Some thing about the virtual adversarial training##################
        self.lip = opt.lip
        self.saliency = opt.saliency
        self.lambda_lp = opt.lambda_lp
        self.ip = ip
        self.xi = xi
        self.d_X = lambda x, x_hat: torch.norm((x - x_hat).view(x.size(0), -1), p=2, dim=1, keepdim=True)
        self.d_Y = lambda y, y_hat: kl_divergence(logits_p=y, logits_q=y_hat)
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.lambda_Rm = opt.lambda_Rm
        self.lambda_Rs = opt.lambda_Rs
        self.lambda_reg = opt.lambda_reg
        self.beta_reg = opt.beta_reg
        self.L = L
        self.wb = opt.wb
        self.K = K
        self.factor = factor
        if eps_min == eps_max:
            self.eps = lambda x: eps_min * torch.ones(x.size(0), 1, 1, 1, device=x.device)
        else:
            self.eps = lambda x: eps_min + (eps_max - eps_min) * torch.rand(x.size(0), 1, 1, device=x.device)
        ##################################################################

    def run_test(self, predict, labels):
        correct = 0
        pred = predict.data.max(1)[1]
        correct = pred.eq(labels.data).cpu().sum()
        return correct, len(labels.data)

    def mixup_beta_batch(self, x1, x2, c1=0.4, c2=0.6):
        k = torch.distributions.Beta(c1, c2).sample((x1.size(0),))
        k_prime = torch.max(k, 1-k)
        k_prime = k_prime.view(-1, 1, 1).cuda()
        mixed_x = k_prime*x1 + (1 - k_prime)*x2
        return mixed_x

    def train(self, tot_epochs, train_loader, train_loader_label, val_loader, test_loader, opt):
        #### Training procedure #####
        patience = opt.patience
        early_stopping = EarlyStopping(patience, verbose=True,
                                       checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.learning_rate)
        train_max_epoch, train_best_acc, val_best_acc = 0, 0, 0
        lipresults = list()
        for epoch in range(tot_epochs):
            self.model.train()
            self.ema_model.train()
            acc_label, acc_unlabel, loss_label, loss_unlabel, alp = list(), list(), list(), list(), list()
            for i, (data_labeled, data_unlabel) in enumerate(zip(cycle(train_loader_label), train_loader)):
                self.global_step += 1
                # label_data and # unlabel_data
                x, targets = data_labeled
                aug1, aug2, targetAug = data_unlabel
                x, targets, aug1, aug2, targetAug = x.cuda(), targets.cuda(), \
                                        aug1.float().cuda(), aug2.float().cuda(), targetAug.cuda()
                ###############################################
                # supervised loss
                outputs = self.model(x)
                loss = self.ce_loss(outputs, targets)
                prediction = outputs.argmax(-1)
                correct = prediction.eq(targets.view_as(prediction)).sum()
                loss_label.append(loss.item())
                acc_label.append(100.0 * correct / len(targets))
                self.wb.log({'loss_label': loss, 'acc_label':100.0*correct/len(targets)})

                # unsupervised consistency loss
                self.update_ema(self.model, self.ema_model, opt.ema_decay, self.global_step)

                output_aug = self.model(aug1)
                # aug1 = self.mixup_beta_batch(aug1, aug2)
                # aug1_hat = self.get_adversarial_perturbations(self.model, aug1)

                # output_aug_hat = self.model(aug1_hat)
                # alp_loss = self.get_alp_loss(x=aug1, x_hat=aug1_hat, y=output_aug, y_hat=output_aug_hat)
                # loss += alp_loss
                # alp.append(alp_loss.item())

                with torch.no_grad():
                    output_aug_ema = self.ema_model(aug2)
                    output_aug_ema = output_aug_ema.detach()
                cons_loss = kl_divergence(output_aug_ema, output_aug)

                prediction = output_aug.argmax(-1)
                correct = prediction.eq(targetAug).sum()
                loss_unlabel.append(cons_loss.item())
                acc_unlabel.append(100.0 * correct / len(targetAug))
                loss += cons_loss
                self.wb.log({'loss_unlabel': cons_loss, 'acc_unlabel': 100.0 * correct / len(targetAug)})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc_epoch_label = sum(acc_label) / len(acc_label)
            acc_epoch_unlabel = sum(acc_unlabel) / len(acc_unlabel)
            loss_epoch_unlabel = sum(loss_unlabel) / len(loss_unlabel)
            alp_epoch = 0
            if acc_epoch_unlabel > train_best_acc:
                train_best_acc = acc_epoch_unlabel
                train_max_epoch = epoch

            acc_vals, acc_tests = list(), list()
            self.model.eval()
            with torch.no_grad():
                for i, (x, target) in enumerate(val_loader):
                    x, target = x.cuda(), target.cuda()
                    output = self.model(x)
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
                        output = self.model(x)
                        prediction = output.argmax(-1)
                        correct = prediction.eq(target.view_as(prediction)).sum()
                        accuracy = (100.0 * correct / len(target))
                        acc_tests.append(accuracy.item())
                    test_acc = sum(acc_tests) / len(acc_tests)

                    print('[Test-{}] Val ACC:{:.2f}%, Best Test ACC.: {:.2f}% in Epoch {}'
                          .format(epoch, val_acc, test_acc, val_best_epoch))
                    self.wb.log({'best_test_acc' : test_acc})

            early_stopping(val_acc, self.model)
            if(early_stopping.early_stop):
                print("Early stopping")
                break

            if (epoch + 1) % opt.save_freq == 0:
                print("[INFO] save backbone at epoch {}!".format(epoch))
                torch.save(self.model.state_dict(), '{}/backbone_{}.tar'.format(opt.ckpt_dir, epoch))
            alp_epoch = round(alp_epoch, 4)
            lipresults.append(str(alp_epoch))
            print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch Label ACC.= {:.2f}%, UnLabel ACC.= {:.2f}%, '
                  'Train Unlabel Best ACC.= {:.1f}%, Train Max Epoch={}' \
                  .format(epoch + 1, opt.model_name, opt.dataset_name, loss_epoch_unlabel,
                          acc_epoch_label, acc_epoch_unlabel, train_best_acc, train_max_epoch))
            self.wb.log({'acc_epoch_label': acc_epoch_label, 'acc_epoch_unlabel': acc_epoch_unlabel,
                         'loss_epoch_unlabel': loss_epoch_unlabel})

        # # record lipschitz constant
        # if opt.lip and not opt.saliency:
        #     result = "\nAdv, "
        # elif opt.lip and opt.saliency:
        #     result = "\nSS, "
        # else:
        #     result = "\nw/o SS, "
        # s = ","
        # lipresults = lipresults[:600]
        # f = open("./result.txt", "a")
        # f.write(result + s.join(lipresults))
        # f.write("\r\n")
        # f.close()

        return test_acc, acc_epoch_unlabel, val_best_epoch

    def get_alp_loss(self, x, x_hat, y, y_hat):
        y_diff = self.d_Y(y, y_hat)
        x_diff = self.d_X(x, x_hat)
        nan_count = torch.sum(y_diff != y_diff).item()
        inf_count = torch.sum(y_diff == float("inf")).item()
        neg_count = torch.sum(y_diff < 0).item()
        lip_ratio = y_diff / x_diff
        alp = torch.clamp(lip_ratio - self.L, min=0)
        nonzeros = torch.nonzero(alp)
        alp_count = nonzeros.size(0)
        alp_l1 = torch.mean(alp)
        alp_loss = self.lambda_Rs * alp_l1
        return alp_loss

    def update_ema(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step +1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1-alpha, param.data)

    def get_adversarial_perturbations(self, f, x):
        # Time series shape [batch_size, length, feature]
        r_adv = self.virtual_adversarial(f=f, x=x.detach())
        x_hat = x + r_adv
        return x_hat

    def get_SS(self, f, x):
        mask_L = int(x.shape[1] / self.factor)
        shape = (x.shape[0], self.K, mask_L, x.shape[2])
        x_hat = perturbation(x, 'noise')
        self.mask = Variable(torch.zeros(shape).cuda(), requires_grad=True)
        normalize = lambda vector: F.normalize(vector.view(x.shape[0], -1, 1), p=2, dim=1).view_as(vector)
        y = f(x)
        data_X = torch.zeros((x.shape[0], self.K, x.shape[1], x.shape[2])).cuda()
        for _ in range(self.ip):
            self.mask.requires_grad_()
            m1 = self.mask.view(x.shape[0] * self.K, 1, mask_L, -1)
            m_scale = F.interpolate(m1, size=(x.shape[1], x.shape[2]), mode="bilinear")
            m_scale = m_scale.view(x.shape[0], self.K, x.shape[1], -1)
            data_X[:, 0, ...], data_X[:, 1, ...] = x, x_hat
            m1 = torch.sigmoid(m_scale)
            sum_masks = m1.sum(1, keepdim=True)
            m1 = m1 / sum_masks
            mixed_data = m1 * data_X
            x_processed = mixed_data.sum(1)
            y_hat = f(x_processed)
            y_diff = self.d_Y(y, y_hat)
            y_diff = torch.mean(y_diff)
            y_diff.backward(retain_graph=True)
            with torch.no_grad():
                self.mask -= normalize(self.mask.grad).detach()

        data_X[:, 0, ...], data_X[:, 1, ...] = x, x_hat
        m1 = self.mask.view(x.shape[0] * self.K, 1, mask_L, -1)
        m_scale = F.interpolate(m1, size=(x.shape[1], x.shape[2]), mode="bilinear")
        m_scale = m_scale.view(x.shape[0], self.K, x.shape[1], -1)

        m1 = torch.sigmoid(m_scale)
        sum_masks = m1.sum(1, keepdim=True)
        m1 = m1 / sum_masks
        mixed_data = m1 * data_X
        x_processed = mixed_data.sum(1)
        return x_processed

    def virtual_adversarial(self, f, x):
        batch_size = x.size(0)
        f.zero_grad()
        normalize = lambda vector: F.normalize(vector.view(batch_size, -1, 1), p=2, dim=1).view_as(x)
        d = torch.rand_like(x) - 0.5
        d = normalize(d)
        for _ in range(self.ip):
            d.requires_grad_()
            x_hat = torch.clamp(x + self.xi*d, min=-1, max=1)
            y = f(x)
            y_hat = f(x_hat)
            lds_loss = mse_with_softmax(y, y_hat)
            # y_diff = self.d_Y(y, y_hat)

            reg1 = torch.norm((x-x_hat), p=2)
            temp = (x[:,:-2,:] - x_hat[:,1:-1,:] + x[:,2:,:] - x_hat[:,1:-1,:])
            temp = temp.reshape(batch_size, -1)
            diff1 = torch.norm(temp, dim=1, p=2)
            reg2 = torch.mean(diff1)
            reg_loss = (0.5*reg1 + self.lambda_reg *reg2)

            # y_diff = lds_loss
            y_diff = lds_loss + self.beta_reg * reg_loss
            y_diff.backward()
            d = normalize(d.grad).detach()
            f.zero_grad()
            self.wb.log({'lds_loss':lds_loss, 'reg1':reg1, 'reg2':reg2})

        r_adv = normalize(d) * self.eps(x)
        r_adv[r_adv != r_adv] = 0
        r_adv[r_adv == float("inf")] = 0
        temp = torch.lt(torch.norm(r_adv.view(batch_size, -1, 1), p=2, dim=1, keepdim=True), self.eps_min).float() \
                + torch.gt(torch.norm(r_adv.view(batch_size, -1, 1), p=2, dim=1, keepdim=True), self.eps_max).float()
        r_adv_mask = torch.clamp(temp, min=0, max=1)
        r_adv_mask = r_adv_mask.expand_as(x)
        r_adv = (1 - r_adv_mask) * r_adv + r_adv_mask * normalize(torch.rand_like(x) - 0.5)
        return r_adv

def kl_divergence(logits_p, logits_q):
    ptemp = torch.distributions.categorical.Categorical(logits=logits_p)
    qtemp = torch.distributions.categorical.Categorical(logits=logits_q)
    result = torch.distributions.kl.kl_divergence(
        p=ptemp, q=qtemp
    )
    loss_result = torch.mean(result)
    return loss_result

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

def mse_with_softmax(logit1, logit2):
    assert logit1.size() == logit2.size()
    return F.mse_loss(F.softmax(logit1, 1), F.softmax(logit2, 1))

def perturbation(X, method, std=0.01, mean=0.):
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
