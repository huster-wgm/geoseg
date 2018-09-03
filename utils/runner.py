#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @CreateTime:   2018-01-26T16:50:00+09:00
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import sys
sys.path.append('./utils')
import time
import torch
import metrics
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable


Utils_DIR = os.path.dirname(os.path.abspath(__file__))
Logs_DIR = os.path.join(Utils_DIR, '../logs')
Checkpoint_DIR = os.path.join(Utils_DIR, '../checkpoint')

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def load_checkpoint(name):
    assert os.path.exists("{}/{}".format(Checkpoint_DIR, name)
                          ), "{} not exists.".format(name)
    print("Loading checkpoint: {}".format(name))
    return torch.load("{}/{}".format(Checkpoint_DIR, name))


class Base(object):
    def __init__(self, args, method, is_multi=False):
        self.args = args
        self.method = method
        self.is_multi = is_multi
        self.date = time.strftime("%h%d_%H")
        self.epoch = 0
        self.iter = 0
        self.logs = []
        self.headers = ["epoch", "iter", "train_loss", "train_acc", "train_time(sec)", "train_fps", "val_loss", "val_acc", "val_time(sec)", "val_fps"]

    def logging(self, verbose=True):
        self.logs.append([self.epoch, self.iter] +
                         self.train_log + self.val_log)
        if verbose:
            print("Epoch:{:02d}, Iter:{:05d}, train_loss:{:0.3f}, train_acc:{:0.3f}, val_loss:{:0.3f}, val_acc:{:0.3f}."
                  .format(self.epoch, self.iter, self.train_log[0], self.train_log[1], self.val_log[0], self.val_log[1]))

    def save_log(self):
        if not os.path.exists(os.path.join(Logs_DIR, 'raw')):
            os.makedirs(os.path.join(Logs_DIR, 'raw'))

        self.logs = pd.DataFrame(self.logs,
                                 columns=self.headers)

        self.logs.to_csv("{}/raw/{}_{}_{}.csv".format(Logs_DIR, self.method, self.args.trigger,
                                                      self.args.terminal), index=False, float_format='%.3f')

    def save_checkpoint(self, model, name=None):
        if self.args.cuda:
            model.cpu()
        if name:
            model_name = "{}_{}_{}_{}.pth".format(
                self.method, name, self.args.trigger, self.args.terminal)
        model_name = "{}_{}_{}.pth".format(
            self.method, self.args.trigger, self.args.terminal)
        if not os.path.exists(Checkpoint_DIR):
            os.mkdir(Checkpoint_DIR)
        torch.save(model, os.path.join(Checkpoint_DIR, model_name))
        print("Saving checkpoint: {}".format(model_name))

    def learning_curve(self, labels=["train_loss", "train_acc", "val_loss", "val_acc"]):
        if not os.path.exists(os.path.join(Logs_DIR, "curve")):
            os.mkdir(os.path.join(Logs_DIR, "curve"))
        # set style
        sns.set_context("paper", font_scale=1.5,)
        sns.set_style("ticks", {
            "font.family": "Times New Roman",
            "font.serif": ["Times", "Palatino", "serif"]})

        for _label in labels:
            plt.plot(self.logs[self.args.trigger],
                     self.logs[_label], label=_label)
        plt.ylabel("BCE-Loss / Overall Accuracy")
        if self.args.trigger == 'epoch':
            plt.xlabel("Epochs")
        else:
            plt.xlabel("Iterations")
        plt.suptitle("Training log of {}".format(self.method))
        # remove top&left line
        # sns.despine()
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        plt.savefig(os.path.join(Logs_DIR, 'curve', '{}_{}_{}.png'.format(self.method, self.args.trigger, self.args.terminal)),
                    format='png', bbox_inches='tight', dpi=1200)
        #plt.savefig('curve/{}_curve.eps'.format(fig_title), format='eps', bbox_inches='tight', dpi=1200)

        return 0


class Trainer(Base):
    def training(self, net, datasets):
        """
          input:
            net: (object) model & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        net.model.train()
        steps = len(datasets[0]) // args.batch_size
        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        train_loss, train_acc = 0, 0
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True)
            batch_iterator = iter(data_loader)
            for step in range(steps):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # convert numpy.ndarray into pytorch tensor
                x, y = next(batch_iterator)
                x = Variable(x)
                y = Variable(y)
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # training
                gen_y = net.model(x)
                if self.is_multi:
                    gen_y = gen_y[0]
                loss = F.binary_cross_entropy(gen_y, y)
                # Update generator parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                train_loss += loss.data[0]
                train_acc += metrics.overall_accuracy(gen_y.data, y.data)
                # logging
                if self.iter % args.iter_interval == 0:
                    _time = time.time() - start
                    nb_samples = args.iter_interval * args.batch_size
                    train_log = [train_loss / args.iter_interval, train_acc /
                                 args.iter_interval, _time, nb_samples / _time]
                    self.train_log = [round(x, 3) for x in train_log]
                    self.validating(net.model, datasets[1])
                    self.logging(verbose=True)
                    # reinitialize
                    start = time.time()
                    train_loss, train_acc = 0, 0

    def validating(self, model, dataset):
        """
          input:
            model: (object) pytorch model
            batch_size: (int)
            dataset : (object) dataset
          return [val_acc, val_loss]
        """
        args = self.args
        val_loss, val_acc = 0, 0
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False)
        batch_iterator = iter(data_loader)
        steps = len(dataset) // args.batch_size
        model.eval()
        start = time.time()
        for step in range(steps):
            x, y = next(batch_iterator)
            x = Variable(x, volatile=True)
            y = Variable(y, volatile=True)
            if args.cuda:
                x = x.cuda()
                y = y.cuda()

            # calculate pixel accuracy of generator
            gen_y = model(x)
            if self.is_multi:
                gen_y = gen_y[0]
            val_loss += F.binary_cross_entropy(gen_y, y).data[0]
            val_acc += metrics.overall_accuracy(gen_y.data, y.data)

        _time = time.time() - start
        nb_samples = steps * args.batch_size
        val_log = [val_loss / steps, val_acc /
                   steps, _time, nb_samples / _time]
        self.val_log = [round(x, 3) for x in val_log]

    def evaluating(self, model, dataset, split):
        """
          input:
            model: (object) pytorch model
            dataset: (object) dataset
            split: (str) split of dataset in ['train', 'val', 'test']
          return [overall_accuracy, precision, recall, f1-score, jaccard, kappa]
        """
        args = self.args
        oa, precision, recall, f1, jac, kappa = 0, 0, 0, 0, 0, 0
        model.eval()
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False)
        batch_iterator = iter(data_loader)
        steps = len(dataset) // args.batch_size

        start = time.time()
        for step in range(steps):
            x, y = next(batch_iterator)
            x = Variable(x, volatile=True)
            y = Variable(y, volatile=True)
            if args.cuda:
                x = x.cuda()
                y = y.cuda()
            # calculate pixel accuracy of generator
            gen_y = model(x)
            if self.is_multi:
                gen_y = gen_y[0]
            oa += metrics.overall_accuracy(gen_y.data, y.data)
            precision += metrics.precision(gen_y.data, y.data)
            recall += metrics.recall(gen_y.data, y.data)
            f1 += metrics.f1_score(gen_y.data, y.data)
            jac += metrics.jaccard(gen_y.data, y.data)
            kappa += metrics.kappa(gen_y.data, y.data)

        _time = time.time() - start

        if not os.path.exists(os.path.join(Logs_DIR, 'statistic')):
            os.makedirs(os.path.join(Logs_DIR, 'statistic'))

        # recording performance of the model
        nb_samples = steps * args.batch_size
        basic_info = [self.date, self.method,
                      self.epoch, self.iter, nb_samples, _time]
        basic_info_names = ['date', 'method', 'epochs',
                            'iters', 'nb_samples', 'time(sec)']

        perform = [round(idx / steps, 3)
                   for idx in [oa, precision, recall, f1, jac, kappa]]
        perform_names = ["overall_accuracy", "precision",
                         "recall", "f1-score", "jaccard", "kappa"]
        cur_log = pd.DataFrame([basic_info + perform],
                               columns=basic_info_names + perform_names)
        # save performance
        if os.path.exists(os.path.join(Logs_DIR, 'statistic', "{}.csv".format(split))):
            logs = pd.read_csv(os.path.join(
                Logs_DIR, 'statistic', "{}.csv".format(split)))
        else:
            logs = pd.DataFrame([])
        logs = logs.append(cur_log, ignore_index=True)
        logs.to_csv(os.path.join(Logs_DIR, 'statistic',
                                 "{}.csv".format(split)), index=False, float_format='%.3f')


class brTrainer(Trainer):
    def training(self, net, datasets):
        """
          input:
            net: (object) model & optimizer
            datasets : (list)['train', 'val'] dataset object
        """
        args = self.args
        net.model.train()
        steps = len(datasets[0]) // args.batch_size
        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        train_loss, train_acc = 0, 0
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True)
            batch_iterator = iter(data_loader)
            for step in range(steps):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # convert numpy.ndarray into pytorch tensor
                x, y, y_sub = next(batch_iterator)
                x = Variable(x)
                y = Variable(y)
                y_sub = Variable(y_sub)
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                    y_sub = y_sub.cuda()
                # training
                gen_y, gen_y_sub = net.model(x)
                loss_seg = F.binary_cross_entropy(gen_y, y)
                # TODO: this should be replace by hausdorff distance
                loss_edge = F.binary_cross_entropy(gen_y_sub, y_sub)
                loss = (1 - args.alpha) * loss_seg + args.alpha * loss_edge
                # Update generator parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                train_loss += loss.data[0]
                train_acc += metrics.overall_accuracy(gen_y.data, y.data)
                # logging
                if self.iter % args.iter_interval == 0:
                    _time = time.time() - start
                    nb_samples = args.iter_interval * args.batch_size
                    train_log = [train_loss / args.iter_interval, train_acc /
                                 args.iter_interval, _time, nb_samples / _time]
                    self.train_log = [round(x, 3) for x in train_log]
                    self.validating(net.model, datasets[1])
                    self.logging(verbose=True)
                    # reinitialize
                    start = time.time()
                    train_loss, train_acc = 0, 0


class zeroTrainer(Trainer):
    def training(self, net, datasets):
        """
          input:
            net: (object) model & optimizer
            datasets : (list)['train', 'val'] dataset object
        """
        args = self.args
        net.model.train()
        steps = len(datasets[0]) // args.batch_size
        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        train_loss, train_acc = 0, 0
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True)
            batch_iterator = iter(data_loader)
            for step in range(steps):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # convert numpy.ndarray into pytorch tensor
                x, y = next(batch_iterator)
                y_zero = torch.zeros(y.shape)[:, :1, :, :]
                x = Variable(x)
                y = Variable(y)
                y_zero = Variable(y_zero)
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                    y_zero = y_zero.cuda()
                # training
                gen_y, gen_y_zero = net.model(x)
                loss_seg = F.binary_cross_entropy(gen_y, y)
                # TODO: this should be replace by hausdorff distance
                loss_zero = F.binary_cross_entropy(gen_y_zero, y_zero)
                loss = loss_seg + args.alpha * loss_zero
                # Update generator parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                train_loss += loss.data[0]
                train_acc += metrics.overall_accuracy(gen_y.data, y.data)
                # logging
                if self.iter % args.iter_interval == 0:
                    _time = time.time() - start
                    nb_samples = args.iter_interval * args.batch_size
                    train_log = [train_loss / args.iter_interval, train_acc /
                                 args.iter_interval, _time, nb_samples / _time]
                    self.train_log = [round(x, 3) for x in train_log]
                    self.validating(net.model, datasets[1])
                    self.logging(verbose=True)
                    # reinitialize
                    start = time.time()
                    train_loss, train_acc = 0, 0


class mcTrainer(Trainer):
    def training(self, net, datasets):
        """
          input:
            net: (object) model & optimizer
            datasets : (list)['train', 'val'] dataset object
        """
        args = self.args
        net.model.train()
        steps = len(datasets[0]) // args.batch_size
        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        train_loss, train_acc = 0, 0
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True)
            batch_iterator = iter(data_loader)
            for step in range(steps):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # convert numpy.ndarray into pytorch tensor
                x, y, y_sub = next(batch_iterator)
                x = Variable(x)
                y = Variable(y)
                y_sub = Variable(y_sub)
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                    y_sub = y_sub.cuda()
                # training
                gens = net.model(x)
                gen_y, gen_y_sub = gens[0], gens[3]
                loss_main = F.binary_cross_entropy(gen_y, y)
                loss_sub_4 = F.binary_cross_entropy(gen_y_sub, y_sub)
                loss = (1 - args.alpha) * loss_main + args.alpha * loss_sub_4
                # Update generator parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                train_loss += loss.data[0]
                train_acc += metrics.overall_accuracy(gen_y.data, y.data)
                # logging
                if self.iter % args.iter_interval == 0:
                    _time = time.time() - start
                    nb_samples = args.iter_interval * args.batch_size
                    train_log = [train_loss / args.iter_interval, train_acc /
                                 args.iter_interval, _time, nb_samples / _time]
                    self.train_log = [round(x, 3) for x in train_log]
                    self.validating(net.model, datasets[1])
                    self.logging(verbose=True)
                    # reinitialize
                    start = time.time()
                    train_loss, train_acc = 0, 0


class cganTrainer(Trainer):
    def training(self, net, datasets):
        """
          input:
            net: (object) generator/discriminator model & optimizer
            datasets : (list)['train', 'val'] dataset object
        """
        args = self.args
        net.generator.train()
        net.discriminator.train()
        steps = len(datasets[0]) // args.batch_size
        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        train_loss, train_acc = 0.0, 0.0
        patch_sizes = [args.batch_size, 1, (datasets[0]).img_rows // (
            2**args.patch_layers), (datasets[0]).img_cols // (2**args.patch_layers)]
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True)
            batch_iterator = iter(data_loader)
            for step in range(steps):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # prepare training Variable
                x, y = next(batch_iterator)
                x = Variable(x)
                y = Variable(y)
                posi_label = Variable(torch.ones(
                    (*patch_sizes)), requires_grad=False)
                nega_label = Variable(torch.zeros(
                    (*patch_sizes)), requires_grad=False)
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                    posi_label = posi_label.cuda()
                    nega_label = nega_label.cuda()

                gen_y = net.generator(x)
                if self.is_multi:
                    gen_y = gen_y[0]

                ############################
                # Update Discriminator network: \
                # maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                net.discriminator.zero_grad()

                # train with fake
                fake_pair = torch.cat([x, gen_y.detach()], 1)
                gen_logit_y = net.discriminator(fake_pair)
                d_gen_patch_error = F.mse_loss(gen_logit_y, nega_label)

                # train with real
                real_pair = torch.cat([x, y], 1)
                real_logit_y = net.discriminator(real_pair)
                d_real_patch_error = F.mse_loss(real_logit_y, posi_label)

                if self.iter % 10 == 0:
                    # combined error & update parameters
                    d_patch_error = (d_gen_patch_error +
                                     d_real_patch_error) * 0.5
                    d_patch_error.backward()
                    net.d_optimizer.step()

                ############################
                # Update Generator network: \
                # maximize log(D(G(z)))
                ###########################

                net.generator.zero_grad()

                # G(A) should fake the discriminator
                fake_pair = torch.cat([x, gen_y], 1)
                fake_logit_y = net.discriminator(fake_pair)
                gan_error = F.mse_loss(fake_logit_y, posi_label)

                # G(A) should be consistent with B
                cls_error = F.binary_cross_entropy(gen_y, y)

                # # combined error & update paramenters
                g_error = cls_error + gan_error
                g_error.backward()
                # update paramenters
                net.g_optimizer.step()

                print("\t Discriminator (Gen err : {:0.3f} ; Real err : {:0.3f} );  Generator (Cls err : {:0.3f} ; GAN err : {:0.3f} ); ".format(
                    d_gen_patch_error.data[0], d_real_patch_error.data[0], cls_error.data[0], gan_error.data[0]))

                # compute generator accuracy
                # gan_acc = metrics.overall_accuracy(
                #     fake_logit_y.data, posi_label.data)
                train_acc += metrics.overall_accuracy(gen_y.data, y.data)
                train_loss += g_error.data[0]

                # logging
                if self.iter % args.iter_interval == 0:
                    _time = time.time() - start
                    nb_samples = args.iter_interval * args.batch_size
                    train_log = [train_loss / args.iter_interval, train_acc /
                                 args.iter_interval, _time, nb_samples / _time]
                    self.train_log = [round(x, 3) for x in train_log]
                    self.validating(net.generator, datasets[1])
                    self.logging(verbose=True)
                    # reinitialize
                    start = time.time()
                    train_loss, train_acc = 0, 0


if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    print("Hello")
