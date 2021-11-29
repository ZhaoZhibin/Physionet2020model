#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings

import torch
from torch import nn
from torch import optim
import numpy as np

import models
import datasets
from utils.save import Save_Tool
from utils.freeze import set_freeze_by_id
from utils.metrics import *
from loss.focal_loss import binary_focal_loss
from loss.weight_BCE_loss import WeightedMultilabel

class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir


    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # Load the datasets
        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}
        self.datasets['train'], self.datasets['val'] = Dataset(args.data_dir, args.split).data_preprare()
        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False),
                                                           drop_last=True)
                            for x in ['train', 'val']}

        # Define the model
        self.num_classes = Dataset.num_classes
        self.model = getattr(models, args.model_name)(in_channel=Dataset.inputchannel, out_channel=Dataset.num_classes)
        # self.model = getattr(models, args.model_name)()
        # self.model.fc = torch.nn.Linear(self.model.fc.in_features, Dataset.num_classes)
        # parameter_list = self.model.parameter_list(args.lr)

        if args.layer_num_last != 0:
            set_freeze_by_id(self.model, args.layer_num_last)
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'cos':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, steps, 0)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # Define the monitoring accuracy
        if args.monitor_acc == 'acc':
            self.cal_acc = None
        elif args.monitor_acc == 'AUC':
            self.cal_acc = RocAucEvaluation
        elif args.monitor_acc == 'ecgAcc':
            self.cal_acc = cal_Acc
        else:
            raise Exception("monitor_acc is not implement")

        # Load the checkpoint
        self.start_epoch = 0

        self.criterion = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid.to(self.device)
        self.model.to(self.device)




    def train(self):
        """
        Training process
        :return:
        """
        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0

        batch_loss = 0.0
        step_start = time.time()

        for epoch in range(self.start_epoch, args.max_epoch):

            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                #self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))


            # Each epoch has a training and val phase
            for phase in ['train','val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0
                batch_length = 0
                batch_acc = 0

                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'train'):
                        # forward
                        logits = self.model(inputs)
                        logits_prob = self.sigmoid(logits)
                        if batch_idx == 0:
                            labels_all = labels
                            logits_prob_all = logits_prob
                        else:
                            labels_all = torch.cat((labels_all, labels), 0)
                            logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)
                        # print(logits_prob.shape, labels.shape)
                        # auroc, auprc, accuracy, f_measure, Fbeta_measure, Gbeta_measure = self.cal_acc(labels, logits_prob, threshold=0.5, num_classes=self.num_classes)
                        loss = self.criterion(logits_prob, labels)
                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp

                        # Calculate the training information
                        if phase == 'train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            # batch_acc += accuracy
                            batch_count += inputs.size(0)
                            batch_length += 1
                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                # batch_acc = batch_acc / batch_length
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0*batch_count/train_time
                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx*len(inputs), len(self.dataloaders[phase].dataset),
                                    batch_loss, sample_per_sec, batch_time
                                ))
                                # batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1


                challenge_metric = self.cal_acc(labels_all, logits_prob_all, threshold=0.5, num_classes=self.num_classes)

                # Print the train and val information via each epoch
                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                logging.info('Epoch: {} {}-Loss: {:.4f} {}-challenge_metric: {:.4f}, Cost {:.1f} sec'.
                    format(epoch, phase, epoch_loss, phase, challenge_metric, time.time() - epoch_start))
                # This depends on the weights
                epoch_acc = challenge_metric

                # save the model
                if phase == 'val':
                    # save the checkpoint for other learning
                    model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    # save the best model according to the val accuracy
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, CM: {:.4f}".
                                     format(epoch, challenge_metric))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))


            if self.lr_scheduler is not None:
                self.lr_scheduler.step()















