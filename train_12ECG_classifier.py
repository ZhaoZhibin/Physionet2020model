#!/usr/bin/env python
import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_utils_clip_ag import train_utils
import torch
import shutil

from stratification_split_single import read_and_split_data, prepare_datacsv

def train_12ECG_classifier(input_directory, output_directory):
    model_list = ['./load_model/48-0.6740-split0.pth',
                  './load_model/42-0.6701-split1.pth',
                  './load_model/40-0.6777-split2.pth',
                  './load_model/42-0.6749-split3.pth',
                  './load_model/47-0.6791-split4.pth']
    for i in range(5):
        shutil.copy(model_list[i], output_directory)
    split_list = ['0', '1', '2', '3', '4']

    # read and split the data
    prepare_datacsv(input_directory)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    save_dir = output_directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(5):
        args = parse_args()
        args.data_dir = input_directory
        args.split = split_list[i]
        shutil.copy(model_list[i], output_directory)
        args.load_model = model_list[i]
        # set the logger
        setlogger(os.path.join(save_dir, 'train'+args.split+'.log'))

        # save the args
        for k, v in args.__dict__.items():
            logging.info("{}: {}".format(k, v))

        trainer = train_utils(args, save_dir)
        trainer.setup()
        trainer.train()




def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--model_name', type=str, default='seresnet18_1d_ag', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='ECGag', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default='./', help='the directory of the data')
    parser.add_argument('--split', type=str, default='0', help='The number of split')
    parser.add_argument('--monitor_acc', type=str, default='ecgAcc', help='the directory of the data')
#    parser.add_argument('--label_dir', type=str, default='/Users/michael', help='the directory of the labels')
    #parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_seresnet18_clip_1d_split0',
                        help='the directory to save the model')
#    parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    # optimization information
    parser.add_argument('--layer_num_last', type=int, default=0, help='the number of last layers which unfreeze')
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.0003, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix', 'cos'], default='step',
                        help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='2', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--load_model', type=str, default='./load_model/48-0.6294-best_model.pth', help='the directory of the resume training model')
    parser.add_argument('--max_epoch', type=int, default=4, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=200, help='the interval of log training information')

    args, _ = parser.parse_known_args()
    return args

