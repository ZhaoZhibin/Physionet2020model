#!/usr/bin/env python

import numpy as np
import os
import torch
from torch import nn
import warnings
import models
from scipy.signal import resample
import math
import pandas as pd
import shutil
from get_12ECG_features import get_12ECG_features

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if torch.cuda.is_available():
    device = torch.device("cuda")
    device_count = torch.cuda.device_count()
else:
    warnings.warn("gpu is not available")
    device = torch.device("cpu")

def Resample(input_signal, src_fs, tar_fs):
    '''
    :param input_signal:输入信号
    :param src_fs:输入信号采样率
    :param tar_fs:输出信号采样率
    :return:输出信号
    '''
    dtype = input_signal.dtype
    audio_len = input_signal.shape[1]
    audio_time_max = 1.0 * (audio_len) / src_fs
    src_time = 1.0 * np.linspace(0, audio_len, audio_len) / src_fs
    tar_time = 1.0 * np.linspace(0, np.int(audio_time_max * tar_fs), np.int(audio_time_max * tar_fs)) / tar_fs
    for i in range(input_signal.shape[0]):
        if i == 0:
            output_signal = np.interp(tar_time, src_time, input_signal[i, :]).astype(dtype)
            output_signal = output_signal.reshape(1, len(output_signal))
        else:
            tmp = np.interp(tar_time, src_time, input_signal[i, :]).astype(dtype)
            tmp = tmp.reshape(1, len(tmp))
            output_signal = np.vstack((output_signal, tmp))
    return output_signal

def processing_data(data, win_length, src_fs, tar_fs):

    """
    Add any preprocessing at here
    """
    data = Resample(data, src_fs, tar_fs)
    num = data.shape[1]
    if num < win_length:
        zeros_padding = np.zeros(shape=(data.shape[0], win_length - num), dtype=np.float32)
        data = np.hstack((data, zeros_padding))
    data = data.astype(np.float32)
    data = torch.from_numpy(data)
    data = torch.unsqueeze(data, 0)

    return data

def prepare_data(age, gender):
    data = np.zeros(5,)
    if age >= 0:
        data[0] = age / 100
    if 'F' in gender:
        data[2] = 1
        data[4] = 1
    elif gender == 'Unknown':
        data[4] = 0
    elif 'f' in gender:
        data[2] = 1
        data[4] = 1
    else:
        data[3] = 1
        data[4] = 1
    return data
    
def read_ag(header_data):
    for lines in header_data:
        if lines.startswith('#Age'):
            tmp = lines.split(': ')[1].strip()
            if tmp == 'NaN':
                age = -1
            else:
                age = int(tmp)
        if lines.startswith('#Sex'):
            tmp = lines.split(': ')[1].strip()
            if tmp == 'NaN':
                gender = 'Unknown'
            else:
                gender = tmp
    data = prepare_data(age, gender)
    data = torch.from_numpy(data).float()
    data = torch.unsqueeze(data, 0)
    return data

def output_label(logits_prob, threshold, num_classes):
    pred_label = np.zeros(num_classes, dtype=int)
    _, y_pre_label = torch.max(logits_prob, 1)
    y_pre_label = y_pre_label.cpu().detach().numpy()
    pred_label[y_pre_label] = 1

    score_tmp = logits_prob.cpu().detach().numpy()

    y_pre = (score_tmp - threshold) >= 0
    pred_label = pred_label + y_pre
    pred_label[pred_label > 1.1] = 1
    return score_tmp, pred_label

def run_12ECG_classifier(data, header_data, model):
    weight_list = ['./magic_weight0.npz', './magic_weight1.npz', './magic_weight2.npz',
                   './magic_weight3.npz', './magic_weight4.npz']
    num_classes = 24
    tar_fs = 257
    src_fs = int(header_data[0].split(' ')[2].strip())
    ag = read_ag(header_data)
    ag = ag.to(device)

    win_length = 4096


    m = nn.Sigmoid()
    data = processing_data(data, win_length, src_fs, tar_fs)
    inputs = data.to(device)
    # Use your classifier here to obtain a label and score for each class.


    val_length = inputs.shape[2]
    overlap = 256
    patch_number = math.ceil(abs(val_length - win_length) / (win_length - overlap)) + 1
    if patch_number > 1:
        start = int((val_length - win_length) / (patch_number - 1))
    score = 0
    combined_label = 0
    for j in range(len(model)):
        model_one = model[j]
        for i in range(patch_number):
            if i == 0:
                logit = model_one(inputs[:, :, 0: val_length], ag)
                logits_prob = m(logit)
            elif i == patch_number - 1:
                logit = model_one(inputs[:, :, val_length - win_length: val_length], ag)
                logits_prob_tmp = m(logit)
                logits_prob = (logits_prob + logits_prob_tmp) / patch_number
            else:
                logit = model_one(inputs[:, :, i * start:i * start + win_length], ag)
                logits_prob_tmp = m(logit)
                logits_prob = logits_prob + logits_prob_tmp

        # using the threshold to check each model
        A = np.load(weight_list[j])
        threshold = A['arr_0']
        score_tmp, pred_label = output_label(logits_prob, threshold, num_classes)

        # the label
        combined_label = combined_label + pred_label

        # The probability
        score = score + score_tmp

    score = score / len(model)
    combined_label = combined_label / len(model)
    max_index = np.argmax(combined_label, 1)
    combined_label[0, max_index] = 1
    threshold_tmp = 0.5
    combined_label[combined_label >= threshold_tmp] = 1
    combined_label[combined_label < threshold_tmp] = 0


    current_label = np.squeeze(combined_label.astype(np.int))
    current_score = np.squeeze(score)

    # Get the label
    label_file_dir = './utils/dx_mapping_scored.csv'
    label_file = pd.read_csv(label_file_dir)
    equivalent_classes = ['59118001', '63593006', '17338001']
    classes = sorted(list(set([str(name) for name in label_file['SNOMED CT Code']]) - set(equivalent_classes)))

    return current_label, current_score, classes

def load_12ECG_model(model_input):
    # load the model from disk
    # load the model from disk
    model_list = ['./load_model/48-0.6740-split0.pth',
                  './load_model/42-0.6701-split1.pth',
                  './load_model/40-0.6777-split2.pth',
                  './load_model/42-0.6749-split3.pth',
                  './load_model/47-0.6791-split4.pth']
    for i in range(5):
        shutil.copy(model_list[i], model_input)
    model_list = lsdir(rootdir=model_input, suffix=".pth")
    split_list = ['split0', 'split1', 'split2', 'split3', 'split4']
    resumes = []
    for split in split_list:
        sub_list = [i for i in model_list if split in i]
        accuracy = np.array([float(i.split('-')[-2]) for i in sub_list])
        resumes.append(sub_list[int(np.argmax(accuracy))])

    model_all = []

    for resume in resumes:
        model = getattr(models, 'seresnet18_1d_ag')(in_channel=12, out_channel=24)
        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            if device_count > 1:
                model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(resume))
        else:
            model.load_state_dict(torch.load(resume, map_location=device))

        model.to(device)
        model.eval()
        model_all.append(model)

    return model_all


def lsdir(rootdir="", suffix=".png"):
    file_list = []
    assert os.path.exists(rootdir)
    for r, y, names in os.walk(rootdir):
        for name in names:
            if str(name).endswith(suffix):
                file_list.append(os.path.join(r, name))
    return file_list
