#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np

def Resample(input_signal, src_fs, tar_fs):
    '''
    :param input_signal:输入信号
    :param src_fs:输入信号采样率
    :param tar_fs:输出信号采样率
    :return:输出信号
    '''
    if src_fs != tar_fs:
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
    else:
        output_signal = input_signal
    return output_signal

def load_data(case, src_fs, tar_fs=257):
    x = loadmat(case)
    data = np.asarray(x['val'], dtype=np.float64)
    data = Resample(data, src_fs, tar_fs)
    return data

class dataset(Dataset):

    def __init__(self, anno_pd, test=False, transform=None, data_dir=None, loader=load_data):
        self.test = test
        if self.test:
            self.data = anno_pd['filename'].tolist()
            self.fs = anno_pd['fs'].tolist()
        else:
            self.data = anno_pd['filename'].tolist()
            labels = anno_pd.iloc[:, 4:].values
            self.multi_labels = [labels[i, :] for i in range(labels.shape[0])]
            self.fs = anno_pd['fs'].tolist()

        self.transforms = transform
        self.data_dir = data_dir
        self.loader = loader


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.test:
            img_path = self.data[item]
            fs = self.fs[item]
            img = self.loader(self.data_dir + img_path, src_fs=fs)
            img = self.transforms(img)
            return img, img_path
        else:
            img_name = self.data[item]
            fs = self.fs[item]
            img = self.loader(img_name, src_fs=fs)
            label = self.multi_labels[item]
            """
            for i in range(img.shape[1]):
                img[:, i] = ecg_preprocessing(img[:, i], wfun='db6', levels=9, type=2)
            """
            img = self.transforms(img)
            return img, torch.from_numpy(label).float()

if __name__ == '__main__':
    """
    img = cv2.imread('../ODIR-5K_training/0_left.jpg')
    #cv2.flip(img, 1, dst=None)
    cv2.namedWindow("resized", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("resized", 640, 480)
    cv2.imshow('resized', img)
    cv2.waitKey(5)
    # cv2.destoryAllWindows()
    """
