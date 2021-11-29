
import numpy as np
import random
from scipy.signal import resample


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Reshape(object):
    def __call__(self, seq):
        #print(seq.shape)
        return seq.transpose()


class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)


class AddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class RandomAddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class Scale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
        scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
        return seq*scale_matrix


class RandomScale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2): #np.random.randint(2):
            return seq
        else:
            scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
            scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
            return seq*scale_matrix




def amplify(x):
    """
    # 随机变幅值
    :param x: 二维数组， 序列长度*通道数
    :return: 增强样本
    """
    alpha = (random.random()-0.5)
    factor = -alpha*x + (1+alpha)
    return x*factor

class RandomAmplify(object):

    def __call__(self, seq):
        if np.random.randint(2): #np.random.randint(2):
            return seq
        else:
            return amplify(seq)




class RandomStretch(object):
    def __init__(self, sigma=0.3):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2): # np.random.randint(2):
            return seq
        else:
            seq_aug = np.zeros(seq.shape)
            len = seq.shape[1]
            length = int(len * (1 + (random.random()-0.5)*self.sigma))
            for i in range(seq.shape[0]):
                y = resample(seq[i, :], length)
                if length < len:
                    seq_aug[i, :length] = y
                else:
                    seq_aug[i, :] = y[:len]

            return seq_aug


class RandomCrop(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            max_index = seq.shape[1] - self.crop_len
            random_index = np.random.randint(max_index)
            seq[:, random_index:random_index+self.crop_len] = 0
            return seq

class Normalize(object):
    def __init__(self, type="0-1"):
        self.type = type

    def __call__(self, seq):
        if self.type == "0-1":
            for i in range(seq.shape[0]):
                if np.sum(seq[i, :]) == 0:
                    seq[i, :] = seq[i, :]
                else:
                    seq[i, :] = (seq[i, :]-seq[i, :].min())/(seq[i, :].max()-seq[i, :].min())
        elif self.type == "mean-std":
            for i in range(seq.shape[0]):
                if np.sum(seq[i, :]) == 0:
                    seq[i, :] = seq[i, :]
                else:
                    seq[i, :] = (seq[i, :]-seq[i, :].mean())/seq[i, :].std()
        elif self.type == "none":
            seq = seq
        else:
            raise NameError('This normalization is not included!')
        return seq


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



from scipy import signal

"""
def Resample(sig, target_point_num=None):
    '''
    对原始信号进行重采样
    :param sig: 原始信号
    :param target_point_num:目标型号点数
    :return: 重采样的信号
    '''
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig
"""


class DownSample(object):
    def __init__(self, src, tar):
        self.src = src
        self.tar = tar
    def __call__(self, seq):

        return Resample(seq, self.src, self.tar)


class RandomStart(object):
    def __init__(self, num=2048):
        self.num = num

    def __call__(self, seq):
        start = random.randint(0, seq.shape[1] - self.num)
        return seq[:, start:start+self.num]

class ConstantStart(object):
    def __init__(self, start=0, num=2048):
        self.start = start
        self.num = num

    def __call__(self, seq):

        return seq[:, self.start:self.start+self.num]

class ZerosPadding(object):
    def __init__(self, len=72000):
        self.len = len

    def __call__(self, seq):
        if seq.shape[1] >= self.len:
            seq = seq[:, 0:self.len]
        else:
            zeros_padding = np.zeros(shape=(seq.shape[0], self.len - seq.shape[1]), dtype=np.float32)
            seq = np.hstack((seq, zeros_padding))
        return seq

class RandomClip(object):
    def __init__(self, len=72000):
        self.len = len

    def __call__(self, seq):
        if seq.shape[1] >= self.len:
            start = random.randint(0, seq.shape[1] - self.len)
            seq = seq[:, start:start+self.len]
        else:
            left = random.randint(0, self.len - seq.shape[1])
            right = self.len - seq.shape[1] - left
            zeros_padding1 = np.zeros(shape=(seq.shape[0], left), dtype=np.float32)
            zeros_padding2 = np.zeros(shape=(seq.shape[0], right), dtype=np.float32)
            seq = np.hstack((zeros_padding1, seq, zeros_padding2))
        return seq


class ValClip(object):
    def __init__(self, len=72000):
        self.len = len

    def __call__(self, seq):
        if seq.shape[1] >= self.len:
            seq = seq
        else:
            zeros_padding = np.zeros(shape=(seq.shape[0], self.len - seq.shape[1]), dtype=np.float32)
            seq = np.hstack((seq, zeros_padding))
        return seq


def verflip(sig):
    '''
    信号竖直翻转
    :param sig:
    :return:
    '''
    return sig[:, ::-1]

def shift(sig, interval=20):
    '''
    上下平移
    :param sig:
    :return:
    '''
    for col in range(sig.shape[0]):
        offset = np.random.choice(range(-interval, interval))
        sig[col, :] += offset
    return sig


class Randomverflip(object):
    def __call__(self, seq):
        if np.random.randint(2):#np.random.randint(2):
            return seq
        else:
            return seq[:, ::-1]


class Randomshift(object):
    def __init__(self, interval=20):
        self.interval = interval

    def __call__(self, seq):
        if np.random.randint(2):#np.random.randint(2):
            return seq
        else:
            for col in range(seq.shape[0]):
                offset = np.random.choice(range(-self.interval, self.interval))
                seq[col, :] += offset
            return seq