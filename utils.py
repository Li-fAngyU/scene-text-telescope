import numpy as np
import string
import paddle
from math import exp
import paddle
import paddle.nn as nn
from IPython import embed

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 1]

    mse = ((img1[:,:3,:,:]*255 - img2[:,:3,:,:]*255)**2).mean()
    if mse == 0:
        return float('inf')
    return 20 * paddle.log10(255.0 / paddle.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = paddle.to_tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    #window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return _2D_window.expand([channel, 1, window_size, window_size])


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = nn.functional.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = nn.functional.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = nn.functional.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Layer):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        _2D_window = create_window(window_size, self.channel)
        window = self.create_parameter([self.channel, 1, self.window_size, self.window_size], nn.initializer.Assign(_2D_window.numpy()))
        self.add_parameter('window',window)


    def forward(self, img1, img2):
        img1 = img1[:,:3,:,:]
        img2 = img2[:,:3,:,:]
        (_, channel, _, _) = img1.shape
        return _ssim(img1, img2, self.window, self.window_size, channel, self.size_average)

def parse_crnn_data(imgs_input):
    imgs_input = paddle.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
    R = imgs_input[:, 0:1, :, :]
    G = imgs_input[:, 1:2, :, :]
    B = imgs_input[:, 2:3, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B
    return tensor

def get_crnn_pred(outputs):
    alphabet = '-0123456789abcdefghijklmnopqrstuvwxyz'
    predict_result = []
    for output in outputs:
        x_numpy = output.numpy()
        max_index = [np.where(x_numpy == value)[1][0] for value in np.max(x_numpy, 1)]
        out_str = ""
        last = ""
        for i in max_index:
            if alphabet[i] != last:
                if i != 0:
                    out_str += alphabet[i]
                    last = alphabet[i]
                else:
                    last = ""
        predict_result.append(out_str)
    return predict_result

def str_filt(str_, voc_type):
    alpha_dict = {
        'digit': string.digits,
        'lower': string.digits + string.ascii_lowercase,
        'upper': string.digits + string.ascii_letters,
        'all':   string.digits + string.ascii_letters + string.punctuation
    }
    if voc_type == 'lower':
        str_ = str_.lower()
    for char in str_:
        if char not in alpha_dict[voc_type]:
            str_ = str_.replace(char, '')
    return str_

def get_vocabulary(voc_type, EOS='EOS', PADDING='PADDING', UNKNOWN='UNKNOWN'):
    '''
    voc_type: str: one of 'LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'
    '''
    voc = None
    types = ['digit', 'lower', 'upper', 'all']
    if voc_type == 'digit':
        voc = list(string.digits)
    elif voc_type == 'lower':
      voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'upper':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'all':
        voc = list(string.digits + string.ascii_letters + string.punctuation)
    else:
        raise KeyError('voc_type Error')

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    return voc

class AsterInfo(object):
    def __init__(self, voc_type):
        super(AsterInfo, self).__init__()
        self.voc_type = voc_type
        assert voc_type in ['digit', 'lower', 'upper', 'all']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)