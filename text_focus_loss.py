# Textfocusloss
import paddle
import string
import numpy as np
import paddle.nn as nn
from transformer import Transformer
import pickle as pkl

standard_alphebet = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
standard_dict = {}
for index in range(len(standard_alphebet)):
    standard_dict[standard_alphebet[index]] = index

def load_confuse_matrix():
    f = open('./dataset/mydata/confuse.pkl', 'rb')
    data = pkl.load(f)
    f.close()
    number = data[:10]
    upper = data[10:36]
    lower = data[36:]
    end = np.ones((1,62))
    pad = np.ones((63,1))
    rearrange_data = np.concatenate((end, number, lower, upper), axis=0)
    rearrange_data = np.concatenate((pad, rearrange_data), axis=1)
    rearrange_data = 1 / rearrange_data
    rearrange_data[rearrange_data==np.inf] = 1
    rearrange_data = paddle.to_tensor(rearrange_data)

    lower_alpha = 'abcdefghijklmnopqrstuvwxyz'
    # upper_alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(63):
        for j in range(63):
            if i != j and standard_alphebet[j] in lower_alpha:
                rearrange_data[i][j] = max(rearrange_data[i][j], rearrange_data[i][j+26])
    rearrange_data = rearrange_data[:37,:37]

    return rearrange_data

weight_table = load_confuse_matrix()
def weight_cross_entropy(pred, gt):
    global weight_table
    batch = gt.shape[0]
    weight = weight_table[gt]
    pred_exp = paddle.exp(pred)
    pred_exp_weight = weight * pred_exp
    loss = 0
    for i in range(len(gt)):
        loss -= paddle.log(pred_exp_weight[i][gt[i]] / paddle.sum(pred_exp_weight, 1)[i])
    return loss / batch

# ----------------------------------------------

def to_gray_tensor(tensor):
    R = tensor[:, 0:1, :, :]
    G = tensor[:, 1:2, :, :]
    B = tensor[:, 2:3, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B
    return tensor

def str_filt(str_, voc_type):
    alpha_dict = {
        'digit': string.digits,
        'lower': string.digits + string.ascii_lowercase,
        'upper': string.digits + string.ascii_letters,
        'all': string.digits + string.ascii_letters + string.punctuation
    }
    if voc_type == 'lower':
        str_ = str_.lower()
    for char in str_:
        if char not in alpha_dict[voc_type]:
            str_ = str_.replace(char, '')
    str_ = str_.lower()
    return str_

class TextFocusLoss(nn.Layer):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.english_alphabet = '-0123456789abcdefghijklmnopqrstuvwxyz'
        self.english_dict = {}
        for index in range(len(self.english_alphabet)):
            self.english_dict[self.english_alphabet[index]] = index

        self.build_up_transformer()

    def build_up_transformer(self):
        transformer = Transformer()
        transformer = paddle.DataParallel(transformer)
        transformer.load_dict(paddle.load('data/data165753/transformer.pdparams'))
        # for p in transformer.parameters():
        #     p.stop_gradient = True
        # transformer.train()
        transformer.eval()
        self.transformer = transformer

    def label_encoder(self, label):
        batch = len(label)

        length = [len(i) for i in label]
        length_tensor = paddle.to_tensor(length,dtype=paddle.int64)

        max_length = max(length)
        input_tensor = np.zeros((batch, max_length))
        for i in range(batch):
            for j in range(length[i] - 1):
                input_tensor[i][j + 1] = self.english_dict[label[i][j]]

        text_gt = []
        for i in label:
            for j in i:
                text_gt.append(self.english_dict[j])
        text_gt = paddle.to_tensor(text_gt, dtype=paddle.int64)

        input_tensor = paddle.to_tensor(input_tensor, dtype=paddle.int64)
        return length_tensor, input_tensor, text_gt


    def forward(self,sr_img, hr_img, label):

        mse_loss = self.mse_loss(sr_img, hr_img)

        if self.args.text_focus:
            label = [str_filt(i, 'lower')+'-' for i in label]
            length_tensor, input_tensor, text_gt = self.label_encoder(label)
            hr_pred, word_attention_map_gt, hr_correct_list = self.transformer(to_gray_tensor(hr_img), length_tensor,
                                                                          input_tensor, test=False)
            sr_pred, word_attention_map_pred, sr_correct_list = self.transformer(to_gray_tensor(sr_img), length_tensor,
                                                                            input_tensor, test=False)
            attention_loss = self.l1_loss(word_attention_map_gt, word_attention_map_pred)
            # recognition_loss = self.l1_loss(hr_pred, sr_pred)
            recognition_loss = weight_cross_entropy(sr_pred, text_gt)
            loss = mse_loss + attention_loss * 10 + recognition_loss * 0.001
            return loss, mse_loss, attention_loss, recognition_loss
        else:
            attention_loss = -1
            recognition_loss = -1
            loss = mse_loss
            return loss, mse_loss, attention_loss, recognition_loss