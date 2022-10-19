from datetime import datetime
import paddle 
import paddle.nn as nn
from utils import str_filt, parse_crnn_data, get_crnn_pred
print('test trainer!!! -- 1')

def eval(model, val_loader, image_crit, index, recognizer, aster_info, mode):
    global easy_test_times
    global medium_test_times
    global hard_test_times

    model.eval()
    recognizer.eval()  # key
    n_correct = 0
    n_correct_lr = 0
    sum_images = 0
    metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0,
                    'images_and_labels': []}
    image_start_index = 0
    for i, data in (enumerate(val_loader)):
        images_hr, images_lr, label_strs = data
        val_batch_size = images_lr.shape[0]
        images_sr = model(images_lr)

        recognizer_dict_sr = parse_crnn_data(images_sr[:, :3, :, :])
        recognizer_output_sr = recognizer(recognizer_dict_sr)
        outputs_sr = recognizer_output_sr.transpose([1, 0, 2])
        predict_result_sr = get_crnn_pred(outputs_sr)

        cnt = 0
        for pred, target in zip(predict_result_sr, label_strs):
            if pred == str_filt(target, 'lower'):
                n_correct += 1
            cnt += 1

        sum_images += val_batch_size
        paddle.device.cuda.empty_cache()
    #logging.info('save display images')
    print('save display images')
    accuracy = round(n_correct / sum_images, 4)
    #logging.info('sr_accuray: %.2f%%' % (accuracy * 100))
    print('sr_accuray: %.2f%%' % (accuracy * 100))
    metric_dict['accuracy'] = accuracy
    
    
    model.train()
    
    return metric_dict

def train(model, config, aster, train_loader, val_loader_list):
    model = paddle.DataParallel(model)

    best_history_acc = dict(
        zip([val_loader_dir.split('/')[-1] for val_loader_dir in config.TRAIN.VAL.val_data_dir],
            [0] * len(val_loader_list)))
    best_model_acc = copy.deepcopy(best_history_acc)
    best_model_psnr = copy.deepcopy(best_history_acc)
    best_model_ssim = copy.deepcopy(best_history_acc)
    best_acc = 0
    converge_list = []

    cfg = config.TRAIN
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=0.25)
    optim = paddle.optimizer.Adam(learning_rate=cfg.lr, beta1=cfg.beta1, beta2=0.999, epsilon=1e-08, parameters=model.parameters(), grad_clip=clip)

    times = 0
    easy_test_times = 0
    medium_test_times = 0
    hard_test_times = 0
    for epoch in range(cfg.epochs):
        for j, data in enumerate(train_loader):
            model.train()
            iters = len(train_loader) * epoch + j
            images_hr, images_lr, label_strs = data
            images_lr.stop_gradient = False
            sr_img = model(images_lr)
            loss, mse_loss, attention_loss, recognition_loss = image_crit(sr_img, images_hr, label_strs)

            #global times

            times += 1
            loss_im = loss * 100
            loss_im.backward()

            optim.step()
            optim.clear_grad()
            
            if iters % cfg.displayInterval == 0:
                #logging.info('[{}]\t'
                print('[{}]\t'
                        'Epoch: [{}][{}/{}]\t' 
                        'total_loss {:.3f} \t'
                        'mse_loss {:.3f} \t'
                        'attention_loss {:.3f} \t'
                        'recognition_loss {:.3f} \t'
                        .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                epoch, j + 1, len(train_loader),
                                float(loss_im.numpy()),
                                mse_loss.numpy()[0],
                                attention_loss.numpy()[0],
                                recognition_loss.numpy()[0]
                                ))
                                
        #logging.info('======================================================')
        print('======================================================')
        current_acc_dict = {}
        for k, val_loader in enumerate(val_loader_list):
            data_name = config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
            #logging.info('evaling %s' % data_name)
            print('evaling %s' % data_name)
            metrics_dict = eval(model, val_loader, image_crit, iters, aster, aster_info, data_name)
            converge_list.append({'iterator': iters,
                                    'acc': metrics_dict['accuracy'],
                                    'psnr': metrics_dict['psnr_avg'],
                                    'ssim': metrics_dict['ssim_avg']})
            acc = metrics_dict['accuracy']
            current_acc_dict[data_name] = float(acc)
            if acc > best_history_acc[data_name]:

                data_for_evaluation = metrics_dict['images_and_labels']

                best_history_acc[data_name] = float(acc)
                best_history_acc['epoch'] = epoch
                #logging.info('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))
                print('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))

            else:
                #logging.info('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))
                print('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))
        if sum(current_acc_dict.values()) > best_acc:
            best_acc = sum(current_acc_dict.values())
            best_model_acc = current_acc_dict
            best_model_acc['epoch'] = epoch
            best_model_psnr[data_name] = metrics_dict['psnr_avg']
            best_model_ssim[data_name] = metrics_dict['ssim_avg']
            best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
            #logging.info('saving best model')
            paddle.save(model.state_dict(), 'best_model.pdparams')
            print('saving best model!!!')

        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
        paddle.save(model.state_dict(), 'tmp_model.pdparams')
        print("model save!!!")

class tmpargs():
    def __init__(self) -> None:
        super().__init__()
        self.arch = 'tbsrn'
        self.exp_name = 'EXP_NAME'
        self.test = False
        self.text_focus = True
        self.test_data_dir = 'dataset/mydata/test'
        self.batch_size = 16
        self.resume = ''
        self.rec = 'crnn'
        self.STN = True
        self.syn = False
        self.mixed = False
        self.mask = False
        self.hd_u = 32
        self.srb = 5
        self.demo = False
        self.demo_dir = 'demo/'

import os
import yaml
from easydict import EasyDict
import paddle.distributed as dist


if __name__ == '__main__':
    print('mission start!!!')
    dist.init_parallel_env()
    args = tmpargs()
    config_path = 'super_resolution.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)
    from get_data import get_test_data, get_train_data, get_val_data
    train_dataset, train_loader = get_train_data(config)
    val_dataset_list, val_loader_list = get_val_data(config)
    from tbsrn import TBSRN
    import copy
    from text_focus_loss import TextFocusLoss
    model = TBSRN()
    #model = paddle.DataParallel(model)
    #model.load_dict(paddle.load('tmp_model1.pdparams'))
    image_crit = TextFocusLoss(args)
    print('model load success!!!')

    from utils import AsterInfo
    aster_info = AsterInfo(config.TRAIN.voc_type)

    from crnn import CRNN
    aster = CRNN(32, 1, 37, 256)
    aster.load_dict(paddle.load('data/data165753/crnn.pdparams'))
    # for p in aster.parameters():
    #     p.stop_gradient = True
    # aster.train()
    train(model, config, aster, train_loader, val_loader_list)








