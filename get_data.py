# dataset file Done!
import paddle
import numpy as np
from PIL import Image
from paddle.io import Dataset
import lmdb
import string
import six


class resizeNormalize(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.mask = mask
        self.totensor = paddle.vision.transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img_tensor = self.totensor(img)
        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.totensor(mask)
            #img = np.concatenate((img, mask), 0)
            img_tensor = paddle.concat((img_tensor, mask), 0)
        return img_tensor.numpy()

class alignCollate_real(object):
    def __init__(self, imgH=64, imgW=256, down_sample_scale=4, keep_ratio=False, min_ratio=1, mask=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask

    def __call__(self, batch):
        images_HR, images_lr, label_strs = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH), self.mask)
        transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
        images_HR = [transform(image) for image in images_HR]
        images_HR = np.concatenate([np.expand_dims(t,0) for t in images_HR], 0)

        images_lr = [transform2(image) for image in images_lr]
        images_lr = np.concatenate([np.expand_dims(t,0) for t in images_lr], 0)
        #images_HR = np.transpose(images_HR, (0, 3,1 ,2))
        #images_lr = np.transpose(images_lr, (0, 3,1 ,2))
        return images_HR.astype(np.float32), images_lr.astype(np.float32), label_strs

def buf2PIL(txn, key, type='RGB'):
    imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf).convert(type)
    return im

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

class lmdbDataset_real_val(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super().__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,)
        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)
            
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        try:
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            img_lr = buf2PIL(txn, img_lr_key, 'RGB')
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)
        return img_HR, img_lr, label_str

class lmdbDataset_real_train(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super().__init__()
        self.env = lmdb.open(
            root[0],
            max_readers=1,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            map_size=int(1e9))
        env1 = lmdb.open(
            root[1],
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)
            
        self.txn = self.env.begin(write = True)
        txn1 = env1.begin(write = False)
        size = int(self.txn.get(b'num-samples'))
        for key,value in txn1.cursor():
            if key[:8] == b'image_hr':
                tmp_key = b'image_hr-%09d' % (int(key[9:])+size)
            elif key[:8] == b'image_lr':
                tmp_key = b'image_lr-%09d' % (int(key[9:])+size)
            elif key[:5] == b'label':
                tmp_key = b'label-%09d' % (int(key[9:])+size)
            elif key == b'num-samples':
                tmp_key = key
                value = b'%d' %(size + int(value))
            self.txn.put(tmp_key,value)
        self.nSamples = int(self.txn.get(b'num-samples'))
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        label_key = b'label-%09d' % index
        word = str(self.txn.get(label_key).decode())
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        try:
            img_HR = buf2PIL(self.txn, img_HR_key, 'RGB')
            img_lr = buf2PIL(self.txn, img_lr_key, 'RGB')
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)
        return img_HR, img_lr, label_str

# Done
def get_train_data(config):
    cfg = config.TRAIN
    if isinstance(cfg.train_data_dir, list):
        train_dataset = lmdbDataset_real_train(root=cfg.train_data_dir, voc_type=cfg.voc_type, max_len=cfg.max_len)
    else:
        raise TypeError('check trainRoot')

    train_loader = paddle.io.DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=int(cfg.workers),
        collate_fn=alignCollate_real(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                        mask=False),
        drop_last=True)
    return train_dataset, train_loader

def get_train_data_distribution(config):
    cfg = config.TRAIN
    if isinstance(cfg.train_data_dir, list):
        train_dataset = lmdbDataset_real_train(root=cfg.train_data_dir, voc_type=cfg.voc_type, max_len=cfg.max_len)
    else:
        raise TypeError('check trainRoot')

    train_loader = paddle.io.DistributedBatchSampler(
            dataset=train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=alignCollate_real(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                        mask=False),
            num_workers = int(cfg.workers)
        )
    return train_dataset, train_loader

# Done
def get_val_data(config):
    cfg = config.TRAIN
    assert isinstance(cfg.VAL.val_data_dir, list)
    dataset_list = []
    loader_list = []
    for data_dir_ in cfg.VAL.val_data_dir:
        val_dataset, val_loader = get_test_data(config, data_dir_)
        dataset_list.append(val_dataset)
        loader_list.append(val_loader)
    return dataset_list, loader_list

# Done
def get_test_data(config, dir_):
    cfg = config.TRAIN
    test_dataset = lmdbDataset_real_val(root=dir_,
                                        voc_type=cfg.voc_type,
                                        max_len=cfg.max_len,
                                        test=True,
                                        )
    test_loader = paddle.io.DataLoader(
        test_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=int(cfg.workers),
        collate_fn=alignCollate_real(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                        mask=False),
        drop_last=False)
    return test_dataset, test_loader