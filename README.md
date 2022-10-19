# Paddle-TBSRN

## 目录

- [1. 简介]()
- [2. 数据集准备]()
- [3. 复现精度]()
- [4. 模型目录与环境]()
    - [4.1 目录介绍]()
    - [4.2 准备环境]()
- [5. 开始使用]()
    - [5.1 模型训练]()

## 1. 简介
**论文:** [Scene Text Telescope: Text-Focused Scene Image Super-Resolution](https://ieeexplore.ieee.org/document/9578891/)

该文提出了一个聚焦文本的超分辨率框架，称为场景文本Telescope(STT)。在文本级布局方面，本文提出了一个基于Transformer的超分辨网络(TBSRN)，包含一个自注意模块来提取序列信息，对任意方向的文本具有鲁棒性。在字符级的细节方面，本文提出了一个位置感知模块和一个内容感知模块来突出每个字符的位置和内容。通过观察一些字符在低分辨率条件下看起来难以区分，本文使用加权交叉熵损失解决。

**官方repo:** [scene-text-telescope](https://github.com/FudanVI/FudanOCR/tree/main/scene-text-telescope)


## 2. 数据集准备

TextZoom中的数据集来自两个超分数据集RealSR和SR-RAW，两个数据集都包含LR-HR对，TextZoom有17367对训数据和4373对测试数据，

[官方下载地址](https://pan.baidu.com/share/init?surl=P_SCcQG74fiQfTnfidpHEw)，提取码为stt6。

[AI Studio下载地址](https://aistudio.baidu.com/aistudio/datasetdetail/171370), 注：由[超级码立](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/249413)所提供。

* TextZoom dataset
* Pretrained weights of CRNN 
* Pretrained weights of Transformer-based recognizer

数据集目录结构：
```
mydata
├── train1
├── train2
├── confuse.pkl
├── crnn.pdparams
├── pretrain_transformer.pdparams
└── test
    ├── easy
    ├── medium
    └── hard
```

## 3. 复现精度

|        Methods       	 |         easy      	|   medium      |    hard       |    avg      	|
|:------------------:    |:------------------:	|:---------:	|:------:   	|:---------:	|
|        官方repo         | 	      0.5979        |   0.4507  	|    0.3418   	|    0.4634  	|
|        复现repo         | 	      0.5707        |   0.4422  	|    0.3306   	|    0.4478  	|

## 4. 模型目录与环境

### 4.1 目录介绍

```
    |--dataset                              # 训练和测试数据集
    |--text_focus_loss.py                   # 训练损失 
    |--utils.py                             # 模型工具文件
    |--crnn.py                              # crnn 模型文件
    |--get_data.py                          # 数据集获取文件
    |--tbsrn.py                             # tbsrn 模型文件
    |--transformer.py                       # transfomer 模型文件
    |--super_resolution.yaml                # 训练超参数文件
    |--requirement.txt                      # 相关依赖
    |--inverse_kernel.pkl                   
    |--target_coordinate_repr.pkl
    |--trainer.py                           # 训练代码
    |----README.md                          # 用户手册
```

### 4.2 准备环境

- 框架：
  - PaddlePaddle >= 2.3.1
- 环境配置：使用`pip install -r requirement.txt`安装依赖。
  
## 5. 开始使用
### 5.1 模型训练

`python trainer.py`

部分训练日志如下所示：
```
[2022-10-01 12:15:09]	Epoch: [8][21/1085]	total_loss 5.133 	mse_loss 0.014 	attention_loss 0.004 	recognition_loss 1.339 	
[2022-10-01 12:15:32]	Epoch: [8][71/1085]	total_loss 4.433 	mse_loss 0.008 	attention_loss 0.004 	recognition_loss 1.180 	
[2022-10-01 12:15:55]	Epoch: [8][121/1085]	total_loss 4.160 	mse_loss 0.009 	attention_loss 0.003 	recognition_loss 0.990 	
[2022-10-01 12:16:18]	Epoch: [8][171/1085]	total_loss 5.220 	mse_loss 0.012 	attention_loss 0.004 	recognition_loss 1.722 	
[2022-10-01 12:16:41]	Epoch: [8][221/1085]	total_loss 4.676 	mse_loss 0.009 	attention_loss 0.004 	recognition_loss 1.187 	
[2022-10-01 12:17:03]	Epoch: [8][271/1085]	total_loss 4.330 	mse_loss 0.009 	attention_loss 0.003 	recognition_loss 1.252 	
[2022-10-01 12:17:25]	Epoch: [8][321/1085]	total_loss 4.445 	mse_loss 0.011 	attention_loss 0.003 	recognition_loss 0.939 	
[2022-10-01 12:17:49]	Epoch: [8][371/1085]	total_loss 4.104 	mse_loss 0.009 	attention_loss 0.003 	recognition_loss 0.887 	
[2022-10-01 12:18:12]	Epoch: [8][421/1085]	total_loss 3.812 	mse_loss 0.006 	attention_loss 0.003 	recognition_loss 0.515 	
[2022-10-01 12:18:34]	Epoch: [8][471/1085]	total_loss 4.544 	mse_loss 0.009 	attention_loss 0.004 	recognition_loss 0.869 	
[2022-10-01 12:18:57]	Epoch: [8][521/1085]	total_loss 4.209 	mse_loss 0.008 	attention_loss 0.003 	recognition_loss 1.261 	
[2022-10-01 12:19:19]	Epoch: [8][571/1085]	total_loss 4.074 	mse_loss 0.008 	attention_loss 0.003 	recognition_loss 0.602 
```

# 注：该使用手册参考[超级码立](https://github.com/Lieberk/Paddle-TextSR-STT/blob/0802e7acefd1018c7fcf814d7f7877549eedb20c/README.md)进行修改。