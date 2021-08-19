# show-attend-and-tell-paddle

基于[paddle](https://github.com/PaddlePaddle/Paddle)框架的[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)实现

## 一、简介

本项目使用[paddle](https://github.com/PaddlePaddle/Paddle)框架复现[Show, Attend and Tell](https://arxiv.org/abs/1502.03044)模型。该论文首次将注意力机制引入到`image captioning`任务中，使得模型在生成不同单词的过程中，能够关注图像的不同区域，取得了不错的效果。

**注: AI Studio项目地址: [https://aistudio.baidu.com/aistudio/projectdetail/2288384](https://aistudio.baidu.com/aistudio/projectdetail/2288384).**

**您可以使用[AI Studio](https://aistudio.baidu.com/)平台在线运行该项目!**

**论文:**

* [1] K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhutdinov, R. Zemel, Y. Bengio, "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention", ICML, 2015.

**参考项目:**

* [a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) [Pytorch实现]

## 二、复现精度

> 所有指标均为模型在[Flickr8K](https://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b)的测试集评估而得

| 指标 | BlEU-1 | BlEU-2 | BlEU-3 | BlEU-4 |
| :---: | :---: | :---: | :---: | :---: |
| 原论文 | 0.670 | 0.457 | 0.314 | 0.213 |
| 复现精度 | 0.677 | 0.494 | 0.350 | 0.243 |

## 三、数据集

本项目所使用的数据集为[Flickr8K](https://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b)。该数据集共包含8000张图像，每张图像对应5个标题。训练集、验证集和测试集分别为6000、1000、1000张图像及其对应的标题（我们提供了脚本下载该数据集的标题以及图像特征，见[download_dataset.sh](https://github.com/fuqianya/show-attend-and-tell-paddle/blob/main/download_dataset.sh)）。

## 四、环境依赖

* 硬件：CPU、GPU

* 软件：
    * Python 3.8
    * Java 1.8.0
    * PaddlePaddle == 2.1.0

## 五、快速开始

### step1: clone 

```bash
# clone this repo
git clone https://github.com/fuqianya/show-attend-and-tell-paddle.git
cd show-attend-and-tell-paddle
```

### step2: 安装环境及依赖

```bash
pip install -r requirements.txt
```

### step3: 下载数据

```bash
# 下载数据集
bash ./download_dataset.sh
```

### step4: 数据集预处理

```python
python prepro.py
```

### step5: 训练

```bash
python train.py
```

### step6: 测试


```bash
python eval.py --eval_model ./checkpoint/epoch_27.pth
```

### 使用预训练模型进行预测

模型下载: [谷歌云盘](https://drive.google.com/file/d/1LmIlgf3XHuHVEkOTVscxlbXBi-h1YMdy/view?usp=sharing)

将下载的模型权重以及训练信息放到`checkpoint`目录下, 运行`step6`的指令进行测试。

## 六、代码结构与详细说明

```bash
├── checkpoint      　   # 存储训练的模型
├── config
│　 └── config.py        # 模型的参数设置
├── data            　   # 预处理的数据
├── images            　 # 数据集图像
├── model
│   └── encoder.py    　 # 编码器
│   └── decoder.py    　 # 解码器
│   └── dataloader.py  　# 加载训练数据
│   └── loss.py        　# 定义损失函数
├── pyutils 
│   └── cap_eval       　# 计算评价指标工具
├── result            　 # 存放生成的标题
├── utils 
│   └── eval_utils.py  　# 测试工具
├── download_dataset.sh　# 数据集下载脚本
├── prepro.py          　# 数据预处理
├── train.py           　# 训练主函数
├── eval.py            　# 测试主函数
└── requirement.txt   　 # 依赖包
```

模型、训练的所有参数信息都在`config.py`中进行了详细注释，详情见`config/config.py`。

## 七、模型信息

关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| :---: | :---: |
| 发布者 | fuqianya |
| 时间 | 2021.08 |
| 框架版本 | Paddle 2.1.0 |
| 应用场景 | 多模态 |
| 支持硬件 | GPU、CPU |
| 下载链接 | [预训练模型](https://drive.google.com/file/d/1LmIlgf3XHuHVEkOTVscxlbXBi-h1YMdy/view?usp=sharing) \| [训练日志](https://drive.google.com/file/d/1mHRINgJG6hzxNinUhDUMaJMBvSXlpw8i/view?usp=sharing)  |