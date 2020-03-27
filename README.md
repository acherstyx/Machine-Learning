# 机器学习

一些机器学习算法的代码实现，如果没有特殊说明，均为采用`TensorFlow v2`编写。  

## 项目列表

- ### Image classification | 图像分类  

  - #### [MNIST digits classification](https://github.com/AcherStyx/Machine-Learning/tree/master/Image%20classification/MNIST%20digits%20classification)

    MNIST数字识别，属于早期项目，采用`TensorFlow v1`进行实现，模型采用卷积神经网络。  
  
  - #### [MNIST digits classification with RNN](https://github.com/AcherStyx/Machine-Learning/tree/master/Image%20classification/MNIST%20digits%20classification%20with%20RNN)

    通过RNN来实现的MNIST数字识别，拥有同样高的准确率。  

  - #### [CIFAR-10 image classification](https://github.com/AcherStyx/Machine-Learning/tree/master/Image%20classification/CIFAR-10%20image%20classification)

    采用`TensorFlow v1`，用于测试ResNet和调参的影响。  
    在训练过程中不断输出了每一层网络的梯度最大值，因而会产生较大的日志文件，注意空间占用。  

  - #### [Poker recognition](https://github.com/AcherStyx/Machine-Learning/tree/master/Image%20classification/Poker%20recognition)

    采用`TensorFlow v1`，实现了对于扑克牌花色、数字的识别。  
    模型主要通过迁移学习（Inception V3）来实现快速训练，主要思路在于利用数据增广来增加泛化性能，数据集只需要提供每一张牌对应一张图片即可。  
    注意模型设计并不合理，对于花色和数字的识别分开实现、运行了两个模型，实际上可以进行合并。但运行预测时，通过GPU加速仍然可以实时处理视频流。

- ### Object detection | 目标检测  

  - #### [Face detection with OpenCV](https://github.com/AcherStyx/Machine-Learning/tree/master/Object%20detection/Face%20detection%20with%20OpenCV)

    通过`OpenCV`的`Haar Cascade`实现，最大的特点就是快，缺点是不是很准。  
    替换`.xml`格式的配置文件之后可以进行其他内容的检测，很方便。  

  - #### [YOLO V1](https://github.com/AcherStyx/Machine-Learning/tree/master/Object%20detection/YOLO%20V1)

    基于`TensorFlow v2`的YOLO V1实现。  

- ### Text | 文本  

  - #### [Text generation](https://github.com/AcherStyx/Machine-Learning/tree/master/Text/Text%20generation)

    基于RNN的文本生成，用于对RNN文本学习能力的简单测试。可以基本实现对预料的词汇的学习，但是生成的文本缺乏语义，难以体会出其含义。  
    只使用了最为简单的模型，但是在`model`中可以扩充、替换更强大的模型。  

- ### Generation | 生成  

  - #### [Handwritten digit generation with DCGAN](https://github.com/AcherStyx/Machine-Learning/tree/master/Generation/Handwritten%20digit%20generation%20with%20DCGAN)

    使用DCGAN来以随机噪声为输入，生成手写数字。  
    新增以指定的数字+随机噪声为随机扰动，生成所指定的数字图像。

  - #### [Pix2Pix common framwork](https://github.com/AcherStyx/Machine-Learning/tree/master/Generation/Pix2Pix%20common%20framework)

    Pix2Pix的实现，算法本身利用了GAN的思想，旨在构建图像统一的端到端训练模型。  

## 目录约定

非特殊情况下（历史原因、特殊文件），各个项目内的文件按如下方式组织：

```shell
Project_root.
├── .gitignore  # 如果有额外的忽略文件，添加在这里
├── Train.py    # 主要功能的调用
├── README.md   # 当前项目的说明
├── ...
├── logs
│   └── ...     # 日志文件
├── templates
│   └── ...     # 模板
├── configs
│   └── ...     # 设置
├── data_loaders
│   └── ...     # 数据集加载
├── models
│   └── ...     # 模型
├── trainers
│   └── ...     # 训练
├── utils
│   └── ...     # 功能性代码
└── ...
```

`data_loader`、`model`和`trainer`均采用Python类构建，继承`templates`下的统一模板。  
`config`定义在每一个`data_loader`、`model`等的同文件下，在`config`中创建实例并赋值各个设置参数。  
