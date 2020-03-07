# 手写数字生成

通过使用深度卷积生成对抗网络，学习从随机噪声中生成手写数字，参考[Tensorflow示例](https://www.tensorflow.org/tutorials/generative/dcgan)。  
所使用的MNIST数据集默认下载在`~/.keras/datasets`，因而训练不需要手动下载任何的数据集，直接运行即可。  


## 生成效果

训练30个Epoch之后：  

![generate_sample](_images/generate_history.gif)

## 文件结构

```shell
.
├── configs
│   ├── default.py
│   └── __init__.py
├── data_loaders
│   ├── __init__.py
│   └── load_mnist.py
├── logs
│   └──...
├── models
│   ├── dcgan_discriminator.py
│   ├── dcgan_generator.py
│   └── __init__.py
├── templates
│   ├── __init__.py
│   ├── data_loader_template.py
│   ├── model_templet.py
│   └── trainer_template.py
├── trainers
│   ├── from_random_noise.py
│   └── __init__.py
├── README.md
└── train.py
```