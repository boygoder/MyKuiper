# README

C++实现深度学习推理框架。

## 目录

**source**是源码目录。

1. **data/** 是张量类Tensor的实现和Tensor初始化方法。
2. **layer/** 是神经网络层的定义和具体实现。
3. **parser/** 是Pnnx表达式的解析类。
4. **runtime/** 是计算图结构，包含计算图、算子、操作数。

**test**是单元测试目录。

## 使用的技术和开发环境

- 开发语言：C++ 17
- 数学库：Armadillo + OpenBlas
- 加速库：OpenMP
- 单元测试：Google Test
- 日志：Google Glog
- 设计模式：单例模式、工厂模式、策略模式、适配器模式


## 开发进度

### 支持的算子

- Convolution ：卷积层
- AvgPooling ：平均池化层
- MaxPooling ：最大池化层
- Expression：抽象语法树，张量的逐向量相乘、相加。
- Flatten：维度展平
- ReLU：relu激活函数,$relu(x) = max(0,x)$
- Linear：线性层，$y = Ax + b$
- Softmax ：softmax激活函数,$softmax(x_{i}) = \dfrac{e^{x_{i}}}{\sum_{j=1}^{n} e^{x_{j}}}$
- Upsample ：上采样层，将输入放大scale倍，具体实现为最近邻插值。
- SiLU ：silu激活函数，$silu(x) = \dfrac{x}{e^{-x}+1}$
- Concat：张量拼接层

  

### 支持的模型

  支持resnet18和yolov5s模型的推理。

## 性能测试

假定编译出来的程序名test.exe, 比如我们现在想要只跑case2这几个测试，那么一个命令行参数就可以搞定：

./test --gtest_filter=case2.*

### 测试设备

i5-13500hx

### 编译环境

g++ (Ubuntu 13.1.0-8ubuntu1~20.04.2) 13.1.0

### 性能结果

耗时通过连续100次运行,并以求平均的方式计算。

| **input_size**  | **模型名称** | **C++耗时** |
| -------------- | ------------- | ---------- |
| $224\times224$  batch_size=1 | resnet_batch1 | 1.41476 s |
| $600 \times 600$ batch_size=1 | yolov5s |  3.89041s |

