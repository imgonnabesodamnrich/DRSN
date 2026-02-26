# 论文笔记——深度残差收缩网络 (Deep Residual Shrinkage Networks, DRSN)

> **A Deep Learning Framework for Highly Noised Vibration Signals**
>
> **Paper Title:** *Deep Residual Shrinkage Networks for Fault Diagnosis*
> 
> **Journal:** *IEEE Transactions on Industrial Informatics*, Vol. 16, No. 7, 2020
> 
> **DOI:**[10.1109/TII.2019.2943898](https://doi.org/10.1109/TII.2019.2943898)
> 

![DOI](https://img.shields.io/badge/DOI-10.1109%2FTII.2019.2943898-blue) ![Status](https://img.shields.io/badge/Status-Published-brightgreen) ![Topic](https://img.shields.io/badge/Topic-Deep_Learning-orange) ![Topic](https://img.shields.io/badge/Topic-Fault_Diagnosis-orange)

---


## 目录

- [1. 背景](#1-背景)
- [2. 核心机制](#2-核心机制)
    - [2.1 软阈值化 (Soft Thresholding)](#21-软阈值化-soft-thresholding)
    - [2.2 自适应阈值计算子网络](#22-自适应阈值计算子网络)
- [3. 网络结构变体](#3-网络结构变体)
    - [3.1 DRSN-CS（通道共享阈值）](#31-drsn-cs通道共享阈值)
    - [3.2 DRSN-CW（通道独立阈值）](#32-drsn-cw通道独立阈值)
- [4. 实验结论](#4-实验结论)
- [5. 文献来源](#5-文献来源)

---


## 1. 背景

工业旋转机械（如齿轮箱、轴承）的故障诊断通常依赖于振动信号分析。实际运行环境中存在大量背景噪声，导致故障初期的微弱信号容易被掩盖。

传统的深度学习模型（如卷积神经网络 CNN、残差网络 ResNet）在处理高噪声信号时，容易将噪声干扰提取为特征，导致诊断准确率下降。深度残差收缩网络（Deep Residual Shrinkage Networks, DRSN）基于这一问题提出。该方法将传统信号处理中的软阈值化（Soft Thresholding）机制作为非线性变换层引入深度神经网络结构中，以消除噪声相关特征，提高模型在强噪声环境下的特征学习能力。

<div align="center">
  <img width="70%" src="https://github.com/user-attachments/assets/1ede7ae0-3219-4413-bc85-8d1e5e84f966" />
  <p><em>图1 实验所用的动力传动系统故障诊断模拟器 </em></p>
</div>

## 2. 核心机制

DRSN 的核心思想是在 ResNet 的残差模块内部，集成一个能够自动计算阈值并执行特征收缩的子模块。

### 2.1 软阈值化 (Soft Thresholding)

软阈值化是信号降噪领域（如小波去噪）的经典方法。其基本数学逻辑是：
1. 将绝对值小于特定阈值的特征置为零。
2. 将绝对值大于阈值的特征向零的方向进行收缩（即减去设定的阈值）。

通过软阈值化，模型可以将接近于零的噪声特征剔除，同时保留绝对值较大的关键故障特征。由于该操作的导数在非零区域恒为 1，它同样具备防止梯度消失的作用。

<div align="center">
  <img width="80%" src="https://github.com/user-attachments/assets/9f79323d-8344-48bf-aecc-f0e86f57e0e2" />
  <p><em>图2 软阈值化函数及其导数示意图 </em></p>
</div>

### 2.2 自适应阈值计算子网络

传统信号处理方法通常依赖专家经验手动设定阈值，且无法适应神经网络中逐层变化的特征分布。DRSN 设计了一个特定的子网络，通过梯度下降算法自动学习并确定软阈值。

其计算流程如下：
1. **全局特征提取：** 对输入特征图取绝对值，并执行全局平均池化（GAP），将其压缩为一个一维向量。
2. **缩放系数生成：** 将该一维向量输入一个两层的全连接（FC）网络，并通过 Sigmoid 激活函数，输出一个范围在 (0, 1) 之间的缩放系数（Scaling Parameter）。
3. **阈值计算：** 将生成的缩放系数与输入特征的绝对值平均数相乘，得出最终的阈值。

该机制确保了阈值始终为正数，且不会大于特征的最大绝对值。同时，每段输入的振动信号样本均能根据自身的噪声水平，动态生成专属的阈值。

## 3. 网络结构变体

根据特征图通道对阈值的使用策略，论文提出了两种具体的网络架构设计：

### 3.1 DRSN-CS（通道共享阈值）
在 DRSN-CS（Channel-Shared）结构中，特征图的所有通道共用同一个软阈值。这种方式计算开销相对较小，适用于各通道噪声分布相对均匀的场景。

<div align="center">
  <img width="45%" src="https://github.com/user-attachments/assets/e387fe43-00f9-4380-87ed-b56058cf322f" />
  <p><em>图3 具有通道共享阈值的残差收缩构建单元 (RSBU-CS) 结构 </em></p>
</div>

### 3.2 DRSN-CW（通道独立阈值）
在 DRSN-CW（Channel-Wise）结构中，全连接网络的输出神经元数量与特征图的通道数一致。这意味着每一个通道都会计算出一个独立的缩放系数，从而获得独立的阈值。由于不同的卷积通道通常包含不同程度的噪声信息，通道独立的阈值设定更为灵活，理论去噪能力与特征筛选能力更强。

<div align="center">
  <img width="45%" src="https://github.com/user-attachments/assets/202bcf78-2b4a-4e4b-9eb7-c077eb4feab1" />
  <p><em>图4 具有通道独立阈值的残差收缩构建单元 (RSBU-CW) 结构 </em></p>
</div>

## 4. 实验结论

论文通过行星齿轮箱传动系统故障诊断模拟器采集了包含健康状态、轴承故障与齿轮故障在内的8种状态数据。

在人为添加不同信噪比（SNR 从 5dB 到 -5dB）的高斯噪声、拉普拉斯噪声和粉红噪声的条件下，DRSN 展现出了优越的诊断性能。实验数据表明，引入自适应软阈值化机制的 DRSN-CS 和 DRSN-CW 的测试准确率均显著高于传统的 CNN 与 ResNet。其中，由于 DRSN-CW 对不同通道特征具备更精细的收缩控制能力，其总体诊断准确率略高于 DRSN-CS。

可视化结果同样证实，DRSN 提取的高层特征在二维空间中的类内聚合度更高，类间区分度更明显。

## 5. 文献来源

- **标题 :** Deep Residual Shrinkage Networks for Fault Diagnosis
- **期刊 :** IEEE Transactions on Industrial Informatics (Volume 16, Issue 7, July 2020, Pages 4681-4690)
- **DOI :** [10.1109/TII.2019.2943898](https://doi.org/10.1109/TII.2019.2943898)
