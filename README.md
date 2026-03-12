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
- [5. 代码复现](#5-代码复现)
- [文献来源](#文献来源)
---


## 1. 背景

工业旋转机械（如齿轮箱、轴承）的故障诊断通常依赖于振动信号分析。实际运行环境中存在大量背景噪声，导致故障初期的微弱信号容易被掩盖。

传统的深度学习模型（如卷积神经网络 CNN、残差网络 ResNet）在处理高噪声信号时，容易将噪声干扰提取为特征，导致诊断准确率下降。深度残差收缩网络（Deep Residual Shrinkage Networks, DRSN）基于这一问题提出。该方法将传统信号处理中的软阈值化（Soft Thresholding）机制作为非线性变换层引入深度神经网络结构中，以消除噪声相关特征，提高模型在强噪声环境下的特征学习能力。

## 2. 核心机制

DRSN 的核心思想是在 ResNet 的残差模块内部，集成一个能够自动计算阈值并执行特征收缩的子模块。

### 2.1 软阈值化 (Soft Thresholding)

软阈值化是信号降噪领域（如小波去噪）的经典方法。其基本数学逻辑是：
1. 将绝对值小于特定阈值的特征置为零。
2. 将绝对值大于阈值的特征向零的方向进行收缩（即减去设定的阈值）。

通过软阈值化，模型可以将接近于零的噪声特征剔除，同时保留绝对值较大的关键故障特征。由于该操作的导数在非零区域恒为 1，它同样具备防止梯度消失的作用。

<div align="center">
  <img width="80%" src="https://github.com/user-attachments/assets/9f79323d-8344-48bf-aecc-f0e86f57e0e2" />
  <p><em>图1 软阈值化函数及其导数示意图 </em></p>
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
  <p><em>图2 具有通道共享阈值的残差收缩构建单元 (RSBU-CS) 结构 </em></p>
</div>

### 3.2 DRSN-CW（通道独立阈值）
在 DRSN-CW（Channel-Wise）结构中，全连接网络的输出神经元数量与特征图的通道数一致。这意味着每一个通道都会计算出一个独立的缩放系数，从而获得独立的阈值。由于不同的卷积通道通常包含不同程度的噪声信息，通道独立的阈值设定更为灵活，理论去噪能力与特征筛选能力更强。

<div align="center">
  <img width="45%" src="https://github.com/user-attachments/assets/202bcf78-2b4a-4e4b-9eb7-c077eb4feab1" />
  <p><em>图3 具有通道独立阈值的残差收缩构建单元 (RSBU-CW) 结构 </em></p>
</div>

## 4. 实验结论

论文通过行星齿轮箱传动系统故障诊断模拟器采集了包含健康状态、轴承故障与齿轮故障在内的8种状态数据。

在人为添加不同信噪比（SNR 从 5dB 到 -5dB）的高斯噪声、拉普拉斯噪声和粉红噪声的条件下，DRSN 展现出了优越的诊断性能。实验数据表明，引入自适应软阈值化机制的 DRSN-CS 和 DRSN-CW 的测试准确率均显著高于传统的 CNN 与 ResNet。其中，由于 DRSN-CW 对不同通道特征具备更精细的收缩控制能力，其总体诊断准确率略高于 DRSN-CS。

可视化结果同样证实，DRSN 提取的高层特征在二维空间中的类内聚合度更高，类间区分度更明显。

## 5. 代码复现
利用PyTorch在加噪的凯斯西储大学轴承数据集上复现了论文的方法，完整代码如下：
<details>
<summary><b>📦 点击展开：DRSN-CW 项目原理与完整源码</b></summary>

```python
"""
项目名称：深度残差收缩网络 (DRSN-CW) - 旋转机械故障诊断框架
论文引用：Zhao, M., et al. "Deep Residual Shrinkage Networks for Fault Diagnosis," IEEE TII, 2020.

技术核心与实现原理：
1. 自适应软阈值化 (Adaptive Soft Thresholding)：
   通过构建非线性收缩算子，模型能够自动识别并抑制振动信号中的噪声分量。该机制在特征映射层面
   将对决策贡献度较低的“准零特征”进行收缩，从而增强模型对强背景噪声的鲁棒性。

2. 通道级注意力机制 (Channel-wise Attention)：
   模型内部集成轻量级子网络，利用全局平均采样与全连接层 (FC)，为每个特征通道独立计算动态阈值 τ。
   这种自适应的降噪策略使网络能够针对不同频率成分的干扰实施差异化去噪。

3. 残差收缩构建块 (RSBU-CW)：
   采用深度残差学习（Residual Learning），通过跨层恒等映射解决深层网络的性能退化问题。
   RSBU 模块将信号降噪与特征提取结合，确保了从原始时间序列到高阶语义特征的平滑转换。

4. 鲁棒性增强策略：
   集成动态数据增强引擎，包括循环移位、瞬态冲击模拟及动态信噪比（SNR）混合训练。
   模拟极端环境（如 -8dB SNR 环境）下的数据特征，提升模型在复杂工业现场的泛化精度。
"""

import os
import sys
import logging
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# =============================================================================
# 1. 环境与计算资源配置
# =============================================================================

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def get_device():
    """
    自动检测可用计算设备：优先 CUDA，其次 MPS (Apple Silicon)，最后 CPU。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
logging.info(f"系统运行设备: {DEVICE}")

# =============================================================================
# 2. 数据工程模块
# =============================================================================

class CWRULoader:
    """
    CWRU 原始数据加载类：负责读取 .mat 文件并提取驱动端振动信号。
    """
    def __init__(self, dataset_root, window_size=1024):
        self.base_directory = os.path.abspath(dataset_root)
        self.sample_length = window_size
        self.sampling_interval = window_size

    def _parse_mat_content(self, target_file):
        try:
            storage = sio.loadmat(target_file)
            for identifier in storage.keys():
                if 'DE_time' in identifier:
                    return storage[identifier].flatten()
        except Exception as e:
            logging.debug(f"文件读取异常 {target_file}: {e}")
            return None
        return None

    def load_data(self, category_dictionary):
        feature_collection, label_collection = [],[]
        is_data_found = False
        
        for class_idx, name_list in category_dictionary.items():
            for filename in name_list:
                full_path = os.path.join(self.base_directory, f"{filename}.mat")
                if not os.path.exists(full_path):
                    continue
                
                series = self._parse_mat_content(full_path)
                if series is None:
                    continue
                
                is_data_found = True
                for pointer in range(0, len(series) - self.sample_length + 1, self.sampling_interval):
                    sub_seq = series[pointer : pointer + self.sample_length]
                    feature_collection.append(sub_seq)
                    label_collection.append(class_idx)
        
        if not is_data_found:
            raise FileNotFoundError("指定路径下未检测到符合命名的 CWRU .mat 文件。")
            
        return np.array(feature_collection, dtype='float32'), np.array(label_collection, dtype='int64')

def add_awgn_torch(signal_tensor, snr_value):
    """
    加性高斯白噪声叠加（PyTorch 实现）：
    根据信噪比公式 P_noise = P_signal / 10^(SNR/10) 计算噪声水平。
    """
    if isinstance(snr_value, (list, tuple)):
        snr = np.random.uniform(snr_value[0], snr_value[1])
    else:
        snr = snr_value
        
    # 计算信号功率：针对 batch 中的每个样本独立计算
    signal_power = torch.mean(signal_tensor**2, dim=2, keepdim=True)
    noise_std = torch.sqrt(signal_power / (10 ** (snr / 10.0)))
    noise = torch.randn_like(signal_tensor) * noise_std
    return signal_tensor + noise

class FaultDataset(Dataset):
    """
    故障诊断数据集：支持训练时的动态数据增强（平移、冲击、噪声）。
    """
    def __init__(self, x, y, is_training=False):
        # 统一形状为 (Batch, Channels, Length)
        self.x = torch.from_numpy(x).float().unsqueeze(1)
        self.y = torch.from_numpy(y).long()
        self.is_training = is_training

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample_x = self.x[idx]
        sample_y = self.y[idx]

        if self.is_training:
            # 1. 随机循环移位 (Rolling)
            offset = np.random.randint(0, sample_x.shape[1])
            sample_x = torch.roll(sample_x, shifts=offset, dims=1)

            # 2. 瞬态冲击干扰 (Impulse Noise)
            if np.random.rand() > 0.9:
                num_spikes = np.random.randint(1, 3)
                positions = np.random.randint(0, sample_x.shape[1], num_spikes)
                spike_mag = torch.std(sample_x) * np.random.uniform(1.5, 2.5)
                sample_x[0, positions] += spike_mag * np.random.choice([-1, 1])

            # 3. 动态 SNR 混合 (Dynamic Noise)
            if np.random.rand() > 0.5:
                # 临时添加 batch 维度以复用 add_awgn_torch
                sample_x = add_awgn_torch(sample_x.unsqueeze(0), (-8, 8)).squeeze(0)

        return sample_x, sample_y

# =============================================================================
# 3. 神经网络架构定义 (DRSN-CW)
# =============================================================================

class SoftThreshold(nn.Module):
    """
    软阈值算子：y = sign(x) * max(|x| - tau, 0)
    """
    def forward(self, x, tau):
        # tau 形状应能通过广播与 x 匹配 (Batch, Channels, 1)
        return torch.sign(x) * F.relu(torch.abs(x) - tau)

class RSBU_CW(nn.Module):
    """
    残差收缩构建块 (Channel-wise)：
    通过注意力机制学习各通道独立的收缩阈值，实现自适应去噪。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(RSBU_CW, self).__init__()
        padding = kernel_size // 2
        
        # 主分支特征提取
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        
        # 注意力子网络 (阈值生成器)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )
        self.threshold_op = SoftThreshold()
        
        # 残差快捷链路
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        # 权重初始化 (He Normal / Kaiming Normal)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        identity = self.shortcut(x)
        
        # 两次预激活卷积变换
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        
        # 计算通道阈值
        # 1. 计算绝对值的均值作为特征统计量
        abs_x = torch.abs(out)
        abs_mean = self.gap(abs_x).view(abs_x.size(0), -1)
        
        # 2. 生成缩放因子并计算最终阈值 tau
        scales = self.fc(abs_mean)
        tau = (scales * abs_mean).unsqueeze(2) # 形状: (Batch, Channels, 1)
        
        # 3. 软阈值降噪
        out = self.threshold_op(out, tau)
        
        return out + identity

class DRSN_CW(nn.Module):
    """
    深度残差收缩网络完整结构。
    """
    def __init__(self, num_classes):
        super(DRSN_CW, self).__init__()
        
        # 初始特征感知层
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        
        # 堆叠 RSBU 块
        self.layer1 = nn.Sequential(
            RSBU_CW(32, 32, 5, stride=2),
            RSBU_CW(32, 32, 5, stride=1)
        )
        self.layer2 = nn.Sequential(
            RSBU_CW(32, 64, 5, stride=2),
            RSBU_CW(64, 64, 5, stride=1)
        )
        self.layer3 = nn.Sequential(
            RSBU_CW(64, 128, 5, stride=2),
            RSBU_CW(128, 128, 5, stride=1)
        )
        
        # 全局池化与分类输出
        self.post_bn = nn.BatchNorm1d(128)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.post_bn(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# =============================================================================
# 4. 训练与评估流程
# =============================================================================

def train_and_evaluate(data_path, window_size=1024):
    # CWRU 类别字典
    label_map = {
        0:['Normal_0', 'Normal_1', 'Normal_2', 'Normal_3'],
        1:['IR007_0', 'IR007_1', 'IR007_2', 'IR007_3'],
        2:['IR014_0', 'IR014_1', 'IR014_2', 'IR014_3'],
        3:['IR021_0', 'IR021_1', 'IR021_2', 'IR021_3'],
        4:['B007_0', 'B007_1', 'B007_2', 'B007_3'],
        5:['B014_0', 'B014_1', 'B014_2', 'B014_3'],
        6:['B021_0', 'B021_1', 'B021_2', 'B021_3'],
        7:['OR007@6_0', 'OR007@6_1', 'OR007@6_2', 'OR007@6_3'],
        8:['OR014@6_0', 'OR014@6_1', 'OR014@6_2', 'OR014@6_3'],
        9:['OR021@6_0', 'OR021@6_1', 'OR021@6_2', 'OR021@6_3']
    }

    loader = CWRULoader(data_path, window_size)
    try:
        raw_x, raw_y = loader.load_data(label_map)
    except Exception as e:
        logging.error(f"数据加载终止: {e}")
        return

    # 划分数据集 (70/15/15)
    x_train, x_temp, y_train, y_temp = train_test_split(raw_x, raw_y, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # 标准化处理 (使用训练集统计量)
    mu, sigma = np.mean(x_train), np.std(x_train)
    x_train = (x_train - mu) / sigma
    x_val = (x_val - mu) / sigma
    x_test = (x_test - mu) / sigma

    # 构建 PyTorch DataLoaders
    train_ds = FaultDataset(x_train, y_train, is_training=True)
    val_ds = FaultDataset(x_val, y_val, is_training=False)
    test_ds = FaultDataset(x_test, y_test, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # 初始化模型、损失函数与优化器
    model = DRSN_CW(num_classes=len(label_map)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)

    # 训练循环
    num_epochs = 100
    logging.info("开始模型训练与评估...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == targets).sum().item()
        
        # 验证集评估 (施加 -8dB 极端噪声验证鲁棒性)
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = add_awgn_torch(inputs, snr_value=-8).to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == targets).sum().item()
        
        epoch_loss = val_loss / len(val_ds)
        scheduler.step(epoch_loss)
        
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] "
                     f"Train Acc: {train_correct/len(train_ds):.4f} | "
                     f"Val Acc (-8dB): {val_correct/len(val_ds):.4f}")

    # 最终测试评估
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = add_awgn_torch(inputs, snr_value=-8).to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(inputs)
            test_correct += (outputs.argmax(1) == targets).sum().item()
    
    print("\n" + "="*50)
    print("模型评估报告 (DRSN-CW PyTorch)")
    print("评估环境：-8dB SNR (强背景噪声)")
    print(f"最终识别准确率: {100.0 * test_correct / len(test_ds):.2f}%")
    print("="*50)

# =============================================================================
# 程序入口
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = os.path.join(os.getcwd(), 'data_path')
    
    if not os.path.exists(DATA_PATH):
        user_input = input("未找到默认目录，请输入 CWRU 数据集 (.mat) 文件夹路径: ").strip()
        if user_input:
            DATA_PATH = user_input
        else:
            logging.critical("路径无效，程序退出。")
            sys.exit(0)

    train_and_evaluate(DATA_PATH)
```
</div>
</details>

在人工添加了-8dB强噪声的情况下，深度残差收缩网络也取得了90%以上的准确率。

<div align="center">
  <img width="45%" src="https://github.com/user-attachments/assets/fa4150d7-1691-4a4d-848e-8b2059d8b26f" />
  <p><em>图4 实验结果 </em></p>
</div>

## 文献来源

- **标题 :** Deep Residual Shrinkage Networks for Fault Diagnosis
- **期刊 :** IEEE Transactions on Industrial Informatics (Volume 16, Issue 7, July 2020, Pages 4681-4690)
- **DOI :** [10.1109/TII.2019.2943898](https://doi.org/10.1109/TII.2019.2943898)
