# RAICOM 2025 智海人工智能算法应用竞赛

本仓库整理自 **2025 年睿抗机器人开发者大赛（RAICOM）全国总决赛——“智海人工智能算法应用竞赛”** 的参赛代码。

## 项目内容

| 赛段 | 任务 | 方法 |
| --- | --- | --- |
| 省赛 | 5 类语音情绪识别 | MFCC、谱质心、谱带宽、色度与频谱对比度特征 + Conformer |
| 全国总决赛 | 足球比赛场景目标检测 | YOLOv5 + HSV、旋转、平移、缩放、Mosaic 与 MixUp 数据增强 |

语音情绪类别为 `anger`、`fear`、`happy`、`neutral`、`sad`；目标检测类别为 `ball`、`goalkeeper`、`player`、`referee`。

## 仓库结构

```text
.
├── provincial_round/
│   ├── model.py             # 特征提取、数据集与 Conformer 模型
│   ├── train.py             # 训练与验证
│   └── predict.py           # 单条音频推理
├── national_final/
│   ├── train.py             # YOLOv5 训练与数据增强
│   ├── data.example.yaml    # 数据集配置示例
│   └── README.md            # 数据准备说明
├── requirements.txt
└── .gitignore
```

## 快速开始

建议使用 Python 3.10 或 3.11。先安装与设备匹配的 PyTorch 和 TorchAudio，再安装其余依赖：

```bash
pip install torch torchaudio
pip install -r requirements.txt
```

### 省赛：语音情绪识别

按以下结构准备 WAV 数据：

```text
data/ser/train/
├── anger/
├── fear/
├── happy/
├── neutral/
└── sad/
```

训练：

```bash
python provincial_round/train.py --data data/ser/train
```

推理：

```bash
python provincial_round/predict.py \
  --audio path/to/sample.wav \
  --checkpoint weights/ser_conformer_best.pth
```

### 全国总决赛：足球目标检测

按照 [national_final/README.md](national_final/README.md) 准备数据集，然后运行：

```bash
python national_final/train.py --data national_final/data.yaml
```

## 公开范围

本仓库公开整理后的核心代码与配置。以下内容没有上传：

- 竞赛提供的语音数据与本地数据副本
- 训练生成的模型权重、日志和缓存
- 获奖证书、讲解视频及包含个人信息的材料
- 与算法任务无关的教学案例

国赛目标检测数据来自 Roboflow 的 [Football Players Detection v9](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/9)，原数据集采用 CC BY 4.0 许可；请从原页面获取并遵守其许可要求。

## 说明

代码由竞赛期间的实验脚本整理而来，主要调整了目录结构、本机绝对路径和训练入口，以便复现。实际结果会受数据划分、软硬件环境和随机种子影响。
