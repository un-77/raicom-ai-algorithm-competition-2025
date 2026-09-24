# 全国总决赛：足球场景目标检测

本赛段使用 YOLOv5 检测足球、守门员、球员和裁判四类目标。

## 数据集

竞赛代码使用 Roboflow 的 [Football Players Detection v9](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/9)。该数据集包含 312 张 YOLO 格式标注图像，原页面标注的许可为 CC BY 4.0。

下载 YOLO 格式的数据后，整理为：

```text
national_final/data/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

复制配置并按需修改路径：

```bash
cp national_final/data.example.yaml national_final/data.yaml
```

Windows PowerShell 可使用：

```powershell
Copy-Item national_final/data.example.yaml national_final/data.yaml
```

从仓库根目录运行：

```bash
python national_final/train.py --data national_final/data.yaml
```

训练默认进行 50 个 epoch，图像尺寸为 640，batch size 为 2，并启用 HSV、旋转、平移、缩放、剪切、水平翻转、Mosaic 和 MixUp 增强。
