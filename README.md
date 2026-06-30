# CIFAR-10 ResNet / PlainNet 消融实验

本仓库提供一个可复现的 PyTorch 图像分类实验，用于比较 CIFAR-10 版 ResNet 与去除残差连接后的 PlainNet。实验包含 ResNet-18 / PlainNet-18 消融对比，并补充 ResNet-34 / PlainNet-34 以观察网络加深后残差连接的影响。

仓库只保留代码、运行脚本和依赖说明；训练结果、曲线图、模型权重、课程报告等生成文件不会上传到 GitHub。重新运行脚本后，这些文件会在本地 `results/` 目录中生成。

## 文件结构

- `src/train_cifar10_resnet.py`：CIFAR-10 版 ResNet/PlainNet 训练、验证、测试与曲线绘制。
- `src/compare_runs.py`：汇总多个实验结果并生成消融对比图。
- `scripts/run_resnet18_ablation.sh`：一键运行 ResNet-18 与去残差 PlainNet-18 消融实验。
- `scripts/run_resnet34_ablation.sh`：一键运行 ResNet-34 与去残差 PlainNet-34 深层消融实验。
- `requirements.txt`：Python 依赖版本要求。

运行过程中会自动下载 CIFAR-10 到本地 `data/`，并把训练指标、曲线图、模型权重和汇总结果保存到本地 `results/`。这两个目录默认被 `.gitignore` 排除。

## 环境依赖

```bash
pip install -r requirements.txt
```

主要依赖：

- `torch>=2.0`
- `torchvision>=0.15`
- `matplotlib>=3.7`

## 快速运行

```bash
bash scripts/run_resnet18_ablation.sh
```

默认运行 5 个 epoch，学习率为 `0.01`，适合快速验证代码流程。若要进行较充分训练：

```bash
EPOCHS=100 BATCH_SIZE=256 LR=0.1 bash scripts/run_resnet18_ablation.sh
EPOCHS=100 BATCH_SIZE=256 LR=0.1 bash scripts/run_resnet34_ablation.sh
```

只跑单个模型示例：

```bash
python src/train_cifar10_resnet.py \
  --model resnet18 \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.1 \
  --output-dir results/resnet18_e100
```

脚本会自动下载 CIFAR-10 到 `data/`，并在输出目录生成：

- `metrics.csv` / `metrics.json`：每个 epoch 的训练与验证指标。
- `curves.png`：训练/验证 loss 与 accuracy 曲线。
- `summary.json`：最佳验证准确率、测试准确率、参数量等摘要。
- `best_model.pt` / `last_model.pt`：模型权重。

## 参考结果

在 100 epoch、batch size 256、初始学习率 0.1 的设置下，已有实验得到如下结果：

| 模型 | 最佳验证准确率 | 测试准确率 |
|---|---:|---:|
| ResNet-18 | 94.76% | 94.39% |
| PlainNet-18 | 94.48% | 93.96% |
| ResNet-34 | 94.88% | 94.56% |
| PlainNet-34 | 93.52% | 93.44% |

结果显示，18 层网络中 ResNet-18 相比 PlainNet-18 有小幅提升；加深到 34 层后，ResNet-34 相比 PlainNet-34 的优势更明显，说明残差连接在更深网络中对优化和泛化更有帮助。
