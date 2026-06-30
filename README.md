# CNN 演变综述与 CIFAR-10 图像分类实践

本目录按 `任务要求.md` 完成一份可运行的 ResNet-18 复现实验与报告，并补充了 ResNet-34 / PlainNet-34 深层消融实验。

## 文件结构

- `src/train_cifar10_resnet.py`：CIFAR-10 版 ResNet/PlainNet 训练、验证、测试与曲线绘制。
- `src/compare_runs.py`：汇总多个实验结果并生成消融对比图。
- `scripts/run_resnet18_ablation.sh`：一键运行 ResNet-18 与去残差 PlainNet-18 消融实验。
- `scripts/run_resnet34_ablation.sh`：一键运行 ResNet-34 与去残差 PlainNet-34 深层消融实验。
- `reports/CNN综述与CIFAR10实践报告.md`：综述、实验设计、运行说明、结果与心得。
- `results/`：运行后保存模型、指标、曲线和对比表。

## 快速运行

```bash
bash scripts/run_resnet18_ablation.sh
```

默认运行 5 个 epoch，学习率为 `0.01`，用于较快得到训练/验证曲线。若想复现本报告中的充分训练结果：

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
- `curves.png`：训练/验证 loss 和 accuracy 曲线。
- `summary.json`：最佳验证准确率、测试准确率、参数量等摘要。
- `best_model.pt` / `last_model.pt`：模型权重。

## 本次 100 Epoch 结果

已按任务要求运行：

```bash
EPOCHS=100 BATCH_SIZE=256 LR=0.1 bash scripts/run_resnet18_ablation.sh
EPOCHS=100 BATCH_SIZE=256 LR=0.1 bash scripts/run_resnet34_ablation.sh
```

结果：

| 模型 | 最佳验证准确率 | 测试准确率 |
|---|---:|---:|
| ResNet-18 | 94.76% | 94.39% |
| PlainNet-18 | 94.48% | 93.96% |
| ResNet-34 | 94.88% | 94.56% |
| PlainNet-34 | 93.52% | 93.44% |

最终报告见 `reports/CNN综述与CIFAR10实践报告.md`，对应图像见：

- `results/resnet18_e100/curves.png`
- `results/plain18_e100/curves.png`
- `results/ablation_comparison.png`
- `results/resnet34_e100/curves.png`
- `results/plain34_e100/curves.png`
- `results/resnet34_e100_comparison/ablation_comparison.png`
- `results/depth_comparison/ablation_comparison.png`
