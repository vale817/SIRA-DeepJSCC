# SIRA-B1-init vs SIRA-B2-init

## 方法定义

- `cnn`: B1 CNN-DeepJSCC，全模型训练。
- `semantic`: B2 Semantic-weighted，全模型训练。
- `sira_b1_init`: 加载 `cnn` checkpoint，冻结 encoder/decoder，只训练 M/R/A。
- `sira_b2_init`: 加载 `semantic` checkpoint，冻结 encoder/decoder，只训练 M/R/A。
- `sira`: 旧版 `sira_b1_init` 别名，仅用于兼容已有 checkpoint。

## 公平对比

`sira_b1_init` 和 `sira_b2_init` 使用：

- 相同的 M/R/A 网络结构；
- 相同的随机种子和数据顺序；
- 相同的学习率、epoch、损失函数和 SNR 采样；
- 不同的冻结 backbone checkpoint。

因此，两者差异主要反映冻结 backbone 表示质量的影响。

## 训练

先确保 B1/B2 checkpoint 已存在：

```bash
python train.py --latent_ch 2 --methods cnn semantic
```

然后训练两个 SIRA 版本：

```bash
python train.py --latent_ch 2 --methods sira_b1_init sira_b2_init
```

如果 checkpoint 已存在并需要重新训练：

```bash
python train.py --latent_ch 2 --methods sira_b1_init sira_b2_init --force
```

训练输出：

```text
checkpoints/sira_b1_init_awgn_c2.pt
checkpoints/sira_b2_init_awgn_c2.pt
```

`.pt` 是最佳验证 checkpoint；`_final.pt` 是最后一轮 checkpoint。

## DIV2K 与 Kodak 评估

```bash
python eval.py \
  --latent_ch 2 \
  --methods cnn semantic sira_b1_init sira_b2_init
```

## COCO Object ROI 评估

```bash
python eval_coco.py \
  --coco_root data/coco \
  --latent_ch 2 \
  --methods cnn semantic sira_b1_init sira_b2_init
```

## Power Map 可视化

```bash
python visualize_power_map.py \
  --latent_ch 2 \
  --method sira_b1_init \
  --dataset coco \
  --index 20 \
  --snrs -2 5 15 \
  --out results/power_map_sira_b1_c2.png

python visualize_power_map.py \
  --latent_ch 2 \
  --method sira_b2_init \
  --dataset coco \
  --index 20 \
  --snrs -2 5 15 \
  --out results/power_map_sira_b2_c2.png
```

## 结果解释

- 若 `SIRA-B2-init > SIRA-B1-init`，说明 SIRA 能受益于更语义友好的冻结 backbone。
- 若两者都优于各自 backbone，说明 M/R/A 作为轻量保护适配器有效。
- 该实验支持 `backbone-compatible` 或 `compatible with different pretrained DeepJSCC backbones`。
- 因为 B1/B2 架构相同，该实验本身不足以严格证明 `architecture-agnostic`。
