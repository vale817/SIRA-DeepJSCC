# SIRA-B2-init w/o R 消融实验

`sira_b2_no_r` 从 `semantic_awgn_c2.pt` 初始化，冻结 encoder/decoder，只训练
语义映射器 M 和保护适配器 A。该模型完全不创建信道可靠性模块 R，也不使用
R 产生的 SNR embedding、temperature 或线性 SNR。两种配置都使用
解码器敏感度 `v` 计算 `m × v`。`w/o R` 使用平方根风险先验，
因此这组对照同时改变了显式 SNR 条件化和依赖 SNR 的注水先验。

## 1. 训练

```bash
cd ~/sira_autodl

export SIRA_DINO_HUB_DIR=/root/autodl-tmp/torch_cache/hub
export SIRA_DINO_SOURCE=local
export SIRA_DINO_REPO_OR_DIR=/root/autodl-tmp/torch_cache/hub/facebookresearch_dinov2_main

python -m scripts.train \
  --latent_ch 2 \
  --methods sira_b2_no_r \
  --batch_size 16
```

输出 checkpoint：

```text
checkpoints/sira_b2_no_r_awgn_c2.pt
checkpoints/sira_b2_no_r_awgn_c2_final.pt
```

## 2. DIV2K + Kodak 评估

```bash
python -m scripts.eval \
  --latent_ch 2 \
  --methods semantic sira_b2_init sira_b2_no_r \
  --result_dir results/ablation_no_r_c2
```

## 3. COCO Object ROI 评估

```bash
python -m scripts.eval_coco \
  --coco_root ~/autodl-pub/COCO2017 \
  --latent_ch 2 \
  --methods semantic sira_b2_init sira_b2_no_r \
  --result_dir results/ablation_no_r_c2
```

## 4. 如何解释

在相同骨干初始化、数据、随机种子和评估脚本下，`Full` 与 `w/o R`
的差值衡量整套 SNR 条件化分配在该条件下的变化。要单独衡量 `v` 的作用，
需要另外比较 `m`、`v` 和 `m × v` 三种风险输入。已归档结果见[实验索引](../results/README.md)。

## 5. 验证 R 是否真正改变 Power Map

```bash
python -m scripts.analyze_r_sensitivity \
  --latent_ch 2 \
  --dataset kodak \
  --methods sira_b2_init sira_b2_no_r \
  --result_dir results/ablation_no_r_c2
```

脚本从训练好的模型读取推理使用的敏感度缓存。对于较早的 `m`-only
模型，缺失的缓存保持默认全 1。重点查看 JSON/终端中的 `-2 dB` 与
`15 dB` 对比：

- `relative_mad` 接近 0 且 `correlation` 接近 1：power map 几乎不随 SNR 改变。
- R 的 embedding/tau 明显变化，但 power map 不变：分配对这段 SNR 变化不敏感。
- power map 明显变化，但 Full 与 w/o R 性能接近：R 改变了策略，但变化没有带来有效收益。
- w/o R 的 `relative_mad` 应为 0、`correlation` 应为 1，可作为实现正确性的检查。
