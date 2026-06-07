# SIRA-B2-init w/o R 消融实验

`sira_b2_no_r` 从 `semantic_awgn_c2.pt` 初始化，冻结 encoder/decoder，只训练
语义映射器 M 和保护适配器 A。该模型完全不创建信道可靠性模块 R，也不使用
R 产生的 SNR embedding 或 temperature。

## 1. 训练

```bash
cd ~/sira_autodl

export SIRA_DINO_HUB_DIR=/root/autodl-tmp/torch_cache/hub
export SIRA_DINO_SOURCE=local
export SIRA_DINO_REPO_OR_DIR=/root/autodl-tmp/torch_cache/hub/facebookresearch_dinov2_main

python train.py \
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
python eval.py \
  --latent_ch 2 \
  --methods semantic sira_b2_init sira_b2_no_r \
  --result_dir results/ablation_no_r_c2
```

## 3. COCO Object ROI 评估

```bash
python eval_coco.py \
  --coco_root ~/autodl-pub/COCO2017 \
  --latent_ch 2 \
  --methods semantic sira_b2_init sira_b2_no_r \
  --result_dir results/ablation_no_r_c2
```

## 4. 如何解释

- `Full > w/o R`：R 的信道可靠性感知带来额外收益。
- `Full ≈ w/o R`：当前收益主要来自语义功率映射 M+A，R 的作用有限。
- `Full < w/o R`：R 可能引入了不必要的 SNR 条件扰动，需要重新设计或训练。

比较时应使用相同初始化、数据、随机种子、训练轮数和评估脚本。

## 5. 验证 R 是否真正改变 Power Map

```bash
python analyze_r_sensitivity.py \
  --latent_ch 2 \
  --dataset kodak \
  --methods sira_b2_init sira_b2_no_r \
  --result_dir results/ablation_no_r_c2
```

重点查看 JSON/终端中的 `-2 dB` 与 `15 dB` 对比：

- `relative_mad` 接近 0 且 `correlation` 接近 1：power map 几乎不随 SNR 改变。
- R 的 embedding/tau 明显变化，但 power map 不变：A 忽略了 R。
- power map 明显变化，但 Full 与 w/o R 性能接近：R 改变了策略，但变化没有带来有效收益。
- w/o R 的 `relative_mad` 应为 0、`correlation` 应为 1，可作为实现正确性的检查。
