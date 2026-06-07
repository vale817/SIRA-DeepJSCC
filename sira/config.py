# ============================================================
# config.py  —  所有超参数和路径，改这一个文件就够了
# ============================================================
import os

# ── 路径 ──────────────────────────────────────────────────────
DIV2K_TRAIN_HR = './data/DIV2K/DIV2K_train_HR'   # 800 张高分辨率图
DIV2K_VAL_HR   = './data/DIV2K/DIV2K_valid_HR'   # 100 张，可选
KODAK_DIR      = './data/kodak'                   # 24 张 Kodak 图
CKPT_DIR       = './checkpoints'
RESULT_DIR     = './results'

# ── 训练超参数 ────────────────────────────────────────────────
CHANNEL        = 'awgn'          # 'awgn' or 'rayleigh'
LATENT_CH      = 4
EPOCHS_B1      = 100             # B1 (cnn) 训练轮数
EPOCHS_B2      = 100             # B2 (semantic) 训练轮数
EPOCHS_SIRA    = 100             # SIRA 训练轮数（只训 M/R/A）
BATCH_SIZE     = 16              # DIV2K 256×256 crop，4090 可以开到 32
LR             = 1e-3
CROP_SIZE      = 256             # 训练时随机 crop 的尺寸
TRAIN_SNR_RANGE = (-2.0, 15.0)
SNR_SWEEP      = [-2, 0, 2, 5, 10, 15]

# ── DINOv2 ────────────────────────────────────────────────────
# AutoDL/Colab 上首次加载 DINOv2 会下载 torch hub repo + checkpoint。
# 可用环境变量覆盖，避免改代码：
#   SIRA_IMPORTANCE_MODE=edge              # 跳过 DINO，使用边缘重要性
#   SIRA_DINO_HUB_DIR=/path/to/torch_hub   # 指定/复用 DINO 缓存目录
#   SIRA_DINO_SOURCE=local
#   SIRA_DINO_REPO_OR_DIR=/path/to/dinov2  # 离线使用已下载的 dinov2 repo
IMPORTANCE_MODE  = os.getenv('SIRA_IMPORTANCE_MODE', 'dino').lower()  # 'edge' or 'dino'
DINO_MODEL_NAME  = os.getenv('SIRA_DINO_MODEL_NAME', 'dinov2_vits14')
DINO_INPUT_SIZE  = int(os.getenv('SIRA_DINO_INPUT_SIZE', '224'))
DINO_TEMPERATURE = float(os.getenv('SIRA_DINO_TEMPERATURE', '0.25'))
DINO_REC_ALPHA   = float(os.getenv('SIRA_DINO_REC_ALPHA', '0.5'))
DINO_M_LAMBDA    = float(os.getenv('SIRA_DINO_M_LAMBDA', '0.1'))
DINO_HUB_DIR     = os.getenv('SIRA_DINO_HUB_DIR', './.torch_hub')
DINO_SOURCE      = os.getenv('SIRA_DINO_SOURCE', 'github')  # 'github' or 'local'
DINO_REPO_OR_DIR = os.getenv('SIRA_DINO_REPO_OR_DIR', 'facebookresearch/dinov2')

# ── 样式（画图用）────────────────────────────────────────────
METHOD_NAMES = {
    'cnn':      'B1: CNN-DeepJSCC',
    'semantic': 'B2: Semantic-weighted',
    'sira':     'Ours: SIRA-B1-init',
    'sira_b1_init': 'Ours: SIRA-B1-init',
    'sira_b2_init': 'Ours: SIRA-B2-init',
    'sira_b2_no_r': 'Ablation: SIRA-B2 w/o R',
}
METHOD_STYLE = {
    'cnn':      ('#6E7781', 'o', '--'),
    'semantic': ('#2C91E0', '^', ':'),
    'sira':     ('#F0A73A', 'D', '-'),
    'sira_b1_init': ('#F0A73A', 'D', '-'),
    'sira_b2_init': ('#3ABF99', 's', '-.'),
    'sira_b2_no_r': ('#9B8AC4', 'v', '--'),
}

# ── 其他 ──────────────────────────────────────────────────────
SEED = 42

os.makedirs(CKPT_DIR,   exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
