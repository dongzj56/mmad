import os, json, time, csv, numpy as np
from collections import Counter, defaultdict   # Counter 仍保留，避免影响其余逻辑
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from datasets.ADNI import ADNI, ADNI_transform
from monai.data import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from utils.metrics import calculate_metrics
from models.unet3d import UNet3DClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.multiprocessing import freeze_support


# -------------------- 配置 --------------------
def load_cfg(path):
    with open(path) as f: 
        return json.load(f)

class Cfg:
    def __init__(self, d):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for k, v in d.items(): 
            setattr(self, k, v)

# ----------------- 创建模型 -------------------
def generate_model(cfg):
    model = UNet3DClassifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.nb_class
    ).to(cfg.device)

    # 参数统计
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_per_param  = 2 if getattr(cfg, 'fp16', False) else 4
    print("--------------------model------------------")
    print(f"Total params(M)    : {total_params:,}")
    print(f"Trainable params(M): {trainable_params:,}")
    print(f"Approx. size       : {total_params*bytes_per_param/1024**2:.2f} MB")
    print("model type:", type(model).__name__)

    return model

# -----------------------测试-----------------------
def load_test_data(cfg, fold):
    full_ds = ADNI(
        cfg.label_file,
        cfg.mri_dir,
        cfg.task,
        cfg.augment
    ).data_dict

    idx_path = os.path.join(cfg.checkpoint_dir, "fold_indices.json")
    with open(idx_path, "r") as f:
        all_indices = json.load(f)

    test_idx = all_indices[str(fold)]["test_idx"]
    test_data = [full_ds[i] for i in test_idx]
    return test_data

def test_models(checkpoint_dir, test_data, fold):
    device = cfg.device

    _, test_tf = ADNI_transform(augment=False)
    ds = Dataset(data=test_data, transform=test_tf)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    model = generate_model(cfg)
    ckpt = os.path.join(checkpoint_dir, f"best_model_fold{fold}.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    print(f"✅ Loaded {ckpt}")

    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['MRI'].to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            labels = batch['label'].long().view(-1).cpu().numpy()
            y_prob.extend(probs)
            y_true.extend(labels)

    y_pred  = (np.array(y_prob) > 0.5).astype(int)
    metrics = calculate_metrics(y_true, y_pred, y_prob)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC Fold {fold} (AUC={metrics["AUC"]:.2f})')
    roc_path = os.path.join(checkpoint_dir, f"roc_fold{fold}.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ ROC curve for fold {fold} saved to {roc_path}")

    return metrics, y_prob, y_true

if __name__ == '__main__':
    if os.name == 'nt':
        freeze_support()

    # train()

#---------------------------------测试---------------------------

    all_metrics = []
    all_probs   = []
    all_labels  = []

    config_path = "config/config_unet3d_single_model2.json"
    cfg = Cfg(load_cfg(config_path))

    results_txt = os.path.join(cfg.checkpoint_dir, "test_results.txt")
    with open(results_txt, "w") as f:
        f.write("Fold\tACC\tPRE\tSEN\tSPE\tF1\tAUC\tMCC\n")

    for fold in range(1, cfg.n_splits + 1):
        print(f"\n=== Testing Fold {fold}/{cfg.n_splits} ===")
        test_data = load_test_data(cfg, fold)
        metrics, probs, labels = test_models(cfg.checkpoint_dir, test_data, fold)

        with open(results_txt, "a") as f:
            f.write(
                f"{fold}\t"
                f"{metrics['ACC']:.4f}\t{metrics['PRE']:.4f}\t"
                f"{metrics['SEN']:.4f}\t{metrics['SPE']:.4f}\t"
                f"{metrics['F1']:.4f}\t{metrics['AUC']:.4f}\t"
                f"{metrics['MCC']:.4f}\n"
            )

        all_metrics.append(metrics)
        all_probs.extend(probs)
        all_labels.extend(labels)

    # 画平均 ROC
    mean_fpr = np.linspace(0, 1, 100)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    plt.plot(mean_fpr, interp_tpr, 'b-', lw=2,
            label=f'Mean ROC (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(cfg.checkpoint_dir, 'mean_test_roc.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 汇总指标：均值 ± 标准差
    print("\n=== Final Test Results (mean ± std) ===")
    for k in ['ACC', 'PRE', 'SEN', 'SPE', 'F1', 'AUC', 'MCC']:
        vals = [m[k] for m in all_metrics]
        print(f"{k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
