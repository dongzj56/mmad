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
    cfg = Cfg(load_cfg())
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

def train():
    # ----------------- 加载数据 -------------------
    config_path = "config/config_unet3d_single_model.json"
    cfg = Cfg(load_cfg(config_path))
    for name, val in vars(cfg).items():
        print(f"{name:15s}: {val}")
    writer = SummaryWriter(cfg.checkpoint_dir)

    fold_loaders = []                  # ⬅️ 所有折的 DataLoader 都收集到这里
    fold_indices = defaultdict(dict)   # 可选：若想保存索引，方便调试

    full_ds = ADNI(cfg.label_file, cfg.mri_dir, cfg.task, cfg.augment).data_dict
    labels  = [d['label'] for d in full_ds]

    outer_cv = StratifiedKFold(
        n_splits=cfg.n_splits,     # 5 折
        shuffle=True,
        random_state=cfg.seed
    )

    for fold, (train_val_idx, test_idx) in enumerate(outer_cv.split(full_ds, labels), start=1):
        train_val_ds = [full_ds[i] for i in train_val_idx]
        test_ds      = [full_ds[i] for i in test_idx]

        # —— 内层 90/10 分出验证集 —— #
        labels_train_val = [d['label'] for d in train_val_ds]
        idxs = np.arange(len(train_val_ds))
        train_idx_, val_idx_ = train_test_split(
            idxs, test_size=0.125, stratify=labels_train_val, random_state=cfg.seed
        )
        train_ds = [train_val_ds[i] for i in train_idx_]
        val_ds   = [train_val_ds[i] for i in val_idx_]

        print(f"\n=== Fold {fold}/{cfg.n_splits} ===")
        print(f"训练集样本数: {len(train_ds)}  ({len(train_ds)/len(full_ds):.1%})")
        print(f"验证集样本数: {len(val_ds)}  ({len(val_ds)/len(full_ds):.1%})")
        print(f"测试集样本数: {len(test_ds)}  ({len(test_ds)/len(full_ds):.1%})")

        # —— 构造 DataLoader —— #
        tr_tf, vl_tf = ADNI_transform(augment=cfg.augment)
        te_tf        = vl_tf      # 测试不做增强

        tr_loader = DataLoader(
            Dataset(train_ds, tr_tf),
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        vl_loader = DataLoader(
            Dataset(val_ds, vl_tf),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )
        test_loader = DataLoader(
            Dataset(test_ds, te_tf),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )

        # —— 保存到列表 —— #
        fold_loaders.append({
            "fold"        : fold,
            "train_loader": tr_loader,
            "val_loader"  : vl_loader,
            "test_loader" : test_loader
        })

        # （可选）保存索引，便于日后溯源
        fold_indices[fold]["train_idx"] = train_idx_
        fold_indices[fold]["val_idx"]   = val_idx_
        fold_indices[fold]["test_idx"]  = test_idx

    # 现在 fold_loaders[0] ~ fold_loaders[4] 就是 5 组 train/val/test DataLoader
    save_path = os.path.join(cfg.checkpoint_dir, "fold_indices.json")
    with open(save_path, "w") as f:
        serializable = {
            str(fold): {
                "train_idx": v["train_idx"].tolist(),
                "val_idx"  : v["val_idx"].tolist(),
                "test_idx" : v["test_idx"].tolist(),
            }
            for fold, v in fold_indices.items()
        }
        json.dump(serializable, f, indent=2)
    print(f"fold indices saved to {save_path}")

    # ----------------- 五折交叉验证训练 -----------------
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for fold_idx in range(cfg.n_splits):              # cfg.n_splits == 5
        fold = fold_idx + 1
        print(f"\n=== Fold {fold}/{cfg.n_splits} ===")

        # —— 每折都重新实例化模型与训练组件 —— #
        model     = generate_model(cfg)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=getattr(cfg, 'weight_decay', 0)
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
        scaler    = GradScaler(enabled=getattr(cfg, 'fp16', False))

        # —— 获取该折的 DataLoader —— #
        tr_loader = fold_loaders[fold_idx]['train_loader']
        vl_loader = fold_loaders[fold_idx]['val_loader']

        # —— 换成标准交叉熵 —— #
        criterion = nn.CrossEntropyLoss()   # ⭐ 不再使用加权交叉熵 ⭐

        # —— 为该折创建专属 CSV —— #
        csv_path = os.path.join(cfg.checkpoint_dir, f"metrics_fold{fold}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_Loss","train_ACC","train_PRE","train_SEN","train_SPE","train_F1","train_AUC","train_MCC",
                "val_Loss","val_ACC"  ,"val_PRE"  ,"val_SEN"  ,"val_SPE"  ,"val_F1"  ,"val_AUC"  ,"val_MCC",
            ])

        best_auc = -np.inf

        # —— Epoch 循环 —— #
        for epoch in range(1, cfg.num_epochs + 1):
            t0 = time.time()

            # -------- Train --------
            model.train()
            tr_loss_sum = 0.0
            tr_batches  = 0
            yt, yp, ys = [], [], []
            for batch in tr_loader:
                x = batch['MRI'].to(cfg.device)
                y = batch['label'].to(cfg.device).long().view(-1)

                optimizer.zero_grad()
                with autocast(device_type='cuda', enabled=getattr(cfg, 'fp16', False)):
                    out  = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                tr_loss_sum += loss.item()
                tr_batches  += 1

                prob = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
                pred = out.argmax(1).detach().cpu().numpy()
                yt.extend(y.cpu().numpy())
                yp.extend(pred)
                ys.extend(prob)

            tr_met  = calculate_metrics(yt, yp, ys)
            tr_loss = tr_loss_sum / tr_batches

            # -------- Validation --------
            model.eval()
            vl_loss_sum = 0.0
            vl_batches  = 0
            yt, yp, ys = [], [], []
            with torch.no_grad():
                for batch in vl_loader:
                    x = batch['MRI'].to(cfg.device)
                    y = batch['label'].to(cfg.device).long().view(-1)

                    with autocast(device_type='cuda', enabled=getattr(cfg, 'fp16', False)):
                        out  = model(x)
                        loss = criterion(out, y)

                    vl_loss_sum += loss.item()
                    vl_batches  += 1

                    prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                    pred = out.argmax(1).cpu().numpy()
                    yt.extend(y.cpu().numpy())
                    yp.extend(pred)
                    ys.extend(prob)

            vl_met  = calculate_metrics(yt, yp, ys)
            vl_loss = vl_loss_sum / vl_batches
            scheduler.step()

            print(f"Fold {fold} | Epoch {epoch:03d} | "
                f"Train Loss={tr_loss:.4f} | Val Loss={vl_loss:.4f} | "
                f"Train ACC={tr_met['ACC']:.4f} | Val ACC={vl_met['ACC']:.4f} | "
                f"Train AUC={tr_met['AUC']:.4f} | Val AUC={vl_met['AUC']:.4f} | "
                f"time={time.time()-t0:.1f}s")

            # —— 保存当前折最佳模型 —— #
            if vl_met['AUC'] > best_auc:
                best_auc = vl_met['AUC']
                torch.save(
                    model.state_dict(),
                    os.path.join(cfg.checkpoint_dir, f"best_model_fold{fold}.pth")
                )
                print("✅ Fold", fold, "saved best model (AUC={:.4f})".format(best_auc))

            # —— 追加写入 CSV —— #
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{tr_loss:.4f}", f"{tr_met['ACC']:.4f}", f"{tr_met['PRE']:.4f}",
                    f"{tr_met['SEN']:.4f}", f"{tr_met['SPE']:.4f}", f"{tr_met['F1']:.4f}", f"{tr_met['AUC']:.4f}", f"{tr_met['MCC']:.4f}",
                    f"{vl_loss:.4f}", f"{vl_met['ACC']:.4f}", f"{vl_met['PRE']:.4f}",
                    f"{vl_met['SEN']:.4f}", f"{vl_met['SPE']:.4f}", f"{vl_met['F1']:.4f}", f"{vl_met['AUC']:.4f}", f"{vl_met['MCC']:.4f}",
                ])

        print(f"=== Fold {fold} 完成，Best AUC={best_auc:.4f} ===")

if __name__ == '__main__':
    if os.name == 'nt':
        freeze_support()

    train()

#---------------------------------测试---------------------------

    all_metrics = []
    all_probs   = []
    all_labels  = []

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
