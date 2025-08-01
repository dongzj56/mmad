import os
import json
import time
import csv
import numpy as np
from multiprocessing import freeze_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef,
                             confusion_matrix)
from sklearn.model_selection import train_test_split, StratifiedKFold
from monai.data import Dataset
from models import resnet
from datasets.ADNI import ADNI, ADNI_transform
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from thop import profile  # 用于计算 FLOPs
import logging  # 新增：导入 logging

# load config.json
def load_config(path="config/config.json"):
    with open(path) as f:
        return json.load(f)

class Config:
    def __init__(self, d):
        for k, v in d.items(): setattr(self, k, v)
        self.weight_decay = getattr(self, 'weight_decay', 1e-4)
        self.dropout_rate = getattr(self, 'dropout_rate', 0.5)
        self.n_splits = getattr(self, 'n_splits', 5)
        self.print_config()

    def print_config(self):
        print("Configuration Parameters:\n" + "=" * 40)
        for k, v in vars(self).items():
            print(f"{k}: {v}")
        print("=" * 40)

# 加载模型
def generate_model(model_type='resnet', model_depth=50,
                   input_W=112, input_H=136, input_D=112,
                   resnet_shortcut='B',
                   pretrain_path='config/pretrain/resnet_50_23dataset.pth',
                   nb_class=2,
                   dropout_rate=0.5,
                   device=torch.device('cpu')):
    assert model_type == 'resnet'
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    fn = {
        10: resnet.resnet10, 18: resnet.resnet18, 34: resnet.resnet34,
        50: resnet.resnet50, 101: resnet.resnet101,
        152: resnet.resnet152, 200: resnet.resnet200
    }[model_depth]

    net = fn(
        sample_input_W=input_W, sample_input_H=input_H, sample_input_D=input_D,
        shortcut_type=resnet_shortcut, no_cuda=True, num_seg_classes=1
    )

    fc_in = {10: 256, 18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048, 200: 2048}[model_depth]
    net.conv_seg = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(fc_in, nb_class)
    )

    net.to(device)
    sd = net.state_dict()
    if os.path.isfile(pretrain_path):
        ckpt = torch.load(pretrain_path, map_location=device)
        state = ckpt.get('state_dict', ckpt)
        pd = {k: v for k, v in state.items() if k in sd}
        sd.update(pd)
        net.load_state_dict(sd)
        print("Loaded pretrained weights.")
    else:
        print(f"[Warning] no pretrained file at {pretrain_path}")
    return net

# 计算指标
def calculate_metrics(y_true, y_pred, y_score):
    """
    返回字典中键的排列顺序：ACC → PRE → SEN → SPE → F1 → AUC → MCC
    """
    # 混淆矩阵：[[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 7 个指标
    acc  = accuracy_score(y_true, y_pred)                       # ACC
    pre  = precision_score(y_true, y_pred, zero_division=0)     # PRE
    sen  = recall_score(y_true, y_pred, zero_division=0)        # SEN (=TPR)
    spe  = tn / (tn + fp + 1e-8)                                # SPE (=TNR)
    f1   = f1_score(y_true, y_pred, zero_division=0)            # F1
    auc  = roc_auc_score(y_true, y_score)                       # AUC
    mcc  = matthews_corrcoef(y_true, y_pred)                    # MCC

    return {
        'ACC': acc, 'PRE': pre, 'SEN': sen, 'SPE': spe,
        'F1': f1, 'AUC': auc, 'MCC': mcc,
        'cm': np.array([[tn, fp],
                        [fn, tp]])    
    }

# 训练
def train():
    cfg = Config(load_config())
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # 加载数据
    dataset = ADNI(cfg.label_file, cfg.mri_dir, cfg.task, cfg.augment).data_dict
    tr_val, test_data = train_test_split(
        dataset, test_size=0.2, random_state=42,
        stratify=[d['label'] for d in dataset]
    )
    labels = [d['label'] for d in tr_val]  # 用于分层交叉验证

    # 初始化 CSV，把所有指标都写进表头
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    csv_path = os.path.join(cfg.checkpoint_dir, 'cv_results.csv')

    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'fold', 'epoch',
            # 训练指标
            'tr_ACC', 'tr_PRE', 'tr_SEN', 'tr_SPE', 'tr_F1', 'tr_AUC', 'tr_MCC', 'tr_loss',
            # 验证指标
            'vl_ACC', 'vl_PRE', 'vl_SEN', 'vl_SPE', 'vl_F1', 'vl_AUC', 'vl_MCC', 'vl_loss',
            # 学习率
            'lr'
        ])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 分层交叉验证
    kf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(tr_val, labels), 1):
        print(f"\n=== Fold {fold}/{cfg.n_splits} ===")
        train_data = [tr_val[i] for i in train_idx]
        val_data   = [tr_val[i] for i in val_idx]

        # 动态预处理（每个fold独立）
        tf_tr, tf_vt = ADNI_transform(augment=cfg.augment)
        ds_tr = Dataset(data=train_data, transform=tf_tr)
        ds_vl = Dataset(data=val_data,   transform=tf_vt)
        loader_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
        loader_vl = DataLoader(ds_vl, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # 初始化模型和优化器
        model = generate_model(
            model_type=cfg.model_type, model_depth=cfg.model_depth,
            input_W=cfg.input_W, input_H=cfg.input_H, input_D=cfg.input_D,
            resnet_shortcut=cfg.resnet_shortcut,
            pretrain_path=cfg.pretrain_path,
            nb_class=2, dropout_rate=cfg.dropout_rate,
            device=device
        )
        class_counts = np.bincount([d['label'] for d in train_data])
        class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )

        # 学习率调度器：只用余弦退火
        min_lr = cfg.lr * 1e-6
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.num_epochs,
            eta_min=min_lr,
            last_epoch=-1
        )

        for epoch in range(1, cfg.num_epochs + 1):
            t0 = time.time()
            model.train()
            loss_sum = 0
            y_true, y_pred, y_score = [], [], []

            for batch in loader_tr:
                x = batch['MRI'].to(device)
                y = batch['label'].to(device).long().view(-1)
                out = model(x)
                loss = criterion(out, y)
                loss_sum += loss.item()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                probs = torch.softmax(out, 1)[:, 1].detach().cpu().numpy()
                preds = out.argmax(1).detach().cpu().numpy()
                y_true.extend(y.cpu().numpy())
                y_score.extend(probs)
                y_pred.extend(preds)

            # 训练集指标
            tr_metrics = calculate_metrics(y_true, y_pred, y_score)
            tr_loss = loss_sum / len(loader_tr)

            # 验证集评估
            model.eval()
            v_true, v_pred, v_score = [], [], []
            vl_loss = 0.0
            with torch.no_grad():
                for batch in loader_vl:
                    x = batch['MRI'].to(device)
                    y = batch['label'].to(device).squeeze().long()
                    out = model(x)
                    loss = nn.CrossEntropyLoss()(out, y)
                    vl_loss += loss.item()

                    probs = torch.softmax(out, 1)
                    v_score.extend(probs[:, 1].cpu().numpy())
                    v_pred.extend(out.argmax(1).cpu().numpy())
                    v_true.extend(y.cpu().numpy())

            vl_metrics = calculate_metrics(v_true, v_pred, v_score)
            vl_loss = vl_loss / len(loader_vl)

            # 学习率更新
            lr_now = scheduler.get_last_lr()[0]
            scheduler.step()

            # 写入 CSV
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    fold, epoch,
                    f"{tr_metrics['ACC']:.6f}", f"{tr_metrics['PRE']:.6f}",
                    f"{tr_metrics['SEN']:.6f}", f"{tr_metrics['SPE']:.6f}",
                    f"{tr_metrics['F1']:.6f}",  f"{tr_metrics['AUC']:.6f}",
                    f"{tr_metrics['MCC']:.6f}", f"{tr_loss:.6f}",
                    f"{vl_metrics['ACC']:.6f}", f"{vl_metrics['PRE']:.6f}",
                    f"{vl_metrics['SEN']:.6f}", f"{vl_metrics['SPE']:.6f}",
                    f"{vl_metrics['F1']:.6f}",  f"{vl_metrics['AUC']:.6f}",
                    f"{vl_metrics['MCC']:.6f}", f"{vl_loss:.6f}",
                    f"{lr_now:.6f}"
                ])

            # 控制台打印
            print(
                f"Fold{fold} Ep{epoch:03d} | "
                f"TR  ACC={tr_metrics['ACC']:.4f} PRE={tr_metrics['PRE']:.4f} "
                f"SEN={tr_metrics['SEN']:.4f} SPE={tr_metrics['SPE']:.4f} "
                f"F1={tr_metrics['F1']:.4f} AUC={tr_metrics['AUC']:.4f} "
                f"MCC={tr_metrics['MCC']:.4f} | "
                f"VL  ACC={vl_metrics['ACC']:.4f} PRE={vl_metrics['PRE']:.4f} "
                f"SEN={vl_metrics['SEN']:.4f} SPE={vl_metrics['SPE']:.4f} "
                f"F1={vl_metrics['F1']:.4f} AUC={vl_metrics['AUC']:.4f} "
                f"MCC={vl_metrics['MCC']:.4f} | "
                f"lr={lr_now:.10f} time={time.time() - t0:.1f}s"
            )

        # 保存最终epoch模型
        torch.save({
            'epoch': cfg.num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'final_metrics': {
                'train_auc': tr_metrics['AUC'],
                'val_auc': vl_metrics['AUC'],
                'val_loss': vl_loss
            }
        }, os.path.join(cfg.checkpoint_dir, f"model_fold{fold}.pth"))

    print("\n=== CV complete ===")

    # 运行测试
    test_models(cfg.checkpoint_dir, test_data)


def test_models(checkpoint_dir, test_data):
    """主测试函数"""
    cfg = Config(load_config())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 数据准备
    test_transforms = ADNI_transform(augment=False)[1]
    test_ds = Dataset(data=test_data, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # 存储结果
    all_metrics = []
    all_fold_probs = []
    all_fold_labels = []

    # 打开结果文件
    results_path = os.path.join(checkpoint_dir, 'test_result.txt')
    with open(results_path, 'w') as result_file:
        result_file.write("=== Test Results by Fold ===\n\n")

        plt.figure(figsize=(10, 8))
        for fold in range(1, cfg.n_splits + 1):
            # 加载模型
            model = generate_model(
                model_type=cfg.model_type,
                model_depth=cfg.model_depth,
                input_W=cfg.input_W,
                input_H=cfg.input_H,
                input_D=cfg.input_D,
                resnet_shortcut=cfg.resnet_shortcut,
                nb_class=2,
                dropout_rate=cfg.dropout_rate,
                device=device
            )
            ckpt_path = os.path.join(checkpoint_dir, f"model_fold{fold}.pth")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # —— 新增：计算 Params, FLOPs, Memory —— 
            dummy_input = torch.randn(1, 1, cfg.input_W, cfg.input_H, cfg.input_D).to(device)
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            # 参数量与内存开销
            total_params = params
            mem_bytes = total_params * 4  # 假设 float32，每个参数 4 字节
            mem_mb = mem_bytes / (1024**2)
            flops_g = flops / 1e9  # 转为 GFLOPs

            info_line = (
                f"Fold {fold} Model Complexity:\n"
                f"  Params: {total_params:,} ({mem_mb:.2f} MB)\n"
                f"  FLOPs: {flops_g:.3f} GFLOPs\n\n"
            )
            print(info_line, end='')
            result_file.write(info_line)
            # — end 新增 —

            # 预测
            fold_probs = []
            y_true = []
            with torch.no_grad():
                for batch in test_loader:
                    x = batch['MRI'].to(device)
                    y = batch['label'].squeeze().cpu().numpy()
                    y_true.extend(y)
                    out = model(x)
                    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                    fold_probs.extend(probs)

            all_fold_probs.extend(fold_probs)
            all_fold_labels.extend(y_true)

            # 计算指标
            y_pred = (np.array(fold_probs) > 0.5).astype(int)
            metrics = calculate_metrics(y_true, y_pred, fold_probs)
            all_metrics.append(metrics)

            # ROC 曲线
            fpr, tpr, _ = roc_curve(y_true, fold_probs)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {fold} (AUC={auc(fpr, tpr):.2f})')

            # 打印 & 写入文件：本折结果
            header = f"--- Fold {fold} Test Metrics ---\n"
            print(header, end='')
            result_file.write(header)
            for k in ['ACC','PRE','SEN','SPE','F1','AUC','MCC']:
                line = f"{k}: {metrics[k]:.4f}\n"
                print(line, end='')
                result_file.write(line)
            cm_line = f"Confusion Matrix:\n{metrics['cm']}\n\n"
            print(cm_line, end='')
            result_file.write(cm_line)

        # 绘制与保存合并 ROC
        fpr, tpr, _ = roc_curve(all_fold_labels, all_fold_probs)
        roc_auc = auc(fpr, tpr)
        mean_fpr = np.linspace(0, 1, 100)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        plt.plot(mean_fpr, interp_tpr, 'b-', lw=2, label=f'Mean ROC (AUC={roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Test ROC Curves')
        plt.legend(loc="lower right")
        roc_path = os.path.join(checkpoint_dir, 'test_roc_curves.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nROC曲线已保存至: {roc_path}")

        # 计算总体指标
        result_file.write("=== Overall Test Results ===\n\n")
        print("\n=== Overall Test Results ===")
        for k in ['ACC','PRE','SEN','SPE','F1','AUC','MCC']:
            vals = [m[k] for m in all_metrics]
            mean, std = np.mean(vals), np.std(vals)
            line = f"{k}: {mean:.4f} ± {std:.4f}\n"
            print(line, end='')
            result_file.write(line)

    print(f"\n测试结果已保存至: {results_path}")
    return (
        {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0] if k != 'cm'},
        {k: np.std([m[k] for m in all_metrics])  for k in all_metrics[0] if k != 'cm'}
    )


if __name__ == '__main__':
    freeze_support()
    train()

# cd baseline_MRI_3D
# python train_ResNet3D.py