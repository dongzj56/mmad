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
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from monai.data import Dataset
from models import resnet
from datasets.ADNI import ADNI, ADNI_transform
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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


def generate_model(model_type='resnet', model_depth=50,
                   input_W=224, input_H=224, input_D=224,
                   resnet_shortcut='B',
                   pretrain_path='config/pretrain/resnet_50_23dataset.pth',
                   nb_class=2,  # 修改为2分类输出
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
        nn.Linear(fc_in, nb_class)  # 输出维度与nb_class=2匹配
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


def calculate_metrics(y_true, y_pred, y_score):
    return {
        'acc': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_score),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'cm': confusion_matrix(y_true, y_pred)
    }


def train():
    torch.manual_seed(42)
    np.random.seed(42)
    cfg = Config(load_config())

    # 加载数据
    dataset = ADNI(cfg.label_file, cfg.mri_dir, cfg.task, cfg.augment).data_dict
    tr_val, test_data = train_test_split(dataset, test_size=0.2, random_state=42,
                                         stratify=[d['label'] for d in dataset])
    labels = [d['label'] for d in tr_val]  # 用于分层交叉验证

    # 初始化日志
    writer = SummaryWriter(cfg.checkpoint_dir)
    csv_path = os.path.join(cfg.checkpoint_dir, 'cv_results.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['fold', 'epoch',
             'tr_acc', 'tr_auc', 'tr_loss',
             'vl_acc', 'vl_auc', 'vl_loss', 'lr']
        )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 分层交叉验证
    kf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(tr_val, labels), 1):
        print(f"\n=== Fold {fold}/{cfg.n_splits} ===")
        train_data = [tr_val[i] for i in train_idx]
        val_data = [tr_val[i] for i in val_idx]

        # 动态预处理（每个fold独立）
        tf_tr, tf_vt = ADNI_transform(augment=cfg.augment)
        ds_tr = Dataset(data=train_data, transform=tf_tr)
        ds_vl = Dataset(data=val_data, transform=tf_vt)
        loader_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        loader_vl = DataLoader(ds_vl, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # 初始化模型和优化器
        model = generate_model(
            model_type=cfg.model_type, model_depth=cfg.model_depth,
            input_W=cfg.input_W, input_H=cfg.input_H, input_D=cfg.input_D,
            resnet_shortcut=cfg.resnet_shortcut,
            pretrain_path=cfg.pretrain_path,
            nb_class=2,
            dropout_rate=cfg.dropout_rate,
            device=device
        )
        
        # 计算类别权重
        class_counts = np.bincount([d['label'] for d in train_data])
        class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)  # 带权重的损失函数

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )

        # 超参
        warmup_epochs = max(1, min(10, int(cfg.num_epochs * 0.1)))
        total_epochs = cfg.num_epochs
        cosine_epochs = total_epochs - warmup_epochs
        min_lr = cfg.lr * 1e-4

        # 学习率调度器（保持原有结构）
        warmup_sched = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        cosine_sched = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=min_lr,
            last_epoch=-1  # 确保每次从头开始
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs]
        )

        # 早停机制
        best_metric = -np.inf  # 初始化为负无穷
        patience = 5
        no_improve = 0

        for epoch in range(1, cfg.num_epochs + 1):
            t0 = time.time()
            model.train()
            loss_sum = 0
            y_true, y_pred, y_score = [], [], []

            for batch in loader_tr:
                x = batch['MRI'].to(device)
                y = batch['label'].to(device).squeeze().long()  # 确保为LongTensor
                out = model(x)
                loss = criterion(out, y)
                loss_sum += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                probs = torch.softmax(out, 1)[:, 1].detach().cpu().numpy()
                preds = out.argmax(1).detach().cpu().numpy()
                y_true.extend(y.cpu().numpy())
                y_score.extend(probs)
                y_pred.extend(preds)

            # 训练集指标
            tr_metrics = calculate_metrics(y_true, y_pred, y_score)
            tr_loss = loss_sum / len(loader_tr)

            # 验证集评估（优化后的softmax计算）
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

                    probs = torch.softmax(out, 1)  # 只计算一次softmax
                    v_score.extend(probs[:, 1].cpu().numpy())
                    v_pred.extend(out.argmax(1).cpu().numpy())
                    v_true.extend(y.cpu().numpy())

            vl_metrics = calculate_metrics(v_true, v_pred, v_score)
            vl_loss = vl_loss / len(loader_vl)

            # 学习率更新
            lr_now = scheduler.get_last_lr()[0]
            scheduler.step()

            # 记录日志
            writer.add_scalar(f'fold{fold}/train/acc', tr_metrics['acc'], epoch)
            writer.add_scalar(f'fold{fold}/val/acc', vl_metrics['acc'], epoch)
            writer.add_scalar(f'fold{fold}/train/loss', tr_loss, epoch)
            writer.add_scalar(f'fold{fold}/val/loss', vl_loss, epoch)
            writer.add_scalar(f'fold{fold}/lr', lr_now, epoch)

            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    fold, epoch,
                    f"{tr_metrics['acc']:.6f}", f"{tr_metrics['auc']:.6f}", f"{tr_loss:.6f}",
                    f"{vl_metrics['acc']:.6f}", f"{vl_metrics['auc']:.6f}", f"{vl_loss:.6f}",
                    f"{lr_now:.6f}"
                ])

            print(f"Fold{fold} Ep{epoch:03d} | "
                  f"tr_acc={tr_metrics['acc']:.6f}(AUC={tr_metrics['auc']:.6f}) "
                  f"vl_acc={vl_metrics['acc']:.6f}(AUC={vl_metrics['auc']:.6f}) "
                  f"lr={lr_now:.7f} time={time.time() - t0:.3f}s")

            # 早停判断
            current_metric = 0.5 * vl_metrics['auc'] + 0.5 * vl_metrics['acc']
            if current_metric > best_metric:
                best_metric = current_metric
                no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'metrics': {
                        'train_auc': tr_metrics['auc'],
                        'val_auc': vl_metrics['auc'],
                        'val_loss': vl_loss,
                        'current_metric': current_metric
                    },
                    'config': vars(cfg)  # 保存当前配置
                }, os.path.join(cfg.checkpoint_dir, f"best_fold{fold}.pth"))
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # 保存最终epoch模型
        torch.save({
            'epoch': cfg.num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'final_metrics': {
                'train_auc': tr_metrics['auc'],
                'val_auc': vl_metrics['auc'],
                'val_loss': vl_loss
            }
        }, os.path.join(cfg.checkpoint_dir, f"model_fold{fold}_final.pth"))

    print("\n=== CV complete ===")
    
    '''测试集'''
    # 指定checkpoint目录
    checkpoint_dir = cfg.checkpoint_dir
    
    # 运行测试
    test_models(checkpoint_dir, test_data)
    writer.close()

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
        
        # 加载checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"best_fold{fold}.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 测试评估
        all_probs = []
        y_true = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch['MRI'].to(device)
                y = batch['label'].squeeze().cpu().numpy()
                y_true.extend(y)
                out = model(x)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)

        # 存储当前折结果
        all_fold_probs.extend(all_probs)
        all_fold_labels.extend(y_true)

        # 计算指标
        y_pred = (np.array(all_probs) > 0.5).astype(int)
        metrics = calculate_metrics(y_true, y_pred, all_probs)
        all_metrics.append(metrics)

        # 绘制单折ROC曲线（半透明）
        fpr, tpr, _ = roc_curve(y_true, all_probs)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, 
                label=f'Fold {fold} (AUC={auc(fpr, tpr):.2f})')

        # 打印结果
        print(f"\n=== Fold {fold} Test Metrics ===")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Accuracy: {metrics['acc']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print("Confusion Matrix:\n", metrics['cm'])

    # 绘制合并后的平滑ROC曲线
    fpr, tpr, _ = roc_curve(all_fold_labels, all_fold_probs)
    roc_auc = auc(fpr, tpr)
    mean_fpr = np.linspace(0, 1, 100)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    plt.plot(mean_fpr, interp_tpr, 'b-', lw=2, 
            label=f'Mean ROC (AUC={roc_auc:.2f})')

    # 计算平均指标
    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics]) 
        for k in all_metrics[0].keys() if k != 'cm'
    }
    std_metrics = {
        k: np.std([m[k] for m in all_metrics])
        for k in all_metrics[0].keys() if k != 'cm'
    }

    # 完善图表
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC Curves')
    plt.legend(loc="lower right")
    
    # 保存结果
    roc_path = os.path.join(checkpoint_dir, 'test_roc_curves.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nROC曲线已保存至: {roc_path}")

    # 打印最终结果
    print("\n=== Final Test Results ===")
    for metric in ['auc', 'acc', 'f1', 'precision', 'recall']:
        print(f"{metric.upper()}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")
    
    return avg_metrics, std_metrics

if __name__ == '__main__':
    freeze_support()
    train()

# cd baseline_MRI_3D
# python train_ResNet3D.py