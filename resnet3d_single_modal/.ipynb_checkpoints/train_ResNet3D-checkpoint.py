import os
import json
import time
import csv
import numpy as np
from multiprocessing import freeze_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from ptflops import get_model_complexity_info  # pip install ptflops
from sklearn.model_selection import train_test_split, StratifiedKFold
from monai.data import Dataset
from models import resnet
from datasets.ADNI import ADNI, ADNI_transform


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
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'cm': confusion_matrix(y_true, y_pred)
    }


def train():
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
            nb_class=2,  # 强制设为2分类
            dropout_rate=cfg.dropout_rate,
            device=device
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=cfg.lr * 1e-2)

        # 早停机制
        best_vl_auc = 0
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
                loss = nn.CrossEntropyLoss()(out, y)
                loss_sum += loss.item()
                optimizer.zero_grad();
                loss.backward();
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

                    probs = torch.softmax(out, 1)[:, 1].cpu().numpy()
                    preds = out.argmax(1).cpu().numpy()
                    v_true.extend(y.cpu().numpy())
                    v_score.extend(probs)
                    v_pred.extend(preds)

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
                    f"{tr_metrics['acc']:.4f}", f"{tr_metrics['auc']:.4f}", f"{tr_loss:.4f}",
                    f"{vl_metrics['acc']:.4f}", f"{vl_metrics['auc']:.4f}", f"{vl_loss:.4f}",
                    f"{lr_now:.6f}"
                ])

            print(f"Fold{fold} Ep{epoch:03d} | "
                  f"tr_acc={tr_metrics['acc']:.3f}(AUC={tr_metrics['auc']:.3f}) "
                  f"vl_acc={vl_metrics['acc']:.3f}(AUC={vl_metrics['auc']:.3f}) "
                  f"lr={lr_now:.6f} time={time.time() - t0:.1f}s")

            # 早停判断
            if vl_metrics['auc'] > best_vl_auc:
                best_vl_auc = vl_metrics['auc']
                no_improve = 0
                torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, f"best_fold{fold}.pth"))
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # 保存最终epoch模型
        torch.save(model.state_dict(),
                   os.path.join(cfg.checkpoint_dir, f"model_fold{fold}_final.pth"))

    print("\n=== CV complete ===")

    # 测试集评估示例（需补充测试集DataLoader）
    test_transforms = ADNI_transform(augment=False)[1]
    test_ds = Dataset(data=test_data, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    def ensemble_evaluation(test_loader, cfg, device):
        models = []
        # 加载所有fold的最佳模型
        for fold in range(1, cfg.n_splits + 1):
            model_path = os.path.join(cfg.checkpoint_dir, f"best_fold{fold}.pth")
            model = generate_model(
                model_type=cfg.model_type, model_depth=cfg.model_depth,
                input_W=cfg.input_W, input_H=cfg.input_H, input_D=cfg.input_D,
                resnet_shortcut=cfg.resnet_shortcut,
                pretrain_path=cfg.pretrain_path,
                nb_class=2,
                dropout_rate=cfg.dropout_rate,
                device=device
            )
            model.load_state_dict(torch.load(model_path))
            model.eval()
            models.append(model)

        # 集成预测
        all_probs = []
        y_true = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch['MRI'].to(device)
                y = batch['label'].squeeze().cpu().numpy()
                y_true.extend(y)

                # 各模型预测概率取平均
                prob = 0
                for model in models:
                    out = model(x)
                    prob += torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                prob /= len(models)
                all_probs.extend(prob)

        # 计算指标
        y_pred = (np.array(all_probs) > 0.5).astype(int)
        metrics = calculate_metrics(y_true, y_pred, all_probs)

        print("\n=== Test Set Metrics ===")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Accuracy: {metrics['acc']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print("Confusion Matrix:\n", metrics['cm'])

    ensemble_evaluation(test_loader, cfg, device)


if __name__ == '__main__':
    freeze_support()
    train()