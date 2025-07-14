import torch
from torch import nn
from models.unet3d import SharedConv3DBlock, Conv3DBlock, cen_exchange, UNet3DEncoder


class ImageEncoder(nn.Module):
    """
    两路 3D‑U‑Net 编码器，流内独立学习 → 特征拼接，返回融合向量表示。
    """
    def __init__(
        self,
        in_channels_per_modality: int = 1,
        level_channels: list[int]  = [64, 128, 256],
        bottleneck_channel: int    = 512
    ):
        super().__init__()
        # 两个独立的 Encoder（参数不共享）
        self.mri_enc = UNet3DEncoder(
            in_channels_per_modality,
            level_channels,
            bottleneck_channel
        )
        self.pet_enc = UNet3DEncoder(
            in_channels_per_modality,
            level_channels,
            bottleneck_channel
        )
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, mri: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        # 流内编码
        f_mri = self.mri_enc(mri)  # [B, C, D, H, W]
        f_pet = self.pet_enc(pet)
        # 流内池化 & 展平
        v_mri = self.global_pool(f_mri).flatten(1)  # [B, C]
        v_pet = self.global_pool(f_pet).flatten(1)  # [B, C]
        # 特征融合
        # fused = torch.cat([v_mri, v_pet], dim=1)   # [B, 2C]
        # return fused
        return v_mri, v_pet

class ImageEncoder_CEN(nn.Module):
    """
    双流 3D‑U‑Net + CEN 共享编码器，返回影像特征向量（不含拼接和分类头）。
    """
    def __init__(
        self,
        in_ch_modality: int = 1,
        level_channels: list[int] = [64, 128, 256],
        bottleneck_ch: int = 512,
        share_layers: int = 2,             # 共享前几层
        cen_ratios: tuple[float, ...] = (0.2, 0.1)   # 每层交换比例
    ):
        super().__init__()
        assert share_layers == len(cen_ratios), \
            "len(cen_ratios) 必须等于 share_layers"

        l1, l2, l3 = level_channels

        # ─── 动态构建共享 + CEN 层 ───
        self.shared_blocks = nn.ModuleList()
        in_out_pairs = [
            (in_ch_modality, l1),
            (l1, l2),
            (l2, l3)
        ]
        for i in range(share_layers):
            ic, oc = in_out_pairs[i]
            ratio = cen_ratios[i]
            self.shared_blocks.append(
                SharedConv3DBlock(ic, oc, half_ratio=ratio)
            )

        last_out = [l1, l2, l3][share_layers - 1]

        # ─── 深层独立编码 ───
        self.mri_block3 = Conv3DBlock(last_out, l3)
        self.pet_block3 = Conv3DBlock(last_out, l3)
        self.mri_bneck = Conv3DBlock(l3, bottleneck_ch, bottleneck=True)
        self.pet_bneck = Conv3DBlock(l3, bottleneck_ch, bottleneck=True)

        # ─── 全局池化 ───
        self.gap = nn.AdaptiveAvgPool3d(1)

    def forward(self, mri: torch.Tensor, pet: torch.Tensor):
        # ----- Shared layers + CEN -----
        for blk in self.shared_blocks:
            mri, pet, _, _ = blk(mri, pet)

        # ----- Private deep layers -----
        mri, _ = self.mri_block3(mri)
        pet, _ = self.pet_block3(pet)
        mri, _ = self.mri_bneck(mri)
        pet, _ = self.pet_bneck(pet)

        # ----- 特征向量提取 -----
        vm = self.gap(mri).flatten(1)  # [B, bottleneck_ch]
        vp = self.gap(pet).flatten(1)  # [B, bottleneck_ch]
        # 拼接 MRI 与 PET 向量        ↓ 维度 [B, 2*bottleneck_ch]
        v = torch.cat([vm, vp], dim=1)

        return v
        #
        # # 返回 MRI 和 PET 的特征向量
        # return vm, vp

#
# class MultiModalClassifier(nn.Module):
#     """
#     多模态融合分类器：接受 MRI 特征、PET 特征和表格特征，将三者拼接后进行线性分类。
#
#     Inputs:
#       - vm: [B, C] MRI 特征向量
#       - vp: [B, C] PET 特征向量
#       - tab: [B, T] 表格特征向量
#     Outputs:
#       - logits: [B, num_classes]
#     """
#     def __init__(
#         self,
#         feature_dim: int,
#         tabular_dim: int,
#         num_classes: int,
#         dropout: float = 0.0
#     ):
#         super().__init__()
#         fused_dim = feature_dim * 2 + tabular_dim
#         layers = []
#         if dropout > 0:
#             layers.append(nn.Dropout(dropout))
#         layers.append(nn.Linear(fused_dim, num_classes))
#         self.classifier = nn.Sequential(*layers)
#
#     def forward(self, vm: torch.Tensor, vp: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
#         # vm, vp: [B, C]; tab: [B, T]
#         fused = torch.cat([vm, vp, tab], dim=1)
#         logits = self.classifier(fused)
#         return logits

# -------------- 定义多模态分类器（简单 MLP 示例） --------------
class MultiModalClassifier(nn.Module):
    """img_feat [B, img_dim] & table [B, tab_dim] -> logits [B, num_cls]"""
    def __init__(self, img_dim: int, tab_dim: int, num_classes: int):
        super().__init__()
        self.img_dim   = img_dim
        self.tab_dim   = tab_dim
        self.num_cls   = num_classes
        self.fc = nn.Sequential(
            nn.Linear(self.img_dim + self.tab_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_cls)
        )

    def forward(self, img_feat: torch.Tensor, table_feat: torch.Tensor):
        x = torch.cat([img_feat, table_feat], dim=1)
        return self.fc(x)