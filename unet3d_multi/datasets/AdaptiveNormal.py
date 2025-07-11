import os
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
)
import nibabel as nib
import numpy as np
import torch
from os.path import join

import os


def adaptive_normal(img):  # 定义一个名为adaptive_normal的函数，接收一个图像数组作为输入参数
    min_p = 0.001  # 设置最小分位点（1%的像素值）
    max_p = 0.999  # 设置最大分位点（99%的像素值），这些分位数决定了归一化的范围

    imgArray = img  # 将输入图像赋值给imgArray变量，作为后续的处理对象
    imgPixel = imgArray[imgArray >= 0]  # 过滤掉图像中小于0的像素值（可能是无效值或背景）
    imgPixel, _ = torch.sort(imgPixel)  # 对过滤后的像素值进行排序，_表示不使用排序后的索引

    # 计算最小分位数值对应的像素索引
    index = int(round(len(imgPixel) - 1) * min_p + 0.5)  # 计算最小分位数的位置
    if index < 0:  # 防止索引越界
        index = 0
    if index > (len(imgPixel) - 1):  # 防止索引越界
        index = len(imgPixel) - 1
    value_min = imgPixel[index]  # 获取最小分位值

    # 计算最大分位数值对应的像素索引
    index = int(round(len(imgPixel) - 1) * max_p + 0.5)  # 计算最大分位数的位置
    if index < 0:  # 防止索引越界
        index = 0
    if index > (len(imgPixel) - 1):  # 防止索引越界
        index = len(imgPixel) - 1
    value_max = imgPixel[index]  # 获取最大分位值

    mean = (value_max + value_min) / 2.0  # 计算归一化的均值，即最大值与最小值的平均值
    stddev = (value_max - value_min) / 2.0  # 计算归一化的标准差，即最大值与最小值的差的一半

    imgArray = (imgArray - mean) / stddev  # 对图像数组进行归一化，使其均值为0，标准差为1
    imgArray[imgArray < -1] = -1.0  # 将归一化后小于-1的像素值限制为-1
    imgArray[imgArray > 1] = 1.0  # 将归一化后大于1的像素值限制为1

    return imgArray  # 返回归一化后的图像数组
