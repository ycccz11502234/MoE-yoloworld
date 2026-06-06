#!/usr/bin/env python3
"""
使用ultralytics内置功能下载和转换Objects365数据集
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置数据集目录
from ultralytics.utils import SETTINGS
DATASETS_ROOT = "/apdcephfs_gy8/share_304660203/hunyuan/yccczwang/dev/dataset"
SETTINGS.update(datasets_dir=DATASETS_ROOT)

from ultralytics.data.utils import check_det_dataset

def main():
    # 使用ultralytics内置的检查功能
    # 这会自动下载缺失的文件并生成标签
    print("Checking and downloading Objects365 dataset...")
    
    # 检查数据集，这会触发下载和标签生成
    data_info = check_det_dataset("Objects365.yaml")
    
    print("Dataset check completed!")
    print(f"Training images: {data_info.get('train', {}).get('nc', 'N/A')} classes")
    print(f"Validation images: {data_info.get('val', {}).get('nc', 'N/A')} classes")

if __name__ == "__main__":
    main()