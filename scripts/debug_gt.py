#!/usr/bin/env python
"""
调试脚本：检查 GT 计算是否正确
"""
import sys
sys.path.insert(0, '/data/student/Fengjunming/GlueStick')
sys.path.insert(0, '/data/student/Fengjunming/GlueStick/glue-factory')

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入模型
from gluestick import GLUESTICK_ROOT
from gluestick.models.two_view_pipeline import TwoViewPipeline

# 导入 GlueFactory GT 计算
from gluefactory.geometry.gt_generation import (
    gt_matches_from_homography,
    gt_line_matches_from_homography,
)

# 导入数据集
from dataset.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset

def process_real_batch(batch_data, device):
    """处理真实数据 batch"""
    fix_tensor, moving_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1 = batch_data
    
    if fix_tensor.dim() == 3:
        fix_tensor = fix_tensor.unsqueeze(0)
        moving_tensor = moving_tensor.unsqueeze(0)
        T_0to1 = T_0to1.unsqueeze(0)
    
    if fix_tensor.shape[1] == 3:
        fix_gray = (fix_tensor[:, 0:1] * 0.299 + 
                   fix_tensor[:, 1:2] * 0.587 + 
                   fix_tensor[:, 2:3] * 0.114)
    else:
        fix_gray = fix_tensor
    
    if moving_tensor.shape[1] == 3:
        moving_gray = (moving_tensor[:, 0:1] * 0.299 + 
                      moving_tensor[:, 1:2] * 0.587 + 
                      moving_tensor[:, 2:3] * 0.114)
    else:
        moving_gray = moving_tensor
    
    batch = {
        'image0': fix_gray.to(device),
        'image1': moving_gray.to(device),
        'T_0to1': T_0to1.to(device),
    }
    
    return batch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 加载模型
    print("\n加载模型...")
    pretrained_path = str(GLUESTICK_ROOT / "resources/weights/checkpoint_GlueStick_MD.tar")
    model_conf = {
        "name": "two_view_pipeline",
        "extractor": {
            "name": "wireframe",
            "trainable": False,
            "sp_params": {
                "has_detector": True,
                "has_descriptor": True,
                "descriptor_dim": 256,
                "nms_radius": 4,
                "detection_threshold": 0.005,
                "max_num_keypoints": 1000,
                "force_num_keypoints": True,
                "remove_borders": 4,
                "return_all": True,
            },
            "wireframe_params": {
                "merge_points": True,
                "merge_line_endpoints": True,
                "nms_radius": 3,
                "max_n_junctions": 500,
            },
            "max_n_lines": 250,
            "min_length": 15,
        },
        "matcher": {
            "name": "gluestick",
            "trainable": True,
            "weights": pretrained_path,
            "input_dim": 256,
            "descriptor_dim": 256,
            "GNN_layers": ["self", "cross"] * 9,
            "num_line_iterations": 1,
            "line_attention": False,
            "filter_threshold": 0.2,
            "checkpointed": False,
        },
        "filter": {"name": None},
        "solver": {"name": None},
    }
    model = TwoViewPipeline(model_conf).to(device)
    model.eval()
    
    # 加载数据集
    print("\n加载数据集...")
    train_dataset = CFFADataset(
        root_dir='/data/student/Fengjunming/LoFTR/data/operation_pre_filtered_cffa',
        split='train',
        img_size=512
    )
    val_dataset = CFFADataset(
        root_dir='/data/student/Fengjunming/LoFTR/data/operation_pre_filtered_cffa',
        split='val',
        img_size=512
    )
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    def analyze_dataset(loader, name):
        print(f"\n{'='*60}")
        print(f"分析 {name} 数据集 ({len(loader)} 样本)")
        print(f"{'='*60}")
        
        all_num_pos = []
        all_num_neg0 = []
        all_num_neg1 = []
        all_num_kpts0 = []
        all_num_kpts1 = []
        all_num_ignored = []
        
        # 检查 H 矩阵
        identity_count = 0
        
        for batch_data in tqdm(loader, desc=f"Processing {name}"):
            batch = process_real_batch(batch_data, device)
            H = batch['T_0to1']
            
            # 检查是否为单位矩阵
            is_identity = torch.allclose(H, torch.eye(3, device=device).unsqueeze(0), atol=1e-5)
            if is_identity:
                identity_count += 1
            
            with torch.no_grad():
                pred = model(batch)
            
            kpts0 = pred['keypoints0']  # [B, N, 2]
            kpts1 = pred['keypoints1']  # [B, M, 2]
            
            all_num_kpts0.append(kpts0.shape[1])
            all_num_kpts1.append(kpts1.shape[1])
            
            # 计算 GT
            gt_points = gt_matches_from_homography(
                kpts0, kpts1, H, pos_th=3.0, neg_th=5.0
            )
            
            assignment = gt_points['assignment']  # [B, N, M]
            matches0 = gt_points['matches0']      # [B, N]
            matches1 = gt_points['matches1']      # [B, M]
            
            num_pos = assignment.sum().item()
            num_neg0 = (matches0 == -1).sum().item()
            num_neg1 = (matches1 == -1).sum().item()
            num_ignored0 = (matches0 == -2).sum().item()
            num_ignored1 = (matches1 == -2).sum().item()
            
            all_num_pos.append(num_pos)
            all_num_neg0.append(num_neg0)
            all_num_neg1.append(num_neg1)
            all_num_ignored.append(num_ignored0 + num_ignored1)
        
        print(f"\n统计结果:")
        print(f"  单位矩阵 H 数量: {identity_count}/{len(loader)}")
        print(f"  平均 keypoints0: {np.mean(all_num_kpts0):.1f}")
        print(f"  平均 keypoints1: {np.mean(all_num_kpts1):.1f}")
        print(f"  平均正样本数 (匹配): {np.mean(all_num_pos):.1f}")
        print(f"  平均负样本0 (未匹配): {np.mean(all_num_neg0):.1f}")
        print(f"  平均负样本1 (未匹配): {np.mean(all_num_neg1):.1f}")
        print(f"  平均忽略样本: {np.mean(all_num_ignored):.1f}")
        print(f"  正样本比例: {np.mean(all_num_pos) / (np.mean(all_num_kpts0) + 1e-8) * 100:.2f}%")
        
        # 打印详细分布
        print(f"\n  正样本分布:")
        print(f"    min: {min(all_num_pos)}, max: {max(all_num_pos)}, median: {np.median(all_num_pos)}")
        
        return all_num_pos, all_num_neg0, all_num_neg1
    
    train_stats = analyze_dataset(train_loader, "训练集")
    val_stats = analyze_dataset(val_loader, "验证集")
    
    # 额外检查：打印第一个样本的详细信息
    print("\n" + "="*60)
    print("第一个训练样本的详细分析")
    print("="*60)
    
    batch_data = next(iter(train_loader))
    batch = process_real_batch(batch_data, device)
    
    H = batch['T_0to1']
    print(f"\n单应性矩阵 H:\n{H.squeeze().cpu().numpy()}")
    
    with torch.no_grad():
        pred = model(batch)
    
    kpts0 = pred['keypoints0']
    kpts1 = pred['keypoints1']
    
    print(f"\nkeypoints0 shape: {kpts0.shape}")
    print(f"keypoints1 shape: {kpts1.shape}")
    
    if kpts0.shape[1] > 0:
        print(f"keypoints0 范围: x=[{kpts0[..., 0].min():.1f}, {kpts0[..., 0].max():.1f}], y=[{kpts0[..., 1].min():.1f}, {kpts0[..., 1].max():.1f}]")
    if kpts1.shape[1] > 0:
        print(f"keypoints1 范围: x=[{kpts1[..., 0].min():.1f}, {kpts1[..., 0].max():.1f}], y=[{kpts1[..., 1].min():.1f}, {kpts1[..., 1].max():.1f}]")
    
    # 计算 GT
    gt_points = gt_matches_from_homography(
        kpts0, kpts1, H, pos_th=3.0, neg_th=5.0
    )
    
    proj_0to1 = gt_points['proj_0to1']  # kpts0 投影到 image1
    proj_1to0 = gt_points['proj_1to0']  # kpts1 投影到 image0
    
    print(f"\nproj_0to1 范围: x=[{proj_0to1[..., 0].min():.1f}, {proj_0to1[..., 0].max():.1f}], y=[{proj_0to1[..., 1].min():.1f}, {proj_0to1[..., 1].max():.1f}]")
    print(f"proj_1to0 范围: x=[{proj_1to0[..., 0].min():.1f}, {proj_1to0[..., 0].max():.1f}], y=[{proj_1to0[..., 1].min():.1f}, {proj_1to0[..., 1].max():.1f}]")
    
    # 检查投影是否在合理范围内
    valid_proj_0to1 = (proj_0to1[..., 0] >= 0) & (proj_0to1[..., 0] < 512) & \
                      (proj_0to1[..., 1] >= 0) & (proj_0to1[..., 1] < 512)
    valid_proj_1to0 = (proj_1to0[..., 0] >= 0) & (proj_1to0[..., 0] < 512) & \
                      (proj_1to0[..., 1] >= 0) & (proj_1to0[..., 1] < 512)
    
    print(f"\n投影后在图像范围内的比例:")
    print(f"  proj_0to1: {valid_proj_0to1.float().mean() * 100:.1f}%")
    print(f"  proj_1to0: {valid_proj_1to0.float().mean() * 100:.1f}%")
    
    assignment = gt_points['assignment']
    matches0 = gt_points['matches0']
    
    print(f"\n匹配统计:")
    print(f"  正样本数: {assignment.sum().item()}")
    print(f"  负样本数 (unmatch): {(matches0 == -1).sum().item()}")
    print(f"  忽略样本数 (ignore): {(matches0 == -2).sum().item()}")

if __name__ == '__main__':
    main()
