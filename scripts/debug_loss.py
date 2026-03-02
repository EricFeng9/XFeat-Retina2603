#!/usr/bin/env python
"""
调试脚本：对比训练和验证时的 loss 计算
"""
import sys
sys.path.insert(0, '/data/student/Fengjunming/GlueStick')
sys.path.insert(0, '/data/student/Fengjunming/GlueStick/glue-factory')

import torch
import numpy as np
from torch.utils.data import DataLoader

from gluestick import GLUESTICK_ROOT
from gluestick.models.two_view_pipeline import TwoViewPipeline

from gluefactory.geometry.gt_generation import (
    gt_matches_from_homography,
    gt_line_matches_from_homography,
)
from gluefactory.models.matchers.gluestick import GlueStick as GFGlueStick

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


def compute_loss(pred, batch, loss_model):
    """计算 loss（模拟训练脚本中的逻辑）"""
    H = batch["T_0to1"]
    kpts0 = pred["keypoints0"]
    kpts1 = pred["keypoints1"]
    
    # 点 GT
    gt_points = gt_matches_from_homography(
        kpts0, kpts1, H, pos_th=3.0, neg_th=5.0
    )
    
    # 线 GT
    if "lines0" in pred and "lines1" in pred:
        lines0 = pred["lines0"]
        lines1 = pred["lines1"]
        valid_lines0 = pred.get("valid_lines0", torch.ones_like(lines0[..., 0], dtype=torch.bool))
        valid_lines1 = pred.get("valid_lines1", torch.ones_like(lines1[..., 0], dtype=torch.bool))
        
        line_assignment, line_m0, line_m1 = gt_line_matches_from_homography(
            lines0, lines1, valid_lines0, valid_lines1,
            batch["image0"].shape, batch["image1"].shape, H,
            npts=50, dist_th=5, overlap_th=0.2, min_visibility_th=0.5,
        )
    else:
        bsize = kpts0.shape[0]
        line_assignment = torch.zeros(bsize, 0, 0, dtype=torch.bool, device=kpts0.device)
        line_m0 = torch.full((bsize, 0), -1, device=kpts0.device, dtype=torch.long)
        line_m1 = torch.full((bsize, 0), -1, device=kpts0.device, dtype=torch.long)
    
    data_for_loss = {
        "keypoints0": kpts0,
        "keypoints1": kpts1,
        "lines0": pred.get("lines0", torch.zeros_like(line_m0[..., None, None])),
        "lines1": pred.get("lines1", torch.zeros_like(line_m1[..., None, None])),
        "gt_assignment": gt_points["assignment"],
        "gt_matches0": gt_points["matches0"],
        "gt_matches1": gt_points["matches1"],
        "gt_line_assignment": line_assignment,
        "gt_line_matches0": line_m0,
        "gt_line_matches1": line_m1,
    }
    
    losses, _ = loss_model.loss(pred, data_for_loss)
    return losses, gt_points


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
    
    # 构建 loss 模块
    loss_model_conf = {
        "input_dim": 256,
        "descriptor_dim": 256,
        "weights": None,
        "loss": {
            "nll_weight": 1.0,
            "nll_balancing": 0.5,
        },
    }
    loss_model = GFGlueStick(loss_model_conf).to(device)
    
    # 加载数据集
    print("\n加载数据集...")
    train_dataset = CFFADataset(
        root_dir='/data/student/Fengjunming/LoFTR/data/operation_pre_filtered_cffa',
        split='train',
        img_size=512
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    # 获取第一个 batch
    batch_data = next(iter(train_loader))
    batch = process_real_batch(batch_data, device)
    
    print("\n" + "="*60)
    print("测试 1: model.eval() 模式")
    print("="*60)
    model.eval()
    with torch.no_grad():
        pred_eval = model(batch)
        losses_eval, gt_eval = compute_loss(pred_eval, batch, loss_model)
    
    print(f"\nlog_assignment shape: {pred_eval['log_assignment'].shape}")
    print(f"gt_assignment shape: {gt_eval['assignment'].shape}")
    print(f"\nLosses (eval mode):")
    print(f"  total: {losses_eval['total'].item():.4f}")
    if 'assignment_nll' in losses_eval:
        print(f"  assignment_nll: {losses_eval['assignment_nll'].item():.4f}")
    if 'line_assignment_nll' in losses_eval:
        print(f"  line_assignment_nll: {losses_eval['line_assignment_nll'].item():.4f}")
    
    # 检查 log_assignment 的值
    log_a = pred_eval['log_assignment']
    print(f"\nlog_assignment 统计:")
    print(f"  min: {log_a.min().item():.4f}")
    print(f"  max: {log_a.max().item():.4f}")
    print(f"  mean: {log_a.mean().item():.4f}")
    
    # 检查 dustbin
    print(f"\ndustbin 列 (最后一列) 的 log 概率:")
    print(f"  mean: {log_a[:, :, -1].mean().item():.4f}")
    print(f"dustbin 行 (最后一行) 的 log 概率:")
    print(f"  mean: {log_a[:, -1, :].mean().item():.4f}")
    
    print("\n" + "="*60)
    print("测试 2: model.train() 模式 (extractor.eval(), BatchNorm.eval())")
    print("="*60)
    model.train()
    if hasattr(model, 'extractor'):
        model.extractor.eval()
        for param in model.extractor.parameters():
            param.requires_grad = False
    
    # 【关键修复】冻结所有 BatchNorm 层
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.eval()
    
    # 不使用 no_grad，模拟训练
    pred_train = model(batch)
    losses_train, gt_train = compute_loss(pred_train, batch, loss_model)
    
    print(f"\nlog_assignment shape: {pred_train['log_assignment'].shape}")
    print(f"gt_assignment shape: {gt_train['assignment'].shape}")
    print(f"\nLosses (train mode):")
    print(f"  total: {losses_train['total'].item():.4f}")
    if 'assignment_nll' in losses_train:
        print(f"  assignment_nll: {losses_train['assignment_nll'].item():.4f}")
    if 'line_assignment_nll' in losses_train:
        print(f"  line_assignment_nll: {losses_train['line_assignment_nll'].item():.4f}")
    
    # 检查 log_assignment 的值
    log_a = pred_train['log_assignment']
    print(f"\nlog_assignment 统计:")
    print(f"  min: {log_a.min().item():.4f}")
    print(f"  max: {log_a.max().item():.4f}")
    print(f"  mean: {log_a.mean().item():.4f}")
    
    # 对比两者
    print("\n" + "="*60)
    print("对比 eval 和 train 模式")
    print("="*60)
    print(f"log_assignment 差异:")
    diff = (pred_eval['log_assignment'] - pred_train['log_assignment']).abs()
    print(f"  max diff: {diff.max().item():.6f}")
    print(f"  mean diff: {diff.mean().item():.6f}")
    
    print(f"\n预测的 matches0 相同？{torch.equal(pred_eval['matches0'], pred_train['matches0'])}")
    
    # 检查 keypoints 是否相同
    kp_diff = (pred_eval['keypoints0'] - pred_train['keypoints0']).abs().max()
    print(f"keypoints0 差异: {kp_diff.item():.6f}")
    
    print("\n" + "="*60)
    print("测试 3: 模拟训练一步后再计算 loss (with BatchNorm fix)")
    print("="*60)
    
    # 用第一个 batch 做一次前向+反向
    model.train()
    if hasattr(model, 'extractor'):
        model.extractor.eval()
    # 冻结 BatchNorm
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.eval()
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-6
    )
    
    # 第一次前向
    pred1 = model(batch)
    losses1, _ = compute_loss(pred1, batch, loss_model)
    loss1 = losses1['total'].mean()
    
    print(f"\n第一次前向 (训练前):")
    print(f"  total loss: {loss1.item():.4f}")
    
    # 反向传播并更新
    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()
    
    # 第二次前向
    pred2 = model(batch)
    losses2, _ = compute_loss(pred2, batch, loss_model)
    loss2 = losses2['total'].mean()
    
    print(f"\n第二次前向 (训练后一步):")
    print(f"  total loss: {loss2.item():.4f}")
    print(f"  loss 变化: {loss2.item() - loss1.item():.4f}")
    
    # 检查参数是否变化
    print("\n检查参数变化...")
    total_change = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_change += param.grad.abs().sum().item()
    print(f"  总梯度绝对值和: {total_change:.4f}")


if __name__ == '__main__':
    main()
