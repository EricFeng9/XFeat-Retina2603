import sys
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import pprint
from pathlib import Path
from loguru import logger
import cv2
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.strategies import DDPStrategy
import logging
from types import SimpleNamespace

# 添加父目录到 sys.path 以支持导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# 导入 XFeat 相关模块
from modules.xfeat import XFeat

# 导入真实数据集（用于测试）
from dataset.CFFA.cffa_dataset import CFFADataset

# 导入指标计算模块（与 train_onPureGen_v2.py 对齐，使用 v2）
from scripts.v1.metrics import (
    compute_homography_errors, 
    aggregate_metrics,
    set_metrics_verbose
)

# ==========================================
# 配置函数（与 train_onPureGen_v2.py 完全一致）
# ==========================================
def get_default_config():
    """获取默认配置（适配XFeat）"""
    conf = SimpleNamespace()
    conf.TRAINER = SimpleNamespace()
    conf.TRAINER.CANONICAL_BS = 4
    conf.TRAINER.CANONICAL_LR = 1e-4
    conf.TRAINER.TRUE_LR = 1e-4
    conf.TRAINER.RANSAC_PIXEL_THR = 3.0
    conf.TRAINER.SEED = 66
    conf.TRAINER.WORLD_SIZE = 1
    conf.TRAINER.TRUE_BATCH_SIZE = 4
    conf.TRAINER.PLOT_MODE = 'evaluation'
    
    # XFeat 配置
    conf.XFEAT = {
        'top_k': 4096,  # 最大关键点数量
        'detection_threshold': 0.05,  # 检测阈值
        'weights': 'weights/xfeat.pt',  # 预训练权重路径
        'min_cossim': 0.82,  # MNN匹配的最小余弦相似度阈值
    }
    
    return conf

# ==========================================
# 工具函数
# ==========================================
def is_valid_homography(H, scale_min=0.1, scale_max=10.0, perspective_threshold=0.005):
    """单应矩阵防爆锁"""
    if H is None:
        return False
    if np.isnan(H).any() or np.isinf(H).any():
        return False
    
    det = np.linalg.det(H[:2, :2])
    if det < scale_min or det > scale_max:
        return False
    
    if abs(H[2, 0]) > perspective_threshold or abs(H[2, 1]) > perspective_threshold:
        return False
    
    return True

def filter_valid_area(img1, img2):
    """筛选有效区域：只保留两张图片都不为纯黑像素的部分"""
    assert img1.shape[:2] == img2.shape[:2], "两张图片的尺寸必须一致"
    if len(img1.shape) == 3:
        mask1 = np.any(img1 > 10, axis=2)
    else:
        mask1 = img1 > 0
    if len(img2.shape) == 3:
        mask2 = np.any(img2 > 10, axis=2)
    else:
        mask2 = img2 > 0
    valid_mask = mask1 & mask2
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return img1, img2
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    filtered_img1 = img1[row_min:row_max+1, col_min:col_max+1].copy()
    filtered_img2 = img2[row_min:row_max+1, col_min:col_max+1].copy()
    valid_mask_cropped = valid_mask[row_min:row_max+1, col_min:col_max+1]
    filtered_img1[~valid_mask_cropped] = 0
    filtered_img2[~valid_mask_cropped] = 0
    return filtered_img1, filtered_img2

def compute_corner_error(H_est, H_gt, height, width):
    """计算四个角点的平均重投影误差（MACE）"""
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    corners_homo = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1)
    corners_gt_homo = (H_gt @ corners_homo.T).T
    corners_gt = corners_gt_homo[:, :2] / (corners_gt_homo[:, 2:] + 1e-6)
    corners_est_homo = (H_est @ corners_homo.T).T
    corners_est = corners_est_homo[:, :2] / (corners_est_homo[:, 2:] + 1e-6)
    try:
        errors = np.sqrt(np.sum((corners_est - corners_gt)**2, axis=1))
        mace = np.mean(errors)
    except:
        mace = float('inf')
    return mace

def create_chessboard(img1, img2, grid_size=4):
    """创建棋盘格对比图"""
    H, W = img1.shape
    cell_h = H // grid_size
    cell_w = W // grid_size
    chessboard = np.zeros((H, W), dtype=img1.dtype)
    for i in range(grid_size):
        for j in range(grid_size):
            y_start, y_end = i * cell_h, (i + 1) * cell_h
            x_start, x_end = j * cell_w, (j + 1) * cell_w
            if (i + j) % 2 == 0:
                chessboard[y_start:y_end, x_start:x_end] = img1[y_start:y_end, x_start:x_end]
            else:
                chessboard[y_start:y_end, x_start:x_end] = img2[y_start:y_end, x_start:x_end]
    return chessboard

# ==========================================
# 共享指标计算函数（可被 train 脚本导入）
# ==========================================
def compute_auc_rop(s_error):
    """
    计算 mAUC (mean Area Under Curve)
    基于 mean_auc.md 中描述的算法：
    对每个整数像素阈值 [1, 25]，计算误差小于阈值的样本比例
    mAUC = 累加比例 / (25 * 100)，范围 [0, 1]
    """
    s_error = np.array(s_error)
    if len(s_error) == 0:
        return 0.0
    limit = 25
    accum_s = 0
    for i in range(1, limit + 1):
        accum_s += np.sum(s_error < i) * 100 / len(s_error)
    return accum_s / (limit * 100)

def compute_batch_registration_metrics(batch, H_ests):
    """
    计算一个 batch 内每个样本的配准指标（共享函数，可被 train 脚本导入）。
    
    Returns:
        list of dicts, 每个 dict 包含:
            - mse, rmse, mace: float or None (匹配失败时为 None)
            - match_success: bool
            - corner_error: float (匹配失败时为 1e6)
    """
    batch_size = batch['image0'].shape[0]
    Ts_gt = batch['T_0to1'].cpu().numpy()
    results = []
    
    for i in range(batch_size):
        H_est = H_ests[i]
        
        # 判断匹配是否成功
        originally_identity = np.allclose(H_est, np.eye(3), atol=1e-6)
        valid_H = is_valid_homography(H_est)
        match_success = (not originally_identity) and valid_H
        
        if not valid_H:
            H_est = np.eye(3)
        
        img0 = (batch['image0'][i, 0].cpu().numpy() * 255).astype(np.uint8)
        img1 = (batch['image1'][i, 0].cpu().numpy() * 255).astype(np.uint8)
        img1_gt = (batch['image1_gt'][i, 0].cpu().numpy() * 255).astype(np.uint8)
        
        h, w = img0.shape
        
        if match_success:
            mace = compute_corner_error(H_est, Ts_gt[i], h, w)
            try:
                H_inv = np.linalg.inv(H_est)
                img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
            except:
                img1_result = img1.copy()
            try:
                res_f, orig_f = filter_valid_area(img1_result, img1_gt)
                mask = (res_f > 0)
                mse = np.mean((res_f[mask].astype(np.float64) - orig_f[mask].astype(np.float64))**2) if np.any(mask) else 0.0
                rmse = np.sqrt(mse)
            except:
                mse = 0.0
                rmse = 0.0
            results.append({'mse': mse, 'rmse': rmse, 'mace': mace,
                            'match_success': True, 'corner_error': mace})
        else:
            results.append({'mse': None, 'rmse': None, 'mace': None,
                            'match_success': False, 'corner_error': 1e6})
    
    return results

def aggregate_epoch_metrics(sample_metrics, epoch_t_errs):
    """
    聚合所有样本的指标，计算 epoch 级别的汇总（共享函数，可被 train 脚本导入）。
    """
    from scripts.v1.metrics import error_auc
    
    total = len(sample_metrics)
    matched = [m for m in sample_metrics if m['match_success']]
    matched_count = len(matched)
    failed_count = total - matched_count
    failure_rate = failed_count / total if total > 0 else 0.0
    
    # MSE/RMSE/MACE 只在匹配成功的样本上计算
    avg_mse = float(np.mean([m['mse'] for m in matched])) if matched else 0.0
    avg_rmse = float(np.mean([m['rmse'] for m in matched])) if matched else 0.0
    avg_mace = float(np.mean([m['mace'] for m in matched])) if matched else float('inf')
    
    # mAUC 在所有样本上计算（匹配失败的样本 corner_error = 1e6）
    corner_errors = [m['corner_error'] for m in sample_metrics]
    mauc = compute_auc_rop(corner_errors) if corner_errors else 0.0
    
    # AUC from t_errs
    if len(epoch_t_errs) > 0:
        auc_dict = error_auc(epoch_t_errs, [5, 10, 20])
    else:
        auc_dict = {'auc@5': 0.0, 'auc@10': 0.0, 'auc@20': 0.0}
    
    auc5 = auc_dict.get('auc@5', 0.0)
    auc10 = auc_dict.get('auc@10', 0.0)
    auc20 = auc_dict.get('auc@20', 0.0)
    combined_auc = (auc5 + auc10 + auc20) / 3.0
    inverse_mace = 1.0 / (1.0 + avg_mace)
    
    return {
        'mse': avg_mse, 'rmse': avg_rmse, 'mace': avg_mace,
        'mauc': mauc, 'failure_rate': failure_rate,
        'matched_count': matched_count, 'failed_count': failed_count, 'total_count': total,
        'auc@5': auc5, 'auc@10': auc10, 'auc@20': auc20,
        'combined_auc': combined_auc, 'inverse_mace': inverse_mace,
    }

# ==========================================
# 辅助类: RealDatasetWrapper (格式转换，与 train_onPureGen_v2.py 一致)
# ==========================================
class RealDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1 = self.base_dataset[idx]
        # 数据集返回的已是归一化到 [0, 1] 的 fix，和 [-1, 1] 的 moving
        moving_original_tensor = (moving_original_tensor + 1) / 2
        moving_gt_tensor = (moving_gt_tensor + 1) / 2
        
        # 转换为灰度图 [1, H, W]
        if fix_tensor.shape[0] == 3:
            fix_gray = 0.299 * fix_tensor[0] + 0.587 * fix_tensor[1] + 0.114 * fix_tensor[2]
            fix_gray = fix_gray.unsqueeze(0)
        else:
            fix_gray = fix_tensor
            
        if moving_gt_tensor.shape[0] == 3:
            moving_gray = 0.299 * moving_gt_tensor[0] + 0.587 * moving_gt_tensor[1] + 0.114 * moving_gt_tensor[2]
            moving_gray = moving_gray.unsqueeze(0)
        else:
            moving_gray = moving_gt_tensor
            
        if moving_original_tensor.shape[0] == 3:
            moving_orig_gray = 0.299 * moving_original_tensor[0] + 0.587 * moving_original_tensor[1] + 0.114 * moving_original_tensor[2]
            moving_orig_gray = moving_orig_gray.unsqueeze(0)
        else:
            moving_orig_gray = moving_original_tensor
        
        fix_name = os.path.basename(fix_path)
        moving_name = os.path.basename(moving_path)
        
        # 数据集内部计算的 T_0to1 是从 Moving 到 Fix 的变换
        # GlueStick 输出是从 Image0(Fix) -> Image1(Moving) 的变换
        # 所以这里取逆
        try:
            T_fix_to_moving = torch.inverse(T_0to1)
        except:
            T_fix_to_moving = T_0to1
            
        return {
            'image0': fix_gray,
            'image1': moving_orig_gray,
            'image1_gt': moving_gray,
            'T_0to1': T_fix_to_moving,
            'pair_names': (fix_name, moving_name),
            'dataset_name': 'MultiModal'
        }

class MultimodalDataModule(pl.LightningDataModule):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': True
        }

    def setup(self, stage=None):
        # 测试阶段使用真实数据集（CFFA），与 train_onPureGen_v2.py 验证集对齐
        script_dir = Path(__file__).parent.parent.parent
        
        # 测试集：真实数据（路径与 train_onPureGen_v2.py 一致）
        test_data_dir = script_dir / 'dataset' / 'CFFA'
        test_base = CFFADataset(root_dir=str(test_data_dir), split=self.args.split, mode='fa2cf')
        self.test_dataset = RealDatasetWrapper(test_base)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, shuffle=False, **self.loader_params)

# ==========================================
# 核心模型: PL_GlueStick_Gen（与 train_onPureGen_v2.py 完全对齐）
# ==========================================
class PL_XFeat_Gen(pl.LightningModule):
    """XFeat 的 PyTorch Lightning 封装（用于生成数据训练，测试用）"""
    def __init__(self, config, result_dir=None):
        super().__init__()
        self.config = config
        self.result_dir = result_dir
        self.save_hyperparameters({'config': str(config)})
        
        # 初始化 XFeat 模型
        # 获取预训练权重的绝对路径
        script_dir = Path(__file__).parent.parent.parent
        weights_path = script_dir / config.XFEAT['weights']
        
        if weights_path.exists():
            logger.info(f"加载 XFeat 预训练权重: {weights_path}")
            self.xfeat = XFeat(
                weights=str(weights_path),
                top_k=config.XFEAT['top_k'],
                detection_threshold=config.XFEAT['detection_threshold']
            )
        else:
            logger.warning(f"预训练权重不存在: {weights_path}，使用随机初始化")
            self.xfeat = XFeat(
                weights=None,
                top_k=config.XFEAT['top_k'],
                detection_threshold=config.XFEAT['detection_threshold']
            )
        
        # 用于控制是否强制可视化
        self.force_viz = False
        
        # 课程学习权重（测试时不需要，但为了兼容 checkpoint 加载）
        self.vessel_loss_weight = 10.0
        
        # 最小余弦相似度阈值（用于MNN匹配）
        self.min_cossim = config.XFEAT['min_cossim']

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        lr = self.config.TRAINER.TRUE_LR
        optimizer = torch.optim.Adam(self.xfeat.net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "combined_auc",
            },
        }

    def forward(self, batch):
        """前向传播 - 使用 XFeat 提取特征和匹配"""
        # 提取特征
        with torch.no_grad():
            feats0_list = self.xfeat.detectAndCompute(batch['image0'], top_k=self.config.XFEAT['top_k'])
            feats1_list = self.xfeat.detectAndCompute(batch['image1'], top_k=self.config.XFEAT['top_k'])
        
        B = len(feats0_list)
        
        # 将 List[Dict] 转换为批量张量格式
        max_kpts0 = max([len(f['keypoints']) for f in feats0_list])
        max_kpts1 = max([len(f['keypoints']) for f in feats1_list])
        
        device = batch['image0'].device
        
        # 初始化批量张量
        keypoints0 = torch.zeros(B, max_kpts0, 2, device=device)
        keypoints1 = torch.zeros(B, max_kpts1, 2, device=device)
        descriptors0 = torch.zeros(B, max_kpts0, 64, device=device)
        descriptors1 = torch.zeros(B, max_kpts1, 64, device=device)
        scores0 = torch.zeros(B, max_kpts0, device=device)
        scores1 = torch.zeros(B, max_kpts1, device=device)
        
        # 填充数据
        for b in range(B):
            n0 = len(feats0_list[b]['keypoints'])
            n1 = len(feats1_list[b]['keypoints'])
            keypoints0[b, :n0] = feats0_list[b]['keypoints']
            keypoints1[b, :n1] = feats1_list[b]['keypoints']
            descriptors0[b, :n0] = feats0_list[b]['descriptors']
            descriptors1[b, :n1] = feats1_list[b]['descriptors']
            scores0[b, :n0] = feats0_list[b]['scores']
            scores1[b, :n1] = feats1_list[b]['scores']
        
        # 更新 batch
        batch.update({
            'keypoints0': keypoints0,
            'keypoints1': keypoints1,
            'descriptors0': descriptors0,
            'descriptors1': descriptors1,
            'keypoint_scores0': scores0,
            'keypoint_scores1': scores1,
        })
        
        # 使用 XFeat 的 MNN 匹配器进行匹配
        matches_list = self.xfeat.batch_match(descriptors0, descriptors1, min_cossim=self.min_cossim)
        
        # 将匹配结果转换为 matches0 格式 [B, M]，-1 表示无匹配
        matches0 = torch.full((B, max_kpts0), -1, dtype=torch.long, device=device)
        for b, (idx0, idx1) in enumerate(matches_list):
            matches0[b, idx0] = idx1
        
        return {
            'matches0': matches0,
            'keypoints0': keypoints0,
            'keypoints1': keypoints1,
            'descriptors0': descriptors0,
            'descriptors1': descriptors1,
        }

    def _compute_gt_matches(self, kpts0, kpts1, T_0to1, dist_th=3.0):
        """计算几何 Ground Truth 匹配对"""
        B, M, _ = kpts0.shape
        B, N, _ = kpts1.shape
        device = kpts0.device
        
        kpts0_h = torch.cat([kpts0, torch.ones(B, M, 1, device=device)], dim=-1)
        kpts0_warped_h = torch.matmul(kpts0_h, T_0to1.transpose(1, 2))
        kpts0_warped = kpts0_warped_h[..., :2] / (kpts0_warped_h[..., 2:] + 1e-8)
        
        dist = torch.cdist(kpts0_warped, kpts1)
        min_dist, matched_indices = torch.min(dist, dim=-1)
        mask = min_dist < dist_th
        matches_gt = torch.where(mask, matched_indices, torch.tensor(-1, device=device))
        
        return matches_gt

    def _compute_loss(self, outputs, batch):
        """计算损失（基于匹配对的监督学习）"""
        kpts0 = batch['keypoints0']
        kpts1 = batch['keypoints1']
        T_0to1 = batch['T_0to1']
        
        # 计算点匹配的 GT
        matches_gt = self._compute_gt_matches(kpts0, kpts1, T_0to1)
        
        # 获取预测的匹配
        matches_pred = outputs['matches0']  # [B, M]
        
        # 计算匹配损失
        B, M = matches_pred.shape
        device = kpts0.device
        
        descriptors0 = batch['descriptors0']  # [B, M, 64]
        descriptors1 = batch['descriptors1']  # [B, N, 64]
        
        # 过滤有效的关键点（分数>0）
        valid_mask = batch['keypoint_scores0'] > 0
        
        # 计算所有可能的余弦相似度 [B, M, N]
        desc0_norm = F.normalize(descriptors0, dim=-1)
        desc1_norm = F.normalize(descriptors1, dim=-1)
        
        # 计算余弦相似度矩阵
        cossim = torch.bmm(desc0_norm, desc1_norm.transpose(1, 2))  # [B, M, N]
        
        loss = torch.tensor(0.0, device=device)
        valid_count = 0
        
        for b in range(B):
            valid_b = valid_mask[b]
            matches_gt_b = matches_gt[b]
            
            # 只考虑有效的关键点
            valid_indices = torch.where(valid_b)[0]
            
            for idx in valid_indices:
                gt_match = matches_gt_b[idx]
                
                if gt_match >= 0:  # 有GT匹配
                    logits = cossim[b, idx, :]  # [N]
                    target = gt_match
                    
                    # 交叉熵损失
                    loss += F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
                    valid_count += 1
        
        if valid_count > 0:
            loss = loss / valid_count
        
        return loss

    def test_step(self, batch, batch_idx):
        """测试步骤（与 train_onPureGen_v2.py 的 validation_step 完全对齐）"""
        outputs = self(batch)
        
        # 计算测试损失
        loss = self._compute_loss(outputs, batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # 获取预测的匹配对
        matches0 = outputs['matches0']
        kpts0 = batch['keypoints0']
        kpts1 = batch['keypoints1']
        
        B = kpts0.shape[0]
        H_ests = []
        
        mkpts0_f_list = []
        mkpts1_f_list = []
        m_bids_list = []
        
        valid_mask = matches0 > -1
        
        for b in range(B):
            valid_b = valid_mask[b]
            if not valid_b.any():
                H_ests.append(np.eye(3))
                continue
                
            m_indices_0 = torch.where(valid_b)[0]
            m_indices_1 = matches0[b, valid_b]
            
            pts0 = kpts0[b, m_indices_0].cpu().numpy()
            pts1 = kpts1[b, m_indices_1].cpu().numpy()
            
            mkpts0_f_list.append(kpts0[b, m_indices_0])
            mkpts1_f_list.append(kpts1[b, m_indices_1])
            m_bids_list.append(torch.full((len(pts0),), b, dtype=torch.long, device=kpts0.device))
            
            if len(pts0) >= 4:
                try:
                    H, _ = cv2.findHomography(pts0, pts1, cv2.RANSAC, self.config.TRAINER.RANSAC_PIXEL_THR)
                    if H is None:
                        H = np.eye(3)
                except:
                    H = np.eye(3)
            else:
                H = np.eye(3)
            H_ests.append(H)
        
        # 构建 metrics.py 需要的 batch 格式
        metrics_batch = {
            'mkpts0_f': torch.cat(mkpts0_f_list, dim=0) if mkpts0_f_list else torch.empty(0, 2),
            'mkpts1_f': torch.cat(mkpts1_f_list, dim=0) if mkpts1_f_list else torch.empty(0, 2),
            'm_bids': torch.cat(m_bids_list, dim=0) if m_bids_list else torch.empty(0, dtype=torch.long),
            'T_0to1': batch['T_0to1'],
            'image0': batch['image0'],
            'dataset_name': batch['dataset_name']
        }
        
        # 使用 metrics.py 计算指标
        set_metrics_verbose(True)
        compute_homography_errors(metrics_batch, self.config)
        
        # 与 train_onPureGen_v2.py 对齐：逐batch计算AUC并log
        from scripts.v1.metrics import error_auc
        if len(metrics_batch.get('t_errs', [])) > 0:
            auc_dict = error_auc(metrics_batch['t_errs'], [5, 10, 20])
            for k, v in auc_dict.items():
                self.log(k, v, on_epoch=True, prog_bar=False, logger=True)
        
        return {
            'H_est': H_ests,
            'kpts0': kpts0,
            'kpts1': kpts1,
            'matches0': matches0,
            'metrics_batch': metrics_batch
        }

# ==========================================
# 回调逻辑: TestCallback
# ==========================================
class TestCallback(Callback):
    def __init__(self, args, output_dir):
        super().__init__()
        self.args = args
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_mses = []
        self.epoch_maces = []

        import csv
        self.csv_path = self.output_dir / "test_metrics.csv"
        with open(self.csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Batch", "Test Loss", "MSE", "MACE", "AUC@5", "AUC@10", "AUC@20"])

    def on_test_epoch_start(self, trainer, pl_module):
        self.epoch_mses = []
        self.epoch_maces = []
        self.epoch_t_errs = []
        self.batch_metrics = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        batch_mses, batch_maces = self._process_batch(trainer, pl_module, batch, outputs, batch_idx, save_images=True)
        self.epoch_mses.extend(batch_mses)
        self.epoch_maces.extend(batch_maces)

    def on_test_epoch_end(self, trainer, pl_module):
        if not self.epoch_mses:
            logger.warning("没有收集到任何测试指标")
            return
        
        avg_mse = sum(self.epoch_mses) / len(self.epoch_mses)
        avg_mace = sum(self.epoch_maces) / len(self.epoch_maces) if self.epoch_maces else float('inf')
        
        display_metrics = {'mse': avg_mse, 'mace': avg_mace}
        
        # 从 trainer.callback_metrics 获取 AUC（与 train_onPureGen_v2.py 对齐）
        metrics = trainer.callback_metrics
        for k in ['auc@5', 'auc@10', 'auc@20']:
            if k in metrics:
                display_metrics[k] = metrics[k].item()
            else:
                display_metrics[k] = 0.0
        
        if 'test_loss' in metrics:
            display_metrics['test_loss'] = metrics['test_loss'].item()
        
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
        
        # 计算综合 AUC
        auc5 = display_metrics.get('auc@5', 0.0)
        auc10 = display_metrics.get('auc@10', 0.0)
        auc20 = display_metrics.get('auc@20', 0.0)
        combined_auc = (auc5 + auc10 + auc20) / 3.0
        
        inverse_mace = 1.0 / (1.0 + avg_mace)
        
        logger.info(f"测试总结 >> {metric_str} | combined_auc: {combined_auc:.4f} | inverse_mace: {inverse_mace:.6f}")
        
        # 保存总结报告
        summary_path = self.output_dir / "test_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"测试总结\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"测试损失: {display_metrics.get('test_loss', 0.0):.6f}\n")
            f.write(f"MSE: {avg_mse:.6f}\n")
            f.write(f"MACE: {avg_mace:.4f}\n")
            f.write(f"AUC@5: {auc5:.4f}\n")
            f.write(f"AUC@10: {auc10:.4f}\n")
            f.write(f"AUC@20: {auc20:.4f}\n")
            f.write(f"Combined AUC: {combined_auc:.4f}\n")
            f.write(f"Inverse MACE: {inverse_mace:.6f}\n")
        
        logger.info(f"测试总结已保存到: {summary_path}")

    def _process_batch(self, trainer, pl_module, batch, outputs, batch_idx, save_images=False):
        batch_size = batch['image0'].shape[0]
        mses, maces = [], []
        H_ests = outputs.get('H_est', [np.eye(3)] * batch_size)
        Ts_gt = batch['T_0to1'].cpu().numpy()
        
        rejected_count = 0
        
        for i in range(batch_size):
            H_est = H_ests[i]
            
            # 启用防爆锁
            if not is_valid_homography(H_est):
                H_est = np.eye(3)
                rejected_count += 1
            
            img0 = (batch['image0'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1 = (batch['image1'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            img1_gt = (batch['image1_gt'][i, 0].cpu().numpy() * 255).astype(np.uint8)
            
            h, w = img0.shape
            try:
                H_inv = np.linalg.inv(H_est)
                img1_result = cv2.warpPerspective(img1, H_inv, (w, h))
            except:
                img1_result = img1.copy()
            
            try:
                res_f, orig_f = filter_valid_area(img1_result, img1_gt)
                mask = (res_f > 0)
                mse = np.mean((res_f[mask].astype(np.float64) - orig_f[mask].astype(np.float64))**2) if np.any(mask) else 0.0
            except:
                mse = 0.0
            mses.append(mse)
            maces.append(compute_corner_error(H_est, Ts_gt[i], h, w))
            
            if save_images:
                sample_name = f"batch{batch_idx:04d}_sample{i:02d}_{Path(batch['pair_names'][0][i]).stem}_vs_{Path(batch['pair_names'][1][i]).stem}"
                save_path = self.output_dir / sample_name
                save_path.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path / "fix.png"), img0)
                cv2.imwrite(str(save_path / "moving_original.png"), img1)
                cv2.imwrite(str(save_path / "moving_result.png"), img1_result)
                cv2.imwrite(str(save_path / "moving_gt.png"), img1_gt)
                
                # 绘制关键点和匹配（与 train 的 _process_batch 对齐）
                img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
                img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
                
                if 'kpts0' in outputs and 'kpts1' in outputs:
                    kpts0_np = outputs['kpts0'][i].cpu().numpy()
                    kpts1_np = outputs['kpts1'][i].cpu().numpy()
                    
                    # 过滤填充点（分数为0的点）——与 train 对齐
                    scores0 = batch['keypoint_scores0'][i].cpu().numpy()
                    scores1 = batch['keypoint_scores1'][i].cpu().numpy()
                    valid_mask0 = scores0 > 0
                    valid_mask1 = scores1 > 0
                    
                    # 绘制所有有效关键点（白色）
                    for idx, pt in enumerate(kpts0_np):
                        if valid_mask0[idx]:
                            cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
                    for idx, pt in enumerate(kpts1_np):
                        if valid_mask1[idx]:
                            cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
                    
                    # 绘制线段（如果存在）——与 train 对齐
                    if 'lines0' in batch and 'lines1' in batch:
                        lines0 = batch['lines0'][i].cpu().numpy()
                        lines1 = batch['lines1'][i].cpu().numpy()
                        valid_lines0 = batch.get('valid_lines0', None)
                        valid_lines1 = batch.get('valid_lines1', None)
                        
                        if valid_lines0 is not None:
                            valid_lines0 = valid_lines0[i].cpu().numpy()
                        else:
                            valid_lines0 = np.ones(len(lines0), dtype=bool)
                            
                        if valid_lines1 is not None:
                            valid_lines1 = valid_lines1[i].cpu().numpy()
                        else:
                            valid_lines1 = np.ones(len(lines1), dtype=bool)
                        
                        for line_idx, line in enumerate(lines0):
                            if valid_lines0[line_idx]:
                                pt1 = (int(line[0, 0]), int(line[0, 1]))
                                pt2 = (int(line[1, 0]), int(line[1, 1]))
                                cv2.line(img0_color, pt1, pt2, (0, 255, 255), 1)
                        
                        for line_idx, line in enumerate(lines1):
                            if valid_lines1[line_idx]:
                                pt1 = (int(line[0, 0]), int(line[0, 1]))
                                pt2 = (int(line[1, 0]), int(line[1, 1]))
                                cv2.line(img1_color, pt1, pt2, (0, 255, 255), 1)
                    
                    # 绘制匹配点（红色）
                    if 'matches0' in outputs:
                        m0 = outputs['matches0'][i].cpu()
                        valid = m0 > -1
                        m_indices_0 = torch.where(valid)[0].numpy()
                        m_indices_1 = m0[valid].numpy()
                        
                        for idx0 in m_indices_0:
                            if valid_mask0[idx0]:
                                pt = kpts0_np[idx0]
                                cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                        for idx1 in m_indices_1:
                            if valid_mask1[idx1]:
                                pt = kpts1_np[idx1]
                                cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                        
                        # 使用 ConnectionPatch 绘制匹配连线（与 train 对齐）
                        try:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor='black')
                            fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
                            
                            ax1.imshow(img0, cmap='gray')
                            ax1.axis('off')
                            ax2.imshow(img1, cmap='gray')
                            ax2.axis('off')
                            
                            if len(m_indices_0) > 0:
                                from matplotlib.patches import ConnectionPatch
                                for idx0, idx1 in zip(m_indices_0, m_indices_1):
                                    if valid_mask0[idx0] and valid_mask1[idx1]:
                                        pt0 = kpts0_np[idx0]
                                        pt1 = kpts1_np[idx1]
                                        con = ConnectionPatch(
                                            xyA=(pt0[0], pt0[1]), coordsA=ax1.transData,
                                            xyB=(pt1[0], pt1[1]), coordsB=ax2.transData,
                                            color='lime', linewidth=0.5, alpha=0.7
                                        )
                                        fig.add_artist(con)
                            
                            plt.savefig(str(save_path / "matches.png"), dpi=100, facecolor='black',
                                      bbox_inches='tight', pad_inches=0.1)
                            plt.close(fig)
                        except Exception as e:
                            logger.warning(f"绘制匹配图失败: {e}")
                
                cv2.imwrite(str(save_path / "fix_with_kpts.png"), img0_color)
                cv2.imwrite(str(save_path / "moving_with_kpts.png"), img1_color)
                
                try:
                    cb = create_chessboard(img1_result, img0)
                    cv2.imwrite(str(save_path / "chessboard.png"), cb)
                except:
                    pass
                
                # 保存单样本指标
                with open(save_path / "metrics.txt", "w") as f:
                    f.write(f"MSE: {mse:.6f}\n")
                    f.write(f"MACE: {maces[-1]:.4f}\n")
                    f.write(f"Matches: {len(m_indices_0) if 'matches0' in outputs else 0}\n")
        
        if rejected_count > 0:
            logger.info(f"防爆锁触发: {rejected_count}/{batch_size} 个样本的单应矩阵被重置为单位矩阵")
        
        return mses, maces

# ==========================================
# 参数解析和主函数
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="XFeat Gen-Data Testing (aligned with train_onPureGen_v2.py)")
    parser.add_argument('--name', '-n', type=str, required=True, help='训练模型名称（用于定位checkpoint）')
    parser.add_argument('--test_name', type=str, default='test_results', help='测试名称（用于指定输出子目录）')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点路径（默认使用best_checkpoint）')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='测试数据集划分')
    return parser.parse_args()

def main():
    args = parse_args()
    args.mode = 'gen'  # 固定为生成数据模式
    
    config = get_default_config()
    config.TRAINER.SEED = 66
    pl.seed_everything(config.TRAINER.SEED)
    
    # 确定checkpoint路径
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        # 默认使用best_checkpoint
        ckpt_path = Path(f"results/xfeat_{args.mode}/{args.name}/best_checkpoint/model.ckpt")
    
    if not ckpt_path.exists():
        logger.error(f"检查点不存在: {ckpt_path}")
        logger.info(f"请确保训练模型存在，或使用 --checkpoint 指定有效的检查点路径")
        return
    
    logger.info(f"加载检查点: {ckpt_path}")
    
    # 设置输出目录
    output_dir = Path(f"results/xfeat_{args.mode}/{args.name}/{args.test_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "test_log.txt"
    
    # 配置日志
    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, format=log_format, level="INFO")
    logger.add(log_file, format=log_format, level="INFO", mode="w", backtrace=True, diagnose=False)
    logger.info(f"日志将同时保存到: {log_file}")
    
    # 设置环境变量，让 metrics.py 也写入日志文件
    os.environ['LOFTR_LOG_FILE'] = str(log_file)
    
    # GPU 配置
    if ',' in str(args.gpus):
        gpus_list = [int(x) for x in args.gpus.split(',')]
        _n_gpus = len(gpus_list)
    else:
        try:
            gpus_list = [int(args.gpus)]
            _n_gpus = 1
        except:
            gpus_list = 'auto'
            _n_gpus = 1
    
    config.TRAINER.WORLD_SIZE = max(_n_gpus, 1)
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    
    logger.info(f"GPU 配置: devices={gpus_list}, num_gpus={_n_gpus}")
    logger.info(f"测试名称: {args.test_name}")
    logger.info(f"测试数据集划分: {args.split}")
    logger.info(f"输出目录: {output_dir}")
    
    # 从检查点加载模型
    # 确定 map_location
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda(0)
    else:
        map_location = 'cpu'
    
    model = PL_XFeat_Gen.load_from_checkpoint(
        str(ckpt_path),
        config=config,
        result_dir=str(output_dir),
        map_location=map_location
    )
    model.eval()
    
    # 初始化数据模块
    data_module = MultimodalDataModule(args, config)
    
    # TensorBoard 日志
    tb_logger = TensorBoardLogger(save_dir='logs/tb_logs', name=f"xfeat_test_{args.name}_{args.test_name}")
    
    # Trainer 配置
    trainer_kwargs = {
        'accelerator': "gpu" if torch.cuda.is_available() else "cpu",
        'devices': gpus_list,
        'callbacks': [TestCallback(args, output_dir)],
        'logger': tb_logger,
    }
    
    # 只有在多 GPU 时才添加 strategy
    if _n_gpus > 1:
        trainer_kwargs['strategy'] = DDPStrategy(find_unused_parameters=False)
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    logger.info(f"开始测试 (模型: {args.name} | 测试集: CFFA 真实数据 {args.split} split)")
    trainer.test(model, datamodule=data_module)
    
    logger.info(f"测试完成! 结果已保存到: {output_dir}")

if __name__ == '__main__':
    main()


