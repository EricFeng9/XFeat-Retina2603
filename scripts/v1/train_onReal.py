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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
import logging
from types import SimpleNamespace

# 添加父目录到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# 导入 XFeat 相关模块
from modules.xfeat import XFeat
from modules.interpolator import InterpolateSparse2d
import torch.nn.functional as F

# 导入数据集
from dataset.operation_pre_filtered_cffa.operation_pre_filtered_cffa_dataset import CFFADataset as CFFADataset_Filtered
from dataset.CFFA.cffa_dataset import CFFADataset

# 导入指标计算
from scripts.v1.metrics import (
    compute_homography_errors, 
    aggregate_metrics,
    set_metrics_verbose
)

# ==========================================
# 配置函数
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
        'top_k': 4096,
        'detection_threshold': 0.05,
        'weights': 'weights/xfeat.pt',
        'min_cossim': 0.82,
    }
    
    return conf

# ==========================================
# 工具函数
# ==========================================
def is_valid_homography(H, scale_min=0.1, scale_max=10.0, perspective_threshold=0.005):
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
    assert img1.shape[:2] == img2.shape[:2]
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
# 数据集包装类
# ==========================================
class RealDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        fix_tensor, moving_original_tensor, moving_gt_tensor, fix_path, moving_path, T_0to1 = self.base_dataset[idx]
        moving_original_tensor = (moving_original_tensor + 1) / 2
        moving_gt_tensor = (moving_gt_tensor + 1) / 2
        
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
        if stage == 'fit' or stage is None:
            script_dir = Path(__file__).parent.parent.parent
            
            # 训练集：使用 operation_pre_filtered_cffa
            train_data_dir = script_dir / 'dataset' / 'operation_pre_filtered_cffa'
            train_base = CFFADataset_Filtered(root_dir=str(train_data_dir), split='train', mode='fa2cf')
            self.train_dataset = RealDatasetWrapper(train_base)
            
            # 验证集：使用原始 CFFA 数据集
            val_data_dir = script_dir / 'dataset' / 'CFFA'
            val_base = CFFADataset(root_dir=str(val_data_dir), split='val', mode='fa2cf')
            self.val_dataset = RealDatasetWrapper(val_base)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, **self.loader_params)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, shuffle=False, **self.loader_params)

# ==========================================
# 模型类: PL_GlueStick_Real
# ==========================================
class PL_XFeat_Real(pl.LightningModule):
    """XFeat 的 PyTorch Lightning 封装（用于真实数据训练）"""
    def __init__(self, config, result_dir=None):
        super().__init__()
        self.config = config
        self.result_dir = result_dir
        self.save_hyperparameters({'config': str(config)})
        
        # 初始化 XFeat 模型
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
        
        # 将 net 切换回 train 模式（XFeat 初始化时强制 .eval()）
        self.xfeat.net.train()
        
        self.force_viz = False
        self.min_cossim = config.XFEAT['min_cossim']
        
        # 预实例化 interpolators，避免每次 forward 都重建
        self._interp_nearest  = InterpolateSparse2d('nearest')
        self._interp_bilinear = InterpolateSparse2d('bilinear')

    def configure_optimizers(self):
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

    def _train_extract_features(self, x, top_k):
        """
        训练专用特征提取：直接调用 self.xfeat.net 保留梯度计算图。
        绕过 detectAndCompute 上的 @torch.inference_mode() 装饰器，
        使 descriptors 具有 grad_fn，从而支持 loss.backward()。
        """
        x, rh, rw = self.xfeat.preprocess_tensor(x)   # resize to 32x multiple
        B, _, _H, _W = x.shape

        # 直接走 net.forward()，有梯度
        M, K, H = self.xfeat.net(x)                   # M: [B,64,H/8,W/8], K: [B,65,...], H: [B,1,...]
        M = torch.nn.functional.normalize(M, dim=1)   # L2-normalize feature maps

        # 关键点热力图（不需要梯度，只用来定位）
        with torch.no_grad():
            K1h = self.xfeat.get_kpts_heatmap(K)
            mkpts = self.xfeat.NMS(K1h, threshold=self.xfeat.detection_threshold, kernel_size=5)

            # 插值可靠性分数（用预实例化的 interpolators）
            scores = (self._interp_nearest(K1h, mkpts, _H, _W)
                      * self._interp_bilinear(H, mkpts, _H, _W)).squeeze(-1)
            scores[torch.all(mkpts == 0, dim=-1)] = -1

            # Top-K 选点
            idxs = torch.argsort(-scores)
            mkpts_x = torch.gather(mkpts[..., 0], -1, idxs)[:, :top_k]
            mkpts_y = torch.gather(mkpts[..., 1], -1, idxs)[:, :top_k]
            mkpts   = torch.cat([mkpts_x[..., None], mkpts_y[..., None]], dim=-1)
            scores  = torch.gather(scores, -1, idxs)[:, :top_k]
            # 恢复原始坐标
            mkpts = mkpts * torch.tensor([rw, rh], device=mkpts.device).view(1, 1, -1)

        # 插值描述符（有梯度，因为 M 是直接从 net.forward() 来的）
        feats = self.xfeat.interpolator(M, mkpts.detach(), H=_H, W=_W)  # [B, top_k, 64]
        feats = torch.nn.functional.normalize(feats, dim=-1)

        valid = scores > 0
        return mkpts, scores, feats, valid

    def forward(self, batch):
        """前向传播 - 训练时直接调用 net 保留梯度，验证/推断时可用 detectAndCompute"""
        top_k = self.config.XFEAT['top_k']
        device = batch['image0'].device
        B = batch['image0'].shape[0]

        if self.training:
            # ---- 训练模式：有梯度的特征提取 ----
            mkpts0, scores0, feats0, valid0 = self._train_extract_features(batch['image0'], top_k)
            mkpts1, scores1, feats1, valid1 = self._train_extract_features(batch['image1'], top_k)

            # 填充至统一长度
            max_kpts0 = mkpts0.shape[1]
            max_kpts1 = mkpts1.shape[1]
            keypoints0   = mkpts0
            keypoints1   = mkpts1
            descriptors0 = feats0   # 保留梯度！
            descriptors1 = feats1   # 保留梯度！
            kp_scores0   = scores0
            kp_scores1   = scores1
        else:
            # ---- 推断模式：使用原始 detectAndCompute（no_grad，效率高）----
            with torch.no_grad():
                feats0_list = self.xfeat.detectAndCompute(batch['image0'], top_k=top_k)
                feats1_list = self.xfeat.detectAndCompute(batch['image1'], top_k=top_k)

            max_kpts0 = max([len(f['keypoints']) for f in feats0_list])
            max_kpts1 = max([len(f['keypoints']) for f in feats1_list])

            keypoints0   = torch.zeros(B, max_kpts0, 2, device=device)
            keypoints1   = torch.zeros(B, max_kpts1, 2, device=device)
            descriptors0 = torch.zeros(B, max_kpts0, 64, device=device)
            descriptors1 = torch.zeros(B, max_kpts1, 64, device=device)
            kp_scores0   = torch.zeros(B, max_kpts0, device=device)
            kp_scores1   = torch.zeros(B, max_kpts1, device=device)

            for b in range(B):
                n0 = len(feats0_list[b]['keypoints'])
                n1 = len(feats1_list[b]['keypoints'])
                keypoints0[b, :n0]   = feats0_list[b]['keypoints']
                keypoints1[b, :n1]   = feats1_list[b]['keypoints']
                descriptors0[b, :n0] = feats0_list[b]['descriptors']
                descriptors1[b, :n1] = feats1_list[b]['descriptors']
                kp_scores0[b, :n0]   = feats0_list[b]['scores']
                kp_scores1[b, :n1]   = feats1_list[b]['scores']

        # 更新 batch（供 loss 和 validation 使用）
        batch.update({
            'keypoints0': keypoints0,
            'keypoints1': keypoints1,
            'descriptors0': descriptors0,
            'descriptors1': descriptors1,
            'keypoint_scores0': kp_scores0,
            'keypoint_scores1': kp_scores1,
        })

        # MNN 匹配（training 时用带梯度的 descriptors 做 match，推断时同理）
        # 注意：batch_match 也有 @inference_mode，用手动实现替代
        desc0_norm = torch.nn.functional.normalize(descriptors0.detach(), dim=-1)
        desc1_norm = torch.nn.functional.normalize(descriptors1.detach(), dim=-1)
        with torch.no_grad():
            cossim_mat = torch.bmm(desc0_norm, desc1_norm.transpose(1, 2))  # [B, M, N]
            match12 = torch.argmax(cossim_mat, dim=-1)       # [B, M]
            match21 = torch.argmax(cossim_mat.permute(0,2,1), dim=-1)  # [B, N]
            idx0_base = torch.arange(match12.shape[1], device=device)

        max_kpts0 = descriptors0.shape[1]
        matches0 = torch.full((B, max_kpts0), -1, dtype=torch.long, device=device)
        for b in range(B):
            mutual = match21[b][match12[b]] == idx0_base
            if self.min_cossim > 0:
                cossim_max = cossim_mat[b].max(dim=1).values
                good = cossim_max > self.min_cossim
                sel = mutual & good
            else:
                sel = mutual
            matches0[b, sel] = match12[b][sel]

        return {
            'matches0': matches0,
            'keypoints0': keypoints0,
            'keypoints1': keypoints1,
            'descriptors0': descriptors0,
            'descriptors1': descriptors1,
        }

    def _compute_gt_matches(self, kpts0, kpts1, T_0to1, dist_th=3.0):
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
        """计算损失 —— 完全向量化，无 Python for 循环，单次 cross_entropy 调用"""
        kpts0    = batch['keypoints0']
        kpts1    = batch['keypoints1']
        T_0to1   = batch['T_0to1']
        desc0    = batch['descriptors0']   # [B, M, 64]
        desc1    = batch['descriptors1']   # [B, N, 64]
        valid_kp = batch['keypoint_scores0'] > 0  # [B, M]

        # ① GT 匹配（no_grad，仅用于监督标签生成）
        with torch.no_grad():
            matches_gt = self._compute_gt_matches(kpts0.detach(), kpts1.detach(), T_0to1)
            # 有效 mask：关键点本身有效 AND 存在 GT 匹配
            gt_match_mask = (matches_gt >= 0) & valid_kp   # [B, M]

        if not gt_match_mask.any():
            # 无有效匹配对时返回有梯度的零损失
            return (desc0 * 0).sum()

        # ② 归一化描述符并计算相似度矩阵 [B, M, N]
        desc0_norm = F.normalize(desc0, dim=-1)
        desc1_norm = F.normalize(desc1, dim=-1)
        cossim = torch.bmm(desc0_norm, desc1_norm.transpose(1, 2))  # [B, M, N]

        # ③ 一次性取出所有有效样本的 logits 和 targets（无 Python 循环）
        b_idx, m_idx = torch.where(gt_match_mask)        # [K], [K]
        targets = matches_gt[b_idx, m_idx]               # [K]  目标列索引
        logits  = cossim[b_idx, m_idx, :]                # [K, N]  整行 logits

        # ④ 一次 cross_entropy 完成所有样本
        loss = F.cross_entropy(logits, targets)
        return loss

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self._compute_loss(outputs, batch)
        
        self.log('train/loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(batch)
        
        loss = self._compute_loss(outputs, batch)
        self.log('val_loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        matches0 = outputs['matches0']
        kpts0 = batch['keypoints0']
        kpts1 = batch['keypoints1']
        
        B = kpts0.shape[0]
        H_ests = []
        
        mkpts0_f_list = []
        mkpts1_f_list = []
        m_bids_list = []
        
        for b in range(B):
            m0 = matches0[b]
            valid = m0 > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[valid]
            
            pts0 = kpts0[b][m_indices_0].cpu().numpy()
            pts1 = kpts1[b][m_indices_1].cpu().numpy()
            
            if len(pts0) > 0:
                mkpts0_f_list.append(torch.from_numpy(pts0).float())
                mkpts1_f_list.append(torch.from_numpy(pts1).float())
                m_bids_list.append(torch.full((len(pts0),), b, dtype=torch.long))
            
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
        
        metrics_batch = {
            'mkpts0_f': torch.cat(mkpts0_f_list, dim=0) if mkpts0_f_list else torch.empty(0, 2),
            'mkpts1_f': torch.cat(mkpts1_f_list, dim=0) if mkpts1_f_list else torch.empty(0, 2),
            'm_bids': torch.cat(m_bids_list, dim=0) if m_bids_list else torch.empty(0, dtype=torch.long),
            'T_0to1': batch['T_0to1'],
            'image0': batch['image0'],
            'dataset_name': batch['dataset_name']
        }
        
        set_metrics_verbose(True)
        compute_homography_errors(metrics_batch, self.config)
        
        # 不在这里计算AUC，而是返回metrics_batch，在callback中累积t_errs后统一计算
        
        return {
            'H_est': H_ests,
            'kpts0': kpts0,
            'kpts1': kpts1,
            'matches0': matches0,
            'metrics_batch': metrics_batch
        }

# ==========================================
# 回调类: MultimodalValidationCallback
# ==========================================
class MultimodalValidationCallback(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.best_val = -1.0
        self.result_dir = Path(f"results/xfeat_{args.mode}/{args.name}")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_mses = []
        self.epoch_maces = []
        self.epoch_t_errs = []  # 累积所有 batch 的 t_errs

        import csv
        self.csv_path = self.result_dir / "metrics.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val MSE", "Val MACE", "Val AUC@5", "Val AUC@10", "Val AUC@20", "Val Combined AUC", "Val Inverse MACE"])
        
        self.current_train_metrics = {}
        self.current_val_metrics = {}

    def _try_write_csv(self, epoch):
        if epoch in self.current_train_metrics and epoch in self.current_val_metrics:
            t = self.current_train_metrics.pop(epoch)
            v = self.current_val_metrics.pop(epoch)
            import csv
            with open(self.csv_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    t.get('loss', ''),
                    v.get('val_loss', ''),
                    v['mse'],
                    v['mace'],
                    v['auc5'],
                    v['auc10'],
                    v['auc20'],
                    v['combined_auc'],
                    v['inverse_mace']
                ])

    def on_validation_epoch_start(self, trainer, pl_module):
        self.epoch_mses = []
        self.epoch_maces = []
        self.epoch_t_errs = []  # 重置 t_errs

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        batch_mses, batch_maces = self._process_batch(trainer, pl_module, batch, outputs, None, save_images=False)
        self.epoch_mses.extend(batch_mses)
        self.epoch_maces.extend(batch_maces)
        # 累积 t_errs
        metrics_batch = outputs.get('metrics_batch', {})
        if 't_errs' in metrics_batch:
            self.epoch_t_errs.extend(metrics_batch['t_errs'])

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        display_metrics = {}
        
        if 'train/loss_epoch' in metrics:
            display_metrics['loss'] = metrics['train/loss_epoch'].item()
        
        if display_metrics:
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
            logger.info(f"Epoch {epoch} 训练总结 >> {metric_str}")
        
        self.current_train_metrics[epoch] = display_metrics
        self._try_write_csv(epoch)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.epoch_mses:
            return
        
        avg_mse = sum(self.epoch_mses) / len(self.epoch_mses)
        avg_mace = sum(self.epoch_maces) / len(self.epoch_maces) if self.epoch_maces else float('inf')
        
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        
        display_metrics = {'mse': avg_mse, 'mace': avg_mace}
        
        # 在 epoch 结束时一次性计算 AUC（全局计算，不是batch平均）
        from scripts.v1.metrics import error_auc
        if len(self.epoch_t_errs) > 0:
            auc_dict = error_auc(self.epoch_t_errs, [5, 10, 20])
            for k, v in auc_dict.items():
                display_metrics[k] = v
        else:
            for k in ['auc@5', 'auc@10', 'auc@20']:
                display_metrics[k] = 0.0
        
        if 'val_loss' in metrics:
            display_metrics['val_loss'] = metrics['val_loss'].item()
        
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in display_metrics.items()])
        
        auc5 = display_metrics.get('auc@5', 0.0)
        auc10 = display_metrics.get('auc@10', 0.0)
        auc20 = display_metrics.get('auc@20', 0.0)
        combined_auc = (auc5 + auc10 + auc20) / 3.0
        
        inverse_mace = 1.0 / (1.0 + avg_mace)
        
        pl_module.log("val_mse", avg_mse, on_epoch=True, prog_bar=False, logger=True)
        pl_module.log("val_mace", avg_mace, on_epoch=True, prog_bar=False, logger=True)
        pl_module.log("combined_auc", combined_auc, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("inverse_mace", inverse_mace, on_epoch=True, prog_bar=False, logger=True)
        
        logger.info(f"Epoch {epoch} 验证总结 >> {metric_str} | combined_auc: {combined_auc:.4f}")
        
        self.current_val_metrics[epoch] = {
            'mse': avg_mse,
            'mace': avg_mace,
            'auc5': auc5,
            'auc10': auc10,
            'auc20': auc20,
            'combined_auc': combined_auc,
            'inverse_mace': inverse_mace,
            'val_loss': display_metrics.get('val_loss', 0.0)
        }
        self._try_write_csv(epoch)
        
        latest_path = self.result_dir / "latest_checkpoint"
        latest_path.mkdir(exist_ok=True)
        trainer.save_checkpoint(latest_path / "model.ckpt")
            
        is_best = False
        if combined_auc > self.best_val:
            self.best_val = combined_auc
            is_best = True
            best_path = self.result_dir / "best_checkpoint"
            best_path.mkdir(exist_ok=True)
            trainer.save_checkpoint(best_path / "model.ckpt")
            with open(best_path / "log.txt", "w") as f:
                f.write(f"Epoch: {epoch}\nBest Combined AUC: {combined_auc:.4f}\n")
                f.write(f"AUC@5: {auc5:.4f}\nAUC@10: {auc10:.4f}\nAUC@20: {auc20:.4f}\n")
                f.write(f"MACE: {avg_mace:.4f}\nMSE: {avg_mse:.6f}\n")
            logger.info(f"发现新的最优模型! Epoch {epoch}, Combined AUC: {combined_auc:.4f}")

        if is_best or (epoch % 5 == 0):
            self._trigger_visualization(trainer, pl_module, is_best, epoch)

    def _trigger_visualization(self, trainer, pl_module, is_best, epoch):
        pl_module.force_viz = True
        target_dir = self.result_dir / (f"epoch{epoch}_best" if is_best else f"epoch{epoch}")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        val_dataloader = trainer.val_dataloaders[0] if isinstance(trainer.val_dataloaders, list) else trainer.val_dataloaders
        pl_module.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if batch_idx > 5:
                    break
                batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = pl_module.validation_step(batch, batch_idx)
                self._process_batch(trainer, pl_module, batch, outputs, target_dir, save_images=True)
        pl_module.force_viz = False

    def _process_batch(self, trainer, pl_module, batch, outputs, epoch_dir, save_images=False):
        batch_size = batch['image0'].shape[0]
        mses, maces = [], []
        H_ests = outputs.get('H_est', [np.eye(3)] * batch_size)
        Ts_gt = batch['T_0to1'].cpu().numpy()
        
        rejected_count = 0
        
        for i in range(batch_size):
            H_est = H_ests[i]
            
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
                sample_name = f"{Path(batch['pair_names'][0][i]).stem}_vs_{Path(batch['pair_names'][1][i]).stem}"
                save_path = epoch_dir / sample_name
                save_path.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path / "fix.png"), img0)
                cv2.imwrite(str(save_path / "moving_result.png"), img1_result)
                
                img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
                img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
                
                if 'kpts0' in outputs and 'kpts1' in outputs:
                    kpts0_np = outputs['kpts0'][i].cpu().numpy()
                    kpts1_np = outputs['kpts1'][i].cpu().numpy()
                    
                    # 过滤填充点（分数为0的点）
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
                    
                    # 绘制线段（如果存在）
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
                        
                        # 绘制有效线段（黄色）
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
                    
                    if 'matches0' in outputs:
                        m0 = outputs['matches0'][i].cpu()
                        valid = m0 > -1
                        m_indices_0 = torch.where(valid)[0].numpy()
                        m_indices_1 = m0[valid].numpy()
                        
                        # 绘制匹配点（红色），同时过滤无效点
                        for idx0 in m_indices_0:
                            if valid_mask0[idx0]:  # 只画有效点
                                pt = kpts0_np[idx0]
                                cv2.circle(img0_color, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                        for idx1 in m_indices_1:
                            if valid_mask1[idx1]:  # 只画有效点
                                pt = kpts1_np[idx1]
                                cv2.circle(img1_color, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                        
                        try:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor='black')
                            fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
                            
                            ax1.imshow(img0, cmap='gray')
                            ax1.axis('off')
                            ax2.imshow(img1, cmap='gray')
                            ax2.axis('off')
                            
                            # 绘制匹配线
                            if len(m_indices_0) > 0:
                                from matplotlib.patches import ConnectionPatch
                                for idx0, idx1 in zip(m_indices_0, m_indices_1):
                                    # 只画有效点的连线
                                    if valid_mask0[idx0] and valid_mask1[idx1]:
                                        pt0 = kpts0_np[idx0]
                                        pt1 = kpts1_np[idx1]
                                        
                                        # 使用 ConnectionPatch 正确绘制两个子图之间的连线
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
        
        if rejected_count > 0 and save_images:
            logger.info(f"防爆锁触发: {rejected_count}/{batch_size}")
        
        return mses, maces

# ==========================================
# 早停机制
# ==========================================
class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch
    
    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module)

# ==========================================
# 主函数
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="XFeat CFFA Real-Data Training")
    parser.add_argument('--name', '-n', type=str, default='xfeat_cffa_baseline', help='训练名称')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--start_point', type=str, default=None, help='从检查点恢复')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--gpus', type=str, default='1')
    return parser.parse_args()

def main():
    args = parse_args()
    args.mode = 'cffa'
    
    config = get_default_config()
    config.TRAINER.SEED = 66
    pl.seed_everything(config.TRAINER.SEED)
    
    result_dir = Path(f"results/xfeat_{args.mode}/{args.name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    log_file = result_dir / "log.txt"
    
    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sys.stderr, format=log_format, level="INFO")
    logger.add(log_file, format=log_format, level="INFO", mode="a", backtrace=True, diagnose=False)
    logger.info(f"日志将同时保存到: {log_file}")
    
    os.environ['LOFTR_LOG_FILE'] = str(log_file)
    
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
    
    model = PL_XFeat_Real(config, result_dir=str(result_dir))
    data_module = MultimodalDataModule(args, config)
    
    tb_logger = TensorBoardLogger(save_dir='logs/tb_logs', name=f"xfeat_{args.name}")
    
    early_stop_callback = DelayedEarlyStopping(
        start_epoch=0,
        monitor='combined_auc',
        mode='max',
        patience=10,
        min_delta=0.0001
    )
    
    logger.info("早停配置: monitor=combined_auc, start_epoch=0, patience=10, min_delta=0.0001")
    
    if not hasattr(args, 'mode'):
        args.mode = 'cffa'
    
    logger.info(f"GPU 配置: devices={gpus_list}, num_gpus={_n_gpus}")
    logger.info(f"学习率: {config.TRAINER.TRUE_LR:.6f} (scaled from {config.TRAINER.CANONICAL_LR})")
    
    trainer_kwargs = {
        'max_epochs': args.max_epochs,
        'accelerator': "gpu" if torch.cuda.is_available() else "cpu",
        'devices': gpus_list,
        'num_sanity_val_steps': 0,
        'check_val_every_n_epoch': 1,
        'callbacks': [
            MultimodalValidationCallback(args), 
            LearningRateMonitor(logging_interval='step'), 
            early_stop_callback
        ],
        'logger': tb_logger,
    }
    
    if _n_gpus > 1:
        trainer_kwargs['strategy'] = DDPStrategy(find_unused_parameters=False)
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    ckpt_path = args.start_point if args.start_point else None
    
    logger.info(f"开始 XFeat 真实数据训练 (训练集: operation_pre_filtered_cffa | 验证集: CFFA): {args.name}")
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()

