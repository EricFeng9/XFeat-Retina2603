"""
随机多强度域随机化增强模块 (Random Multi-Strength Domain Randomization)

改进策略:
1. 每次随机应用轻度/中度/强度三种增强级别之一
2. 根据训练进度动态调整采样概率（前期偏轻，后期偏重）
3. 同一batch内混合不同难度样本，提高泛化能力
4. 避免同时应用所有变换
"""

import torch
import torch.nn.functional as F
import numpy as np
import random

class RandomMultiStrengthAugmentation:
    """随机多强度域随机化"""
    
    def __init__(self, total_epochs=200):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        """设置当前训练轮次"""
        self.current_epoch = epoch
        
    def get_strength_probabilities(self):
        """
        根据训练进度计算三种强度的采样概率
        返回: (light_prob, medium_prob, heavy_prob)
        """
        progress = self.current_epoch / self.total_epochs
        
        if progress < 0.3:
            # 前30% epoch: 主要轻度 (70%), 少量中度 (25%), 极少强度 (5%)
            return (0.70, 0.25, 0.05)
        elif progress < 0.6:
            # 30-60% epoch: 均衡分布 (40%, 40%, 20%)
            return (0.40, 0.40, 0.20)
        else:
            # 后40% epoch: 偏向强度 (20%, 40%, 40%)
            return (0.20, 0.40, 0.40)
    
    def sample_augmentation_strength(self):
        """
        随机采样增强强度级别
        返回: 'light', 'medium', 或 'heavy'
        """
        probs = self.get_strength_probabilities()
        choice = random.choices(['light', 'medium', 'heavy'], weights=probs)[0]
        return choice
    
    def get_current_avg_strength(self):
        """获取当前平均增强强度 (用于日志监控)"""
        probs = self.get_strength_probabilities()
        # 轻度=0.2, 中度=0.5, 强度=0.8
        return probs[0] * 0.2 + probs[1] * 0.5 + probs[2] * 0.8
    
    def augment_image(self, image):
        """随机多强度域随机化增强"""
        # 随机选择增强级别
        strength_level = self.sample_augmentation_strength()
        
        # 根据级别设置具体的增强参数
        if strength_level == 'light':
            strength = random.uniform(0.1, 0.3)
        elif strength_level == 'medium':
            strength = random.uniform(0.4, 0.6)
        else:  # heavy
            strength = random.uniform(0.7, 0.9)
        
        # 确保是 tensor 格式 [B, C, H, W], 范围 [0, 1]
        is_numpy = isinstance(image, np.ndarray)
        if is_numpy:
            if image.ndim == 2:
                img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
            else:
                img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
            if img_tensor.max() > 2.0:
                img_tensor = img_tensor / 255.0
        else:
            img_tensor = image.clone().float()
            if img_tensor.ndim == 2:
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
            elif img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)
        
        original_shape = img_tensor.shape
        B, C, H, W = img_tensor.shape
        
        # 对batch中的每张图独立应用随机增强
        for b in range(B):
            img_b = img_tensor[b:b+1]
            
            # 为当前图像随机采样增强强度
            img_strength_level = self.sample_augmentation_strength()
            if img_strength_level == 'light':
                img_strength = random.uniform(0.1, 0.3)
            elif img_strength_level == 'medium':
                img_strength = random.uniform(0.4, 0.6)
            else:  # heavy
                img_strength = random.uniform(0.7, 0.9)
            
            # === 应用增强变换（基于采样的强度） ===
            
            # 1. 对比度调整 (始终应用，强度随机)
            if random.random() < 0.6:
                contrast_range = 1.0 + img_strength * 0.5  # light: 1.05-1.15, heavy: 1.35-1.45
                contrast = random.uniform(1.0 / contrast_range, contrast_range)
                mean_val = img_b.mean()
                img_b = (img_b - mean_val) * contrast + mean_val
                img_b = img_b.clamp(0, 1)
            
            # 2. 亮度调整
            if random.random() < 0.4:
                brightness_range = 0.05 + img_strength * 0.1  # 0.05 -> 0.15
                brightness_shift = random.uniform(-brightness_range, brightness_range)
                img_b = (img_b + brightness_shift).clamp(0, 1)
            
            # 3. Gamma 变换 (中度及以上)
            if img_strength >= 0.35 and random.random() < 0.4:
                gamma_range = 1.2 + img_strength * 0.5  # 1.2 -> 1.7
                gamma = random.uniform(1.0 / gamma_range, gamma_range)
                img_b = torch.pow(img_b.clamp(1e-6, 1.0), gamma)
            
            # 4. 下采样 (中度及以上，概率和倍数随强度增加)
            if img_strength >= 0.4 and random.random() < 0.2:
                # 下采样倍数: light不触发, medium: 2-3, heavy: 2-5
                if img_strength < 0.6:
                    max_scale = 3
                else:
                    max_scale = 5
                h_scale = random.randint(2, max_scale)
                w_scale = random.randint(2, max_scale)
                low_res = F.interpolate(img_b, size=(H // h_scale, W // w_scale), 
                                      mode='bilinear', align_corners=False)
                img_b = F.interpolate(low_res, size=(H, W), mode='bilinear', align_corners=False)
            
            # 5. 图像反色 (只有强度级别才触发，且概率很低)
            if img_strength >= 0.7 and random.random() < 0.12:
                img_b = 1.0 - img_b
            
            # 6. 不均匀噪声 (强度及以上)
            if img_strength >= 0.5 and random.random() < 0.35:
                noise_std = 0.01 + 0.025 * (img_strength - 0.5) / 0.5  # 0.01 -> 0.035
                noise_map = torch.rand(1, 1, 4, 4, device=img_b.device)
                noise_map = torch.where(noise_map > 0.8, noise_map, torch.zeros_like(noise_map))
                noise_map = F.interpolate(noise_map, size=(H, W), mode='bicubic', align_corners=False)
                noise = torch.randn_like(img_b) * (noise_map * noise_std)
                img_b = (img_b + noise).clamp(0, 1)
            
            # 7. 全局噪声 (少量)
            if random.random() < 0.08:
                noise = torch.randn_like(img_b) * random.uniform(0.002, 0.008)
                img_b = (img_b + noise).clamp(0, 1)
            
            # 动态范围保护
            img_min = img_b.min()
            img_max = img_b.max()
            dynamic_range = img_max - img_min
            
            if dynamic_range < 0.2:
                img_b = (img_b - img_min) / (dynamic_range + 1e-6)
                img_b = img_b * 0.8 + 0.1
            
            img_tensor[b:b+1] = img_b.clamp(0, 1)
        
        # 转换回原格式
        if is_numpy:
            img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0
            if image.ndim == 2:
                img_np = img_np[:, :, 0]
            return img_np.clip(0, 255).astype(np.uint8)
        else:
            return img_tensor.view(original_shape)


# 全局实例
_progressive_augmenter = RandomMultiStrengthAugmentation(total_epochs=200)

def set_augmentation_epoch(epoch):
    """设置当前训练轮次"""
    global _progressive_augmenter
    _progressive_augmenter.set_epoch(epoch)

def apply_progressive_augmentation(image):
    """应用随机多强度域随机化"""
    global _progressive_augmenter
    return _progressive_augmenter.augment_image(image)

def get_current_augmentation_strength():
    """获取当前平均增强强度 (用于日志)"""
    global _progressive_augmenter
    return _progressive_augmenter.get_current_avg_strength()

def get_strength_probabilities():
    """获取当前三种强度的采样概率 (用于日志)"""
    global _progressive_augmenter
    return _progressive_augmenter.get_strength_probabilities()
