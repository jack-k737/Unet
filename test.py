import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion


class BoundaryLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(BoundaryLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def extract_boundary_mask(self, mask):
        """
        通过膨胀和腐蚀操作生成边界掩码。
        """
        mask_np = mask.cpu().numpy()
        dilated = binary_dilation(mask_np, iterations=2)
        eroded = binary_erosion(mask_np, iterations=2)
        boundary = torch.tensor(dilated ^ eroded, dtype=torch.float32, device=mask.device)
        return boundary

    def extract_features(self, features, mask):
        """
        使用掩码对特征进行平均池化，提取区域级别特征。
        """
        masked_features = features * mask.unsqueeze(1)
        feature_sum = masked_features.sum(dim=(2, 3))
        mask_sum = mask.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        return feature_sum / mask_sum

    def forward(self, features, target):
        """
        Args:
            features: Tensor of shape (B, C, H, W) — 提取的特征图。
            target: Tensor of shape (B, 1, H, W) — 分割标签。

        Returns:
            loss: Boundary loss value (scalar)
        """
        B, C, H, W = features.shape

        # 提取边界掩码和背景掩码
        boundary_mask = self.extract_boundary_mask(target.squeeze(1))
        background_mask = (1 - target).squeeze(1) * boundary_mask

        # 提取边界和背景特征
        boundary_features = self.extract_features(features, boundary_mask)
        background_features = self.extract_features(features, background_mask)

        # 计算 InfoNCE 损失
        pos_similarity = self.cosine_similarity(boundary_features, boundary_features)
        neg_similarity = self.cosine_similarity(boundary_features, background_features)

        pos_loss = -torch.log(torch.exp(pos_similarity / self.temperature))
        neg_loss = -torch.log(1 - torch.exp(neg_similarity / self.temperature))

        loss = (pos_loss + neg_loss).mean()

        return loss

import PIL.Image as Image
import torchvision

# 示例使用
if __name__ == "__main__":
    # 随机生成特征图和标签
    features = Image.open('ODOC/Domain1/test/imgs/gdrishtiGS_001.png')
    features = torchvision.transforms.ToTensor()(features).unsqueeze(0)
    target = Image.open('ODOC/Domain1/test/mask/gdrishtiGS_001.png')
    target = torchvision.transforms.ToTensor()(target)
    target = torchvision.transforms.Grayscale()(target).unsqueeze(0)


    # 创建 BoundaryLoss 实例
    criterion = BoundaryLoss(temperature=0.05)

    # 计算边界损失
    loss = criterion(features, target)
    print(f"Boundary Loss: {loss.item()}")
