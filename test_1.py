import torch
import numpy as np

def calculate_hd95(tensor1, tensor2, value, tolerance=1e-3):
    """
    计算两张图片中指定值 (类别) 的 HD95 距离，适用于浮点数值。
    参数：
    - tensor1, tensor2: 两张图片的 PyTorch Tensor。
    - value: 要计算的类别值 (如 0.333 或 0.666)。
    - tolerance: 浮点数比较的容差范围，默认为 1e-3。
    返回：
    - hd95: 对应类别的 HD95 距离。
    """
    # 提取指定值 (前景) 的坐标（允许一定的容忍误差）
    points1 = torch.nonzero((tensor1 >= value - tolerance) & (tensor1 <= value + tolerance), as_tuple=False).numpy()
    points2 = torch.nonzero((tensor2 >= value - tolerance) & (tensor2 <= value + tolerance), as_tuple=False).numpy()

    if len(points1) == 0 or len(points2) == 0:
        return float('inf')  # 如果某类没有前景点，则返回无限大

    # 计算点对的距离
    distances = []
    for p1 in points1:
        distances.append(np.min([np.linalg.norm(p1 - p2) for p2 in points2]))
    for p2 in points2:
        distances.append(np.min([np.linalg.norm(p2 - p1) for p1 in points1]))
    
    # 计算 95% 分位数距离
    hd95 = np.percentile(distances, 95)
    return hd95

# 示例：假设 tensor1 和 tensor2 是您的 PyTorch Tensor
tensor1 = torch.tensor([
    [0.000, 0.333, 0.333, 0.000],
    [0.666, 0.666, 0.000, 0.333],
    [0.666, 0.000, 0.000, 0.333],
    [0.000, 0.000, 0.333, 0.000],
])

tensor2 = torch.tensor([
    [0.000, 0.333, 0.333, 0.000],
    [0.666, 0.000, 0.000, 0.333],
    [0.666, 0.666, 0.000, 0.333],
    [0.000, 0.000, 0.333, 0.000],
])

# 计算 HD95
hd95_0_666 = calculate_hd95(tensor1, tensor2, value=0.666, tolerance=1e-3)
hd95_0_333 = calculate_hd95(tensor1, tensor2, value=0.333, tolerance=1e-3)

print(f"HD95 for 0.666: {hd95_0_666}")
print(f"HD95 for 0.333: {hd95_0_333}")