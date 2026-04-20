import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float, color, metrics
from tqdm import tqdm
import os

# ==========================================
# 1. 定义一个极简的 U-Net 网络结构
# (DIP 极度依赖网络结构本身的先验知识)
# ==========================================
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        # 编码器 (下采样)
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        
        # 解码器 (上采样)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(64 + 32, 32)
        
        # 输出层
        self.out = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid() # 将输出限制在 0-1 之间
        )

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

# ==========================================
# 2. 数据准备与加噪
# ==========================================
# 使用一张 256x256 的图像以加快运行速度
clean_img = data.camera()
clean_img = img_as_float(clean_img)

# 作业要求测试: σ=15, 25, 35, 50，这里先拿 25 测试
sigma = 25 / 255.0  
noisy_img = clean_img + np.random.normal(0, sigma, clean_img.shape)
noisy_img = np.clip(noisy_img, 0, 1)

# 转换为 PyTorch Tensor (1, 1, H, W)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noisy_tensor = torch.from_numpy(noisy_img).float().unsqueeze(0).unsqueeze(0).to(device)
clean_tensor = torch.from_numpy(clean_img).float().unsqueeze(0).unsqueeze(0).to(device)

print(f"使用的设备: {device}")
print(f"初始带噪图像 PSNR: {metrics.peak_signal_noise_ratio(clean_img, noisy_img, data_range=1.0):.2f} dB")

# ==========================================
# 3. DIP 核心逻辑与早停法 (Early Stopping)
# ==========================================
net = SimpleUNet().to(device)
mse_loss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# 【核心机制】生成一张固定的随机噪声作为网络输入
z = torch.rand(1, 1, clean_img.shape[0], clean_img.shape[1]).to(device)

num_iter = 1500 # 迭代总次数
best_psnr = 0
best_img = None
best_iter = 0

# 记录 PSNR 变化用于画图
psnr_history = []

print("\n开始 DIP 训练 (请注意观察 PSNR 的变化趋势)...")
for i in tqdm(range(num_iter)):
    optimizer.zero_grad()
    
    # 网络输入固定噪声 z，尝试输出重建图像
    out = net(z) 
    
    # Loss 仅仅是重建图像与【带噪图像】的差异！(没有使用干净图像)
    loss = mse_loss(out, noisy_tensor) 
    loss.backward()
    optimizer.step()
    
    # 每 10 步计算一次真实的 PSNR（仅用于我们观察早停现象，实际去噪中你没有干净图像）
    if i % 10 == 0:
        out_np = out.detach().cpu().squeeze().numpy()
        current_psnr = metrics.peak_signal_noise_ratio(clean_img, out_np, data_range=1.0)
        psnr_history.append(current_psnr)
        
        # 记录最佳状态 (模拟早停)
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_img = out_np.copy()
            best_iter = i

print(f"\n训练完成！最佳迭代次数: {best_iter}, 最佳 PSNR: {best_psnr:.2f} dB")

# ==========================================
# 4. 结果可视化与保存
# ==========================================
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(clean_img, cmap='gray')
axes[0].set_title('Clean Image')
axes[1].imshow(noisy_img, cmap='gray')
axes[1].set_title(f'Noisy Image (sigma=25)')
axes[2].imshow(best_img, cmap='gray')
axes[2].set_title(f'DIP Result (Iter: {best_iter})')

# 画出 PSNR 变化曲线，完美展示早停法原理
axes[3].plot(range(0, num_iter, 10), psnr_history, color='blue')
axes[3].axvline(x=best_iter, color='red', linestyle='--', label=f'Early Stop at {best_iter}')
axes[3].set_title('PSNR vs. Iterations')
axes[3].set_xlabel('Iterations')
axes[3].set_ylabel('PSNR (dB)')
axes[3].legend()

for ax in axes[:3]:
    ax.axis('off')

# 保存结果
os.makedirs('results', exist_ok=True)
save_path = 'results/dip_result.png'
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"\n结果已保存至: {os.path.abspath(save_path)}")